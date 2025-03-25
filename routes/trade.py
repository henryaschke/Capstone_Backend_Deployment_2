from fastapi import APIRouter, Query, HTTPException, Depends, Path, Body
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta, timezone
import uuid
import math
import pytz
from google.cloud import bigquery

from models.trade import TradeRequest
from dependencies import get_optional_user, get_current_user, DEFAULT_USER_ID, parse_date_string
from database import (
    create_trade, get_user_trades, get_battery_status, create_battery_if_not_exists,
    get_trade_by_id, update_trade_status, update_battery_level, get_pending_trades,
    get_market_data_today, update_portfolio_balance, get_portfolio_by_user_id,
    get_db, PROJECT_ID, DATASET_ID
)

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Define timezone offset for CET/CEST (UTC+1/UTC+2)
# You should adjust this based on your specific needs
# For CET (winter time): timedelta(hours=1)
# For CEST (summer time): timedelta(hours=2)
CET_OFFSET = timedelta(hours=1)

# Define Central European Time timezone
CET = pytz.timezone('Europe/Berlin')

def to_cet(dt):
    """Convert a datetime to CET timezone"""
    if dt.tzinfo is None:
        # If naive datetime, assume it's UTC
        return dt.replace(tzinfo=timezone.utc).astimezone(CET)
    # If it has timezone info, properly convert it to CET
    return dt.astimezone(CET)  # Use pytz for proper DST handling

def from_cet(dt):
    """Convert a CET datetime to UTC"""
    if dt.tzinfo is None:
        # If naive datetime, assume it's CET and convert to UTC
        cet = timezone(CET_OFFSET)
        dt = dt.replace(tzinfo=cet)
    return dt.astimezone(timezone.utc)

def normalize_datetime(dt):
    """Normalize datetime to naive UTC"""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
        dt = dt.replace(tzinfo=None)
    return dt

@router.post("/execute")
async def execute_trade(
    request: TradeRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    test_mode: bool = Query(False, description="Enable test mode to bypass time validation")
):
    """Execute a trade (buy or sell) for the German energy market."""
    try:
        # Extract user_id from the current_user dictionary
        user_id = current_user.get("User_ID")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid user authentication")
            
        logger.info(f"Executing trade for authenticated user_id: {user_id}")
        
        # Check if the execution time is in the future
        # Convert the execution time to a timezone-aware datetime in UTC
        execution_time = datetime.fromisoformat(request.executionTime.replace('Z', '+00:00'))
        
        # Log the execution time with timezone info for debugging
        logger.info(f"Original execution time: {request.executionTime}")
        logger.info(f"Parsed execution time: {execution_time}")
        logger.info(f"Execution time in CET: {to_cet(execution_time)}")
        
        # Current time in UTC with timezone info
        now_utc = datetime.now(timezone.utc)
        logger.info(f"Current time UTC: {now_utc}")
        logger.info(f"Current time CET: {to_cet(now_utc)}")
        
        # Compare time - make sure both are in same timezone (UTC)
        if not test_mode and execution_time < now_utc + timedelta(minutes=5):
            raise HTTPException(
                status_code=400,
                detail="Execution time must be at least 5 minutes in the future"
            )
        
        # Check if the resolution is valid
        if request.resolution not in [15, 30, 60]:
            raise HTTPException(
                status_code=400,
                detail="Resolution must be 15, 30, or 60 minutes"
            )
        
        # Check if the executionTime minutes align with the market intervals (resolution)
        execution_minute = execution_time.minute
        
        if request.resolution == 15:
            # For 15-minute resolution, valid minutes are 0, 15, 30, 45
            valid_minutes = [0, 15, 30, 45]
            if execution_minute not in valid_minutes:
                raise HTTPException(
                    status_code=400,
                    detail=f"For 15-minute resolution, execution time minutes must be one of {valid_minutes}"
                )
        elif request.resolution == 30:
            # For 30-minute resolution, valid minutes are 0, 30
            valid_minutes = [0, 30]
            if execution_minute not in valid_minutes:
                raise HTTPException(
                    status_code=400,
                    detail=f"For 30-minute resolution, execution time minutes must be one of {valid_minutes}"
                )
        elif request.resolution == 60:
            # For 60-minute resolution, valid minutes are 0
            if execution_minute != 0:
                raise HTTPException(
                    status_code=400,
                    detail="For 60-minute resolution, execution time minutes must be 0"
                )
        
        # Check battery capacity for the user
        battery = get_battery_status(user_id)
        
        # If battery doesn't exist, create a new one
        if not battery:
            logger.info(f"No battery found for user {user_id}. Creating a new one.")
            battery = create_battery_if_not_exists(user_id)
            if not battery:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to create battery for user"
                )
        
        # Handle different key naming conventions in the battery data
        # The database might return Total_Capacity, while our code uses total_capacity
        current_level = battery.get("current_level", battery.get("Current_Level", 0))
        total_capacity = battery.get("total_capacity", battery.get("Total_Capacity", 2.5))
        
        # If current_level is represented as a percentage (0-100), convert to absolute MWh
        if current_level > 0 and current_level <= 100 and total_capacity > 0:
            # If current_level seems to be a percentage and total_capacity is in absolute units
            absolute_current_level = (current_level / 100) * total_capacity
            logger.info(f"Converting percentage {current_level}% to absolute: {absolute_current_level} MWh")
            current_level = absolute_current_level
        
        # Validate based on trade type
        if request.type.lower() == "buy":
            # For buy, check if there's enough remaining capacity
            remaining_capacity = total_capacity - current_level
            if request.quantity > remaining_capacity:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient battery capacity. Maximum purchasable: {remaining_capacity} MWh"
                )
        elif request.type.lower() == "sell":
            # For sell, check if there's enough energy in the battery
            if request.quantity > current_level:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient energy in battery. Maximum sellable: {current_level} MWh"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Trade type must be 'buy' or 'sell'"
            )
        
        # Generate a trade ID if not provided
        if request.trade_id:
            try:
                trade_id = int(request.trade_id)
            except (ValueError, TypeError):
                trade_id = int(datetime.now().timestamp() * 1000)  # Use milliseconds since epoch as integer ID
        else:
            # Generate integer trade ID using timestamp
            trade_id = int(datetime.now().timestamp() * 1000)  # Use milliseconds since epoch
        
        # Create trade record with 'pending' status
        trade_data = {
            "Trade_ID": trade_id,  # Now an integer instead of UUID string
            "User_ID": user_id,
            "Market": "Germany",
            "Trade_Type": request.type.lower(),
            "Quantity": request.quantity,
            "Timestamp": execution_time,
            "Resolution": request.resolution,
            "Status": "pending"
        }
        
        success = create_trade(trade_data)
        
        if success:
            return {
                "success": True,
                "trade_id": trade_id,
                "message": f"{request.type.capitalize()} trade scheduled for {request.quantity} MWh at {request.executionTime}"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to create trade record"
            )
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_trade_history_api(
    start_date: str = Query(None),
    end_date: str = Query(None),
    trade_type: str = Query(None),
    status: str = Query(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get trade history (optionally by date range, trade_type, or status)."""
    try:
        logger.info(f"Trade history request: start_date={start_date}, end_date={end_date}, trade_type={trade_type}, status={status}")
        start_time = datetime.now()
        
        # Extract user_id from the current_user dictionary
        user_id = current_user.get("User_ID")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid user authentication")
            
        logger.info(f"Getting trades for authenticated user_id: {user_id}")
        
        start_date_obj = parse_date_string(start_date)
        end_date_obj = parse_date_string(end_date)
        
        # Adjust end_date to include the full day (23:59:59)
        if end_date_obj and end_date_obj.hour == 0 and end_date_obj.minute == 0 and end_date_obj.second == 0:
            logger.info(f"Adjusting end_date from {end_date_obj} to include full day")
            end_date_obj = end_date_obj.replace(hour=23, minute=59, second=59)
            logger.info(f"Adjusted end_date to {end_date_obj}")
        
        logger.info(f"Calling get_user_trades with user_id={user_id}, start_date={start_date_obj}, end_date={end_date_obj}")
        
        # Track time for database query
        query_start = datetime.now()
        trades = get_user_trades(user_id, start_date_obj, end_date_obj, cache_bypass=True)
        query_time = (datetime.now() - query_start).total_seconds()
        logger.info(f"get_user_trades completed in {query_time:.2f} seconds, returned {len(trades)} trades")
        
        # Filter by trade_type if provided
        if trade_type:
            logger.info(f"Filtering by trade_type: {trade_type}")
            trades = [t for t in trades if t.get("trade_type", "").lower() == trade_type.lower()]
            logger.info(f"After trade_type filter: {len(trades)} trades")
        
        # Filter by status if provided
        if status:
            logger.info(f"Filtering by status: {status}")
            
            # Add detailed logging of each trade's status
            logger.info("Status values in all trades before filtering:")
            for index, trade in enumerate(trades):
                status_val = trade.get("status", trade.get("Status", "UNKNOWN"))
                logger.info(f"Trade {index}: ID={trade.get('Trade_ID', 'N/A')}, Status={status_val}, Type={type(status_val)}")
            
            trades = [t for t in trades if t.get("status", "").lower() == status.lower() or t.get("Status", "").lower() == status.lower()]
            logger.info(f"After status filter: {len(trades)} trades")
        
        # Format the trades for the API response
        format_start = datetime.now()
        result = []
        for t in trades:
            # Extract field values with fallbacks for different case conventions (database returned keys)
            trade_id = t.get("trade_id", t.get("Trade_ID", ""))
            trade_type = t.get("trade_type", t.get("Trade_Type", ""))
            quantity = t.get("quantity", t.get("Quantity", 0))
            
            # Timestamp handling
            timestamp = t.get("timestamp", t.get("Timestamp", datetime.now()))
            
            # If timestamp is a datetime, ensure it's properly converted to CET before serialization
            if isinstance(timestamp, datetime):
                # If timestamp has no timezone info, assume it's UTC
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                # Convert to CET
                timestamp_cet = timestamp.astimezone(CET)
                # Format as ISO string with timezone info
                timestamp_iso = timestamp_cet.isoformat()
                logger.info(f"Converted timestamp from {timestamp} to CET: {timestamp_cet}")
            else:
                # It's already a string, leave as is
                timestamp_iso = timestamp
                logger.info(f"Using timestamp string as is: {timestamp_iso}")
            
            # Resolution
            resolution = t.get("resolution", t.get("Resolution", 15))
            
            # Status
            status = t.get("status", t.get("Status", "pending"))
            
            # Market
            market = t.get("market", t.get("Market", "Germany"))
            
            # Created at
            created_at = t.get("created_at", t.get("Created_At", datetime.now()))
            created_at_iso = created_at.isoformat() if isinstance(created_at, datetime) else created_at
            
            # Trade price
            trade_price = t.get("trade_price", t.get("Trade_Price", 0))
            
            logger.info(f"Processing trade: ID={trade_id}, Type={trade_type}, Quantity={quantity}, Price={trade_price}")
            
            # Return field names that match the BigQuery schema to ensure the frontend can find them
            result.append({
                "Trade_ID": trade_id,  # Keep the original field names
                "Trade_Type": trade_type,
                "Quantity": float(quantity) if quantity is not None else 0.0,
                "Timestamp": timestamp_iso,
                "Resolution": resolution,
                "Status": status,
                "Market": market,
                "Created_At": created_at_iso,
                "Trade_Price": float(trade_price) if trade_price is not None else 0.0,
                
                # Also include lowercase versions for backward compatibility
                "trade_id": trade_id,
                "type": trade_type.lower() if isinstance(trade_type, str) else trade_type,
                "quantity": float(quantity) if quantity is not None else 0.0,
                "timestamp": timestamp_iso,
                "resolution": resolution,
                "status": status.lower() if isinstance(status, str) else status,
                "market": market,
                "created_at": created_at_iso,
                "trade_price": float(trade_price) if trade_price is not None else 0.0
            })
        format_time = (datetime.now() - format_start).total_seconds()
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Trade history API completed in {total_time:.2f} seconds. Query: {query_time:.2f}s, Format: {format_time:.2f}s")
        logger.info(f"Returning {len(result)} trades")
        
        return result
    except Exception as e:
        logger.error(f"Error getting trade history: {e}")
        # Log the full stack trace for better debugging
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/execute-pending/{trade_id}")
async def execute_pending_trade(
    trade_id: int = Path(..., description="The ID of the trade to execute"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Execute a specific pending trade by updating its status and related resources."""
    try:
        # Extract user_id from the current_user dictionary
        user_id = current_user.get("User_ID")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid user authentication")
            
        logger.info(f"Executing pending trade for authenticated user_id: {user_id}")
        
        # Get the trade
        trade = get_trade_by_id(trade_id)
        if not trade:
            raise HTTPException(status_code=404, detail=f"Trade with ID {trade_id} not found")
        
        # Log the retrieved trade for debugging
        logger.info(f"Retrieved trade: {trade}")
        
        # Check if trade belongs to the user
        trade_user_id = trade.get("User_ID", trade.get("user_id"))
        logger.info(f"Trade user_id: {trade_user_id}, Current user_id: {user_id}")
        
        if trade_user_id != user_id:
            raise HTTPException(
                status_code=403, 
                detail=f"Trade with ID {trade_id} does not belong to current user"
            )
        
        # Check if trade is already executed
        trade_status = trade.get("Status", trade.get("status", "")).lower()
        if trade_status != "pending":
            raise HTTPException(
                status_code=400, 
                detail=f"Trade with ID {trade_id} is already {trade_status}"
            )
        
        # Log trade_type for debugging
        trade_type = trade.get("Trade_Type", trade.get("trade_type", "")).lower()
        logger.info(f"Trade type: {trade_type}")
        
        # For testing purposes, we'll bypass the timestamp check
        # Instead of checking if execution time is in the past, we'll force execution
        # This allows us to test the pickup function without waiting for the timestamp
        
        # Process the trade execution
        result = await _process_trade_execution(trade_id)
        return result
    
    except HTTPException as http_e:
        # Check if this is a pending case we should preserve
        is_pending_case = (
            "not yet cleared" in str(http_e.detail) or  # Market price not cleared
            "in the future" in str(http_e.detail) or    # Execution time in future
            "Please try again later" in str(http_e.detail) or  # General retry message
            "No market data" in str(http_e.detail)  # No market data available yet for the period
        )
        
        if is_pending_case:
            # For pending cases, update error message in database but keep status as pending
            logger.info(f"Trade {trade_id} remains pending: {http_e.detail}")
            
            # Explicitly update the trade with pending status and error message
            update_data = {
                "Status": "pending",  # Explicitly set status to pending
                "Error_Message": str(http_e.detail)
            }
            try:
                update_trade_status(trade_id, update_data)
            except Exception as update_error:
                logger.error(f"Failed to update trade with error message: {update_error}")
            
            # Return a response indicating the trade remains pending with a reason
            return {
                "success": False,
                "trade_id": trade_id,
                "status": "pending",
                "message": str(http_e.detail)
            }
        
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error executing pending trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/execute-all-pending")
async def execute_all_pending_trades(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Execute all pending trades that are due for execution."""
    try:
        # Extract user_id from the current_user dictionary
        user_id = current_user.get("User_ID")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid user authentication")
            
        logger.info(f"Executing pending trades for authenticated user_id: {user_id}")
        
        # Get all pending trades
        pending_trades = get_pending_trades()
        
        # Filter trades for the current user
        pending_trades = [t for t in pending_trades if t.get("User_ID", t.get("user_id")) == user_id]
        
        if not pending_trades:
            return {"success": True, "message": "No pending trades found ready for execution", "executed_count": 0}
        
        # Process each trade
        results = []
        executed_count = 0
        still_pending_count = 0
        failed_count = 0
        
        for trade in pending_trades:
            try:
                result = await _process_trade_execution(trade.get("Trade_ID", trade.get("trade_id")))
                results.append({
                    "trade_id": trade.get("Trade_ID", trade.get("trade_id")),
                    "success": True,
                    "message": f"Successfully executed {trade.get('Trade_Type', trade.get('trade_type'))} trade"
                })
                executed_count += 1
            except HTTPException as http_e:
                trade_id = trade.get("Trade_ID", trade.get("trade_id"))
                
                # Check for various cases where we should keep the trade as pending
                is_pending_case = (
                    "not yet cleared" in str(http_e.detail) or  # Market price not cleared
                    "in the future" in str(http_e.detail) or    # Execution time in future
                    "Please try again later" in str(http_e.detail) or  # General retry message
                    "No market data" in str(http_e.detail)  # No market data available yet for the period
                )
                
                if is_pending_case:
                    logger.info(f"Trade {trade_id} remains pending: {http_e.detail}")
                    # Update the trade with error message but keep status as pending
                    update_data = {
                        "Status": "pending",  # Explicitly set status to pending
                        "Error_Message": str(http_e.detail)
                    }
                    update_trade_status(trade_id, update_data)
                    
                    results.append({
                        "trade_id": trade_id,
                        "success": False,
                        "status": "pending",
                        "message": str(http_e.detail)
                    })
                    still_pending_count += 1
                else:
                    # For other HTTP exceptions, mark as failed
                    logger.error(f"Error executing trade {trade_id}: {http_e.detail}")
                    # Update the trade status to failed for other errors
                    update_data = {
                        "Status": "failed",
                        "Error_Message": str(http_e.detail)
                    }
                    update_trade_status(trade_id, update_data)
                    
                    results.append({
                        "trade_id": trade_id,
                        "success": False,
                        "status": "failed",
                        "message": str(http_e.detail)
                    })
                    failed_count += 1
            except Exception as e:
                trade_id = trade.get("Trade_ID", trade.get("trade_id"))
                logger.error(f"Error executing trade {trade_id}: {e}")
                
                # Update the trade status to failed
                update_data = {
                    "Status": "failed",
                    "Error_Message": str(e)
                }
                update_trade_status(trade_id, update_data)
                
                results.append({
                    "trade_id": trade_id,
                    "success": False,
                    "status": "failed",
                    "message": str(e)
                })
                failed_count += 1
        
        return {
            "success": True,
            "message": f"Processed {len(pending_trades)} pending trades. {executed_count} executed, {still_pending_count} still pending (price not cleared or future execution time), {failed_count} failed.",
            "executed_count": executed_count,
            "still_pending_count": still_pending_count,
            "failed_count": failed_count,
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Error executing pending trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cancel-all-pending")
async def cancel_all_pending_trades(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Cancel all pending trades for the current user by deleting them."""
    try:
        # Extract user_id from the current_user dictionary
        user_id = current_user.get("User_ID")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid user authentication")
            
        logger.info(f"Deleting pending trades for authenticated user_id: {user_id}")
        
        # Get a database connection
        db = get_db()
        
        # Execute a direct query to get pending trades - case insensitive and covering NULL status
        query = f"""
            SELECT *
            FROM `{PROJECT_ID}.{DATASET_ID}.Trades`
            WHERE User_ID = @user_id
            AND (
                LOWER(IFNULL(Status, '')) = 'pending' 
                OR Status IS NULL 
                OR TRIM(Status) = ''
            )
        """
        params = [bigquery.ScalarQueryParameter("user_id", "INTEGER", user_id)]
        
        logger.info(f"Executing direct query for pending trades: {query}")
        logger.info(f"With parameters: {params}")
        
        # Execute query directly
        pending_trades = db.execute_query(query, params)
        
        logger.info(f"Direct query found {len(pending_trades)} pending trades")
        
        # Debug log each pending trade found
        for trade in pending_trades:
            trade_id = trade.get("Trade_ID", trade.get("trade_id"))
            timestamp = trade.get("Timestamp", trade.get("timestamp"))
            status = trade.get("Status", trade.get("status", "unknown"))
            type_value = trade.get("Trade_Type", trade.get("trade_type", "unknown"))
            quantity = trade.get("Quantity", trade.get("quantity", "unknown"))
            logger.info(f"Found pending trade: ID={trade_id}, Timestamp={timestamp}, Status={status}, Type={type_value}, Quantity={quantity}")
        
        if not pending_trades:
            # Get all user trades to check what's in the database
            all_trades_query = f"""
                SELECT Trade_ID, Status, Trade_Type, Quantity, Timestamp
                FROM `{PROJECT_ID}.{DATASET_ID}.Trades`
                WHERE User_ID = @user_id
                ORDER BY Timestamp DESC
            """
            all_trades = db.execute_query(all_trades_query, params)
            
            logger.info(f"Found {len(all_trades)} total trades for user {user_id}")
            
            # Log each trade for debugging
            for trade in all_trades:
                trade_id = trade.get("Trade_ID", trade.get("trade_id"))
                status_value = trade.get("Status", trade.get("status", "unknown"))
                type_value = trade.get("Trade_Type", trade.get("trade_type", "unknown"))
                quantity = trade.get("Quantity", trade.get("quantity", "unknown"))
                logger.info(f"Trade in database: ID={trade_id}, Status='{status_value}' (type: {type(status_value)}), Type={type_value}, Quantity={quantity}")
            
            logger.info(f"No pending trades found for user {user_id}")
            return {"success": True, "message": "No pending trades found to cancel", "canceled_count": 0}
        
        logger.info(f"Found {len(pending_trades)} pending trades for user {user_id}")
        
        # Delete each pending trade instead of updating status
        results = []
        deleted_count = 0
        failed_count = 0
        
        for trade in pending_trades:
            try:
                trade_id = trade.get("Trade_ID", trade.get("trade_id"))
                logger.info(f"Attempting to delete trade ID: {trade_id}")
                
                # Delete the trade record directly
                delete_success = db.delete_row("Trades", "Trade_ID", trade_id)
                
                if delete_success:
                    logger.info(f"Successfully deleted trade ID: {trade_id}")
                    results.append({
                        "trade_id": trade_id,
                        "success": True,
                        "message": "Trade successfully deleted"
                    })
                    deleted_count += 1
                else:
                    logger.error(f"Failed to delete trade ID: {trade_id}")
                    
                    # As a fallback, try manual SQL delete if the delete_row method fails
                    logger.info(f"Attempting direct SQL DELETE for trade ID: {trade_id}")
                    
                    # Direct SQL DELETE 
                    delete_query = f"""
                        DELETE FROM `{PROJECT_ID}.{DATASET_ID}.Trades`
                        WHERE Trade_ID = @trade_id
                    """
                    delete_params = [bigquery.ScalarQueryParameter("trade_id", "INTEGER", trade_id)]
                    
                    try:
                        job_config = bigquery.QueryJobConfig(query_parameters=delete_params)
                        query_job = db.client.query(delete_query, job_config=job_config)
                        query_job.result()  # Wait for job completion
                        
                        # Check if the delete was successful
                        verify_query = f"""
                            SELECT COUNT(*) as count
                            FROM `{PROJECT_ID}.{DATASET_ID}.Trades`
                            WHERE Trade_ID = @trade_id
                        """
                        verify_job_config = bigquery.QueryJobConfig(query_parameters=delete_params)
                        verify_job = db.client.query(verify_query, job_config=verify_job_config)
                        results_verify = [row for row in verify_job]
                        
                        if len(results_verify) > 0 and results_verify[0].get('count', 0) == 0:
                            logger.info(f"Direct SQL DELETE successful for trade ID: {trade_id}")
                            results.append({
                                "trade_id": trade_id,
                                "success": True,
                                "message": "Trade successfully deleted (via direct SQL)"
                            })
                            deleted_count += 1
                        else:
                            logger.error(f"Direct SQL DELETE verification failed for trade ID: {trade_id}")
                            results.append({
                                "trade_id": trade_id,
                                "success": False,
                                "message": "Failed to delete trade (direct SQL failed)"
                            })
                            failed_count += 1
                    except Exception as sql_error:
                        logger.error(f"Direct SQL DELETE failed for trade ID: {trade_id}: {sql_error}")
                        results.append({
                            "trade_id": trade_id,
                            "success": False,
                            "message": f"Failed to delete trade: {str(sql_error)}"
                        })
                        failed_count += 1
                    
            except Exception as e:
                logger.error(f"Error cancelling trade {trade.get('Trade_ID', trade.get('trade_id'))}: {e}")
                results.append({
                    "trade_id": trade.get("Trade_ID", trade.get("trade_id")),
                    "success": False,
                    "message": str(e)
                })
                failed_count += 1
        
        return {
            "success": True,
            "message": f"Processed {len(pending_trades)} pending trades. {deleted_count} deleted, {failed_count} failed.",
            "canceled_count": deleted_count,
            "failed_count": failed_count,
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Error processing pending trades: {e}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

async def _process_trade_execution(trade_id: int):
    """Internal function to process a trade execution with the current market data."""
    try:
        # 1. Get the trade details
        trade = get_trade_by_id(trade_id)
        
        if not trade:
            logger.error(f"Trade with ID {trade_id} not found")
            raise HTTPException(status_code=404, detail=f"Trade with ID {trade_id} not found")
        
        # Get user ID (handle both camelCase and snake_case keys)
        user_id = trade.get("User_ID", trade.get("user_id", 1))  # Default to user 1 for testing
        
        # Get trade type (handle both camelCase and snake_case keys)
        trade_type = trade.get("Trade_Type", trade.get("trade_type", "buy")).lower()
        
        # Get quantity (handle both camelCase and snake_case keys)
        quantity = float(trade.get("Quantity", trade.get("quantity", 0)))
        
        # Get resolution (handle both camelCase and snake_case keys)
        resolution = int(trade.get("Resolution", trade.get("resolution", 60)))
        
        # Get execution time (handle both camelCase and snake_case keys)
        execution_time = trade.get("Timestamp", trade.get("timestamp"))
        
        # Check if the execution time is in the future - if so, keep the trade as pending
        if execution_time:
            # If execution_time is a string, parse it to datetime
            if isinstance(execution_time, str):
                try:
                    execution_time = datetime.fromisoformat(execution_time.replace('Z', '+00:00'))
                except ValueError:
                    execution_time = datetime.fromisoformat(execution_time)
                    execution_time = execution_time.replace(tzinfo=timezone.utc)
            
            # Compare with current time to see if the execution time is still in the future
            now_utc = datetime.now(timezone.utc)
            
            # Add a small buffer (e.g., 1 minute) to account for slight clock differences
            if execution_time > now_utc + timedelta(minutes=1):
                future_minutes = (execution_time - now_utc).total_seconds() / 60
                error_message = f"Trade scheduled for {execution_time.isoformat()} is {future_minutes:.1f} minutes in the future. Please try again later."
                logger.info(f"Trade {trade_id} remains pending because execution time is in the future: {error_message}")
                # Don't update status - keep it as pending
                raise HTTPException(status_code=400, detail=error_message)
        
        # Normalize execution time to handle timezone
        if execution_time:
            # Log the original execution_time for debugging
            logger.info(f"Raw execution time from database: {execution_time}, type: {type(execution_time)}")
            
            # If execution_time is a string, parse it to datetime
            if isinstance(execution_time, str):
                try:
                    # Try parsing with timezone info
                    execution_time = datetime.fromisoformat(execution_time.replace('Z', '+00:00'))
                    logger.info(f"Parsed execution time with timezone: {execution_time}")
                except ValueError:
                    # If parsing fails, assume it's UTC (BigQuery stores timestamps as UTC)
                    execution_time = datetime.fromisoformat(execution_time)
                    # Explicitly add UTC timezone 
                    execution_time = execution_time.replace(tzinfo=timezone.utc)
                    logger.info(f"Parsed execution time as UTC: {execution_time}")
            
            # Now explicitly convert to CET timezone 
            execution_time_cet = execution_time.astimezone(CET)
            logger.info(f"Converted to CET: {execution_time} → {execution_time_cet}")
            
            # Calculate delivery period based on CET time
            delivery_hour = execution_time_cet.hour
            delivery_minute = execution_time_cet.minute
            delivery_day = execution_time_cet.strftime("%Y-%m-%d")
            
            logger.info(f"Using delivery hour: {delivery_hour}, minute: {delivery_minute} for period calculation")
            
            # Format the delivery period string based on resolution
            if resolution == 15:
                # For 15-minute resolution, format like "12:15 - 12:30"
                start_minute = (delivery_minute // 15) * 15  # Round down to nearest 15
                end_minute = (start_minute + 15) % 60
                end_hour = delivery_hour + (1 if end_minute == 0 else 0)
                delivery_period = f"{delivery_hour:02d}:{start_minute:02d} - {end_hour:02d}:{end_minute:02d}"
            elif resolution == 30:
                # For 30-minute resolution, format like "12:00 - 12:30" or "12:30 - 13:00"
                start_minute = (delivery_minute // 30) * 30  # Either 0 or 30
                end_minute = (start_minute + 30) % 60
                end_hour = delivery_hour + (1 if end_minute == 0 else 0)
                delivery_period = f"{delivery_hour:02d}:{start_minute:02d} - {end_hour:02d}:{end_minute:02d}"
            elif resolution == 60:
                # For 60-minute resolution, format like "12:00 - 13:00"
                delivery_period = f"{delivery_hour:02d}:00 - {(delivery_hour + 1):02d}:00"
            else:
                # Default fallback for unexpected resolution
                delivery_period = str(delivery_hour + 1)  # Legacy format
                
            logger.info(f"Calculated delivery period: '{delivery_period}' for time {execution_time_cet} with resolution {resolution}")
        else:
            # If we don't have execution time, use current hour in CET
            now_cet = to_cet(datetime.now(timezone.utc))
            delivery_hour = now_cet.hour
            delivery_minute = now_cet.minute
            delivery_day = now_cet.strftime("%Y-%m-%d")
            
            # Use the same logic as above to determine delivery period
            if resolution == 15:
                start_minute = (delivery_minute // 15) * 15
                end_minute = (start_minute + 15) % 60
                end_hour = delivery_hour + (1 if end_minute == 0 else 0)
                delivery_period = f"{delivery_hour:02d}:{start_minute:02d} - {end_hour:02d}:{end_minute:02d}"
            elif resolution == 30:
                start_minute = (delivery_minute // 30) * 30
                end_minute = (start_minute + 30) % 60
                end_hour = delivery_hour + (1 if end_minute == 0 else 0)
                delivery_period = f"{delivery_hour:02d}:{start_minute:02d} - {end_hour:02d}:{end_minute:02d}"
            elif resolution == 60:
                delivery_period = f"{delivery_hour:02d}:00 - {(delivery_hour + 1):02d}:00"
            else:
                delivery_period = str(delivery_hour + 1)
        
        logger.info(f"Trade execution time (CET): {execution_time_cet if 'execution_time_cet' in locals() else 'N/A'}, delivery day: {delivery_day}, delivery period: {delivery_period}")
        
        # 2. First check Market_Data_Germany_Today for current data
        db = get_db()
        
        # Build direct query for Market_Data_Germany_Today with precise delivery period
        today_query = f"""
            SELECT *
            FROM `{PROJECT_ID}.{DATASET_ID}.Market_Data_Germany_Today`
            WHERE Delivery_Day = @delivery_day
            AND Delivery_Period = @delivery_period
            AND Resolution_Minutes = @resolution
            AND Cleared = TRUE
        """
        
        today_params = [
            bigquery.ScalarQueryParameter("delivery_day", "STRING", delivery_day),
            bigquery.ScalarQueryParameter("delivery_period", "STRING", delivery_period),
            bigquery.ScalarQueryParameter("resolution", "INTEGER", resolution)
        ]
        
        logger.info(f"Querying Market_Data_Germany_Today with: day={delivery_day}, period='{delivery_period}', resolution={resolution}")
        market_data_today = db.execute_query(today_query, today_params)
        
        # If not found in today's data, check historical data
        market_record = None
        if market_data_today and len(market_data_today) > 0:
            logger.info(f"Found {len(market_data_today)} matching records in Market_Data_Germany_Today")
            market_record = market_data_today[0]
            data_source = "Market_Data_Germany_Today"
        else:
            logger.info("No matching cleared data found in Market_Data_Germany_Today, checking Market_Data_Germany")
            
            # Build query for historical data
            historical_query = f"""
                SELECT *
                FROM `{PROJECT_ID}.{DATASET_ID}.Market_Data_Germany`
                WHERE Delivery_Day = @delivery_day
                AND Delivery_Period = @delivery_period
                AND Resolution_Minutes = @resolution
            """
            
            historical_params = [
                bigquery.ScalarQueryParameter("delivery_day", "STRING", delivery_day),
                bigquery.ScalarQueryParameter("delivery_period", "STRING", delivery_period),
                bigquery.ScalarQueryParameter("resolution", "INTEGER", resolution)
            ]
            
            logger.info(f"Querying Market_Data_Germany with: day={delivery_day}, period='{delivery_period}', resolution={resolution}")
            market_data_historical = db.execute_query(historical_query, historical_params)
            
            if market_data_historical and len(market_data_historical) > 0:
                logger.info(f"Found {len(market_data_historical)} matching records in Market_Data_Germany")
                market_record = market_data_historical[0]
                data_source = "Market_Data_Germany"
            else:
                logger.info("No matching data found in Market_Data_Germany either")
        
        # If still no market data found
        if not market_record:
            error_message = f"No market data available yet for delivery day {delivery_day}, period '{delivery_period}' with resolution {resolution} minutes. The trade will remain pending until data becomes available."
            logger.error(error_message)
            # Keep as pending - don't update status
            raise HTTPException(status_code=400, detail=error_message)
        
        # Check if the market is cleared (only relevant for today's data, historical is assumed cleared)
        is_cleared = True
        if data_source == "Market_Data_Germany_Today":
            is_cleared = market_record.get("Cleared", False)
            
        if not is_cleared:
            error_message = f"The market price for Trade {trade_id} at delivery period '{delivery_period}' with resolution {resolution} minutes is not yet cleared. Please wait for the market to clear or try again later."
            logger.info(f"Trade {trade_id} remains pending because market price is not yet cleared")
            # Keep as pending - don't update status
            raise HTTPException(status_code=400, detail=error_message)
        
        # Select appropriate price based on trade type
        if trade_type == "buy":
            # For buy, use the low price to get the best deal
            trade_price = market_record.get("Low")
            price_field = "Low"
        else:  # sell
            # For sell, use the high price for maximum revenue
            trade_price = market_record.get("High")
            price_field = "High"
        
        # Log the market record for debugging
        logger.info(f"Using {price_field} price from {data_source} for {trade_type} trade: " + 
                  f"Low={market_record.get('Low')}, High={market_record.get('High')}, " +
                  f"VWAP={market_record.get('VWAP')}, Close={market_record.get('Close')}")
        
        if trade_price is None:
            error_message = f"No {price_field} price available for period '{delivery_period}' with resolution {resolution} minutes"
            logger.error(error_message)
            # Keep as pending - don't update status
            raise HTTPException(status_code=400, detail=error_message)
        
        logger.info(f"Found market price: {trade_price} ({price_field}) for period '{delivery_period}', resolution {resolution}")
        
        # 4. Get battery status
        battery = get_battery_status(user_id)
        if not battery:
            battery = create_battery_if_not_exists(user_id)
            if not battery:
                update_data = {
                    "Status": "failed",
                    "Error_Message": "Failed to access or create battery"
                }
                update_trade_status(trade_id, update_data)
                raise HTTPException(status_code=500, detail="Failed to access or create battery")
        
        # Handle different key naming conventions in the battery data
        current_level = battery.get("current_level", battery.get("Current_Level", 0))
        total_capacity = battery.get("total_capacity", battery.get("Total_Capacity", 2.5))
        
        # If current_level is represented as a percentage (0-100), convert to absolute MWh
        if current_level > 0 and current_level <= 100 and total_capacity > 0:
            # If current_level seems to be a percentage and total_capacity is in absolute units
            absolute_current_level = (current_level / 100) * total_capacity
            logger.info(f"Converting percentage {current_level}% to absolute: {absolute_current_level} MWh")
            current_level = absolute_current_level
        
        # 5. Check battery capacity and update battery level
        new_level = current_level
        if trade_type == "buy":
            # Check if there's enough remaining capacity
            remaining_capacity = total_capacity - current_level
            if quantity > remaining_capacity:
                update_data = {
                    "Status": "failed",
                    "Error_Message": f"Insufficient battery capacity. Maximum purchasable: {remaining_capacity} MWh"
                }
                update_trade_status(trade_id, update_data)
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient battery capacity. Maximum purchasable: {remaining_capacity} MWh"
                )
            
            # Charge the battery
            new_level = current_level + quantity
        
        elif trade_type == "sell":
            # Check if there's enough energy in the battery
            if quantity > current_level:
                update_data = {
                    "Status": "failed",
                    "Error_Message": f"Insufficient energy in battery. Maximum sellable: {current_level} MWh"
                }
                update_trade_status(trade_id, update_data)
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient energy in battery. Maximum sellable: {current_level} MWh"
                )
            
            # Discharge the battery
            new_level = current_level - quantity
        else:
            logger.error(f"Invalid trade type: {trade_type}")
            update_data = {
                "Status": "failed",
                "Error_Message": f"Invalid trade type: {trade_type}. Must be 'buy' or 'sell'."
            }
            update_trade_status(trade_id, update_data)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid trade type: {trade_type}. Must be 'buy' or 'sell'."
            )
        
        # Convert absolute MWh to percentage for storage
        if total_capacity > 0:
            new_level_percentage = (new_level / total_capacity) * 100
            logger.info(f"Converting absolute {new_level} MWh to percentage: {new_level_percentage}%")
            new_level = new_level_percentage
        
        # 6. Update battery level
        battery_update_success = update_battery_level(user_id, new_level)
        if not battery_update_success:
            logger.error(f"Failed to update battery level for user {user_id}")
            # Continue with execution despite battery update failure
        
        # 7. Calculate the cost or revenue from the trade
        trade_value = quantity * trade_price
        
        # 8. Update the user's portfolio
        portfolio = get_portfolio_by_user_id(user_id)
        
        # Get current portfolio values
        current_balance = 0
        current_holdings = 0
        cumulative_pnl = 0
        
        if portfolio:
            current_balance = portfolio.get("Current_Balance", 0)
            current_holdings = portfolio.get("Current_Holdings", 0)
            cumulative_pnl = portfolio.get("Cumulative_Profit_Loss", 0)
        
        # Update portfolio based on trade type
        new_balance = current_balance
        new_holdings = current_holdings
        trade_pnl = 0
        
        if trade_type == "buy":
            # Buying energy decreases balance
            new_balance = current_balance - trade_value
            new_holdings = current_holdings + quantity
            
            # For buys, the PnL is unrealized until sold
            trade_pnl = 0
        elif trade_type == "sell":
            # Selling energy increases balance
            new_balance = current_balance + trade_value
            new_holdings = current_holdings - quantity
            
            # For sells, calculate realized PnL
            # Simplified: assume average cost basis
            if current_holdings > 0:
                avg_cost_per_unit = current_balance / current_holdings
                trade_pnl = (trade_price - avg_cost_per_unit) * quantity
            else:
                trade_pnl = trade_value  # If no holdings, all revenue is profit
        
        # Update cumulative P&L
        new_cumulative_pnl = cumulative_pnl + trade_pnl
        
        # Create portfolio update data
        portfolio_update = {
            "Current_Balance": new_balance,
            "Current_Holdings": new_holdings,
            "Cumulative_Profit_Loss": new_cumulative_pnl,
            "Summary_Details": f"Updated after {trade_type} trade {trade_id} execution"
        }
        
        portfolio_update_success = update_portfolio_balance(user_id, portfolio_update)
        if not portfolio_update_success:
            logger.error(f"Failed to update portfolio for user {user_id}")
            # Continue with execution despite portfolio update failure
        
        # 9. Update trade status to executed
        update_data = {
            "Status": "executed",
            "Trade_Price": trade_price
        }
        
        trade_update_success = update_trade_status(trade_id, update_data)
        if not trade_update_success:
            logger.error(f"Failed to update trade status for trade {trade_id}")
            # Continue with execution despite trade update failure
        
        # 10. Return execution results
        return {
            "success": True,
            "trade_id": trade_id,
            "message": f"{trade_type.capitalize()} trade executed for {quantity} MWh",
            "trade_price": trade_price,
            "trade_value": trade_value,
            "new_battery_level": new_level,
            "new_balance": new_balance,
            "profit_loss": trade_pnl
        }
    
    except HTTPException as http_e:
        # Check if this is a pending case we should preserve
        is_pending_case = (
            "not yet cleared" in str(http_e.detail) or  # Market price not cleared
            "in the future" in str(http_e.detail) or    # Execution time in future
            "Please try again later" in str(http_e.detail) or  # General retry message
            "No market data" in str(http_e.detail)  # No market data available yet for the period
        )
        
        if is_pending_case:
            # For pending cases, update error message in database but keep status as pending
            # The actual update happens in the caller functions that catch this exception
            logger.info(f"Trade {trade_id} remains pending: {http_e.detail}")
        
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in _process_trade_execution: {e}")
        # Update trade with error
        try:
            update_data = {
                "Status": "failed",
                "Error_Message": str(e)
            }
            update_trade_status(trade_id, update_data)
        except Exception as update_error:
            logger.error(f"Failed to update trade with error status: {update_error}")
        
        raise Exception(f"Failed to execute trade: {e}") 