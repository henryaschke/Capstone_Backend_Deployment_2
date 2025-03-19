from fastapi import APIRouter, Query, HTTPException, Depends, Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta, timezone
import uuid
import math

from models.trade import TradeRequest
from dependencies import get_optional_user, DEFAULT_USER_ID, parse_date_string
from database import (
    create_trade, get_user_trades, get_battery_status, create_battery_if_not_exists,
    get_trade_by_id, update_trade_status, update_battery_level, get_pending_trades,
    get_market_data_today, update_portfolio_balance, get_portfolio_by_user_id
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

def to_cet(dt):
    """Convert a datetime to CET timezone"""
    if dt.tzinfo is None:
        # If naive datetime, assume it's UTC and add CET offset
        return dt.replace(tzinfo=timezone.utc).astimezone(timezone(CET_OFFSET))
    return dt.astimezone(timezone(CET_OFFSET))

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
    user_id: int = Depends(get_optional_user),
    test_mode: bool = Query(False, description="Enable test mode to bypass time validation")
):
    """Execute a trade (buy or sell) for the German energy market."""
    try:
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
    user_id: int = Depends(get_optional_user)
):
    """Get trade history (optionally by date range, trade_type, or status)."""
    try:
        logger.info(f"Trade history request: start_date={start_date}, end_date={end_date}, trade_type={trade_type}, status={status}")
        start_time = datetime.now()
        
        # Use provided user_id or fallback to default
        if user_id is None:
            user_id = DEFAULT_USER_ID
            logger.info(f"Using default user_id: {DEFAULT_USER_ID}")
        
        # If user_id is a dict (for some reason), extract the User_ID value
        if isinstance(user_id, dict) and 'User_ID' in user_id:
            logger.info(f"Extracted user_id {user_id['User_ID']} from user_id dict")
            user_id = user_id['User_ID']
            
        logger.info(f"Getting trades for user_id: {user_id}")
        
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
            trades = [t for t in trades if t.get("status", "").lower() == status.lower()]
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
            timestamp_iso = timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp
            
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
            
            result.append({
                "trade_id": trade_id,
                "type": trade_type.lower() if isinstance(trade_type, str) else trade_type,
                "quantity": float(quantity) if quantity is not None else 0.0,  # Ensure quantity is a float
                "timestamp": timestamp_iso,
                "resolution": resolution,
                "status": status.lower() if isinstance(status, str) else status,
                "market": market,
                "created_at": created_at_iso,
                "trade_price": float(trade_price) if trade_price is not None else 0.0  # Ensure price is a float
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
    user_id: int = Depends(get_optional_user)
):
    """Execute a specific pending trade by updating its status and related resources."""
    try:
        # Use provided user_id or fallback to default
        if user_id is None:
            user_id = DEFAULT_USER_ID
        
        # Get the trade
        trade = get_trade_by_id(trade_id)
        if not trade:
            raise HTTPException(status_code=404, detail=f"Trade with ID {trade_id} not found")
        
        # Log the retrieved trade for debugging
        logger.info(f"Retrieved trade: {trade}")
        
        # For testing, we're skipping user validation
        # In production, you'd want to enforce this
        # Check if trade belongs to the user
        trade_user_id = trade.get("User_ID", trade.get("user_id"))
        logger.info(f"Trade user_id: {trade_user_id}, Current user_id: {user_id}")
        # Temporarily skip this check for testing
        '''
        if trade_user_id != user_id:
            raise HTTPException(
                status_code=403, 
                detail=f"Trade with ID {trade_id} does not belong to current user"
            )
        '''
        
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
    
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error executing pending trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/execute-all-pending")
async def execute_all_pending_trades(
    user_id: int = Depends(get_optional_user)
):
    """Execute all pending trades that are due for execution."""
    try:
        # Use provided user_id or fallback to default
        if user_id is None:
            user_id = DEFAULT_USER_ID
        
        # Get all pending trades
        pending_trades = get_pending_trades()
        
        # Filter trades for the current user if needed
        if user_id != DEFAULT_USER_ID:  # Only filter if not admin
            pending_trades = [t for t in pending_trades if t.get("User_ID", t.get("user_id")) == user_id]
        
        if not pending_trades:
            return {"success": True, "message": "No pending trades found ready for execution", "executed_count": 0}
        
        # Process each trade
        results = []
        executed_count = 0
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
            except Exception as e:
                logger.error(f"Error executing trade {trade.get('Trade_ID', trade.get('trade_id'))}: {e}")
                results.append({
                    "trade_id": trade.get("Trade_ID", trade.get("trade_id")),
                    "success": False,
                    "message": str(e)
                })
                failed_count += 1
        
        return {
            "success": True,
            "message": f"Processed {len(pending_trades)} pending trades. {executed_count} executed, {failed_count} failed.",
            "executed_count": executed_count,
            "failed_count": failed_count,
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Error executing pending trades: {e}")
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
        
        # Normalize execution time to handle timezone
        if execution_time:
            # If it has tzinfo, use it. Otherwise, assume UTC
            if hasattr(execution_time, 'tzinfo') and execution_time.tzinfo is not None:
                # Convert to CET for delivery period calculation
                execution_time_cet = to_cet(execution_time)
                logger.info(f"Execution time with timezone: {execution_time}, CET: {execution_time_cet}")
            else:
                # Assume the naive datetime is in UTC, convert to CET
                execution_time_utc = execution_time.replace(tzinfo=timezone.utc)
                execution_time_cet = to_cet(execution_time_utc)
                logger.info(f"Execution time without timezone (assuming UTC): {execution_time}, CET: {execution_time_cet}")
            
            # Calculate delivery period based on CET time
            delivery_period = execution_time_cet.hour + 1  # Delivery periods are 1-24 for the hours 0-23 in CET
            delivery_day = execution_time_cet.strftime("%Y-%m-%d")
        else:
            # If we don't have execution time, use current hour in CET
            now_cet = to_cet(datetime.now(timezone.utc))
            delivery_period = now_cet.hour + 1
            delivery_day = now_cet.strftime("%Y-%m-%d")
        
        logger.info(f"Trade execution time (CET): {execution_time_cet if 'execution_time_cet' in locals() else 'N/A'}, delivery day: {delivery_day}, delivery period: {delivery_period}")
        
        # 2. Check market data to see if the product is cleared
        market_data = get_market_data_today(delivery_period, resolution)
        
        if not market_data:
            # No market data found - create mock data for testing
            logger.warning(f"No market data found for delivery period {delivery_period} - using mock data for testing")
            # Mock data will be created by get_market_data_today automatically
            market_data = get_market_data_today(delivery_period, resolution)
        
        if not market_data:
            # If still no data after mock attempt
            error_message = f"No market data found for delivery period {delivery_period} with resolution {resolution}"
            update_data = {
                "Status": "failed",
                "Error_Message": error_message
            }
            update_trade_status(trade_id, update_data)
            raise HTTPException(status_code=400, detail=error_message)
        
        # Get the corresponding market record for the correct resolution and period
        market_record = None
        for record in market_data:
            # Check if the record matches our resolution (try both naming conventions)
            if (record.get("Resolution_Minutes") == resolution or 
                record.get("resolution") == resolution or 
                record.get("ResolutionMinutes") == resolution):
                market_record = record
                break
                
        if not market_record:
            # If we don't have exact resolution match, take the first record
            market_record = market_data[0]
        
        # For testing, always consider the market cleared (try both naming conventions)
        is_cleared = False
        for cleared_field in ["Cleared", "IsCleared", "is_cleared"]:
            if cleared_field in market_record:
                is_cleared = market_record[cleared_field]
                if is_cleared:
                    break
        
        # Default to True for testing if we couldn't find a cleared flag
        if not isinstance(is_cleared, bool):
            is_cleared = True
        
        if not is_cleared:
            # Product is not yet cleared in the market
            error_message = f"Market for delivery period {delivery_period} is not yet cleared"
            update_data = {
                "Status": "failed",
                "Error_Message": error_message
            }
            update_trade_status(trade_id, update_data)
            raise HTTPException(status_code=400, detail=error_message)
        
        # 3. Get the clearing price from market data (try different field names)
        trade_price = None
        for price_field in ["clearing_price", "Close", "close", "VWAP", "vwap"]:
            if price_field in market_record and market_record[price_field] is not None:
                trade_price = float(market_record[price_field])
                break
        
        if trade_price is None:
            error_message = f"No price data available for period {delivery_period}"
            update_data = {
                "Status": "failed",
                "Error_Message": error_message
            }
            update_trade_status(trade_id, update_data)
            raise HTTPException(status_code=400, detail=error_message)
        
        logger.info(f"Found market price: {trade_price} for period {delivery_period}, resolution {resolution}")
        
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
    
    except HTTPException:
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