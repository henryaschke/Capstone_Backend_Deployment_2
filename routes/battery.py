from fastapi import APIRouter, BackgroundTasks, Query, Depends, HTTPException
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging

from models.battery import BatteryStatus, BatteryAction
from dependencies import get_current_user
from database import get_battery_status, create_trade, update_battery_level, create_battery_if_not_exists, recalculate_battery_from_trades

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get("/status")
async def get_battery_status_api(
    recalculate: bool = Query(False, description="Force recalculation of battery level from executed trades"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get the battery status for the authenticated user.
    Optionally recalculate the battery level from executed trades for accuracy.
    """
    try:
        # Extract user_id from the current_user dictionary
        user_id = current_user.get("User_ID")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid user authentication")
            
        logger.info(f"Getting battery status for authenticated user_id: {user_id}")
        
        # Get the battery for this user
        battery = get_battery_status(user_id)
        
        # If requested or if battery doesn't exist, recalculate from trade history
        if recalculate or (battery and battery.get("Current_Level") is None):
            logger.info(f"Recalculating battery level from trades for user {user_id}")
            recalculation_success = recalculate_battery_from_trades(user_id)
            if recalculation_success:
                logger.info("Battery level recalculated successfully")
                # Get updated battery after recalculation
                battery = get_battery_status(user_id)
            else:
                logger.warning("Failed to recalculate battery level from trades")
                
        # If still no battery, create default one
        if not battery:
            logger.warning(f"No battery found for user {user_id}. Creating default battery.")
            battery = create_battery_if_not_exists(user_id)
            
        # Format the response
        battery_data = format_battery_response(battery)
        
        # Log the battery data for debugging
        logger.info(f"Battery status for user {user_id}: {battery_data}")
        
        return battery_data
    
    except Exception as e:
        logger.error(f"Error getting battery status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def format_battery_response(battery: Dict[str, Any]) -> Dict[str, Any]:
    """Format the battery response with additional derived fields."""
    if not battery:
        return {
            "level": 0,
            "capacity": {
                "total": 2.5,
                "usable": 2.0,
                "percentage": 0
            }
        }
    
    # Extract the basic fields
    current_level = battery.get("Current_Level", 0)
    total_capacity = battery.get("Total_Capacity", 2.5)
    usable_capacity = battery.get("Usable_Capacity", 2.0)
    
    # Calculate derived fields
    level_percentage = current_level  # The level is already stored as percentage in the database
    
    # Calculate absolute values
    absolute_level = (current_level / 100) * total_capacity if current_level <= 100 else current_level
    
    return {
        "level": current_level,  # This is already a percentage
        "capacity": {
            "total": total_capacity,
            "usable": usable_capacity,
            "percentage": (usable_capacity / total_capacity) * 100 if total_capacity > 0 else 0
        },
        "absolute_level": absolute_level,  # Added for clarity in MWh
        "last_updated": battery.get("Last_Updated", datetime.now().isoformat())
    }

@router.get("/history")
async def get_battery_history(
    days: int = Query(7, description="Number of days of history"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get battery level history (sample/dummy data)."""
    try:
        # Extract user_id from the current_user dictionary
        user_id = current_user.get("User_ID")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid user authentication")
            
        logger.info(f"Getting battery history for authenticated user_id: {user_id}")
        
        today = datetime.now()
        # In a real app, you would query a BatteryHistory table here
        return [
            {
                "time": (today - timedelta(hours=i)).isoformat(),
                "level": 50 + (10 * (i % 5 - 2))
            }
            for i in range(24 * days)
        ]
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error getting battery history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/charge")
async def charge_battery(
    request: BatteryAction,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Charge the battery by a certain quantity."""
    try:
        # Extract user_id from the current_user dictionary
        user_id = current_user.get("User_ID")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid user authentication")
            
        logger.info(f"Charging battery for authenticated user_id: {user_id}")
        
        battery = get_battery_status(user_id)
        if not battery:
            battery = create_battery_if_not_exists(user_id)
            if not battery:
                raise HTTPException(status_code=500, detail="Failed to create battery for user")
        
        current_level = battery.get("current_level", battery.get("Current_Level", 50.0))
        new_level = current_level + request.quantity
        
        # Cap at 100%
        if new_level > 100:
            new_level = 100
        
        # Update the battery level in the database
        update_success = update_battery_level(user_id, new_level)
        if not update_success:
            logger.error(f"Failed to update battery level for user {user_id}")
            raise HTTPException(status_code=500, detail="Failed to update battery level")
        
        # Create a "charge" trade record
        trade_data = {
            "Trade_ID": int(datetime.now().timestamp() * 1000),
            "User_ID": user_id,
            "Market": "Battery",
            "Trade_Type": "charge",
            "Quantity": request.quantity,
            "Trade_Price": 0,
            "Timestamp": datetime.now(),
            "Status": "executed"
        }
        create_trade(trade_data)
        
        return {"success": True, "message": f"Battery charged by {request.quantity}", "newLevel": new_level}
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error charging battery: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/discharge")
async def discharge_battery(
    request: BatteryAction,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Discharge the battery by a certain quantity."""
    try:
        # Extract user_id from the current_user dictionary
        user_id = current_user.get("User_ID")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid user authentication")
            
        logger.info(f"Discharging battery for authenticated user_id: {user_id}")
        
        battery = get_battery_status(user_id)
        if not battery:
            battery = create_battery_if_not_exists(user_id)
            if not battery:
                raise HTTPException(status_code=500, detail="Failed to create battery for user")
        
        current_level = battery.get("current_level", battery.get("Current_Level", 50.0))
        new_level = current_level - request.quantity
        
        # Cap at 0%
        if new_level < 0:
            new_level = 0
        
        # Update the battery level in the database
        update_success = update_battery_level(user_id, new_level)
        if not update_success:
            logger.error(f"Failed to update battery level for user {user_id}")
            raise HTTPException(status_code=500, detail="Failed to update battery level")
        
        # Create a "discharge" trade record
        trade_data = {
            "Trade_ID": int(datetime.now().timestamp() * 1000),
            "User_ID": user_id,
            "Market": "Battery",
            "Trade_Type": "discharge",
            "Quantity": request.quantity,
            "Trade_Price": 0,
            "Timestamp": datetime.now(),
            "Status": "executed"
        }
        create_trade(trade_data)
        
        return {"success": True, "message": f"Battery discharged by {request.quantity}", "newLevel": new_level}
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error discharging battery: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recalculate")
async def recalculate_battery_api(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Force recalculation of battery level from executed trades.
    This ensures the battery level accurately reflects the history of executed trades.
    """
    try:
        # Extract user_id from the current_user dictionary
        user_id = current_user.get("User_ID")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid user authentication")
            
        logger.info(f"Recalculating battery for user {user_id} from executed trades")
        
        # Perform the recalculation
        success = recalculate_battery_from_trades(user_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to recalculate battery from trades")
        
        # Get the updated battery status
        updated_battery = get_battery_status(user_id)
        
        return {
            "success": True,
            "message": "Battery level recalculated from executed trades",
            "battery": format_battery_response(updated_battery)
        }
    
    except Exception as e:
        logger.error(f"Error recalculating battery: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 