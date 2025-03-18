from fastapi import APIRouter, BackgroundTasks, Query, Depends
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging

from models.battery import BatteryStatus, BatteryAction
from dependencies import get_optional_user, DEFAULT_USER_ID
from database import get_battery_status, create_trade, update_battery_level, create_battery_if_not_exists

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get("/status")
async def get_battery_status_api(
    user_id: int = DEFAULT_USER_ID, 
    current_user: Dict[str, Any] = Depends(get_optional_user)
):
    """Get current battery status and capacity."""
    try:
        battery = get_battery_status(user_id)
        if battery:
            # Check for different field name conventions (camelCase vs snake_case)
            current_level = battery.get("current_level", battery.get("Current_Level", 50.0))
            total_capacity = battery.get("total_capacity", battery.get("Total_Capacity", 2.5))
            usable_capacity = battery.get("usable_capacity", battery.get("Usable_Capacity", 2.0))
            
            # Log the battery status for debugging
            logger.info(f"Battery status for user {user_id}: {battery}")
            
            return {
                "level": current_level,
                "capacity": {
                    "total": total_capacity,
                    "usable": usable_capacity,
                    "percentage": current_level
                }
            }
        else:
            # Default fallback if none found
            return {
                "level": 50.0,
                "capacity": {
                    "total": 2.5,
                    "usable": 2.0,
                    "percentage": 50.0
                }
            }
    except Exception as e:
        logger.error(f"Error getting battery status: {e}")
        return {
            "level": 50.0,
            "capacity": {
                "total": 2.5,
                "usable": 2.0,
                "percentage": 50.0
            }
        }

@router.get("/history")
async def get_battery_history(
    days: int = Query(7, description="Number of days of history"),
    user_id: int = DEFAULT_USER_ID,
    current_user: Dict[str, Any] = Depends(get_optional_user)
):
    """Get battery level history (sample/dummy data)."""
    try:
        today = datetime.now()
        # In a real app, you would query a BatteryHistory table here
        return [
            {
                "time": (today - timedelta(hours=i)).isoformat(),
                "level": 50 + (10 * (i % 5 - 2))
            }
            for i in range(24 * days)
        ]
    except Exception as e:
        logger.error(f"Error getting battery history: {e}")
        return []

@router.post("/charge")
async def charge_battery(
    request: BatteryAction,
    background_tasks: BackgroundTasks,
    user_id: int = DEFAULT_USER_ID
):
    """Charge the battery by a certain quantity."""
    try:
        battery = get_battery_status(user_id)
        if not battery:
            battery = create_battery_if_not_exists(user_id)
            if not battery:
                return {"success": False, "message": "Failed to create battery"}
        
        current_level = battery.get("current_level", battery.get("Current_Level", 50.0))
        new_level = current_level + request.quantity
        
        # Cap at 100%
        if new_level > 100:
            new_level = 100
        
        # Update the battery level in the database
        update_success = update_battery_level(user_id, new_level)
        if not update_success:
            logger.error(f"Failed to update battery level for user {user_id}")
            return {"success": False, "message": "Failed to update battery level"}
        
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
    except Exception as e:
        logger.error(f"Error charging battery: {e}")
        return {"success": False, "message": str(e)}

@router.post("/discharge")
async def discharge_battery(
    request: BatteryAction,
    background_tasks: BackgroundTasks,
    user_id: int = DEFAULT_USER_ID
):
    """Discharge the battery by a certain quantity."""
    try:
        battery = get_battery_status(user_id)
        if not battery:
            battery = create_battery_if_not_exists(user_id)
            if not battery:
                return {"success": False, "message": "Failed to create battery"}
        
        current_level = battery.get("current_level", battery.get("Current_Level", 50.0))
        new_level = current_level - request.quantity
        
        # Cap at 0%
        if new_level < 0:
            new_level = 0
        
        # Update the battery level in the database
        update_success = update_battery_level(user_id, new_level)
        if not update_success:
            logger.error(f"Failed to update battery level for user {user_id}")
            return {"success": False, "message": "Failed to update battery level"}
        
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
    except Exception as e:
        logger.error(f"Error discharging battery: {e}")
        return {"success": False, "message": str(e)} 