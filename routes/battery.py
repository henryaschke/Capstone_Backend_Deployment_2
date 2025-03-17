from fastapi import APIRouter, BackgroundTasks, Query, Depends
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging

from models.battery import BatteryStatus, BatteryAction
from dependencies import get_optional_user, DEFAULT_USER_ID
from database import get_battery_status, create_trade

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
            return {
                "level": battery.get("current_level", 50.0),
                "capacity": {
                    "total": battery.get("total_capacity", 2.5),
                    "usable": battery.get("usable_capacity", 2.0),
                    "percentage": battery.get("current_level", 50.0)
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
            battery = {
                "user_id": user_id,
                "current_level": 50.0,
                "total_capacity": 2.5,
                "usable_capacity": 2.0
            }
        
        current_level = battery.get("current_level", 50.0)
        new_level = current_level + request.quantity
        if new_level > 100:
            new_level = 100
        
        # Optionally update in the database
        # For demonstration, we just create a "charge" trade
        trade_data = {
            "user_id": user_id,
            "market": "Battery",
            "trade_type": "charge",
            "quantity": request.quantity,
            "trade_price": 0,
            "timestamp": datetime.now(),
            "status": "executed"
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
            battery = {
                "user_id": user_id,
                "current_level": 50.0,
                "total_capacity": 2.5,
                "usable_capacity": 2.0
            }
        
        current_level = battery.get("current_level", 50.0)
        new_level = current_level - request.quantity
        if new_level < 0:
            new_level = 0
        
        trade_data = {
            "user_id": user_id,
            "market": "Battery",
            "trade_type": "discharge",
            "quantity": request.quantity,
            "trade_price": 0,
            "timestamp": datetime.now(),
            "status": "executed"
        }
        create_trade(trade_data)
        
        return {"success": True, "message": f"Battery discharged by {request.quantity}", "newLevel": new_level}
    except Exception as e:
        logger.error(f"Error discharging battery: {e}")
        return {"success": False, "message": str(e)} 