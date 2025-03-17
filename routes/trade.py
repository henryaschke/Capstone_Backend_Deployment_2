from fastapi import APIRouter, Query, HTTPException, Depends
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from models.trade import TradeRequest
from dependencies import get_optional_user, DEFAULT_USER_ID, parse_date_string
from database import create_trade, get_user_trades

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.post("/execute")
async def execute_trade(
    request: TradeRequest,
    user_id: int = DEFAULT_USER_ID
):
    """Execute a trade (buy or sell) at a given price."""
    try:
        # Example: Get the current market price from DB. But here we just use the request price
        # In a real scenario, you'd confirm the actual current price or use db_get_market_data.
        
        trade_data = {
            "user_id": user_id,
            "market": "Energy",
            "trade_type": request.type,
            "quantity": request.quantity,
            "trade_price": request.price,
            "timestamp": datetime.fromisoformat(request.executionTime.replace('Z', '+00:00')),
            "status": "executed"
        }
        success = create_trade(trade_data)
        if success:
            # Generate a pseudo trade_id
            trade_id = hash(f"{user_id}-{request.type}-{request.quantity}-{request.executionTime}")
            return {
                "success": True,
                "trade_id": trade_id,
                "message": f"{request.type.capitalize()} trade executed for {request.quantity} units at €{request.price}"
            }
        else:
            return {"success": False, "message": "Failed to create trade record"}
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return {"success": False, "message": str(e)}

@router.get("/history")
async def get_trade_history_api(
    start_date: str = Query(None),
    end_date: str = Query(None),
    trade_type: str = Query(None),
    user_id: int = DEFAULT_USER_ID
):
    """Get trade history (optionally by date range or trade_type)."""
    try:
        start_date_obj = parse_date_string(start_date)
        end_date_obj = parse_date_string(end_date)
        
        trades = get_user_trades(user_id, start_date_obj, end_date_obj)
        if trade_type:
            trades = [t for t in trades if t.get("trade_type", "").lower() == trade_type.lower()]
        
        result = []
        for t in trades:
            price = t.get("trade_price", 0)
            quantity = t.get("quantity", 0)
            ttype = t.get("trade_type", "")
            profit_loss = 0
            if ttype.lower() == "sell":
                profit_loss = price * quantity
            elif ttype.lower() == "buy":
                profit_loss = -(price * quantity)
            
            result.append({
                "id": t.get("trade_id", 0),
                "type": ttype,
                "quantity": quantity,
                "price": price,
                "timestamp": t.get("timestamp", datetime.now()).isoformat(),
                "status": t.get("status", "executed"),
                "profit_loss": profit_loss
            })
        return result
    except Exception as e:
        logger.error(f"Error getting trade history: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 