from fastapi import APIRouter, Query, HTTPException, Depends
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta

from dependencies import get_current_user, parse_date_string
from database import get_performance_metrics

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get("/metrics")
async def get_performance_metrics_api(
    start_date: str = Query(None),
    end_date: str = Query(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get performance metrics for a user."""
    try:
        # Extract user_id from the current_user dictionary
        user_id = current_user.get("User_ID")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid user authentication")
            
        logger.info(f"Getting performance metrics for authenticated user_id: {user_id}")
        
        start_date_obj = parse_date_string(start_date)
        end_date_obj = parse_date_string(end_date)
        
        # Log the date range for debugging
        logger.info(f"Fetching metrics for date range: {start_date_obj} to {end_date_obj}")
        
        # Get metrics for executed trades
        metrics = get_performance_metrics(user_id, start_date_obj, end_date_obj)
        logger.info(f"Got metrics from database: {metrics}")
        
        # Generate some dummy chart data for demonstration
        chart_data = []
        today = datetime.now()
        for i in range(30):
            day_str = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            chart_data.append({
                "date": day_str,
                "profit": 1000 + (i * 100),
                "revenue": 2500 + (i * 200)
            })
        metrics["chartData"] = sorted(chart_data, key=lambda x: x["date"])
        
        # Add additional metadata to help with debugging
        metrics["dateRange"] = {
            "start": start_date,
            "end": end_date,
            "parsed_start": str(start_date_obj) if start_date_obj else None,
            "parsed_end": str(end_date_obj) if end_date_obj else None
        }
        
        logger.info(f"Returning metrics: {metrics}")
        return metrics
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 