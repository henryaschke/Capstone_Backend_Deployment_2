from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import random

from models.forecast import ForecastRequest, ForecastResponse
from dependencies import get_optional_user, client, DEFAULT_USER_ID
from forecasting import generate_forecasts, test_forecasts_table_insertion, save_forecasts_to_bigquery

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.post("/generate", response_model=ForecastResponse)
@router.get("/generate", response_model=ForecastResponse)
async def generate_price_forecasts(
    request: ForecastRequest = Body(default=ForecastRequest()),
    save_to_database: bool = True,
    user_id: int = 1,
    model_info: str = "RandomForestRegressor",
    current_user: Dict[str, Any] = Depends(get_optional_user)
):
    """
    Generate price forecasts for future time periods that haven't cleared yet.
    This uses a separate forecasting module 'generate_forecasts' for the logic.
    """
    try:
        # Handle both POST with request body and GET with query parameters
        # For GET requests, we'll use query parameters, for POST, we'll use the request body
        if request and isinstance(request, ForecastRequest):
            # Using request body from POST
            resolutions = request.resolutions
            save_to_db = request.save_to_database
            model = request.model_info
            uid = request.user_id
        else:
            # Using query parameters from GET
            resolutions = [15, 30, 60]  # Default resolutions
            save_to_db = save_to_database
            model = model_info
            uid = user_id
            
        logger.info(f"Generating price forecasts for resolutions: {resolutions}")
        
        if not client:
            raise HTTPException(status_code=500, detail="BigQuery client not initialized.")
        
        # Call your custom forecast function from forecasting.py
        forecasts = generate_forecasts(
            client=client,
            resolution_minutes=resolutions,
            save_to_database=save_to_db,
            model_info=model,
            user_id=uid
        )
        
        if forecasts.get("status") == "error":
            err = forecasts.get("error", "Unknown error in forecast generation.")
            raise HTTPException(status_code=500, detail=f"Forecast error: {err}")
        
        return forecasts
    except Exception as e:
        logger.error(f"Error in generate_price_forecasts endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/latest")
async def get_latest_forecasts(current_user: Dict[str, Any] = Depends(get_optional_user)):
    """Retrieve the latest generated forecasts without generating new ones."""
    # For demonstration, we'll just call generate_price_forecasts again.
    return await generate_price_forecasts()

@router.get("/saved")
async def get_saved_forecasts(
    date: Optional[str] = None,
    limit: int = 20,
    user_id: Optional[int] = None,
    resolution: Optional[int] = None,
    active_only: bool = False,
    current_user: Dict[str, Any] = Depends(get_optional_user)
):
    """Retrieve saved forecasts from the 'Forecasts' table."""
    try:
        if not client:
            raise HTTPException(status_code=500, detail="BigQuery client not initialized.")
        
        query = """
        SELECT 
            Forecast_id,
            Market,
            Forecast_Timestamp,
            Forecast_Value_Max,
            Forecast_Value_Min,
            Forecast_Value_Average,
            Generated_At,
            Model_Info,
            Resolution_Minutes,
            Accuracy_Metrics,
            Confidence_Interval_Upper,
            Confidence_Interval_Lower,
            User_ID,
            Is_Active,
            Last_Updated
        FROM `capstone-henry.capstone_db.Forecasts`
        WHERE 1=1
        """
        if date:
            query += f" AND DATE(Forecast_Timestamp) = '{date}'"
        if user_id is not None:
            query += f" AND User_ID = {user_id}"
        if resolution is not None:
            query += f" AND Resolution_Minutes = {resolution}"
        if active_only:
            query += " AND Is_Active = TRUE"
        
        query += f" ORDER BY Forecast_Timestamp DESC LIMIT {limit}"
        
        query_job = client.query(query)
        results = list(query_job.result())
        
        forecasts = []
        for row in results:
            forecast_dict = {
                "forecast_id": float(row.Forecast_id) if row.Forecast_id is not None else None,
                "market": row.Market,
                "timestamp": row.Forecast_Timestamp.isoformat() if row.Forecast_Timestamp else None,
                "max_value": float(row.Forecast_Value_Max) if row.Forecast_Value_Max else None,
                "min_value": float(row.Forecast_Value_Min) if row.Forecast_Value_Min else None,
                "avg_value": float(row.Forecast_Value_Average) if row.Forecast_Value_Average else None,
                "generated_at": row.Generated_At.isoformat() if row.Generated_At else None,
                "model_info": row.Model_Info,
                "resolution_minutes": row.Resolution_Minutes,
                "accuracy": float(row.Accuracy_Metrics) if row.Accuracy_Metrics else None,
                "confidence_upper": float(row.Confidence_Interval_Upper) if row.Confidence_Interval_Upper else None,
                "confidence_lower": float(row.Confidence_Interval_Lower) if row.Confidence_Interval_Lower else None,
                "user_id": row.User_ID,
                "is_active": row.Is_Active,
                "last_updated": row.Last_Updated.isoformat() if row.Last_Updated else None
            }
            forecasts.append(forecast_dict)
        
        return {"forecasts": forecasts, "count": len(forecasts), "status": "success"}
    except Exception as e:
        logger.error(f"Error retrieving saved forecasts: {e}")
        return {"forecasts": [], "count": 0, "status": "error", "error": str(e)} 