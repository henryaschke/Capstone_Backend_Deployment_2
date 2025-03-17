from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class ForecastRequest(BaseModel):
    resolutions: List[int] = Field(default=[15, 30, 60], description="Resolutions in minutes to forecast for")
    lookback_days: int = Field(default=30, description="Days of historical data to use")
    save_to_database: bool = Field(default=True, description="Whether to save forecasts")
    model_info: str = Field(default="RandomForestRegressor", description="Model used")
    user_id: int = Field(default=1, description="ID of user requesting the forecast")

class ForecastPeriod(BaseModel):
    Delivery_Period: str = Field(..., description="Time period (e.g., '12:00-12:15')")
    PredictedHigh: float = Field(..., description="Predicted high price")
    PredictedLow: float = Field(..., description="Predicted low price")
    ResolutionMinutes: int = Field(..., description="Resolution in minutes")
    Confidence_Upper: Optional[float] = Field(None, description="Upper confidence interval")
    Confidence_Lower: Optional[float] = Field(None, description="Lower confidence interval")

class ForecastResponse(BaseModel):
    forecasts: Dict[str, List[ForecastPeriod]] = Field(..., description="Forecast data by resolution")
    timestamp: str = Field(..., description="Time forecasts were generated")
    status: str = Field(..., description="Forecast operation status")
    error: Optional[str] = Field(None, description="Error message if any")
    saved_to_database: Optional[bool] = Field(None, description="Whether forecasts were saved") 