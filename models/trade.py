from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class TradeRequest(BaseModel):
    type: str = Field(..., description="Trade type (buy/sell)")
    quantity: float = Field(..., description="Trade quantity in MWh")
    executionTime: str = Field(..., description="Scheduled time of execution in ISO format")
    resolution: int = Field(..., description="Market resolution in minutes (15, 30, or 60)")
    trade_id: Optional[int] = Field(None, description="Optional trade ID (integer)")
    user_id: Optional[int] = Field(None, description="User ID (can be provided from auth)")
    market: str = Field("Germany", description="Energy market (default: Germany)")

class AlgorithmSettings(BaseModel):
    settings: dict = Field(..., description="Algorithm configuration parameters") 