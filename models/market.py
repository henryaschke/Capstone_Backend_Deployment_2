from pydantic import BaseModel, Field
from typing import Optional, List

class MarketDataPoint(BaseModel):
    deliveryDay: str = Field(..., description="Delivery day in YYYY-MM-DD format")
    deliveryPeriod: str = Field(..., description="Delivery period e.g. '12:00-13:00'")
    cleared: bool = Field(..., description="Whether the contract has cleared")
    market: str = Field(..., description="Market name")
    high: Optional[float] = Field(None, description="Highest price")
    low: Optional[float] = Field(None, description="Lowest price")
    close: Optional[float] = Field(None, description="Closing price")
    open: Optional[float] = Field(None, description="Opening price")
    transactionVolume: Optional[float] = Field(None, description="Transaction volume")

class HistoricalMarketDataPoint(BaseModel):
    id: str = Field(..., description="Unique ID")
    date: str = Field(..., description="Delivery date")
    resolution: str = Field(..., description="Time resolution (e.g. '15min')")
    deliveryPeriod: str = Field(..., description="Time period")
    market: str = Field(..., description="Market identifier")
    highPrice: Optional[float] = Field(None, description="Highest price")
    lowPrice: Optional[float] = Field(None, description="Lowest price")
    averagePrice: Optional[float] = Field(None, description="Average price (VWAP)")
    openPrice: Optional[float] = Field(None, description="Opening price")
    closePrice: Optional[float] = Field(None, description="Closing price")
    buyVolume: Optional[float] = Field(None, description="Buy volume")
    sellVolume: Optional[float] = Field(None, description="Sell volume")
    volume: Optional[float] = Field(None, description="Total volume")
    vwap3h: Optional[float] = Field(None, description="3-hour volume-weighted average price")
    vwap1h: Optional[float] = Field(None, description="1-hour volume-weighted average price")
    contractOpenTime: Optional[str] = Field(None, description="Time contract opened")
    contractCloseTime: Optional[str] = Field(None, description="Time contract closed") 