from pydantic import BaseModel, Field

class TradeRequest(BaseModel):
    type: str = Field(..., description="Trade type (buy/sell)")
    quantity: float = Field(..., description="Trade quantity")
    price: float = Field(..., description="Trade price")
    executionTime: str = Field(..., description="Time of execution")

class AlgorithmSettings(BaseModel):
    settings: dict = Field(..., description="Algorithm configuration parameters") 