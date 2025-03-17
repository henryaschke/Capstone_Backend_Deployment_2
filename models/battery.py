from pydantic import BaseModel, Field

class BatteryStatus(BaseModel):
    level: float = Field(..., description="Current battery level (%)")
    capacity: dict = Field(..., description="Battery capacity details")

class BatteryAction(BaseModel):
    quantity: float = Field(..., description="Amount to charge/discharge") 