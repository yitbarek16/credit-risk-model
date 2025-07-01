from pydantic import BaseModel
from typing import Optional


# Define expected input structure based on model's features
class CustomerData(BaseModel):
    Amount: float
    Value: float
    PricingStrategy: Optional[float]
    FraudResult: Optional[float]
    txn_hour: int
    txn_day: int
    txn_month: int
    txn_year: int
    ProductCategory: str
    ChannelId: str
    ProviderId: str


# Define the output schema
class RiskResponse(BaseModel):
    risk_probability: float
