from pydantic import BaseModel, Field
from typing import Optional

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: int
    email: str
    user_role: str

class TokenData(BaseModel):
    user_id: Optional[int] = None

class UserCreate(BaseModel):
    email: str
    password: str
    user_role: str = "user"

class UserLogin(BaseModel):
    email: str
    password: str 