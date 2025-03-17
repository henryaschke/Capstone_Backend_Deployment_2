from fastapi import APIRouter, Depends, HTTPException, status, Request, Body
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
import logging
from typing import Dict, Any

from models.auth import Token, UserCreate
from dependencies import (
    get_current_user, create_access_token, authenticate_user, 
    get_user_by_email, create_user_in_bigquery, get_user_by_id,
    ACCESS_TOKEN_EXPIRE_MINUTES, SECRET_KEY, ALGORITHM
)
from jose import jwt

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.post("/register", response_model=Dict[str, Any])
async def register(user_data: UserCreate = Body(...)):
    """Register a new user."""
    try:
        logger.info(f"Register attempt for email: {user_data.email}")
        
        existing_user = get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        db_user = create_user_in_bigquery(user_data.email, user_data.password, user_data.user_role)
        if not db_user:
            raise HTTPException(status_code=500, detail="Failed to create user")
        
        return {"message": "User registered successfully", "user_id": db_user["User_ID"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login a user and return a JWT token."""
    try:
        logger.info(f"Login attempt for {form_data.username}")
        user = authenticate_user(form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token({"sub": user["User_ID"]}, access_token_expires)
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": user["User_ID"],
            "email": user["Email"],
            "user_role": user["User_Role"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/refresh", response_model=Token)
async def refresh_token(request: Request):
    """Refresh a user's token even if the current one is expired."""
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid Authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        old_token = auth_header.replace("Bearer ", "")
        # Decode without checking expiration
        payload = jwt.decode(old_token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": False})
        user_id_str = payload.get("sub")
        if not user_id_str:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user_id = int(user_id_str)
        user = get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: user not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create a new token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        new_token = create_access_token({"sub": str(user_id)}, access_token_expires)
        
        return {
            "access_token": new_token,
            "token_type": "bearer",
            "user_id": user_id,
            "email": user["Email"],
            "user_role": user["User_Role"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during token refresh",
        )

@router.get("/me", response_model=Dict[str, Any])
async def read_users_me(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get information about the current user."""
    return {
        "user_id": current_user["User_ID"],
        "email": current_user["Email"],
        "user_role": current_user["User_Role"],
        "created_at": current_user["Created_At"],
        "updated_at": current_user["Updatet_At"]
    }

@router.get("/protected-data")
async def get_protected_data(current_user: dict = Depends(get_current_user)):
    """Example protected endpoint."""
    return {
        "message": f"Hello {current_user['Email']}, this is protected data",
        "user_id": current_user["User_ID"],
        "user_role": current_user["User_Role"]
    } 