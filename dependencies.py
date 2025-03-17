from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from google.cloud import bigquery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# JWT Authentication Configuration
# ------------------------------------------------------------------------------
SECRET_KEY = "not_so_secret_key_please_change_in_production"  # Replace with a secure key in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 5  # Session timeout after 5 minutes of inactivity

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# OAuth2 scheme for optional token authentication (no auto-error if token is missing)
class OptionalOAuth2PasswordBearer(OAuth2PasswordBearer):
    async def __call__(self, request: Request) -> Optional[str]:
        try:
            return await super().__call__(request)
        except HTTPException:
            return None

optional_oauth2_scheme = OptionalOAuth2PasswordBearer(tokenUrl="/api/auth/login")

# Default user ID for testing/demos
DEFAULT_USER_ID = 1

# ------------------------------------------------------------------------------
# Global BigQuery Client Initialization (Using Default Application Credentials)
# ------------------------------------------------------------------------------
try:
    # Simply use bigquery.Client() for default credentials.
    # In Cloud Run, this uses the attached service account automatically.
    client = bigquery.Client()
    logger.info("BigQuery client initialized successfully with default application credentials")
except Exception as e:
    logger.error(f"Error initializing BigQuery client: {e}")
    client = None

# ------------------------------------------------------------------------------
# Helper Functions for Authentication
# ------------------------------------------------------------------------------

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate a password hash."""
    return pwd_context.hash(password)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    
    # Ensure "sub" is a string if present
    if "sub" in to_encode and to_encode["sub"] is not None:
        to_encode["sub"] = str(to_encode["sub"])
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def parse_date_string(date_str: Optional[str]) -> Optional[datetime]:
    """
    Helper function to parse a date or datetime string into a datetime object.
    Tries ISO-8601 and then "YYYY-MM-DD".
    """
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except ValueError:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return None

# ------------------------------------------------------------------------------
# Database-Like Functions for Authentication (Direct BigQuery Queries)
# ------------------------------------------------------------------------------

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get a user by email from the BigQuery 'Users' table."""
    try:
        if not client:
            logger.error("BigQuery client is not initialized.")
            return None
        
        query = """
        SELECT * FROM `capstone-henry.capstone_db.Users`
        WHERE Email = @email
        LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("email", "STRING", email),
            ]
        )
        query_job = client.query(query, job_config=job_config)
        results = query_job.result()
        rows = list(results)
        if not rows:
            return None
        
        return dict(rows[0].items())
    except Exception as e:
        logger.error(f"Error getting user by email: {str(e)}")
        return None

def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Get a user by ID from the BigQuery 'Users' table."""
    try:
        if not client:
            logger.error("BigQuery client is not initialized.")
            return None
        
        query = """
        SELECT * FROM `capstone-henry.capstone_db.Users`
        WHERE User_ID = @user_id
        LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "INT64", int(user_id)),
            ]
        )
        query_job = client.query(query, job_config=job_config)
        results = query_job.result()
        rows = list(results)
        if not rows:
            return None
        
        return dict(rows[0].items())
    except Exception as e:
        logger.error(f"Error getting user by ID: {str(e)}")
        return None

def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate a user by checking their email and password."""
    user = get_user_by_email(email)
    if not user:
        return None
    # Compare hashed password
    if not verify_password(password, user["Hashed_Password"]):
        return None
    return user

def create_user_in_bigquery(email: str, password: str, user_role: str = "user") -> Optional[Dict[str, Any]]:
    """Create a new user in the 'Users' table."""
    try:
        if not client:
            logger.error("BigQuery client is not initialized.")
            return None
        
        # Check if user already exists
        existing = get_user_by_email(email)
        if existing:
            return None
        
        # Generate new user ID
        query = "SELECT MAX(User_ID) as max_id FROM `capstone-henry.capstone_db.Users`"
        query_job = client.query(query)
        results = list(query_job.result())
        
        new_user_id = 1
        if results and results[0]["max_id"]:
            new_user_id = results[0]["max_id"] + 1
        
        # Hash the password
        hashed_password = get_password_hash(password)
        now = datetime.utcnow()
        
        # Insert the new user
        insert_query = """
        INSERT INTO `capstone-henry.capstone_db.Users`
        (User_ID, Email, Hashed_Password, Created_At, Updatet_At, User_Role)
        VALUES (@user_id, @email, @password, @created_at, @updated_at, @user_role)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "INT64", new_user_id),
                bigquery.ScalarQueryParameter("email", "STRING", email),
                bigquery.ScalarQueryParameter("password", "STRING", hashed_password),
                bigquery.ScalarQueryParameter("created_at", "TIMESTAMP", now),
                bigquery.ScalarQueryParameter("updated_at", "TIMESTAMP", now),
                bigquery.ScalarQueryParameter("user_role", "STRING", user_role),
            ]
        )
        client.query(insert_query, job_config=job_config).result()
        
        return {
            "User_ID": new_user_id,
            "Email": email,
            "Created_At": now,
            "Updatet_At": now,
            "User_Role": user_role
        }
    except Exception as e:
        logger.error(f"Error creating user in BigQuery: {e}")
        return None

# ------------------------------------------------------------------------------
# Dependency for Getting the Current User from the Token
# ------------------------------------------------------------------------------

async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """Extract the current user from the JWT token (raises 401 if invalid)."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        subject = payload.get("sub")
        if subject is None:
            raise credentials_exception
        
        # Convert subject to an integer ID
        try:
            user_id = int(subject)
        except ValueError:
            raise credentials_exception
        
    except JWTError as e:
        logger.error(f"JWT decode error: {str(e)}")
        raise credentials_exception
    
    user = get_user_by_id(user_id)
    if not user:
        raise credentials_exception
    
    return user

async def get_optional_user(token: str = Depends(optional_oauth2_scheme)) -> Optional[Dict[str, Any]]:
    """
    Same as get_current_user, but returns None if no/invalid token,
    instead of raising an exception.
    """
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        subject = payload.get("sub")
        if subject is None:
            return None
        
        user_id = int(subject)
        user = get_user_by_id(user_id)
        return user
    except Exception as e:
        logger.warning(f"Ignoring authentication error for optional user: {e}")
        return None 