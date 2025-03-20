from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import logging
from datetime import datetime
import platform

from dependencies import get_current_user
from database import test_bigquery_connection, get_bigquery_client

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Get a BigQuery client instance
client = get_bigquery_client()

@router.get("/status")
async def get_api_status():
    """Get the API status and basic diagnostics."""
    try:
        # Check database connection
        db_connected = test_bigquery_connection()
        
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "database_connected": db_connected,
            "environment": platform.platform()
        }
    except Exception as e:
        logger.error(f"Error getting API status: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/diagnostic/query")
async def test_bigquery_query(
    project_id: str = "capstone-henry",
    dataset_id: str = "capstone_db",
    table_name: str = "Market_Data_Germany_Today",
    limit: int = 5,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Simple diagnostic query endpoint to verify BigQuery configuration."""
    try:
        # Check if the user is authenticated
        user_id = current_user.get("User_ID")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid user authentication")
            
        logger.info(f"Diagnostic query executed by user {current_user.get('Email')} (ID: {user_id})")
        
        if not client:
            return {
                "success": False,
                "error": "BigQuery client not initialized",
                "configuration": {
                    "project_id": project_id,
                    "dataset_id": dataset_id,
                    "table_name": table_name
                }
            }
        
        query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_name}` LIMIT {limit}"
        query_job = client.query(query)
        results = query_job.result()
        
        rows = [dict(r.items()) for r in results]
        columns = [field.name for field in query_job.schema] if query_job.schema else []
        
        return {
            "success": True,
            "query": query,
            "columns": columns,
            "row_count": len(rows),
            "sample_data": rows,
            "configuration": {
                "project_id": project_id,
                "dataset_id": dataset_id,
                "table_name": table_name
            }
        }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Diagnostic query error: {e}")
        return {
            "success": False,
            "error": str(e),
            "configuration": {
                "project_id": project_id,
                "dataset_id": dataset_id,
                "table_name": table_name
            }
        }

@router.get("/test/users-table")
async def test_users_table(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Check the 'Users' table schema."""
    try:
        # Check if the user is authenticated
        user_id = current_user.get("User_ID")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid user authentication")
            
        # Check if the user has admin role
        user_role = current_user.get("User_Role", "").lower()
        if user_role != "admin":
            logger.warning(f"Unauthorized access attempt to users-table by user {current_user.get('Email')} with role {user_role}")
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to access this endpoint. Admin role required."
            )
            
        logger.info(f"Users-table check executed by admin user {current_user.get('Email')} (ID: {user_id})")
        
        if not client:
            return {"error": "BigQuery client not initialized", "message": "Failed"}
        
        table_ref = client.dataset("capstone_db").table("Users")
        table = client.get_table(table_ref)
        schema_info = [{"name": f.name, "type": f.field_type} for f in table.schema]
        
        query = "SELECT * FROM `capstone-henry.capstone_db.Users` LIMIT 1"
        query_job = client.query(query)
        rows = list(query_job.result())
        sample_row = dict(rows[0]) if rows else None
        
        return {"schema": schema_info, "sample_row": sample_row, "message": "Success"}
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error checking Users table: {e}")
        return {"error": str(e), "message": "Failed to retrieve table structure"}

@router.post("/admin/execute-query")
async def admin_execute_query(
    query_data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Admin endpoint to execute raw SQL queries.
    Requires admin authentication.
    """
    try:
        # Check if the user has admin role
        user_role = current_user.get("User_Role", "").lower()
        if user_role != "admin":
            logger.warning(f"Unauthorized access attempt to admin endpoint by user {current_user.get('Email')} with role {user_role}")
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to access this endpoint. Admin role required."
            )
            
        logger.info(f"Admin query execution by user {current_user.get('Email')} (ID: {current_user.get('User_ID')})")
        
        if not client:
            return {"success": False, "error": "BigQuery client not initialized"}
        
        query = query_data.get("query")
        if not query:
            return {"success": False, "error": "No query provided"}
        
        # Execute the query
        logger.info(f"Executing admin query: {query}")
        query_job = client.query(query)
        
        # Try to get results (for SELECT queries)
        try:
            results = list(query_job.result())
            rows = [dict(r.items()) for r in results]
            return {
                "success": True,
                "message": f"Query executed successfully with {len(rows)} results",
                "results": rows[:10],  # Limit results to avoid huge responses
                "executed_by": {
                    "user_id": current_user.get("User_ID"),
                    "email": current_user.get("Email")
                }
            }
        except Exception:
            # For INSERT, UPDATE, etc. queries that don't return results
            return {
                "success": True,
                "message": "Query executed successfully with no results",
                "affected_rows": query_job.num_dml_affected_rows if hasattr(query_job, 'num_dml_affected_rows') else 0,
                "executed_by": {
                    "user_id": current_user.get("User_ID"),
                    "email": current_user.get("Email")
                }
            }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error executing admin query: {e}")
        return {"success": False, "error": str(e)} 