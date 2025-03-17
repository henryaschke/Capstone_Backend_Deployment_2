from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
from datetime import datetime

from dependencies import client

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get("/status")
async def get_api_status():
    """Check API and BigQuery status."""
    try:
        connection_status = "Connected" if client else "Disconnected"
        config_info = {
            "project_id": "capstone-henry",
            "dataset_id": "capstone_db",
            "main_table": "Market_Data_Germany_Today",
            "server_time": datetime.now().isoformat(),
            "database_connection": connection_status
        }
        
        # Test query to check BigQuery access
        if client:
            try:
                test_query = """
                    SELECT COUNT(*) as row_count
                    FROM `capstone-henry.capstone_db.Market_Data_Germany_Today`
                """
                query_job = client.query(test_query)
                result = list(query_job.result())
                row_count = result[0]['row_count'] if result else 0
                config_info["database_access"] = "Success"
                config_info["row_count"] = row_count
            except Exception as e:
                config_info["database_access"] = "Failed"
                config_info["database_error"] = str(e)
        else:
            config_info["database_access"] = "No Client"
        
        return {"status": "OK", "version": "1.0", "config": config_info}
    except Exception as e:
        logger.error(f"Error in get_api_status: {e}")
        return {
            "status": "Error",
            "error": str(e),
            "server_time": datetime.now().isoformat()
        }

@router.get("/diagnostic/query")
async def test_bigquery_query(
    project_id: str = "capstone-henry",
    dataset_id: str = "capstone_db",
    table_name: str = "Market_Data_Germany_Today",
    limit: int = 5
):
    """Simple diagnostic query endpoint to verify BigQuery configuration."""
    try:
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
async def test_users_table():
    """Check the 'Users' table schema."""
    try:
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
    except Exception as e:
        logger.error(f"Error checking Users table: {e}")
        return {"error": str(e), "message": "Failed to retrieve table structure"} 