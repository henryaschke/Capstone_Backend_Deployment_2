from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BigQuery project and dataset info (hard-coded here for simplicity, 
# but you can also get these from environment variables if preferred)
PROJECT_ID = "capstone-henry"
DATASET_ID = "capstone_db"

class BigQueryDatabase:
    def __init__(self):
        """
        Initializes a BigQuery client using default application credentials.
        In Cloud Run, this picks up the service account credentials if the 
        service account is granted the proper BigQuery IAM roles.
        """
        self.client = None
        try:
            # Using default credentials – no need for a service account JSON file.
            self.client = bigquery.Client()
            logger.info("BigQuery client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing BigQuery client: {e}")
            raise
    
    def get_table_ref(self, table_name: str) -> str:
        """Get fully qualified table reference: <PROJECT>.<DATASET>.<TABLE>."""
        return f"{PROJECT_ID}.{DATASET_ID}.{table_name}"
    
    def execute_query(
        self, 
        query: str, 
        params: Optional[List[bigquery.ScalarQueryParameter]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a BigQuery query and return the results as a list of dictionaries.
        Optionally supports parameterized queries via params.
        """
        try:
            job_config = None
            if params:
                job_config = bigquery.QueryJobConfig(query_parameters=params)
            
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            # Convert results to a list of dicts
            return [dict(row.items()) for row in results]
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            logger.error(f"Query: {query}")
            if params:
                logger.error(f"Params: {params}")
            raise
    
    def insert_row(self, table_name: str, row_data: Dict[str, Any]) -> bool:
        """
        Insert a single row (as a dictionary) into a BigQuery table.
        Returns True if successful, False otherwise.
        """
        try:
            table_ref = self.get_table_ref(table_name)
            table = self.client.get_table(table_ref)
            
            errors = self.client.insert_rows_json(table, [row_data])
            if errors:
                logger.error(f"Errors inserting into {table_name}: {errors}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error inserting row into {table_name}: {e}")
            return False
    
    def update_row(
        self, 
        table_name: str, 
        update_data: Dict[str, Any], 
        condition_field: str, 
        condition_value: Any
    ) -> bool:
        """
        Update rows in a BigQuery table using a straightforward UPDATE statement
        (or a MERGE operation if needed). This version demonstrates a simple
        UPDATE approach with query parameters.
        """
        try:
            table_ref = self.get_table_ref(table_name)
            
            # Create SET clause for the UPDATE statement
            set_clause = ", ".join([f"{key} = @{key}" for key in update_data.keys()])
            
            # Build query parameters
            params = [
                bigquery.ScalarQueryParameter(
                    condition_field, 
                    self._get_bigquery_type(condition_value), 
                    condition_value
                )
            ]
            for key, value in update_data.items():
                params.append(
                    bigquery.ScalarQueryParameter(
                        key, 
                        self._get_bigquery_type(value), 
                        value
                    )
                )
            
            # Construct the update query
            query = f"""
                UPDATE `{table_ref}`
                SET {set_clause}
                WHERE {condition_field} = @{condition_field}
            """
            
            job_config = bigquery.QueryJobConfig(query_parameters=params)
            query_job = self.client.query(query, job_config=job_config)
            query_job.result()  # Wait for job completion
            
            return True
        except Exception as e:
            logger.error(f"Error updating row in {table_name}: {e}")
            return False
    
    def _get_bigquery_type(self, value: Any) -> str:
        """
        Helper to map Python types to BigQuery parameter types.
        Adjust as needed for more complex data types.
        """
        if isinstance(value, str):
            return "STRING"
        elif isinstance(value, int):
            return "INTEGER"
        elif isinstance(value, float):
            return "FLOAT"
        elif isinstance(value, bool):
            return "BOOLEAN"
        elif isinstance(value, datetime):
            return "TIMESTAMP"
        else:
            return "STRING"  # Default fallback

# ------------------------------------------------------------------
# Below are top-level helper functions that create and use BigQueryDatabase.
# ------------------------------------------------------------------

def get_db() -> BigQueryDatabase:
    """Instantiate and return a BigQueryDatabase object."""
    return BigQueryDatabase()

# Example user-related functions

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Fetch a user by their email address."""
    db = get_db()
    query = f"""
        SELECT * 
        FROM `{db.get_table_ref("Users")}`
        WHERE Email = @email
    """
    params = [bigquery.ScalarQueryParameter("email", "STRING", email)]
    results = db.execute_query(query, params)
    return results[0] if results else None

def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a user by their integer user ID."""
    db = get_db()
    query = f"""
        SELECT * 
        FROM `{db.get_table_ref("Users")}`
        WHERE User_ID = @user_id
    """
    params = [bigquery.ScalarQueryParameter("user_id", "INTEGER", user_id)]
    results = db.execute_query(query, params)
    return results[0] if results else None

# Portfolio functions

def get_portfolio_by_user_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Get a portfolio record by user ID."""
    db = get_db()
    query = f"""
        SELECT *
        FROM `{db.get_table_ref("Portfolio")}`
        WHERE User_ID = @user_id
    """
    params = [bigquery.ScalarQueryParameter("user_id", "INTEGER", user_id)]
    results = db.execute_query(query, params)
    return results[0] if results else None

# Trade functions

def get_user_trades(
    user_id: int, 
    start_date: Optional[datetime] = None, 
    end_date: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Return a user's trades, optionally filtered by a date range.
    """
    db = get_db()
    query = f"""
        SELECT *
        FROM `{db.get_table_ref("Trades")}`
        WHERE user_id = @user_id
    """
    params = [bigquery.ScalarQueryParameter("user_id", "INTEGER", user_id)]
    
    if start_date:
        query += " AND timestamp >= @start_date"
        params.append(bigquery.ScalarQueryParameter("start_date", "TIMESTAMP", start_date))
    
    if end_date:
        query += " AND timestamp <= @end_date"
        params.append(bigquery.ScalarQueryParameter("end_date", "TIMESTAMP", end_date))
    
    query += " ORDER BY timestamp DESC"
    
    return db.execute_query(query, params)

def create_trade(trade_data: Dict[str, Any]) -> bool:
    """Insert a new trade record."""
    db = get_db()
    return db.insert_row("Trades", trade_data)

# Market data functions

def get_market_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    market: str = "Germany"
) -> List[Dict[str, Any]]:
    """
    Return historical market data from a relevant table, with optional filters.
    """
    db = get_db()
    table_name = "Market_Data" if market != "Germany" else "Market_Data_Germany"
    query = f"""
        SELECT * 
        FROM `{db.get_table_ref(table_name)}`
        WHERE 1=1
    """
    params = []
    
    if start_date:
        query += " AND Delivery_Day >= @start_date"
        params.append(bigquery.ScalarQueryParameter("start_date", "STRING", start_date))
    
    if end_date:
        query += " AND Delivery_Day <= @end_date"
        params.append(bigquery.ScalarQueryParameter("end_date", "STRING", end_date))
    
    if min_price:
        query += " AND Low >= @min_price"
        params.append(bigquery.ScalarQueryParameter("min_price", "FLOAT", min_price))
    
    if max_price:
        query += " AND High <= @max_price"
        params.append(bigquery.ScalarQueryParameter("max_price", "FLOAT", max_price))
    
    query += " ORDER BY Delivery_Day DESC, Delivery_Period ASC"
    query += " LIMIT 1000"  # Keep result size manageable
    
    return db.execute_query(query, params)

# Forecast functions

def get_forecasts(
    market: str = "Germany",
    start_timestamp: Optional[datetime] = None,
    end_timestamp: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Fetch forecast records for a given market from the Forecasts table, 
    optionally filtered by timestamps.
    """
    db = get_db()
    query = f"""
        SELECT *
        FROM `{db.get_table_ref("Forecasts")}`
        WHERE market = @market
    """
    params = [bigquery.ScalarQueryParameter("market", "STRING", market)]
    
    if start_timestamp:
        query += " AND forecast_timestamp >= @start_timestamp"
        params.append(bigquery.ScalarQueryParameter("start_timestamp", "TIMESTAMP", start_timestamp))
    
    if end_timestamp:
        query += " AND forecast_timestamp <= @end_timestamp"
        params.append(bigquery.ScalarQueryParameter("end_timestamp", "TIMESTAMP", end_timestamp))
    
    query += " ORDER BY forecast_timestamp ASC"
    
    return db.execute_query(query, params)

# Battery table example

def get_battery_status(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Get battery status for a user. If the Battery table doesn't exist, return a dummy response.
    """
    db = get_db()
    
    # Check if the Battery table exists
    check_table_query = f"""
        SELECT table_name 
        FROM `{PROJECT_ID}.{DATASET_ID}.INFORMATION_SCHEMA.TABLES` 
        WHERE table_name = 'Battery'
    """
    
    table_exists = db.execute_query(check_table_query)
    
    if not table_exists:
        # Return a dummy response if no table
        return {
            "user_id": user_id,
            "current_level": 50.0,
            "total_capacity": 2.5,
            "usable_capacity": 2.0,
            "last_updated": datetime.now().isoformat()
        }
    
    # If the Battery table does exist, query it
    query = f"""
        SELECT * 
        FROM `{db.get_table_ref("Battery")}`
        WHERE user_id = @user_id
    """
    params = [bigquery.ScalarQueryParameter("user_id", "INTEGER", user_id)]
    results = db.execute_query(query, params)
    return results[0] if results else None

# Performance metrics functions

def get_performance_metrics(
    user_id: int, 
    start_date: Optional[datetime] = None, 
    end_date: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Example performance metrics aggregator that uses trades and portfolio data.
    """
    db = get_db()
    
    # Calculate total revenue, profit, etc. from trades
    trades_query = f"""
        SELECT 
            SUM(CASE WHEN trade_type = 'sell' THEN quantity * trade_price ELSE 0 END) as total_revenue,
            SUM(CASE WHEN trade_type = 'sell' THEN quantity * trade_price ELSE -quantity * trade_price END) as total_profit,
            COUNT(*) as trade_count
        FROM `{db.get_table_ref("Trades")}`
        WHERE user_id = @user_id
    """
    params = [bigquery.ScalarQueryParameter("user_id", "INTEGER", user_id)]
    
    if start_date:
        trades_query += " AND timestamp >= @start_date"
        params.append(bigquery.ScalarQueryParameter("start_date", "TIMESTAMP", start_date))
    
    if end_date:
        trades_query += " AND timestamp <= @end_date"
        params.append(bigquery.ScalarQueryParameter("end_date", "TIMESTAMP", end_date))
    
    trades_results = db.execute_query(trades_query, params)
    
    # Also get portfolio data
    portfolio = get_portfolio_by_user_id(user_id)
    
    # Combine the data
    metrics = {
        "totalRevenue": trades_results[0].get("total_revenue", 0) if trades_results else 0,
        "totalProfit": trades_results[0].get("total_profit", 0) if trades_results else 0,
        "totalCosts": 0,      # We'll derive below
        "profitMargin": 0,
        "totalVolume": 0,     # We'll fetch from a separate query below
        "accuracy": 92,       # Hard-coded example
        "currentBalance": portfolio.get("Current_Balance", 0) if portfolio else 0,
        "cumulativeProfitLoss": portfolio.get("Cumulative_Profit_Loss", 0) if portfolio else 0
    }
    
    # totalCosts = totalRevenue - totalProfit
    if trades_results:
        revenue = trades_results[0].get("total_revenue", 0) or 0
        profit = trades_results[0].get("total_profit", 0) or 0
        metrics["totalCosts"] = revenue - profit
    
    # Calculate profit margin
    if metrics["totalRevenue"] > 0:
        metrics["profitMargin"] = (metrics["totalProfit"] / metrics["totalRevenue"]) * 100
    
    # Get total volume from trades
    volume_query = f"""
        SELECT SUM(quantity) as total_volume
        FROM `{db.get_table_ref("Trades")}`
        WHERE user_id = @user_id
    """
    volume_results = db.execute_query(volume_query, params)
    
    if volume_results and volume_results[0].get("total_volume") is not None:
        metrics["totalVolume"] = volume_results[0]["total_volume"]
    
    return metrics

def test_bigquery_connection() -> bool:
    """
    Simple connectivity check to ensure queries can run without error.
    Returns True if successful, False otherwise.
    """
    try:
        db = get_db()
        query = "SELECT 1 as test"
        db.execute_query(query)
        return True
    except Exception as e:
        logger.error(f"BigQuery connection test failed: {e}")
        return False

# Optional main check
if __name__ == "__main__":
    connection_success = test_bigquery_connection()
    print(f"BigQuery connection test: {'Successful' if connection_success else 'Failed'}")
