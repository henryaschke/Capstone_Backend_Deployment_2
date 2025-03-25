from google.cloud import bigquery
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BigQuery project and dataset info
PROJECT_ID = "capstone-henry"
DATASET_ID = "capstone_db"

def update_trade_timestamp():
    """Update the timestamp for trade with 7:30 showing as 8:30 in UI"""
    try:
        # Initialize BigQuery client
        client = bigquery.Client()
        
        # Show current information
        query = f"""
            SELECT Trade_ID, User_ID, Timestamp, Trade_Type, Status
            FROM `{PROJECT_ID}.{DATASET_ID}.Trades`
            WHERE User_ID = 12346
            AND Timestamp >= '2025-03-24'
            AND Timestamp < '2025-03-25'
        """
        
        # Run the query
        query_job = client.query(query)
        rows = list(query_job)
        
        print(f"Found {len(rows)} trades on 2025-03-24 for user 12346")
        for row in rows:
            print(f"Trade_ID: {row['Trade_ID']}, Timestamp: {row['Timestamp']}, Type: {row['Trade_Type']}, Status: {row['Status']}")
        
        # Update the timestamp (UTC 7:30 → UTC 6:30 to display as CET 7:30)
        update_query = f"""
            UPDATE `{PROJECT_ID}.{DATASET_ID}.Trades`
            SET Timestamp = '2025-03-24T06:30:00Z'
            WHERE User_ID = 12346
            AND Timestamp >= '2025-03-24'
            AND Timestamp < '2025-03-25'
            AND FORMAT_TIMESTAMP('%H:%M', Timestamp) = '07:30'
        """
        
        update_job = client.query(update_query)
        update_job.result()
        
        print("Updated timestamp. Verifying changes...")
        
        # Verify the update worked
        verify_job = client.query(query)
        updated_rows = list(verify_job)
        
        for row in updated_rows:
            print(f"After update - Trade_ID: {row['Trade_ID']}, Timestamp: {row['Timestamp']}, Type: {row['Trade_Type']}, Status: {row['Status']}")
        
        return True
    except Exception as e:
        print(f"Error updating trade timestamp: {e}")
        return False

if __name__ == "__main__":
    update_trade_timestamp() 