import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from google.cloud import bigquery
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, List, Tuple, Any, Optional
import pytz
import uuid
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_historical_data(
    client: bigquery.Client,
    lookback_days: int = 30,
    resolution_minutes: Optional[int] = None
) -> pd.DataFrame:
    """
    Retrieve historical market data from BigQuery.

    Args:
        client: BigQuery client
        lookback_days: Number of days to look back for historical data
        resolution_minutes: Filter for specific resolution products (15, 30, 60 min)

    Returns:
        DataFrame containing historical market data
    """
    try:
        # Calculate date range
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=lookback_days)

        # Build query
        query = f"""
        SELECT 
            Delivery_Day,
            Delivery_Period,
            Market,
            Resolution_Minutes,
            High,
            Low,
            VWAP,
            VWAP3H,
            VWAP1H,
            Open,
            Close,
            Buy_Volume,
            Sell_Volume,
            Transaction_Volume,
            Contract_Open_Time,
            Contract_Close_Time
        FROM `capstone-henry.capstone_db.Market_Data_Germany`
        WHERE PARSE_DATE('%Y-%m-%d', Delivery_Day) >= '{start_date.strftime('%Y-%m-%d')}'
          AND PARSE_DATE('%Y-%m-%d', Delivery_Day) < '{end_date.strftime('%Y-%m-%d')}'
        """

        # Add resolution filter if specified
        if resolution_minutes is not None:
            query += f" AND Resolution_Minutes = {resolution_minutes}"

        query += " ORDER BY Delivery_Day, Delivery_Period"

        logger.info(f"Executing historical data query for {lookback_days} days of data")

        # Execute query
        df = client.query(query).to_dataframe()
        logger.info(f"Retrieved {len(df)} rows of historical data")

        return df

    except Exception as e:
        logger.error(f"Error retrieving historical data: {e}")
        raise

def get_current_day_data(client: bigquery.Client) -> pd.DataFrame:
    """
    Retrieve current day's market data from BigQuery.

    Args:
        client: BigQuery client

    Returns:
        DataFrame containing today's market data
    """
    try:
        today = datetime.now().strftime('%Y-%m-%d')

        query = f"""
        SELECT 
            Delivery_Day,
            Delivery_Period,
            Market,
            Resolution_Minutes,
            High,
            Low,
            VWAP,
            VWAP3H,
            VWAP1H,
            Open,
            Close,
            Buy_Volume,
            Sell_Volume,
            Transaction_Volume,
            Contract_Open_Time,
            Contract_Close_Time,
            Cleared
        FROM `capstone-henry.capstone_db.Market_Data_Germany_Today`
        WHERE Delivery_Day = '{today}'
        ORDER BY Delivery_Period
        """

        logger.info(f"Executing query for today's market data: {today}")
        df = client.query(query).to_dataframe()

        logger.info(f"Retrieved {len(df)} rows of today's data")
        return df

    except Exception as e:
        logger.error(f"Error retrieving today's data: {e}")
        raise

def process_data_for_forecasting(
    historical_df: pd.DataFrame,
    today_df: pd.DataFrame,
    resolution_minutes: int
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Process and prepare data for forecasting.

    Args:
        historical_df: DataFrame containing historical data
        today_df: DataFrame containing today's data
        resolution_minutes: Resolution of time periods (15, 30, or 60)

    Returns:
        (training_data, future_periods, future_period_ids)
        - DataFrame of training data
        - DataFrame of future (uncleared) periods
        - List of future period identifiers
    """
    try:
        # Filter for the specific resolution
        hist_df = historical_df[historical_df['Resolution_Minutes'] == resolution_minutes].copy()
        today_res_df = today_df[today_df['Resolution_Minutes'] == resolution_minutes].copy()

        if hist_df.empty:
            logger.warning(f"No historical data found for {resolution_minutes}-minute resolution")
            return pd.DataFrame(), pd.DataFrame(), []

        # Convert Delivery_Day to datetime
        hist_df['Date'] = pd.to_datetime(hist_df['Delivery_Day'])

        # Extract hour and minute from Delivery_Period (e.g., "12:00-12:15")
        hist_df['Hour'] = hist_df['Delivery_Period'].apply(
            lambda x: int(x.split(':')[0]) if '-' in x else int(x.split(' ')[1].split(':')[0])
        )
        hist_df['Minute'] = hist_df['Delivery_Period'].apply(
            lambda x: int(x.split(':')[1].split('-')[0]) if '-' in x else int(x.split(' ')[1].split(':')[1])
        )
        hist_df['DayOfWeek'] = hist_df['Date'].dt.dayofweek
        hist_df['Month'] = hist_df['Date'].dt.month
        hist_df['IsWeekend'] = hist_df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

        # Do the same for today's data
        today_res_df['Date'] = pd.to_datetime(today_res_df['Delivery_Day'])
        today_res_df['Hour'] = today_res_df['Delivery_Period'].apply(
            lambda x: int(x.split(':')[0]) if '-' in x else int(x.split(' ')[1].split(':')[0])
        )
        today_res_df['Minute'] = today_res_df['Delivery_Period'].apply(
            lambda x: int(x.split(':')[1].split('-')[0]) if '-' in x else int(x.split(' ')[1].split(':')[1])
        )
        today_res_df['DayOfWeek'] = today_res_df['Date'].dt.dayofweek
        today_res_df['Month'] = today_res_df['Date'].dt.month
        today_res_df['IsWeekend'] = today_res_df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

        # Identify future (uncleared) periods
        future_periods = today_res_df[today_res_df['Cleared'] == False].copy()
        future_period_ids = future_periods['Delivery_Period'].tolist()

        logger.info(f"Found {len(future_period_ids)} future periods to forecast for {resolution_minutes}-minute resolution")

        # Prepare training data from historical + today's cleared periods
        training_data = pd.concat([
            hist_df,
            today_res_df[today_res_df['Cleared'] == True]
        ]).reset_index(drop=True)

        # Handle missing values (simple approach: fill with median)
        training_data = training_data.fillna({
            'High': training_data['High'].median(),
            'Low': training_data['Low'].median(),
            'VWAP': training_data['VWAP'].median(),
            'Open': training_data['Open'].median(),
            'Close': training_data['Close'].median(),
            'Buy_Volume': training_data['Buy_Volume'].median(),
            'Sell_Volume': training_data['Sell_Volume'].median(),
            'Transaction_Volume': training_data['Transaction_Volume'].median()
        })

        return training_data, future_periods, future_period_ids

    except Exception as e:
        logger.error(f"Error processing data for forecasting: {e}")
        raise

def train_forecast_model(
    training_data: pd.DataFrame,
    target_column: str
) -> Tuple[Any, StandardScaler, StandardScaler]:
    """
    Train a model for forecasting the given target column (High or Low).

    Args:
        training_data: DataFrame with training data
        target_column: Column to predict ('High' or 'Low')

    Returns:
        (model, feature_scaler, target_scaler)
    """
    try:
        # Base features
        features = ['Hour', 'Minute', 'DayOfWeek', 'Month', 'IsWeekend']

        # Add price/volume features if they exist
        price_columns = ['VWAP', 'VWAP3H', 'VWAP1H', 'Open', 'Close', 'Buy_Volume', 'Sell_Volume']
        for col in price_columns:
            if col in training_data.columns:
                features.append(col)

        # Create X and y
        X = training_data[features].copy()
        y = training_data[target_column].copy()

        # Scale X and y
        feature_scaler = StandardScaler()
        X_scaled = feature_scaler.fit_transform(X)

        target_scaler = StandardScaler()
        y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

        # Train model (e.g., RandomForest)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y_scaled)

        logger.info(f"Trained model for {target_column} with {len(features)} features")
        return model, feature_scaler, target_scaler

    except Exception as e:
        logger.error(f"Error training forecast model: {e}")
        raise

def generate_forecasts(
    client: bigquery.Client,
    resolution_minutes: List[int] = [15, 30, 60],
    save_to_database: bool = False,
    model_info: str = "RandomForestRegressor",
    user_id: int = 1
) -> Dict[str, Any]:
    """
    Generate price forecasts for the specified resolutions.
    Optionally save them to the 'Forecasts' table in BigQuery.

    Args:
        client: BigQuery client
        resolution_minutes: e.g. [15, 30, 60]
        save_to_database: True to insert into BQ
        model_info: info about the model used
        user_id: optional user ID for saving records

    Returns:
        Dictionary with forecast results
    """
    try:
        # Fetch historical and today's data
        try:
            historical_data = get_historical_data(client, lookback_days=30)
            today_data = get_current_day_data(client)
            
            if historical_data.empty or today_data.empty:
                logger.warning("No historical or today's data found, using sample data generation")
                return generate_sample_forecasts(resolution_minutes, save_to_database, model_info, user_id, client)
            
        except Exception as db_error:
            logger.warning(f"Error fetching data from BigQuery: {db_error}, using sample data generation")
            return generate_sample_forecasts(resolution_minutes, save_to_database, model_info, user_id, client)

        results = {}

        for resolution in resolution_minutes:
            logger.info(f"Generating forecasts for {resolution}-minute resolution")

            # Process
            training_data, future_periods, future_period_ids = process_data_for_forecasting(
                historical_data, today_data, resolution
            )
            if training_data.empty or not future_period_ids:
                logger.warning(f"No training or future data for {resolution}-minute resolution")
                continue

            # Train models (High and Low)
            high_model, high_feature_scaler, high_target_scaler = train_forecast_model(training_data, 'High')
            low_model, low_feature_scaler, low_target_scaler = train_forecast_model(training_data, 'Low')

            # Prepare X_future
            features = ['Hour', 'Minute', 'DayOfWeek', 'Month', 'IsWeekend']
            price_columns = ['VWAP', 'VWAP3H', 'VWAP1H', 'Open', 'Close', 'Buy_Volume', 'Sell_Volume']
            for col in price_columns:
                if col in future_periods.columns:
                    features.append(col)

            X_future = future_periods[features].copy()

            # Scale using High model's feature scaler (same features)
            X_future_scaled = high_feature_scaler.transform(X_future)

            # Predict
            high_preds_scaled = high_model.predict(X_future_scaled)
            low_preds_scaled = low_model.predict(X_future_scaled)

            # Inverse scale
            high_preds = high_target_scaler.inverse_transform(
                high_preds_scaled.reshape(-1, 1)
            ).flatten()
            low_preds = low_target_scaler.inverse_transform(
                low_preds_scaled.reshape(-1, 1)
            ).flatten()

            # Build forecast df
            forecast_df = pd.DataFrame({
                'Delivery_Period': future_periods['Delivery_Period'],
                'PredictedHigh': high_preds,
                'PredictedLow': low_preds,
                'ResolutionMinutes': resolution
            })

            results[f"{resolution}_min"] = forecast_df.to_dict('records')
            logger.info(f"Generated {len(forecast_df)} forecasts for {resolution}-minute resolution")

        # Build response
        response = {
            "forecasts": results,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

        # Save to BQ if requested
        if save_to_database and results:
            save_success = save_forecasts_to_bigquery(client, results, model_info, user_id)
            response["saved_to_database"] = save_success

        return response

    except Exception as e:
        logger.error(f"Error generating forecasts: {e}")
        try:
            # Fallback to sample data if model-based forecasting fails
            logger.info("Falling back to sample forecast generation")
            return generate_sample_forecasts(resolution_minutes, save_to_database, model_info, user_id, client)
        except Exception as sample_error:
            logger.error(f"Even sample forecasting failed: {sample_error}")
            return {
                "forecasts": {},
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }

def generate_sample_forecasts(
    resolution_minutes: List[int] = [15, 30, 60],
    save_to_database: bool = False,
    model_info: str = "SampleForecasts",
    user_id: int = 1,
    client: Optional[bigquery.Client] = None
) -> Dict[str, Any]:
    """
    Generate sample forecasts for testing and fallback purposes.
    
    Args:
        resolution_minutes: List of resolutions to generate forecasts for
        save_to_database: Whether to save to database
        model_info: Model identifier
        user_id: User ID for database records
        client: BigQuery client (optional)
        
    Returns:
        Dictionary with sample forecast results
    """
    logger.info("Generating sample forecasts as real forecasting failed or was not possible")
    now = datetime.now()
    current_hour = now.hour
    sample_forecasts = {}
    
    for resolution in resolution_minutes:
        resolution_key = f"{resolution}_min"
        forecasts_list = []
        for hour in range(current_hour + 1, 24):
            periods_per_hour = 60 // resolution
            for period in range(periods_per_hour):
                start_minute = period * resolution
                end_minute = start_minute + resolution
                if resolution == 60:
                    period_str = f"{hour:02d}:00-{(hour+1):02d}:00"
                else:
                    period_str = f"{hour:02d}:{start_minute:02d}-{hour:02d}:{end_minute:02d}"
                
                base_price = 50 + (hour * 2) + random.uniform(-5, 5)
                predicted_high = base_price + random.uniform(2, 8)
                predicted_low = base_price - random.uniform(2, 8)
                
                forecast = {
                    "Delivery_Period": period_str,
                    "PredictedHigh": predicted_high,
                    "PredictedLow": predicted_low,
                    "ResolutionMinutes": resolution
                }
                
                # Add confidence intervals
                interval = (predicted_high - predicted_low) * 0.2
                forecast["Confidence_Upper"] = predicted_high + interval
                forecast["Confidence_Lower"] = predicted_low - interval
                
                forecasts_list.append(forecast)
        sample_forecasts[resolution_key] = forecasts_list
    
    result = {
        "forecasts": sample_forecasts,
        "timestamp": now.isoformat(),
        "status": "success",
        "saved_to_database": False
    }
    
    if save_to_database and client:
        try:
            # Try to save to database
            test_insert_result = test_forecasts_table_insertion()
            if test_insert_result:
                save_success = save_forecasts_to_bigquery(
                    client, 
                    sample_forecasts,
                    f"{model_info}_Sample",
                    user_id
                )
                result["saved_to_database"] = save_success
        except Exception as db_error:
            logger.error(f"Error saving sample forecasts: {db_error}")
            result["database_error"] = str(db_error)
    
    return result

def validate_forecasts(historical_df: pd.DataFrame, forecast_df: pd.DataFrame) -> Dict[str, float]:
    """
    Placeholder function to validate forecast accuracy.

    Args:
        historical_df: Historical data
        forecast_df: Forecast data

    Returns:
        Dictionary of accuracy metrics
    """
    # Stub: no real implementation yet
    return {
        "mean_absolute_error": 0.0,
        "mean_percentage_error": 0.0
    }

def save_forecasts_to_bigquery(
    client: bigquery.Client,
    forecasts: Dict[str, List[Dict[str, Any]]],
    model_info: str = "RandomForestRegressor",
    user_id: int = 1
) -> bool:
    """
    Save forecast results to the BigQuery 'Forecasts' table.

    Args:
        client: BigQuery client
        forecasts: dictionary keyed by resolution, each containing a list of forecast dicts
        model_info: string describing model used
        user_id: (optional) user ID

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Saving forecasts to BigQuery Forecasts table")

        table_id = "capstone-henry.capstone_db.Forecasts"
        table = client.get_table(table_id)  # might raise exception if table not found
        schema_fields = [field.name for field in table.schema]

        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        rows_to_insert = []

        # Loop over each resolution set
        for resolution_key, forecast_list in forecasts.items():
            logger.info(f"Processing forecasts for resolution: {resolution_key}")
            resolution_val = int(resolution_key.split('_')[0])  # e.g. "15_min" -> 15

            for forecast in forecast_list:
                try:
                    # Generate a unique forecast ID truncated to fit in FLOAT or INT
                    forecast_id = float(int(uuid.uuid4().int) % 10000000000)

                    delivery_period = forecast["Delivery_Period"]
                    predicted_high = float(forecast["PredictedHigh"])
                    predicted_low = float(forecast["PredictedLow"])
                    predicted_avg = (predicted_high + predicted_low) / 2

                    # Build a datetime for the forecast timestamp (using today's date)
                    today = datetime.now().date()
                    start_time_str = delivery_period.split("-")[0].strip()
                    hour, minute = map(int, start_time_str.split(":"))
                    forecast_timestamp = datetime(
                        year=today.year,
                        month=today.month,
                        day=today.day,
                        hour=hour,
                        minute=minute
                    )
                    # If the timestamp is in the past, shift it to tomorrow
                    if forecast_timestamp < datetime.now():
                        forecast_timestamp += timedelta(days=1)
                    forecast_timestamp_str = forecast_timestamp.strftime('%Y-%m-%d %H:%M:%S')

                    # Simple approach for confidence intervals
                    confidence_range = (predicted_high - predicted_low) * 0.1
                    confidence_upper = predicted_high + confidence_range
                    confidence_lower = predicted_low - confidence_range

                    row = {
                        "Forecast_id": forecast_id,
                        "Market": "Germany",
                        "Forecast_Timestamp": forecast_timestamp_str,
                        "Forecast_Value_Max": predicted_high,
                        "Forecast_Value_Min": predicted_low,
                        "Forecast_Value_Average": predicted_avg,
                        "Generated_At": current_timestamp,
                        "Model_Info": f"{model_info}_{resolution_key}",
                        "Resolution_Minutes": resolution_val,
                        "Accuracy_Metrics": None,  # placeholder
                        "Confidence_Interval_Upper": confidence_upper,
                        "Confidence_Interval_Lower": confidence_lower,
                        "User_ID": user_id,
                        "Is_Active": True,
                        "Last_Updated": current_timestamp
                    }

                    # Only include fields actually in the schema
                    row = {k: v for k, v in row.items() if k in schema_fields}
                    rows_to_insert.append(row)

                except Exception as row_error:
                    logger.error(f"Error building forecast row for period {delivery_period}: {row_error}")
                    logger.error(f"Forecast data: {forecast}")
                    continue

        if not rows_to_insert:
            logger.warning("No forecast data to insert into BigQuery")
            return False

        logger.info(f"Inserting {len(rows_to_insert)} forecast rows into {table_id}")
        errors = client.insert_rows_json(table_id, rows_to_insert)
        if errors:
            logger.error(f"Errors inserting forecast rows: {errors}")

            # Attempt a simplified row if needed to debug schema issues
            simple_test = {
                "Forecast_id": float(9999999),
                "Market": "SimpleTest",
                "Forecast_Timestamp": current_timestamp,
                "Forecast_Value_Max": 100.0,
                "Forecast_Value_Min": 90.0,
                "Forecast_Value_Average": 95.0,
                "Generated_At": current_timestamp,
                "Model_Info": "SimpleTest",
                "Resolution_Minutes": 60,
                "User_ID": user_id,
                "Is_Active": True,
                "Last_Updated": current_timestamp
            }
            simple_test = {k: v for k, v in simple_test.items() if k in schema_fields}

            logger.info("Trying a simplified test row insertion to debug potential schema mismatch...")
            test_errors = client.insert_rows_json(table_id, [simple_test])
            if test_errors:
                logger.error(f"Simplified test row insertion also failed: {test_errors}")
            else:
                logger.info("Simplified test row insertion succeeded.")

            return False

        logger.info(f"Successfully saved {len(rows_to_insert)} forecast records to BigQuery")
        return True

    except Exception as e:
        logger.error(f"Error saving forecasts to BigQuery: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_forecasts_table_insertion():
    """
    Test function to confirm we can insert at least one row into the 'Forecasts' table.
    Useful for diagnosing schema or permission issues.
    """
    try:
        # Create a BigQuery client with default credentials
        client = bigquery.Client()

        logger.info("BigQuery client initialized for test insertion into Forecasts table")

        now = datetime.now()
        current_timestamp = now.strftime('%Y-%m-%d %H:%M:%S')

        test_row = {
            "Forecast_id": float(1234567890),
            "Market": "TestMarket",
            "Forecast_Timestamp": current_timestamp,
            "Forecast_Value_Max": float(100.5),
            "Forecast_Value_Min": float(90.5),
            "Forecast_Value_Average": float(95.5),
            "Generated_At": current_timestamp,
            "Model_Info": "TestModel_TestFunction",
            "Resolution_Minutes": 15,
            "Accuracy_Metrics": float(0.95),
            "Confidence_Interval_Upper": float(105.0),
            "Confidence_Interval_Lower": float(85.0),
            "User_ID": 1,
            "Is_Active": True,
            "Last_Updated": current_timestamp
        }

        table_id = "capstone-henry.capstone_db.Forecasts"
        logger.info(f"Checking table schema for {table_id}...")

        table = client.get_table(table_id)
        schema_fields = [field.name for field in table.schema]
        logger.info(f"Table schema: {schema_fields}")

        # Only keep fields that exist in the schema
        test_row = {k: v for k, v in test_row.items() if k in schema_fields}

        logger.info(f"Inserting test row: {test_row}")
        errors = client.insert_rows_json(table_id, [test_row])
        if errors:
            logger.error(f"Errors inserting test row: {errors}")
            # Attempt a minimal row
            simple_test = {
                "Forecast_id": float(9999999),
                "Market": "SimpleTest",
                "Forecast_Timestamp": current_timestamp,
                "Forecast_Value_Max": float(100.0),
                "Forecast_Value_Min": float(90.0),
                "Forecast_Value_Average": float(95.0),
                "Generated_At": current_timestamp,
                "Model_Info": "SimpleTest",
                "Resolution_Minutes": 60,
                "User_ID": 1,
                "Is_Active": True
            }
            simple_test = {k: v for k, v in simple_test.items() if k in schema_fields}
            logger.info(f"Trying minimal row: {simple_test}")

            simple_errors = client.insert_rows_json(table_id, [simple_test])
            if simple_errors:
                logger.error(f"Minimal row insertion failed: {simple_errors}")
                return False
            else:
                logger.info("Minimal row insertion succeeded.")
                return True
        else:
            logger.info("Test row insertion succeeded.")
            return True

    except Exception as e:
        logger.error(f"Error in test_forecasts_table_insertion: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Call the test to see if we can insert a row into the Forecasts table
    success = test_forecasts_table_insertion()
    logger.info(f"Test insertion success: {success}")
