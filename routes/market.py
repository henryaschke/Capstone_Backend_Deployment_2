from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta
import random
import math

from dependencies import get_optional_user, client, DEFAULT_USER_ID, parse_date_string
from database import get_market_data

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get("/")
async def get_market_data_api(
    start_date: str = Query(None),
    end_date: str = Query(None),
    min_price: float = Query(None),
    max_price: float = Query(None),
    market: str = Query("Germany")
):
    """Get market data with optional filters."""
    try:
        market_data = get_market_data(start_date, end_date, min_price, max_price, market)
        return market_data
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("")
async def get_market_data(
    date: str = None, 
    market: str = "Germany", 
    current_user: Dict[str, Any] = Depends(get_optional_user)
):
    """
    Get single-day market data for a given date and market.
    Demonstrates usage of Market_Data_Germany_Today.
    """
    try:
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        logger.info(f"Fetching market data for date: {date}, market: {market}")
        
        if not client:
            raise HTTPException(status_code=500, detail="BigQuery client not initialized.")
        
        query = f"""
        SELECT 
            Delivery_Day,
            Delivery_Period,
            Cleared, 
            Market,
            High,
            Low,
            Close,
            Open,
            Transaction_Volume
        FROM `capstone-henry.capstone_db.Market_Data_Germany_Today`
        WHERE Delivery_Day = '{date}' AND Market = '{market}'
        ORDER BY Delivery_Period
        """
        
        query_job = client.query(query)
        results = query_job.result()
        
        market_data = []
        for row in results:
            data_point = {
                "deliveryDay": row.Delivery_Day,
                "deliveryPeriod": row.Delivery_Period,
                "cleared": bool(row.Cleared),
                "market": row.Market,
                "high": float(row.High) if row.High is not None else None,
                "low": float(row.Low) if row.Low is not None else None,
                "close": float(row.Close) if row.Close is not None else None,
                "open": float(row.Open) if row.Open is not None else None,
                "transactionVolume": float(row.Transaction_Volume) if row.Transaction_Volume is not None else None
            }
            market_data.append(data_point)
        
        if not market_data:
            logger.warning(f"No data found for date {date}, market {market}; generating sample data.")
            market_data = generate_sample_market_data(date, market)
        
        return market_data
    except Exception as e:
        logger.error(f"Error in get_market_data endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_sample_market_data(date: str, market: str):
    """Generate sample data if no real data is found."""
    sample_data = []
    current_hour = datetime.now().hour
    for hour in range(24):
        is_cleared = hour <= current_hour
        base_price = 50 + random.uniform(-5, 5)
        data_point = {
            "deliveryDay": date,
            "deliveryPeriod": f"{hour:02d}:00-{(hour+1):02d}:00",
            "cleared": is_cleared,
            "market": market,
            "high": base_price + random.uniform(0, 5),
            "low": base_price - random.uniform(0, 5),
            "close": base_price + random.uniform(-2, 2),
            "open": base_price + random.uniform(-2, 2),
            "transactionVolume": random.uniform(100, 500) if is_cleared else 0
        }
        sample_data.append(data_point)
    return sample_data

@router.get("/germany")
async def get_germany_market_data(
    start_date: str = None,
    end_date: str = None,
    resolution: int = None,
    limit: int = 1000,
    current_user: Dict[str, Any] = Depends(get_optional_user)
):
    """
    Get historical market data (Market_Data_Germany) with optional date range and resolution.
    """
    try:
        if not client:
            raise HTTPException(status_code=500, detail="BigQuery client not initialized.")
        
        # If not specified, default to last 7 days
        if not start_date:
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        query = f"""
        SELECT 
            ID,
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
        WHERE 1=1
        """
        
        if start_date:
            query += f" AND Delivery_Day >= '{start_date}'"
        if end_date:
            query += f" AND Delivery_Day <= '{end_date}'"
        if resolution:
            query += f" AND Resolution_Minutes = {resolution}"
        
        query += f" ORDER BY Delivery_Day DESC, Delivery_Period LIMIT {limit}"
        
        query_job = client.query(query)
        results = query_job.result()
        
        market_data = []
        for row in results:
            def safe_float(value):
                if value is None:
                    return None
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return None
            
            resolution_str = "15min"
            if row.Resolution_Minutes == 30:
                resolution_str = "30min"
            elif row.Resolution_Minutes == 60:
                resolution_str = "1h"
            
            data_point = {
                "id": row.ID,
                "date": row.Delivery_Day,
                "resolution": resolution_str,
                "deliveryPeriod": row.Delivery_Period,
                "market": row.Market,
                "highPrice": safe_float(row.High),
                "lowPrice": safe_float(row.Low),
                "averagePrice": safe_float(row.VWAP),
                "openPrice": safe_float(row.Open),
                "closePrice": safe_float(row.Close),
                "buyVolume": safe_float(row.Buy_Volume),
                "sellVolume": safe_float(row.Sell_Volume),
                "volume": safe_float(row.Transaction_Volume),
                "vwap3h": safe_float(row.VWAP3H),
                "vwap1h": safe_float(row.VWAP1H),
                "contractOpenTime": row.Contract_Open_Time,
                "contractCloseTime": row.Contract_Close_Time
            }
            market_data.append(data_point)
        
        if not market_data:
            logger.warning(f"No market data found for {start_date} to {end_date}, generating sample data.")
            market_data = generate_sample_germany_market_data(start_date, end_date)
        
        return market_data
    except Exception as e:
        logger.error(f"Error in get_germany_market_data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_sample_germany_market_data(start_date, end_date):
    """Generate sample historical data for Germany if none found."""
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        end = datetime.now()
        start = end - timedelta(days=7)
    
    sample_data = []
    day_count = (end - start).days + 1
    for day_offset in range(day_count):
        current_date = (start + timedelta(days=day_offset)).strftime('%Y-%m-%d')
        for resolution in [15, 30, 60]:
            periods_per_day = 24 * 60 // resolution
            for period in range(periods_per_day):
                minutes_offset = period * resolution
                hour = minutes_offset // 60
                minute = minutes_offset % 60
                if resolution == 60:
                    period_str = f"{hour:02d}:00-{(hour+1):02d}:00"
                else:
                    end_minute = (minute + resolution) % 60
                    end_hour = hour + 1 if minute + resolution >= 60 else hour
                    period_str = f"{hour:02d}:{minute:02d}-{end_hour:02d}:{end_minute:02d}"
                
                base_price = 50 + 10 * math.sin(hour / 12 * math.pi) + random.uniform(0, 10)
                data_point = {
                    "id": f"sample-{current_date}-{period_str}",
                    "date": current_date,
                    "resolution": f"{resolution}min" if resolution < 60 else "1h",
                    "deliveryPeriod": period_str,
                    "market": "Germany",
                    "highPrice": base_price + random.uniform(5, 15),
                    "lowPrice": base_price - random.uniform(0, 5),
                    "averagePrice": base_price + random.uniform(0, 5),
                    "openPrice": base_price - random.uniform(0, 3),
                    "closePrice": base_price + random.uniform(0, 3),
                    "buyVolume": random.uniform(100, 500),
                    "sellVolume": random.uniform(100, 500),
                    "volume": random.uniform(200, 1000),
                    "vwap3h": base_price + random.uniform(0, 2),
                    "vwap1h": base_price + random.uniform(0, 1),
                    "contractOpenTime": f"{max(0, hour-1):02d}:00:00",
                    "contractCloseTime": f"{hour:02d}:00:00"
                }
                sample_data.append(data_point)
    return sample_data

@router.get("/realtime")
async def get_realtime_prices(
    date: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_optional_user)
):
    """
    Get real-time price data for today's date (or a specified date).
    Demonstrates reading from a 'Market_Data_Germany_Today' table.
    """
    try:
        # Default to today's date if none provided
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Fetching price data for date: {date}")
        
        if not client:
            raise HTTPException(status_code=500, detail="BigQuery client not initialized.")
        
        query = f"""
        SELECT 
            Delivery_Day,
            Delivery_Period,
            Cleared, 
            Market,
            High,
            Low,
            Close,
            Open,
            Transaction_Volume
        FROM `capstone-henry.capstone_db.Market_Data_Germany_Today`
        WHERE Delivery_Day = '{date}'
        ORDER BY Delivery_Period
        """
        
        query_job = client.query(query)
        results = query_job.result()
        
        price_data = []
        for row in results:
            try:
                data_point = {
                    "Delivery_Day": row.Delivery_Day,
                    "Delivery_Period": row.Delivery_Period,
                    "Cleared": bool(row.Cleared),
                    "Market": row.Market,
                    "High": float(row.High) if row.High is not None else None,
                    "Low": float(row.Low) if row.Low is not None else None,
                    "Close": float(row.Close) if row.Close is not None else None,
                    "Open": float(row.Open) if row.Open is not None else None,
                    "Transaction_Volume": float(row.Transaction_Volume) if row.Transaction_Volume is not None else None
                }
                price_data.append(data_point)
            except Exception as e:
                logger.error(f"Error processing data point: {str(e)}")
                continue
        
        # If no data found, generate sample data
        if not price_data:
            logger.warning(f"No data found for date {date}, generating sample data.")
            price_data = generate_sample_price_data(date)
        
        return price_data
    except Exception as e:
        logger.error(f"Error in get_realtime_prices: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching price data: {str(e)}")

def generate_sample_price_data(date_str: str):
    """
    Generate sample real-time price data for demonstration purposes.
    Replace with your actual fallback logic if real data is missing.
    """
    sample_data = []
    current_hour = datetime.now().hour
    for hour in range(24):
        is_cleared = hour <= current_hour
        base_price = 60 + random.uniform(-10, 10)
        data_point = {
            "Delivery_Day": date_str,
            "Delivery_Period": f"{hour:02d}:00-{(hour+1):02d}:00",
            "Cleared": is_cleared,
            "Market": "Germany",
            "High": base_price + random.uniform(0, 5),
            "Low": base_price - random.uniform(0, 5),
            "Close": base_price + random.uniform(-2, 2),
            "Open": base_price + random.uniform(-2, 2),
            "Transaction_Volume": random.uniform(100, 500) if is_cleared else 0
        }
        sample_data.append(data_point)
    return sample_data 