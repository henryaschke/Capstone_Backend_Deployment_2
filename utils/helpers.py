import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import random
import math

# Configure logging
logger = logging.getLogger(__name__)

def safe_float(value):
    """Safely convert a value to float, returning None if not possible."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def format_period_string(hour: int, minute: int, resolution: int):
    """Format a period string like '12:00-12:15' based on hour, minute and resolution."""
    end_minute = (minute + resolution) % 60
    end_hour = hour + 1 if minute + resolution >= 60 else hour
    
    if resolution == 60:
        return f"{hour:02d}:00-{(hour+1):02d}:00"
    else:
        return f"{hour:02d}:{minute:02d}-{end_hour:02d}:{end_minute:02d}"

def generate_sample_data(date_str: str, market: str = "Germany", resolution: int = 60):
    """Generate sample data for a specific date and market."""
    sample_data = []
    current_hour = datetime.now().hour
    periods_per_day = 24 * 60 // resolution
    
    for period in range(periods_per_day):
        minutes_offset = period * resolution
        hour = minutes_offset // 60
        minute = minutes_offset % 60
        
        period_str = format_period_string(hour, minute, resolution)
        is_cleared = hour <= current_hour
        
        base_price = 50 + 10 * math.sin(hour / 12 * math.pi) + random.uniform(-5, 5)
        sample_data.append({
            "date": date_str,
            "deliveryPeriod": period_str,
            "cleared": is_cleared,
            "market": market,
            "high": base_price + random.uniform(0, 5),
            "low": base_price - random.uniform(0, 5),
            "open": base_price - random.uniform(0, 3),
            "close": base_price + random.uniform(-2, 2),
            "volume": random.uniform(100, 500) if is_cleared else 0
        })
    
    return sample_data 