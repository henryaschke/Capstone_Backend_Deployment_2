from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn

# Import all route modules
from routes import auth, battery, forecast, market, performance, status, trade

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Energy Trading Platform API",
    description="API for energy trading platform with real-time price data, battery management, and algorithm execution"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers from all route modules
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(battery.router, prefix="/api/battery", tags=["Battery Management"])
app.include_router(forecast.router, prefix="/api/forecast", tags=["Forecasting"])
app.include_router(market.router, prefix="/api/market-data", tags=["Market Data"])
app.include_router(performance.router, prefix="/api/performance", tags=["Performance Metrics"])
app.include_router(status.router, prefix="/api", tags=["Diagnostics & Status"])
app.include_router(trade.router, prefix="/api/trades", tags=["Trading Operations"])

# Main entry point
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True) 