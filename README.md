# Energy Trading Platform Backend

**Author:** Henry Aschke

*This project was inspired by the Venture Lab program and created under the guidance of Solomon Shiferaw in the "Capstone Integration Project" at IE University.*

A robust backend service for an energy trading platform that provides real-time market data, battery management, trading capabilities, and price forecasting using advanced machine learning models.

## Features

- 🔐 **Authentication & Authorization**
  - Secure user registration with email validation
  - JWT-based authentication with token refresh
  - Role-based access control (User, Admin)
  - Session management and token expiration handling
  - Password hashing with secure algorithms

- 📊 **Market Data Management**
  - Real-time market data retrieval with multiple resolutions (15min, 30min, 60min)
  - Historical data analysis with customizable date ranges
  - Multi-market support (Germany and extensible to other markets)
  - Price filtering and aggregation capabilities
  - Time-series data processing and normalization

- 🔋 **Battery Management**
  - Real-time battery status monitoring with capacity tracking
  - Historical battery level visualization
  - Charge/discharge operations with validation
  - Capacity optimization algorithms
  - Battery state recalculation based on trade history
  - Automatic battery creation for new users

- 📈 **Price Forecasting**
  - Advanced machine learning prediction models (RandomForestRegressor)
  - Multiple time resolution forecasting (15min, 30min, 60min)
  - Confidence intervals for predictions
  - Model performance tracking and accuracy metrics
  - Historical vs. predicted price comparison
  - Customizable forecast parameters

- 💹 **Trading System**
  - Scheduled trade execution with timing validation
  - Support for both buy and sell operations
  - Trade history tracking with detailed analytics
  - Pending trade management (execution, cancellation)
  - Integration with battery levels for trade validation
  - Trading portfolio management with balance tracking
  - Automated trading capabilities with custom parameters

- 📊 **Performance Analytics**
  - Comprehensive revenue and profit tracking
  - Time-series performance visualization
  - Trading volume and count metrics
  - Profit margin calculations
  - Historical performance analysis with custom date ranges
  - Accuracy metrics for trading strategies

- 🧪 **System Diagnostics**
  - Comprehensive API status monitoring
  - Database connection diagnostics
  - Query performance metrics
  - Admin-only diagnostic endpoints
  - Environment configuration validation

## Complete API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user with email and password
- `POST /api/auth/login` - Authenticate user and retrieve JWT token
- `POST /api/auth/refresh` - Refresh authentication token even if expired
- `GET /api/auth/me` - Get current authenticated user profile
- `GET /api/auth/protected-data` - Example protected endpoint for testing

### Battery Management
- `GET /api/battery/status` - Get current battery status and capacity
- `GET /api/battery/history` - Get historical battery level data
- `POST /api/battery/charge` - Charge the battery with specified quantity
- `POST /api/battery/discharge` - Discharge the battery with specified quantity
- `POST /api/battery/recalculate` - Force recalculation of battery levels from trade history

### Market Data
- `GET /api/market-data` - Get market data with optional filters (date, price range)
- `GET /api/market` - Get single-day market data for specific date and market
- `GET /api/market/germany` - Get historical Germany market data with customizable date range and resolution
- `GET /api/market/realtime` - Get real-time price data for current trading periods

### Forecasting
- `GET /api/forecast/generate` - Generate price forecasts with default parameters
- `POST /api/forecast/generate` - Generate price forecasts with custom parameters
- `GET /api/forecast/latest` - Get latest generated forecasts without generating new ones
- `GET /api/forecast/saved` - Get saved forecasts from database with flexible filtering
- `GET /api/forecast` - Get energy price forecasts with date range filtering

### Trading
- `POST /api/trade/execute` - Schedule a trade for future execution with validation
- `GET /api/trade/history` - Get trading history with flexible filtering options
- `POST /api/trade/execute-pending/{trade_id}` - Execute a specific pending trade immediately
- `POST /api/trade/execute-all-pending` - Execute all pending trades immediately
- `POST /api/trade/cancel-all-pending` - Cancel all pending trades
- `GET /api/trade/pending` - Get all pending trades awaiting execution
- `GET /api/trade/{trade_id}` - Get details of a specific trade by ID
- `POST /api/trade/cancel/{trade_id}` - Cancel a specific pending trade
- `GET /api/trade/portfolio` - Get user's trading portfolio with balance information
- `POST /api/trade/portfolio/deposit` - Add funds to trading portfolio
- `POST /api/trade/portfolio/withdraw` - Withdraw funds from trading portfolio

### Performance
- `GET /api/performance/metrics` - Get comprehensive performance metrics with optional date filtering

### System Status
- `GET /api/status` - Get API status, version, and connection information
- `GET /api/diagnostic/query` - Execute diagnostic queries against the database
- `GET /api/test/users-table` - Admin-only endpoint to check Users table schema
- `POST /api/admin/execute-query` - Admin-only endpoint to execute custom SQL queries

## Data Models

### User
- User ID, Email, Password (hashed), Role, Created/Updated timestamps

### Battery
- Battery ID, User ID, Current Level, Total/Usable Capacity, Last Updated

### Trade
- Trade ID, User ID, Market, Type (buy/sell), Quantity, Price, Timestamp, Status, Resolution

### Forecast
- Forecast ID, Market, Timestamp, Value (Min/Max/Avg), Model Info, Resolution, Accuracy Metrics, Confidence Intervals

### Portfolio
- Portfolio ID, User ID, Balance, Created/Updated timestamps

## Technical Implementation

- **Database**: BigQuery for high-performance data storage
- **Authentication**: JWT token-based with refresh capability
- **API Framework**: FastAPI with automatic OpenAPI documentation
- **Data Processing**: Pandas for time-series analysis
- **ML Models**: Scikit-learn for forecasting algorithms
- **Validation**: Pydantic models for request/response validation
- **Time Handling**: Timezone-aware datetime processing (CET/CEST)
- **Caching**: In-memory caching for frequently accessed data

## Setup & Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```
4. Run the server:
   ```bash
   python server.py
   ```

## Docker Deployment

Build and run using Docker:
```bash
docker build -t energy-trading-backend .
docker run -p 5000:5000 energy-trading-backend
```

## Development

- Python 3.8+
- FastAPI framework
- SQLAlchemy ORM
- JWT authentication
- BigQuery integration
- Pandas for data analysis
- Scikit-learn for ML models

## Testing

Run tests using:
```bash
python -m pytest
```

## Documentation

Detailed API documentation is available at `/api/docs` when running the server.

## License

MIT License - See LICENSE file for details 