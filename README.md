# Energy Trading Platform Backend

A robust backend service for an energy trading platform that provides real-time market data, battery management, trading capabilities, and price forecasting.

## Features

- 🔐 **Authentication & Authorization**
  - User registration and login
  - JWT-based authentication
  - Role-based access control

- 📊 **Market Data Management**
  - Real-time market data retrieval
  - Historical data analysis
  - Multi-market support (e.g., Germany)

- 🔋 **Battery Management**
  - Real-time battery status monitoring
  - Capacity tracking
  - Usage optimization

- 📈 **Price Forecasting**
  - Advanced price prediction models
  - Multiple time resolutions (15min, 30min, 60min)
  - Model performance tracking

- 💹 **Trading System**
  - Automated trading capabilities
  - Trade execution and management
  - Performance tracking and analytics

- 📊 **Performance Analytics**
  - Revenue and profit tracking
  - Trading performance metrics
  - Historical performance analysis

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - User login
- `POST /api/auth/refresh` - Refresh authentication token
- `GET /api/auth/me` - Get current user info

### Battery Management
- `GET /api/battery/status` - Get battery status and capacity

### Market Data
- `GET /api/market-data` - Get market data for specific date and market
- `GET /api/market/status` - Get market status and configuration

### Forecasting
- `GET/POST /api/forecast/generate` - Generate price forecasts
- `GET /api/forecast/status` - Get forecasting system status

### Trading
- `POST /api/trade/execute` - Execute trades
- `GET /api/trade/history` - Get trading history
- `GET /api/trade/status` - Get trading system status

### Performance
- `GET /api/performance/metrics` - Get performance metrics
- `GET /api/performance/analytics` - Get detailed analytics

### System Status
- `GET /api/status` - Get API and system status
- `GET /api/diagnostic/query` - Execute diagnostic queries

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
- Flask framework
- SQLAlchemy ORM
- JWT authentication
- BigQuery integration

## Testing

Run tests using:
```bash
python -m pytest
```

## Documentation

Detailed API documentation is available at `/api/docs` when running the server.

## License

MIT License - See LICENSE file for details 