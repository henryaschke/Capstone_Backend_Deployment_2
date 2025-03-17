# Energy Trading Platform API

This is a restructured version of the original backend with better organization and modularity.

## Project Structure

```
backend_new/
├── server.py                  # Main entry point
├── database.py                # Database interactions 
├── forecasting.py             # Forecasting algorithms
├── dependencies.py            # Shared dependencies
├── models/                    # Pydantic models
│   ├── __init__.py
│   ├── auth.py                # Auth-related models
│   ├── battery.py             # Battery-related models
│   ├── forecast.py            # Forecast-related models
│   ├── market.py              # Market-related models
│   └── trade.py               # Trade-related models
├── utils/                     # Utility functions
│   ├── __init__.py
│   └── helpers.py             # Common helper functions
└── routes/                    # API endpoints by feature
    ├── __init__.py
    ├── auth.py                # Authentication routes
    ├── battery.py             # Battery management routes
    ├── forecast.py            # Forecasting routes
    ├── market.py              # Market data routes
    ├── performance.py         # Performance metrics routes
    ├── status.py              # Diagnostic & status routes
    └── trade.py               # Trading operations routes
```

## Key Features

- Modular design with separated concerns
- Organized by feature area
- Better maintainability and testability
- Shared dependencies in a central location
- Utility functions for common tasks

## Running the Application

```
python server.py
```

This will start the FastAPI server on port 8000.

## API Documentation

Once the server is running, you can access the auto-generated API documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc 