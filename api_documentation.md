# Energy Trading Platform API Documentation

This document provides a comprehensive overview of all API endpoints available in the Energy Trading Platform backend.

## Authentication Endpoints

### Register User
- **URL**: `/api/auth/register`
- **Method**: `POST`
- **Description**: Register a new user
- **Input**:
  ```json
  {
    "email": "user@example.com",
    "password": "securepassword",
    "user_role": "user"
  }
  ```
- **Output**:
  ```json
  {
    "message": "User registered successfully",
    "user_id": 12345
  }
  ```
- **Error Responses**:
  - `400 Bad Request`: Email already registered
  - `500 Internal Server Error`: Failed to create user

### Login
- **URL**: `/api/auth/login`
- **Method**: `POST`
- **Description**: Authenticate a user and return a JWT token
- **Input**: Form data with fields:
  - `username`: User's email
  - `password`: User's password
- **Output**:
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "user_id": 12345,
    "email": "user@example.com",
    "user_role": "user"
  }
  ```
- **Error Responses**:
  - `401 Unauthorized`: Incorrect email or password
  - `500 Internal Server Error`: Login error

### Refresh Token
- **URL**: `/api/auth/refresh`
- **Method**: `POST`
- **Description**: Refresh a user's token even if the current one is expired
- **Input**: Authorization header with Bearer token
- **Output**:
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "user_id": 12345,
    "email": "user@example.com", 
    "user_role": "user"
  }
  ```
- **Error Responses**:
  - `401 Unauthorized`: Invalid token
  - `500 Internal Server Error`: Token refresh error

### Get Current User
- **URL**: `/api/auth/me`
- **Method**: `GET`
- **Description**: Get information about the current authenticated user
- **Input**: Authorization header with Bearer token
- **Output**:
  ```json
  {
    "user_id": 12345,
    "email": "user@example.com",
    "user_role": "user",
    "created_at": "2025-01-01T00:00:00Z",
    "updated_at": "2025-01-01T00:00:00Z"
  }
  ```

### Protected Data Example
- **URL**: `/api/auth/protected-data`
- **Method**: `GET`
- **Description**: Example protected endpoint
- **Input**: Authorization header with Bearer token
- **Output**:
  ```json
  {
    "message": "Hello user@example.com, this is protected data",
    "user_id": 12345,
    "user_role": "user"
  }
  ```

## Battery Management Endpoints

### Get Battery Status
- **URL**: `/api/battery/status`
- **Method**: `GET`
- **Description**: Get current battery status and capacity
- **Input**: Optional query parameters:
  - `user_id`: User ID (defaults to 1)
- **Output**:
  ```json
  {
    "level": 50.0,
    "capacity": {
      "total": 2.5,
      "usable": 2.0,
      "percentage": 50.0
    }
  }
  ```

## Forecasting Endpoints

### Generate Price Forecasts
- **URL**: `/api/forecast/generate`
- **Method**: `GET` or `POST`
- **Description**: Generate price forecasts for future time periods
- **Input**:
  - POST body (optional):
    ```json
    {
      "resolutions": [15, 30, 60],
      "save_to_database": true,
      "user_id": 1,
      "model_info": "RandomForestRegressor"
    }
    ```
  - Query parameters (for GET):
    - `save_to_database`: Boolean (defaults to true)
    - `user_id`: Integer (defaults to 1)
    - `model_info`: String (defaults to "RandomForestRegressor")
- **Output**:
  ```json
  {
    "success": true,
    "model_info": "RandomForestRegressor",
    "forecasts_generated": 163,
    "timestamp": "2025-03-17T20:56:50.859995",
    "resolutions": [15, 30, 60],
    "samples": {
      "15_min": [{...}, {...}],
      "30_min": [{...}, {...}],
      "60_min": [{...}, {...}]
    }
  }
  ```

## Market Data Endpoints

### Get Market Data
- **URL**: `/api/market-data`
- **Method**: `GET`
- **Description**: Get single-day market data for a given date and market
- **Input**: Query parameters:
  - `date`: ISO date string (defaults to current date)
  - `market`: String (defaults to "Germany")
- **Output**: Array of market data objects:
  ```json
  [
    {
      "deliveryDay": "2025-03-17",
      "deliveryPeriod": "00:00-01:00",
      "cleared": true,
      "market": "Germany",
      "high": 51.38,
      "low": 42.19,
      "close": 49.05,
      "open": 47.75,
      "transactionVolume": 349.01
    },
    ...
  ]
  ```

## Performance Metrics Endpoints

### Get Performance Metrics
- **URL**: `/api/performance/metrics`
- **Method**: `GET`
- **Description**: Get performance metrics for the user
- **Input**: Query parameters:
  - `start_date`: ISO date string (optional)
  - `end_date`: ISO date string (optional)
  - `user_id`: Integer (defaults to 1)
- **Output**:
  ```json
  {
    "totalRevenue": 90.0,
    "totalProfit": -445.6,
    "totalCosts": 535.6,
    "profitMargin": -495.1,
    "totalVolume": 12.3,
    "accuracy": 92,
    "currentBalance": 0,
    "cumulativeProfitLoss": 0,
    "chartData": [
      {
        "date": "2025-02-16",
        "profit": 3900,
        "revenue": 8300
      },
      ...
    ]
  }
  ```

## Diagnostics & Status Endpoints

### Get API Status
- **URL**: `/api/status`
- **Method**: `GET`
- **Description**: Check API and BigQuery status
- **Input**: None
- **Output**:
  ```json
  {
    "status": "OK",
    "version": "1.0",
    "config": {
      "project_id": "capstone-henry",
      "dataset_id": "capstone_db",
      "main_table": "Market_Data_Germany_Today",
      "server_time": "2025-03-17T20:56:50.859995",
      "database_connection": "Connected",
      "database_access": "Success",
      "row_count": 299
    }
  }
  ```

### Diagnostic Query
- **URL**: `/api/diagnostic/query`
- **Method**: `GET`
- **Description**: Execute a diagnostic query against the database
- **Input**: Query parameters:
  - `limit`: Integer (defaults to 10)
- **Output**:
  ```json
  {
    "success": true,
    "query": "SELECT * FROM `capstone-henry.capstone_db.Market_Data_Germany_Today` LIMIT 3",
    "columns": [],
    "row_count": 3,
    "sample_data": [
      {
        "ID": "dd5acb175e8e97f6bf815e00a5b0ff34",
        "Delivery_Day": "2025-03-17",
        "Delivery_Period": "17:45 - 18:00",
        ...
      },
      ...
    ],
    "configuration": {
      "project_id": "capstone-henry",
      "dataset_id": "capstone_db",
      "table_name": "Market_Data_Germany_Today"
    }
  }
  ```

## Trade Endpoints

### POST /api/trades/execute
Execute a trade (buy or sell) for the German energy market.

**Request Body:**
```json
{
  "type": "buy",           // Trade type (buy/sell)
  "quantity": 1.5,         // Trade quantity in MWh
  "executionTime": "2023-11-30T14:30:00Z", // Time when the trade should execute
  "resolution": 15,        // Market resolution in minutes (15, 30, or 60)
  "user_id": 1,            // Optional: User ID (if not provided via auth)
  "trade_id": "abc123",    // Optional: Custom trade ID
  "market": "Germany"      // Default: Germany
}
```

**Response:**
```json
{
  "success": true,
  "trade_id": "abc123",
  "message": "Buy trade scheduled for 1.5 MWh at 2023-11-30T14:30:00Z"
}
```

**Validation Rules:**
- Trade execution time must be at least 5 minutes in the future
- For buy trades, the battery must have sufficient capacity
- For sell trades, the battery must have sufficient energy
- Resolution must be 15, 30, or 60 minutes

### GET /api/trades/history
Get trade history with optional filters.

**Query Parameters:**
- `start_date` (optional): Filter trades from this date
- `end_date` (optional): Filter trades to this date
- `trade_type` (optional): Filter by type (buy/sell)
- `status` (optional): Filter by status (pending/executed/cancelled)

**Response:**
```json
[
  {
    "trade_id": "abc123",
    "type": "buy",
    "quantity": 1.5,
    "timestamp": "2023-11-30T14:30:00Z",
    "resolution": 15,
    "status": "pending",
    "market": "Germany",
    "created_at": "2023-11-30T10:15:30Z"
  }
]
``` 