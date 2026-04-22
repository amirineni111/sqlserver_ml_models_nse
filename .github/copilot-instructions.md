# Copilot Instructions — sqlserver_copilot_nse

## Project Context
This is the **NSE ML training pipeline** — part of a 7-repo stock trading analytics platform. Trains a **single Gradient Boosting classifier (V2)** with hybrid feature engineering to predict Buy/Sell signals for NSE 500 stocks.

## Key Architecture Rules
- Reads from `nse_500_hist_data` (VARCHAR prices — always CAST to FLOAT)
- Writes to `ml_nse_trading_predictions`, `ml_nse_predict_summary`, `ml_nse_technical_indicators`
- **V2 Architecture**: Single GB model + isotonic calibration (simplified from V1's 5-model ensemble)
- **Hybrid Feature Approach**: 76 features → 20 selected (balanced across 4 categories)
- Connected to shared database `stockdata_db` on `192.168.86.28\MSSQLSERVER01` (SQL Auth, remote)

## ML Feature Engineering (CRITICAL - April 2026)
**3-Pronged Hybrid Approach** to prevent market feature dominance:

1. **Market-Neutral Features** (10): Stock strength relative to market/sector
   - Examples: `stock_return_vs_nifty`, `rsi_vs_sector_avg`, `beta_adjusted_return`
   
2. **Interaction Features** (10): Market context combined with stock signals
   - Examples: `outperformance_in_fear`, `contrarian_strength`, `quality_in_volatility`
   - Purpose: "VIX high + Stock strong = Buy" not "VIX high = Sell all"
   
3. **Weighted Selection** (20 features): Force balanced category representation
   - Market Context: 25% (VIX/DXY controlled, not eliminated)
   - Stock-Specific: 35% (core technical indicators prioritized)
   - Relative/Neutral: 25% (comparative metrics)
   - Interactions: 15% (smart combinations)

**Expected**: 40-50% Buy signals (balanced), not 2% or 98%

## Key Technologies
- **Database**: SQL Server (`stockdata_db` on `192.168.86.28\MSSQLSERVER01`, SQL Auth, remote)
- **Language**: Python 3.12+
- **ML Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn
- **Database Connectivity**: pyodbc (SQL Auth via .env credentials)

## Pipeline Flow
1. `calculate_technical_indicators()` — 45 features (RSI, MACD, BB, ATR, etc.)
2. `merge_market_context()` — +11 features (VIX, DXY, yields, NIFTY/S&P returns)
3. `add_market_neutral_features()` — +10 features (relative performance, beta)
4. `add_interaction_features()` — +10 features (smart market+stock combinations)
5. `select_features()` — Weighted selection → 20 balanced features
6. `train_model()` — GB classifier + isotonic calibration
7. `predict_nse_signals_v2.py` — Daily predictions → SQL Server

## Schedule
- Daily 4:30 PM (Mon-Fri): NSE prediction run (after 3 PM EST data fetch)
- Sunday 12:00 PM (noon): Weekly full retrain (classifiers + regressors)

## Data Pipeline
- **NSE Market**: Closes 3:30 PM IST (~5 AM EST)
- **Data Fetch**: 3:00 PM EST daily via yfinance
- **Predictions**: 4:30 PM EST daily (uses latest available trading_date from database)

## Database Notes
- **Server**: `192.168.86.28\MSSQLSERVER01` (named instance, NOT port-based)
- **Auth**: SQL Auth (`remote_user`), `SQL_TRUSTED_CONNECTION=no`
- **WARNING**: Do NOT use IP `192.168.87.27` or port-based format (`,1444`). Wrong `.env` config caused a silent 3-day outage in April 2026.
- Price columns in `nse_500_hist_data` are **VARCHAR** — always use `CAST(close_price AS FLOAT)`
- Predictions include sector and market_cap_category for stratified analysis
- Summary table tracks multi-horizon success rates (1d/5d/10d)

## Code Guidelines
- Use pyodbc for database connections (Windows Integrated Auth)
- Follow PEP 8 Python style guidelines
- Use type hints where appropriate
- Include proper error handling for database operations
- Use environment variables for database credentials

## Sibling Repositories (same database)
- `sqlserver_copilot` — NASDAQ ML (Gradient Boosting, simpler)
- `sqlserver_copilot_forex` — Forex ML (XGBoost/LightGBM, 3-class)
- `stockdata_agenticai` — CrewAI agents that consume predictions
- `streamlit-trading-dashboard` — Visualization
- `sqlserver_mcp` — .NET 8 MCP Server (Microsoft MssqlMcp) with 7 tools: ListTables, DescribeTable, ReadData, CreateTable, DropTable, InsertData, UpdateData. Stdio transport. Use to explore DB schemas and verify query results during development.
- `stockanalysis` — Data ingestion ETL

## MCP Server for Development
Configure in `.vscode/mcp.json` to query stockdata_db directly from your AI IDE:
```json
"MSSQL MCP": {
    "type": "stdio",
    "command": "C:\\Users\\sreea\\OneDrive\\Desktop\\sqlserver_mcp\\SQL-AI-samples\\MssqlMcp\\dotnet\\MssqlMcp\\bin\\Debug\\net8.0\\MssqlMcp.exe",
    "env": {
        "CONNECTION_STRING": "Server=192.168.86.28\\MSSQLSERVER01;Database=stockdata_db;User Id=remote_user;Password=YourStrongPassword123!;TrustServerCertificate=True"
    }
}
```
Useful for: verifying `ml_nse_trading_predictions` output, checking `nse_500_hist_data` schema, exploring ensemble model accuracy.
