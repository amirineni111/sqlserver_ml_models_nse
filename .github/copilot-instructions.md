# Copilot Instructions — sqlserver_copilot_nse

## Project Context
This is the **NSE ML training pipeline** — part of a 7-repo stock trading analytics platform. Trains a 5-model ensemble (RF/GB/ET/LR/Voting) + 4 price regressors to predict Buy/Sell signals for NSE 500 stocks.

## Key Architecture Rules
- Reads from `nse_500_hist_data` (VARCHAR prices — always CAST to FLOAT)
- Writes to `ml_nse_trading_predictions`, `ml_nse_predict_summary`, `ml_nse_technical_indicators`
- 5-model ensemble with 90+ engineered features
- Includes regression models for price target predictions
- Connected to shared database `stockdata_db` on `localhost\MSSQLSERVER01` (Windows Auth)

## Key Technologies
- **Database**: SQL Server (`stockdata_db` on `localhost\MSSQLSERVER01`, Windows Auth)
- **Language**: Python 3.11+
- **ML Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn
- **Database Connectivity**: pyodbc (Trusted_Connection=yes)

## Pipeline Flow
1. `feature_engineering.py` — 90+ features from OHLCV + fundamentals
2. `feature_selection.py` — Feature importance-based selection
3. `ensemble_builder.py` — RF, GB, ET, LR, VotingClassifier
4. `regressor_builder.py` — 4 regression models for price targets
5. `predict_daily.py` — Daily predictions → SQL Server

## Schedule
- Daily 9:30 AM: NSE prediction run
- Sunday 2:00 AM: Weekly full retrain (classifiers + regressors)

## Database Notes
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
- `sqlserver_mcp` — .NET MCP bridge
- `stockanalysis` — Data ingestion ETL
