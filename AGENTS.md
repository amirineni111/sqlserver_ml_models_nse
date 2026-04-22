# AGENTS.md — sqlserver_copilot_nse (NSE ML Pipeline)

## Overview
This repo does NOT contain CrewAI agents. It is a **5-model ensemble ML training pipeline** for NSE 500 stock predictions.

## ML Pipeline Architecture

```
[NSE Market closes 3:30 PM IST → yfinance fetch 3 PM EST]
        │
[nse_500_hist_data] + [nse_500_fundamentals] (SQL Server)
        │
        ▼
  feature_engineering.py  (90+ features)
        │
        ▼
  feature_selection.py  (feature selection)
        │
        ├──────────────────────────┐
        ▼                          ▼
  ensemble_builder.py          regressor_builder.py
  (RF, GB, ET, LR, Voting)    (4 price regressors)
        │                          │
        ▼                          ▼
  models/*.pkl                 models/*_regressor.pkl
        │
        ▼
  predict_daily.py  (daily predictions → SQL Server)
        │          Uses max(trading_date) from database
        ▼
  [ml_nse_trading_predictions] + [ml_nse_predict_summary]
```

## Classifier Ensemble (5 models)
| Model | Algorithm |
|-------|-----------|
| Random Forest | RF Classifier |
| Gradient Boosting | GB Classifier |
| Extra Trees | ET Classifier |
| Logistic Regression | LR |
| VotingClassifier | Soft voting (all 4 above) |

## Key Differences from NASDAQ Pipeline
- **5 models** (vs 1 Gradient Boosting)
- **90+ features** (vs 50+)
- **Regression models** for price targets
- **500 stocks** (vs 100)
- **Sector + market cap** stratification
- Multi-horizon success tracking (1d/5d/10d)

## Database Configuration
- **Server**: `192.168.86.28\MSSQLSERVER01` (named instance, NOT port-based)
- **Database**: `stockdata_db`
- **Auth**: SQL Auth (`remote_user`), `SQL_TRUSTED_CONNECTION=no`
- **WARNING**: Do NOT use IP `192.168.87.27` or port-based format (`,1444`). Wrong `.env` config caused a silent 3-day outage in April 2026.

## Downstream Consumers
- **stockdata_agenticai** — ML Analyst, Strategy Trade, Cross-Strategy agents
- **streamlit-trading-dashboard** — Prediction display and accuracy tracking
- **vw_strategy2_trade_opportunities** view joins predictions with tech signals
