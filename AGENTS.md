# AGENTS.md — sqlserver_copilot_nse (NSE ML Pipeline)

## Overview
This repo does NOT contain CrewAI agents. It is a **V2 ML training pipeline** with a single Gradient Boosting classifier + hybrid feature engineering for NSE 500 stock predictions.

## ML Pipeline Architecture (V2 - April 2026)

```
[NSE Market closes 3:30 PM IST → yfinance fetch 3 PM EST]
        │
[nse_500_hist_data] + [nse_500_fundamentals] (SQL Server)
        │
        ▼
  calculate_technical_indicators()  (45 features)
        │
        ▼
  merge_market_context()  (+11 features: VIX, DXY, yields)
        │
        ├─────────────────────────────────┐
        ▼                                 ▼
  add_market_neutral_features()    add_interaction_features()
  (+10 relative metrics)           (+10 smart combinations)
        │                                 │
        └─────────────┬───────────────────┘
                      ▼
              select_features()  (weighted category-based)
                      │          (76 → 20 balanced features)
                      ▼
              train_model()  (GB + isotonic calibration)
                      │
                      ▼
              nse_gb_model_v2.pkl
                      │
                      ▼
  predict_nse_signals_v2.py  (daily predictions → SQL Server)
                │            Uses max(trading_date) from database
                ▼
  [ml_nse_trading_predictions] + [ml_nse_predict_summary]
```

## V2 Model Architecture (Single Classifier)
| Component | Details |
|-----------|---------|
| Base Model | Gradient Boosting Classifier |
| Calibration | Isotonic regression (stratified cal set) |
| Training Split | 60% train / 20% cal / 20% test |
| Sample Weighting | Class-balanced + time-weighted |
| Feature Selection | Weighted category-based (prevents dominance) |

## HYBRID FEATURE ENGINEERING (Critical - April 2026)

**Problem Solved**: Model was over-relying on market context (VIX, DXY), causing "bearish market = Sell everything"

**Solution**: 3-pronged approach

### 1. Market-Neutral Features (10)
- Stock strength RELATIVE to market/sector
- Examples: `stock_return_vs_nifty`, `rsi_vs_sector_avg`, `beta_adjusted_return`
- Purpose: Find outperformers regardless of market direction

### 2. Interaction Features (10)
- Market context COMBINED with stock signals
- Examples: `outperformance_in_fear` = stock_vs_nifty × (VIX / 15)
- Purpose: "VIX high + Stock strong = Buy" not "VIX high = Sell all"

### 3. Weighted Feature Selection (20 final features)
- Market Context: 25% (controlled, not eliminated)
- Stock-Specific: 35% (prioritized)
- Relative/Neutral: 25%
- Interactions: 15%
- Purpose: Prevent any category from dominating

**Expected Outcome**: 40-50% Buy signals (balanced), not 2% or 98%

## Key Differences from NASDAQ Pipeline
- **V2 architecture** (single GB vs NASDAQ's single GB - same approach)
- **Hybrid features** (76 total → 20 selected vs NASDAQ's 50+ → 20)
- **500 stocks** (vs NASDAQ 100)
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
