# CLAUDE.md — sqlserver_copilot_nse (NSE ML Training Pipeline)

> **Project context file for AI assistants (Claude, Copilot, Cursor).**

---

## 1. SYSTEM OVERVIEW

This is the **NSE ML training pipeline** — one of **7 interconnected repositories** that form an AI-powered stock trading analytics platform. All repos share a single SQL Server database (`stockdata_db`).

### Repository Map

| Layer | Repo | Purpose |
|-------|------|---------|
| Data Ingestion | `stockanalysis` | ETL: yfinance/Alpha Vantage → SQL Server |
| SQL Infrastructure | `sqlserver_mcp` | .NET 8 MCP Server (Microsoft MssqlMcp) — 7 tools (ListTables, DescribeTable, ReadData, CreateTable, DropTable, InsertData, UpdateData) via stdio transport for AI IDE ↔ SQL Server |
| Dashboard | `streamlit-trading-dashboard` | 40+ views, signal tracking, Streamlit UI |
| ML: NASDAQ | `sqlserver_copilot` | Gradient Boosting → `ml_trading_predictions` |
| **ML: NSE** ⭐ | **`sqlserver_copilot_nse`** | **THIS REPO** — 5-model ensemble → `ml_nse_trading_predictions` |
| ML: Forex | `sqlserver_copilot_forex` | XGBoost/LightGBM → `forex_ml_predictions` |
| Agentic AI | `stockdata_agenticai` | 7 CrewAI agents, daily briefing email |

---

## 2. THIS REPO: sqlserver_copilot_nse

### Purpose
Trains a **single Gradient Boosting classifier (V2 architecture)** with isotonic calibration on NSE 500 stocks to predict Buy/Sell signals. Uses **hybrid feature approach** combining market context (25%), stock-specific (35%), relative/neutral (25%), and interaction features (15%) for balanced, context-aware predictions. Writes predictions to `ml_nse_trading_predictions`.

### Daily Schedule (Windows Task Scheduler)
```
4:30 PM (Mon-Fri)   Daily prediction run  → ml_nse_trading_predictions
Sunday 12:00 PM     Weekly full retrain    → Updated model + regressor files
```

**Data Pipeline:**
- NSE market closes: 3:30 PM IST (~5:00 AM EST)
- yfinance data fetch: 3:00 PM EST daily
- ML predictions: 4:30 PM EST daily (after data fetch completes)

### Key Files

```
sqlserver_copilot_nse/
├── retrain_nse_model_v2.py          # V2 training script (single GB + calibration)
├── predict_nse_signals_v2.py        # V2 prediction script (matches training features)
├── daily_nse_automation.py          # Orchestrates daily workflow
├── run_daily_predictions.bat        # Windows Task Scheduler wrapper (4:30 PM Mon-Fri)
├── run_weekly_retrain.bat           # Weekly retrain wrapper (Sunday 12 PM)
├── data/nse_models/
│   ├── nse_gb_model_v2.pkl          # Gradient Boosting model (calibrated)
│   ├── scaler_v2.pkl                # StandardScaler for features
│   ├── label_encoder_v2.pkl         # LabelEncoder (Down=0, Up=1)
│   ├── selected_features_v2.pkl     # 20 selected feature names
│   └── model_metadata_v2.json       # Training metadata + validation results
├── HYBRID_APPROACH_APRIL_21_2026.md # ML feature engineering documentation
├── CLAUDE.md                        # This file - AI assistant reference
└── AGENTS.md                        # Architecture overview
```

---

## 3. ML MODEL DETAILS (V2 ARCHITECTURE - APRIL 2026)

### Model Architecture
**Single Gradient Boosting Classifier** with isotonic probability calibration
- Based on proven NASDAQ approach (simplified from V1's 5-model ensemble)
- Training: 60% train / 20% calibration / 20% test
- Stratified calibration split to prevent bias
- Class + time-based sample weighting

### CRITICAL: Hybrid Feature Engineering Approach (April 21, 2026)

After 5 iterations of bearish bias fixes, we discovered market context features were **valuable but over-dominant**, causing "bearish market = Sell everything" behavior. Solution: **3-pronged hybrid approach**:

#### 1. Market-Neutral Features (10 features - 25% of final selection)
Measure stock strength RELATIVE to market/sector:
- `stock_return_vs_nifty`: Stock return - NIFTY return (outperformance)
- `rsi_vs_sector_avg`: Stock RSI - sector average RSI (relative momentum)
- `beta_adjusted_return`: Return - (beta × market return) (risk-adjusted)
- `volume_anomaly`: Z-score vs 50-day history (stock-specific spikes)
- `relative_strength_20d`: 20-day cumulative outperformance
- `momentum_vs_sector`: Stock vs sector 10-day momentum

**Purpose**: Identify Stock A (+3%) > Stock B (+1%) even when market is down -2%

#### 2. Interaction Features (10 features - 15% of final selection)
Explicitly combine market context WITH stock signals:
- `outperformance_in_fear`: stock_vs_nifty × (VIX / 15) → **High VIX + Outperforming = Strong Buy**
- `contrarian_strength`: stock_vs_nifty × (-NIFTY_return) → **Up when market down = Winner**
- `sector_leader_conviction`: (RSI_vs_sector / 20) × volume_anomaly → Quality + Volume
- `quality_opportunity`: stock_vs_nifty / beta → Low beta outperformers = Quality
- `risk_adjusted_momentum`: stock_vs_nifty_5d / (ATR / price) → Return per unit risk
- `quality_in_volatility`: RSI_vs_sector × (VIX / VIX_20d_avg) → Strong during stress

**Purpose**: Teach model "VIX high + Stock strong = Opportunity" not "VIX high = Sell all"

#### 3. Weighted Feature Selection (20 features total)
Force balanced category representation:
- **Market Context**: 25% (5 features) - VIX/DXY/yields **controlled, not eliminated**
- **Stock-Specific**: 35% (7 features) - Core technical indicators **prioritized**
- **Relative/Neutral**: 25% (5 features) - Comparative metrics
- **Interactions**: 15% (3 features) - Smart combinations

**Mechanism**: Select top N from EACH category (not global top N) to prevent market feature dominance

### Feature Pipeline (76 total → 20 selected)
```
calculate_technical_indicators()     → 45 features (RSI, MACD, BB, ATR, etc.)
merge_market_context()               → +11 features (VIX, DXY, yields, NIFTY/S&P returns)
add_market_neutral_features()        → +10 features (relative performance, beta, anomalies)
add_interaction_features()           → +10 features (smart combinations)
select_features() [weighted]         → 20 features (balanced across categories)
```

### Expected Behavior
- **Bearish market + Weak stock** → Sell
- **Bearish market + Strong stock** → Buy (contrarian opportunity)
- **Bullish market + Weak stock** → Sell (laggard)
- **Bullish market + Strong stock** → Buy (momentum)

**Target**: 40-50% Buy signals (balanced), not 2% or 98%

### Historical Architecture Issues (Documented for Learning)
| Date | Issue | Root Cause | Fix |
|------|-------|-----------|-----|
| Apr 14 | 0% predictions | Undefined variables | Variable definitions |
| Apr 16 | 97.3% Sell | No class balancing | Added class_weight='balanced' |
| Apr 18 | 97.0% Sell | VotingClassifier retraining bug | Removed ensemble, use single GB |
| Apr 21 (Part 1) | 99.6% Sell | Biased calibration set | Stratified calibration split |
| Apr 21 (Part 2) | 98.0% Sell (42 Buy) | Market feature dominance | Hybrid approach (this fix) |

### Output Table: `ml_nse_trading_predictions`
| Column | Type | Description |
|--------|------|-------------|
| ticker | VARCHAR | NSE stock symbol |
| trading_date | DATE | Prediction date |
| predicted_signal | VARCHAR | 'Buy' or 'Sell' |
| confidence_percentage | FLOAT | Ensemble confidence (0-100) |
| signal_strength | VARCHAR | 'Strong'/'Moderate'/'Weak' |
| RSI | FLOAT | Current RSI value |
| buy_probability | FLOAT | P(Buy) from ensemble |
| sell_probability | FLOAT | P(Sell) from ensemble |
| model_name | VARCHAR | Model identifier |
| sector | VARCHAR | Stock sector |
| market_cap_category | VARCHAR | Large/Mid/Small cap |
| high_confidence | BIT | Flag for confidence ≥ 60% (lowered from 70% Apr 2026) |

### Also Writes
- `ml_nse_predict_summary` — Daily aggregates + model_accuracy, success_rate_1d/5d/10d
- `ml_nse_technical_indicators` — Technical indicator snapshots

---

## 4. DATABASE CONTEXT

### Tables This Repo READS
| Table | Purpose |
|-------|---------|
| `nse_500_hist_data` | Historical OHLCV (VARCHAR prices — CAST to FLOAT!) |
| `nse_500` | Ticker master list with sector/industry |
| `nse_500_fundamentals` | 37 fundamental metrics per ticker |

### Tables This Repo WRITES
| Table | Purpose |
|-------|---------|
| `ml_nse_trading_predictions` | Daily Buy/Sell predictions with confidence |
| `ml_nse_predict_summary` | Aggregate stats + accuracy tracking |
| `ml_nse_technical_indicators` | Indicator snapshots |

---

## 5. CODING CONVENTIONS

### .env Configuration (Critical)
The `.env` file **must** use the correct SQL Server address matching the NASDAQ repo:
```
SQL_SERVER=192.168.86.28\MSSQLSERVER01
SQL_DATABASE=stockdata_db
SQL_USERNAME=remote_user
SQL_DRIVER=ODBC Driver 17 for SQL Server
SQL_TRUSTED_CONNECTION=no
```
**WARNING**: Do NOT use IP `192.168.87.27` or port-based format (`,1444`). The correct address is `192.168.86.28\MSSQLSERVER01` (named instance). A wrong IP in `.env` caused a 3-day prediction outage in April 2026 that went undetected because the script exited with code 0 on DB failure.

### Critical Data Issues
- **VARCHAR Price Columns**: Same as NASDAQ repo — `CAST(close_price AS FLOAT)` always
- **500 stocks**: Much larger universe than NASDAQ 100. Feature engineering takes longer.
- **Sector/Cap Stratification**: Summary includes model_accuracy by sector and market_cap_category

### Key Differences from NASDAQ Repo
- 5-model ensemble (vs single Gradient Boosting)
- 90+ features (vs 50+)
- Includes regression models for price targets
- Summary table tracks multi-horizon success rates (1d/5d/10d)
- Predictions include sector and market_cap_category columns

---

## 6. DOWNSTREAM CONSUMERS
- **stockdata_agenticai** — ML Analyst + Strategy Trade + Cross-Strategy agents read these predictions
- **streamlit-trading-dashboard** — Displays predictions and tracks accuracy
- `vw_strategy2_trade_opportunities` view joins `ml_nse_trading_predictions` with tech signal data

---

## 7. MCP SERVER FOR DEVELOPMENT

The `sqlserver_mcp` repo provides an MCP server for AI IDEs to query `stockdata_db` directly during development.

### VS Code Configuration
```json
"MSSQL MCP": {
    "type": "stdio",
    "command": "C:\\Users\\sreea\\OneDrive\\Desktop\\sqlserver_mcp\\SQL-AI-samples\\MssqlMcp\\dotnet\\MssqlMcp\\bin\\Debug\\net8.0\\MssqlMcp.exe",
    "env": {
        "CONNECTION_STRING": "Server=192.168.86.28\\MSSQLSERVER01;Database=stockdata_db;User Id=remote_user;Password=YourStrongPassword123!;TrustServerCertificate=True"
    }
}
```

### 7 MCP Tools: ListTables, DescribeTable, ReadData, CreateTable, DropTable, InsertData, UpdateData

Useful for: checking `ml_nse_trading_predictions` output format, verifying `nse_500_hist_data` schema, exploring prediction accuracy and ensemble model results.
