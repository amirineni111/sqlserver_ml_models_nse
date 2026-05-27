"""Check actual feature values for today's prediction - after scaling."""
import sys, json
sys.path.insert(0, '.')
import retrain_nse_model_v2
from retrain_nse_model_v2 import IsotonicCalibratedModel
import __main__
__main__.IsotonicCalibratedModel = IsotonicCalibratedModel

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os, pyodbc

load_dotenv()
conn_str = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={os.getenv('SQL_SERVER')};"
    f"DATABASE={os.getenv('SQL_DATABASE')};"
    f"UID={os.getenv('SQL_USERNAME')};"
    f"PWD={os.getenv('SQL_PASSWORD')};"
    "TrustServerCertificate=yes"
)
conn = pyodbc.connect(conn_str)

model = joblib.load('data/nse_models/nse_gb_model_v2.joblib')
scaler = joblib.load('data/nse_models/nse_scaler_v2.joblib')
with open('data/nse_models/selected_features_v2.json') as f:
    selected_features = json.load(f)

print(f"Selected features: {selected_features}\n")

# Find prediction date
pred_date_row = pd.read_sql("SELECT MAX(trading_date) as max_date FROM nse_500_hist_data", conn)
prediction_date = pd.to_datetime(pred_date_row.iloc[0]['max_date'])
print(f"Prediction date from DB: {prediction_date.date()}\n")

# Check if selected features exist in a sample row
# Load a few stocks from predict pipeline
sample_sql = f"""
SELECT TOP 30000 
    h.ticker, 
    CONVERT(date, h.trading_date) as trading_date,
    CAST(h.close_price AS FLOAT) as close_price,
    CAST(h.open_price AS FLOAT) as open_price,
    CAST(h.high_price AS FLOAT) as high_price,
    CAST(h.low_price AS FLOAT) as low_price,
    CAST(h.volume AS FLOAT) as volume,
    n.sector, n.industry
FROM nse_500_hist_data h
JOIN nse_500 n ON h.ticker = n.ticker
WHERE h.trading_date >= DATEADD(day, -200, '{prediction_date.strftime('%Y-%m-%d')}')
  AND h.ticker IN (SELECT TOP 10 ticker FROM nse_500)
ORDER BY h.ticker, h.trading_date
"""
df = pd.read_sql(sample_sql, conn)
df['trading_date'] = pd.to_datetime(df['trading_date'])
print(f"Loaded {len(df)} rows for {df['ticker'].nunique()} sample tickers")
print(f"Date range: {df['trading_date'].min().date()} to {df['trading_date'].max().date()}\n")

# Check what dates we have for latest 3 dates
latest_dates = df['trading_date'].sort_values().unique()[-5:]
print(f"Latest 5 dates in stock data: {[str(d.date()) for d in latest_dates]}")

# Check market_context_daily for those dates
mkt_sql = f"""
SELECT trading_date, nifty50_close, india_vix_close, sp500_close, vix_close, dxy_close,
       nifty50_return_1d, vix_change_pct, india_vix_change_pct
FROM market_context_daily
WHERE trading_date >= '{latest_dates[0].strftime('%Y-%m-%d')}'
ORDER BY trading_date
"""
mkt = pd.read_sql(mkt_sql, conn)
mkt['trading_date'] = pd.to_datetime(mkt['trading_date'])
print(f"\nMarket context for those dates:")
print(mkt.to_string(index=False))
conn.close()
