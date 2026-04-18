import pandas as pd
import joblib
import pyodbc

# Load the model artifacts
scaler = joblib.load('data/nse_models/nse_scaler_v2.joblib')
with open('data/nse_models/selected_features_v2.json', 'r') as f:
    import json
    selected_features = json.load(f)

print("=" * 80)
print("INSPECTING APRIL 17 PREDICTION INPUTS")
print("=" * 80)

# Connect and get raw data
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=192.168.86.28\\MSSQLSERVER01;'
    'DATABASE=stockdata_db;'
    'UID=remote_user;'
    'PWD=YourStrongPassword123!;'
    'TrustServerCertificate=yes'
)

# Get a sample stock's data
query = """
WITH latest_fundamentals AS (
    SELECT 
        ticker,
        market_cap,
        ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY fetch_date DESC) as rn
    FROM nse_500_fundamentals
)
SELECT TOP 1
    h.ticker,
    h.trading_date,
    CAST(h.close_price AS FLOAT) as close_price,
    CAST(h.volume AS BIGINT) as volume,
    f.market_cap,
    m.nifty50_close,
    m.nifty50_return_1d,
    m.india_vix_close,
    m.india_vix_change_pct,
    m.sp500_close,
    m.sp500_return_1d,
    m.vix_close,
    m.vix_change_pct,
    m.dxy_close,
    m.dxy_return_1d,
    m.us_10y_yield_close
FROM nse_500_hist_data h
LEFT JOIN latest_fundamentals f ON h.ticker = f.ticker AND f.rn = 1
LEFT JOIN market_context_daily m ON h.trading_date = m.trading_date
WHERE h.trading_date = '2026-04-17'
  AND h.ticker = 'RELIANCE.NS'
"""

df = pd.read_sql(query, conn)
print("\n1. RAW DATA FOR RELIANCE.NS ON APRIL 17")
print("-" * 80)
print(df.T)

# Check what features the model needs
print("\n2. REQUIRED FEATURES (Top 20)")
print("-" * 80)
for i, feat in enumerate(selected_features, 1):
    print(f"{i:2d}. {feat}")

# Check if any required features are missing
print("\n3. CHECKING FOR DATA ISSUES")
print("-" * 80)

# Load full prediction data
query = """
SELECT 
    COUNT(DISTINCT ticker) as ticker_count,
    COUNT(*) as total_rows
FROM nse_500_hist_data
WHERE trading_date = '2026-04-17'
"""
df = pd.read_sql(query, conn)
print(f"Stock data: {df.iloc[0]['ticker_count']} tickers, {df.iloc[0]['total_rows']} rows")

# Check market context
query = """
SELECT COUNT(*) as context_rows
FROM market_context_daily
WHERE trading_date = '2026-04-17'
"""
df = pd.read_sql(query, conn)
print(f"Market context: {df.iloc[0]['context_rows']} row(s)")

conn.close()

# Now let's check prediction script for issues
print("\n4. HYPOTHESIS: What could cause 100% Sell?")
print("-" * 80)
print("""
Possible causes:
1. Missing features filled with 0 (could signal extreme bearish)
2. Feature scaling issue (VIX scaled incorrectly)
3. Model loading wrong version
4. Prediction date filter issue (using wrong data)
5. Class label mapping flipped (Up/Down reversed)

Let me check the prediction script logic...
""")

# Read the prediction script
with open('predict_nse_signals_v2.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
print("\n5. CHECKING CLASS LABEL MAPPING IN PREDICT SCRIPT")
print("-" * 80)

for i, line in enumerate(lines[280:360], start=281):
    if 'up_idx' in line.lower() or 'down_idx' in line.lower() or 'predicted_signal' in line.lower():
        print(f"Line {i}: {line.rstrip()}")

print("\nNote: Encoder classes should be ['Down', 'Up'] with Down=0, Up=1")
print("If prediction uses index 0 as Buy, that would flip everything!")
