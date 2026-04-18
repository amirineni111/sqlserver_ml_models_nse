import pyodbc
import pandas as pd

conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=192.168.86.28\\MSSQLSERVER01;'
    'DATABASE=stockdata_db;'
    'UID=remote_user;'
    'PWD=YourStrongPassword123!;'
    'TrustServerCertificate=yes'
)

print("=" * 80)
print("APRIL 17, 2026 DATA INVESTIGATION")
print("=" * 80)

# Check market context for April 17
print("\n1. MARKET CONTEXT ON APRIL 17")
print("-" * 80)
query = """
SELECT TOP 5
    trading_date,
    nifty50_close,
    nifty50_return_1d,
    india_vix_close,
    india_vix_change_pct,
    sp500_close,
    vix_close,
    dxy_close,
    us_10y_yield_close
FROM market_context_daily
WHERE trading_date >= '2026-04-15'
ORDER BY trading_date DESC
"""
df = pd.read_sql(query, conn)
print(df.to_string(index=False))

# Compare to training period average
print("\n2. TRAINING PERIOD AVERAGES (Jun 2024 - Apr 2026)")
print("-" * 80)
query = """
SELECT 
    AVG(nifty50_close) as avg_nifty50,
    AVG(nifty50_return_1d) as avg_nifty50_return,
    AVG(india_vix_close) as avg_vix,
    AVG(india_vix_change_pct) as avg_vix_change,
    AVG(sp500_close) as avg_sp500,
    AVG(vix_close) as avg_us_vix,
    MAX(india_vix_close) as max_vix,
    MIN(india_vix_close) as min_vix
FROM market_context_daily
WHERE trading_date BETWEEN '2024-06-01' AND '2026-04-15'
"""
df = pd.read_sql(query, conn)
print(df.to_string(index=False))

# Check for missing features
print("\n3. CHECK FOR NULL VALUES IN APRIL 17 DATA")
print("-" * 80)
query = """
SELECT 
    trading_date,
    CASE WHEN nifty50_close IS NULL THEN 'NULL' ELSE 'OK' END as nifty50,
    CASE WHEN india_vix_close IS NULL THEN 'NULL' ELSE 'OK' END as vix,
    CASE WHEN sp500_close IS NULL THEN 'NULL' ELSE 'OK' END as sp500,
    CASE WHEN vix_close IS NULL THEN 'NULL' ELSE 'OK' END as us_vix,
    CASE WHEN dxy_close IS NULL THEN 'NULL' ELSE 'OK' END as dxy,
    CASE WHEN us_10y_yield_close IS NULL THEN 'NULL' ELSE 'OK' END as yield
FROM market_context_daily
WHERE trading_date = '2026-04-17'
"""
df = pd.read_sql(query, conn)
print(df.to_string(index=False))

# Check stock data
print("\n4. SAMPLE STOCK DATA FOR APRIL 17")
print("-" * 80)
query = """
SELECT TOP 5
    ticker,
    CAST(close_price AS FLOAT) as close_price,
    CAST(volume AS BIGINT) as volume
FROM nse_500_hist_data
WHERE trading_date = '2026-04-17'
ORDER BY ticker
"""
df = pd.read_sql(query, conn)
print(df.to_string(index=False))

# Check if stocks went up or down on April 17
print("\n5. ACTUAL APRIL 17 PERFORMANCE (5-day forward return)")
print("-" * 80)
query = """
WITH forward_returns AS (
    SELECT 
        h1.ticker,
        h1.trading_date,
        CAST(h1.close_price AS FLOAT) as close_t0,
        CAST(h2.close_price AS FLOAT) as close_t5,
        ((CAST(h2.close_price AS FLOAT) - CAST(h1.close_price AS FLOAT)) / CAST(h1.close_price AS FLOAT)) * 100 as return_5d
    FROM nse_500_hist_data h1
    LEFT JOIN nse_500_hist_data h2 ON h1.ticker = h2.ticker
        AND h2.trading_date = (
            SELECT MIN(trading_date) 
            FROM nse_500_hist_data 
            WHERE ticker = h1.ticker 
            AND trading_date > h1.trading_date
        )
    WHERE h1.trading_date = '2026-04-17'
)
SELECT 
    COUNT(*) as total_stocks,
    SUM(CASE WHEN return_5d > 0 THEN 1 ELSE 0 END) as stocks_up,
    SUM(CASE WHEN return_5d < 0 THEN 1 ELSE 0 END) as stocks_down,
    AVG(return_5d) as avg_return_5d,
    MAX(return_5d) as max_return,
    MIN(return_5d) as min_return
FROM forward_returns
WHERE close_t5 IS NOT NULL
"""
df = pd.read_sql(query, conn)
print(df.to_string(index=False))

conn.close()

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
print("""
If india_vix_close is extremely high on April 17, that would explain 100% Sell.
The model learned: High VIX = Market Fear = Stocks Go Down

But we need to check if that's ACTUALLY what happened on April 17.
If stocks went UP despite high VIX, the model is wrong.
If stocks went DOWN as predicted, the model is correct (just extreme market conditions).
""")
