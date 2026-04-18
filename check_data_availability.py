import pyodbc
import pandas as pd
from datetime import datetime

# Connect to database
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=192.168.86.28\\MSSQLSERVER01;'
    'DATABASE=stockdata_db;'
    'UID=remote_user;'
    'PWD=YourStrongPassword123!;'
    'TrustServerCertificate=yes'
)

print("=" * 80)
print("DATA AVAILABILITY CHECK - April 18, 2026")
print("=" * 80)

# Check stock price data
print("\n1. NSE STOCK PRICE DATA (nse_500_hist_data)")
print("-" * 80)
query = """
SELECT 
    MAX(trading_date) as max_date,
    MIN(trading_date) as min_date,
    COUNT(DISTINCT ticker) as total_tickers,
    COUNT(*) as total_rows
FROM nse_500_hist_data
"""
df = pd.read_sql(query, conn)
print(df.to_string(index=False))

max_stock_date = df['max_date'].iloc[0]
print(f"\n✅ Stock prices available through: {max_stock_date}")
print(f"   Days behind today (2026-04-18): {(datetime(2026, 4, 18) - pd.Timestamp(max_stock_date)).days} days")

# Check market context data
print("\n2. MARKET CONTEXT DATA (market_context_daily)")
print("-" * 80)
query = """
SELECT 
    MAX(trading_date) as max_date,
    MIN(trading_date) as min_date,
    COUNT(*) as total_rows
FROM market_context_daily
"""
df = pd.read_sql(query, conn)
print(df.to_string(index=False))

max_market_date = df['max_date'].iloc[0]
print(f"\n⚠️ Market indicators available through: {max_market_date}")
print(f"   Days behind today (2026-04-18): {(datetime(2026, 4, 18) - pd.Timestamp(max_market_date)).days} days")

# Check recent stock data by date
print("\n3. RECENT STOCK DATA BY DATE (Feb-Apr 2026)")
print("-" * 80)
query = """
SELECT 
    trading_date,
    COUNT(*) as ticker_count
FROM nse_500_hist_data
WHERE trading_date >= '2026-02-15'
GROUP BY trading_date
ORDER BY trading_date DESC
"""
df = pd.read_sql(query, conn)
print(df.head(30).to_string(index=False))

# Check recent market context by date
print("\n4. RECENT MARKET CONTEXT BY DATE")
print("-" * 80)
query = """
SELECT 
    trading_date,
    nifty50_close,
    india_vix_close
FROM market_context_daily
WHERE trading_date >= '2026-02-15'
ORDER BY trading_date DESC
"""
df = pd.read_sql(query, conn)
print(df.head(30).to_string(index=False))

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

if pd.Timestamp(max_stock_date) > pd.Timestamp(max_market_date):
    gap = (pd.Timestamp(max_stock_date) - pd.Timestamp(max_market_date)).days
    print(f"\n🚨 CRITICAL GAP FOUND!")
    print(f"   Stock data: {max_stock_date}")
    print(f"   Market data: {max_market_date}")
    print(f"   Gap: {gap} days")
    print(f"\n   This is why the model can't train on data after {max_market_date}!")
    print(f"   The model NEEDS market_context_daily for these 8 features:")
    print(f"   - nifty50_return_1d, nifty50_close")
    print(f"   - india_vix_close, india_vix_change_pct")
    print(f"   - And 4 more market indicators")
    print(f"\n   WITHOUT market data, we can't calculate features for stock data after {max_market_date}")
    print(f"\n   SOLUTION: Update market_context_daily table from stockanalysis ETL repo")
else:
    print(f"\n✅ Stock and market data are in sync through {max_market_date}")

conn.close()
