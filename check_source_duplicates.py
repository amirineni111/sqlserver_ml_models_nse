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
print("CHECKING DATA SOURCES FOR DUPLICATES")
print("=" * 80)

# Check nse_500_fundamentals for duplicate tickers
print("\n1. nse_500_fundamentals - Duplicate ticker check:")
query = """
SELECT ticker, COUNT(*) as count 
FROM nse_500_fundamentals 
GROUP BY ticker 
HAVING COUNT(*) > 1 
ORDER BY count DESC
"""
df = pd.read_sql(query, conn)
if df.empty:
    print("✅ No duplicates found in nse_500_fundamentals")
else:
    print(f"⚠️ Found {len(df)} tickers with duplicates:")
    print(df.head(20))

# Check nse_500 for duplicate tickers
print("\n2. nse_500 - Duplicate ticker check:")
query = """
SELECT ticker, COUNT(*) as count 
FROM nse_500 
GROUP BY ticker 
HAVING COUNT(*) > 1 
ORDER BY count DESC
"""
df = pd.read_sql(query, conn)
if df.empty:
    print("✅ No duplicates found in nse_500")
else:
    print(f"⚠️ Found {len(df)} tickers with duplicates:")
    print(df.head(20))

# Check the EXACT query used by predict_nse_signals_v2.py for duplicates
print("\n3. Testing prediction query for duplicates (April 17):")
query = """
WITH price_data AS (
    SELECT 
        h.ticker,
        h.trading_date,
        CAST(h.open_price AS FLOAT) AS open_price,
        CAST(h.high_price AS FLOAT) AS high_price,
        CAST(h.low_price AS FLOAT) AS low_price,
        CAST(h.close_price AS FLOAT) AS close_price,
        CAST(h.volume AS FLOAT) AS volume,
        n.sector,
        n.industry,
        
        CASE 
            WHEN f.market_cap >= 100000000000 THEN 'Large Cap'
            WHEN f.market_cap >= 10000000000 THEN 'Mid Cap'
            ELSE 'Small Cap'
        END AS market_cap_category
        
    FROM nse_500_hist_data h
    INNER JOIN nse_500 n ON h.ticker = n.ticker
    LEFT JOIN nse_500_fundamentals f ON h.ticker = f.ticker
    
    WHERE h.trading_date = '2026-04-17'
      AND h.close_price IS NOT NULL
      AND h.close_price <> '0'
      AND CAST(h.volume AS FLOAT) > 0
)

SELECT 
    ticker, 
    COUNT(*) as row_count
FROM price_data
GROUP BY ticker
HAVING COUNT(*) > 1
ORDER BY row_count DESC
"""
df = pd.read_sql(query, conn)
if df.empty:
    print("✅ No duplicates in prediction query for April 17")
else:
    print(f"⚠️ Found {len(df)} tickers with duplicate rows:")
    print(df.head(20))
    print(f"\n🚨 THIS IS THE PROBLEM! The LEFT JOIN creates duplicate rows!")

conn.close()
