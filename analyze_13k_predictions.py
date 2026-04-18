import pyodbc
import pandas as pd

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
print("WHY 13,022 PREDICTIONS? - Investigation")
print("=" * 80)

# Check V2 predictions
print("\n1. V2 PREDICTIONS FOR APRIL 17, 2026")
print("-" * 80)
query = """
SELECT 
    COUNT(*) as total_predictions,
    COUNT(DISTINCT ticker) as unique_tickers
FROM ml_nse_trading_predictions 
WHERE trading_date = '2026-04-17' 
  AND model_name = 'GradientBoosting_V2_Calibrated'
"""
df = pd.read_sql(query, conn)
print(df.to_string(index=False))

total_preds = df['total_predictions'].iloc[0]
unique_tickers = df['unique_tickers'].iloc[0]

if total_preds > unique_tickers:
    print(f"\n⚠️ DUPLICATES FOUND!")
    print(f"   Total predictions: {total_preds}")
    print(f"   Unique tickers: {unique_tickers}")
    print(f"   Duplicate entries: {total_preds - unique_tickers}")

# Check actual stock data for April 17
print("\n2. ACTUAL STOCK DATA FOR APRIL 17, 2026")
print("-" * 80)
query = """
SELECT 
    COUNT(*) as total_rows,
    COUNT(DISTINCT ticker) as unique_tickers
FROM nse_500_hist_data 
WHERE trading_date = '2026-04-17'
"""
df = pd.read_sql(query, conn)
print(df.to_string(index=False))

actual_stocks = df['unique_tickers'].iloc[0]

# Check for duplicate ticker predictions
print("\n3. DUPLICATE TICKERS IN V2 PREDICTIONS (if any)")
print("-" * 80)
query = """
SELECT 
    ticker, 
    COUNT(*) as prediction_count
FROM ml_nse_trading_predictions 
WHERE trading_date = '2026-04-17' 
  AND model_name = 'GradientBoosting_V2_Calibrated'
GROUP BY ticker 
HAVING COUNT(*) > 1
ORDER BY prediction_count DESC
"""
df = pd.read_sql(query, conn)
if not df.empty:
    print(df.head(20).to_string(index=False))
    print(f"\nTotal tickers with duplicates: {len(df)}")
else:
    print("No duplicate tickers found")

# Check nse_500 master list
print("\n4. NSE_500 MASTER LIST")
print("-" * 80)
query = """
SELECT 
    COUNT(DISTINCT ticker) as total_tickers_in_master
FROM nse_500
"""
df = pd.read_sql(query, conn)
print(df.to_string(index=False))

master_count = df['total_tickers_in_master'].iloc[0]

# Check what the prediction script queried
print("\n5. PREDICTION SCRIPT DATA SOURCE")
print("-" * 80)
print("The script uses this query to get tickers for prediction:")
print("""
    SELECT DISTINCT h.ticker, h.trading_date, h.close_price, ...
    FROM nse_500_hist_data h
    INNER JOIN nse_500 n ON h.ticker = n.ticker
    WHERE h.trading_date = '2026-04-17'
""")

query = """
SELECT COUNT(DISTINCT h.ticker) as tickers_from_query
FROM nse_500_hist_data h
INNER JOIN nse_500 n ON h.ticker = n.ticker
WHERE h.trading_date = '2026-04-17'
"""
df = pd.read_sql(query, conn)
print(f"\nTickers returned by prediction query: {df['tickers_from_query'].iloc[0]}")

# Check if there are multiple entries per ticker in raw data
print("\n6. CHECKING FOR MULTIPLE ENTRIES PER TICKER IN SOURCE DATA")
print("-" * 80)
query = """
SELECT 
    h.ticker,
    COUNT(*) as row_count
FROM nse_500_hist_data h
INNER JOIN nse_500 n ON h.ticker = n.ticker
WHERE h.trading_date = '2026-04-17'
GROUP BY h.ticker
HAVING COUNT(*) > 1
ORDER BY row_count DESC
"""
df = pd.read_sql(query, conn)
if not df.empty:
    print(df.head(20).to_string(index=False))
    print(f"\n🚨 FOUND THE PROBLEM!")
    print(f"   {len(df)} tickers have multiple rows for same date in nse_500_hist_data!")
    print(f"   This causes duplicate predictions")
else:
    print("No duplicate rows found in source data")

# Summary
print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)
print(f"\nExpected tickers (from April 17 data): {actual_stocks}")
print(f"Actual predictions generated: {total_preds}")
print(f"Unique tickers in predictions: {unique_tickers}")
print(f"Master list size (nse_500): {master_count}")

if total_preds > actual_stocks:
    print(f"\n⚠️ PROBLEM: Generated {total_preds - actual_stocks} extra predictions!")
    print(f"\nPossible causes:")
    print(f"1. Duplicate rows in nse_500_hist_data for same ticker+date")
    print(f"2. Script using all tickers from nse_500 master (not just April 17 trades)")
    print(f"3. Cartesian join issue in prediction query")

conn.close()
