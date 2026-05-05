"""
Sample prediction features to see what the model is actually seeing
"""
import sys
import os
import numpy as np
import pandas as pd
import pyodbc
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

# Just check the data that would be fed to the model
conn_str = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={os.getenv('SQL_SERVER')};"
    f"DATABASE={os.getenv('SQL_DATABASE')};"
    f"UID={os.getenv('SQL_USERNAME')};"
    f"PWD={os.getenv('SQL_PASSWORD')}"
)

conn = pyodbc.connect(conn_str)

# Get sample predictions to see feature values
query = """
SELECT TOP 10
    ticker,
    predicted_signal,
    confidence_percentage,
    buy_probability,
    sell_probability,
    sector
FROM ml_nse_trading_predictions
WHERE trading_date = '2026-05-04'
  AND model_name LIKE '%V2%'
ORDER BY buy_probability DESC
"""

df = pd.read_sql(query, conn)

print("="*80)
print("TOP 10 TICKERS BY BUY PROBABILITY")
print("="*80)
print(df.to_string(index=False))

# Check the distribution
query2 = """
SELECT 
    MIN(buy_probability) as min_buy,
    AVG(buy_probability) as avg_buy,
    MAX(buy_probability) as max_buy,
    MIN(sell_probability) as min_sell,
    AVG(sell_probability) as avg_sell,
    MAX(sell_probability) as max_sell
FROM ml_nse_trading_predictions
WHERE trading_date = '2026-05-04'
  AND model_name LIKE '%V2%'
"""

df2 = pd.read_sql(query2, conn)

print("\n" + "="*80)
print("PROBABILITY DISTRIBUTION")
print("="*80)
print(df2.to_string(index=False))

# Check how many are close to threshold
query3 = """
SELECT 
    COUNT(CASE WHEN buy_probability >= 0.49 AND buy_probability <= 0.51 THEN 1 END) as near_threshold,
    COUNT(CASE WHEN buy_probability < 0.30 THEN 1 END) as strong_sell,
    COUNT(CASE WHEN buy_probability > 0.70 THEN 1 END) as strong_buy,
    COUNT(*) as total
FROM ml_nse_trading_predictions
WHERE trading_date = '2026-05-04'
  AND model_name LIKE '%V2%'
"""

df3 = pd.read_sql(query3, conn)

print("\n" + "="*80)
print("PROBABILITY RANGES")
print("="*80)
print(f"Near Threshold (49-51%): {df3['near_threshold'].iloc[0]:,}")
print(f"Strong Sell (<30% Buy): {df3['strong_sell'].iloc[0]:,}")
print(f"Strong Buy (>70% Buy): {df3['strong_buy'].iloc[0]:,}")
print(f"Total: {df3['total'].iloc[0]:,}")

conn.close()

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)
if df3['strong_sell'].iloc[0] > 2000:
    print("✗ Almost ALL stocks are STRONG SELL (<30% Buy)")
    print("  This means the features for May 4 are EXTREMELY bearish")
    print("  Possible causes:")
    print("    1. Genuine market crash conditions on May 4")
    print("    2. NIFTY return data for May 4 is missing/wrong")
    print("    3. Hybrid features are not calculating correctly")
    print("    4. Model calibration is broken")
else:
    print("✓ Probability distribution looks reasonable")

print("\n" + "="*80)
