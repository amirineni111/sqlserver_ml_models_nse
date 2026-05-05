"""
Check market_context_daily data quality
Identify missing values that could break hybrid features
"""
import pyodbc
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

conn_str = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={os.getenv('SQL_SERVER')};"
    f"DATABASE={os.getenv('SQL_DATABASE')};"
    f"UID={os.getenv('SQL_USERNAME')};"
    f"PWD={os.getenv('SQL_PASSWORD')}"
)

conn = pyodbc.connect(conn_str)

# Check data completeness
query = """
SELECT 
    COUNT(*) as total_rows,
    COUNT(nifty50_return_1d) as has_nifty_return,
    COUNT(india_vix_close) as has_india_vix,
    COUNT(vix_close) as has_vix,
    MIN(trading_date) as earliest_date,
    MAX(trading_date) as latest_date
FROM market_context_daily
WHERE trading_date >= '2024-06-01'
"""

df = pd.read_sql(query, conn)

print("="*80)
print("MARKET CONTEXT DATA QUALITY")
print("="*80)
print(df.to_string(index=False))

# Check recent missing values
query2 = """
SELECT TOP 30
    trading_date,
    CASE WHEN nifty50_return_1d IS NULL THEN 'MISSING' ELSE 'OK' END as nifty_return,
    CASE WHEN india_vix_close IS NULL THEN 'MISSING' ELSE 'OK' END as india_vix,
    CASE WHEN vix_close IS NULL THEN 'MISSING' ELSE 'OK' END as vix,
    CASE WHEN nifty50_close IS NULL THEN 'MISSING' ELSE 'OK' END as nifty_close
FROM market_context_daily
WHERE trading_date >= '2026-04-01'
ORDER BY trading_date DESC
"""

df2 = pd.read_sql(query2, conn)

print("\n" + "="*80)
print("RECENT DATA COMPLETENESS (April-May 2026)")
print("="*80)
print(df2.to_string(index=False))

# Check for consecutive missing periods
missing_nifty = df2[df2['nifty_return'] == 'MISSING']
if len(missing_nifty) > 0:
    print("\n" + "="*80)
    print("⚠ WARNING: MISSING NIFTY RETURN DATA")
    print("="*80)
    print(f"Missing dates: {missing_nifty['trading_date'].tolist()}")
    print("\nIMPACT:")
    print("  • stock_return_vs_nifty will be ZERO (not calculated)")
    print("  • beta_adjusted_return will use default beta=1.0")
    print("  • relative_strength_20d will be ZERO")
    print("  • Hybrid features will be INEFFECTIVE")
    print("\nFIX REQUIRED:")
    print("  1. Update market_context_daily with NIFTY data")
    print("  2. Or modify training to forward-fill missing NIFTY returns")

conn.close()

print("\n" + "="*80)
