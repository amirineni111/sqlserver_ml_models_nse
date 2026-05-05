"""
Sample prediction data to verify hybrid features are working
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

# Check if May 4 is actually a trading day
query = """
SELECT COUNT(DISTINCT ticker) as tickers,
       COUNT(*) as records
FROM nse_500_hist_data
WHERE trading_date = '2026-05-04'
"""

df = pd.read_sql(query, conn)

print("="*80)
print("MAY 4, 2026 DATA AVAILABILITY")
print("="*80)
print(df.to_string(index=False))

if df['records'].iloc[0] == 0:
    print("\n⚠ May 4 is NOT a trading day (no data in nse_500_hist_data)")
    print("\nMay 4, 2026 is a SUNDAY - markets are closed!")
    print("\nThe prediction script is using the latest AVAILABLE date.")
    print("Check what date predictions are actually for:")
    
    query2 = """
    SELECT TOP 1 trading_date, COUNT(*) as tickers
    FROM ml_nse_trading_predictions
    WHERE model_name LIKE '%V2%'
    GROUP BY trading_date
    ORDER BY trading_date DESC
    """
    
    df2 = pd.read_sql(query2, conn)
    print(df2.to_string(index=False))

# Check latest actual trading date
query3 = """
SELECT TOP 5
    trading_date,
    COUNT(DISTINCT ticker) as tickers
FROM nse_500_hist_data
GROUP BY trading_date
ORDER BY trading_date DESC
"""

df3 = pd.read_sql(query3, conn)

print("\n" + "="*80)
print("LATEST ACTUAL TRADING DATES")
print("="*80)
print(df3.to_string(index=False))

conn.close()

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)
print("If May 4 is a weekend/holiday:")
print("  • Predictions would use forward-filled data")
print("  • All stocks get SAME market context → no differentiation")
print("  • Hybrid features become useless (all zeros)")
print("\nSOLUTION:")
print("  • Wait for next trading day (May 5?)")
print("  • Or manually specify prediction date: --date 2026-05-01")
print("\n" + "="*80)
