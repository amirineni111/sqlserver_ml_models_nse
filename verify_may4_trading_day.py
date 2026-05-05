"""
Check if May 4, 2026 has actual trading data or if it's a non-trading day
"""
import pyodbc
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

conn_str = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={os.getenv('SQL_SERVER')};"
    f"DATABASE={os.getenv('SQL_DATABASE')};"
    f"UID={os.getenv('SQL_USERNAME')};"
    f"PWD={os.getenv('SQL_PASSWORD')}"
)

conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# Check May 4, 2026
print("="*80)
print("MAY 4, 2026 TRADING DATA CHECK")
print("="*80)
print(f"May 4, 2026 is a {datetime(2026, 5, 4).strftime('%A')}")

# Check if data exists for May 4
cursor.execute("""
    SELECT COUNT(DISTINCT ticker) as ticker_count,
           COUNT(*) as row_count
    FROM nse_500_hist_data
    WHERE trading_date = '2026-05-04'
""")
result = cursor.fetchone()
print(f"\nMay 4 data in nse_500_hist_data:")
print(f"  Tickers: {result[0]}")
print(f"  Rows: {result[1]}")

if result[0] == 0:
    print("\n⚠️ MAY 4, 2026 HAS NO TRADING DATA!")
    print("   This is likely a non-trading day (holiday/weekend)")
    
    # Find last actual trading date
    cursor.execute("""
        SELECT TOP 1 trading_date, COUNT(DISTINCT ticker) as tickers
        FROM nse_500_hist_data
        WHERE trading_date < '2026-05-04'
        GROUP BY trading_date
        ORDER BY trading_date DESC
    """)
    last_date = cursor.fetchone()
    print(f"\n   Last trading date: {last_date[0]} ({last_date[1]} tickers)")
    
    print("\n" + "="*80)
    print("ROOT CAUSE")
    print("="*80)
    print("When there's no new trading data:")
    print("  • Script uses forward-filled data from previous day")
    print("  • All stocks have IDENTICAL market context")
    print("  • stock_return_vs_nifty = 0 (no new NIFTY return)")
    print("  • Hybrid features become INEFFECTIVE (all zeros)")
    print("  • Model falls back to absolute market context only")
    print("  • Result: All stocks get same signal")
    
    print("\n" + "="*80)
    print("SOLUTION")
    print("="*80)
    print(f"Run predictions for the LAST TRADING DATE:")
    print(f"  python predict_nse_signals_v2.py --date {last_date[0].strftime('%Y-%m-%d')}")
    print("\nOr wait for next trading day (May 5? May 6?)")

else:
    print(f"\n✓ May 4 has trading data for {result[0]} tickers")
    print("  Issue might be elsewhere - check feature calculation")

conn.close()
print("\n" + "="*80)
