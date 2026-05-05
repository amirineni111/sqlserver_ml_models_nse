"""
Find last actual trading date and run predictions
"""
import subprocess
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

# Find last trading date with >2000 tickers
cursor.execute("""
    SELECT TOP 1 trading_date, COUNT(DISTINCT ticker) as tickers
    FROM nse_500_hist_data
    WHERE trading_date < CAST('2026-05-01' AS DATE)
    GROUP BY trading_date
    HAVING COUNT(DISTINCT ticker) > 2000
    ORDER BY trading_date DESC
""")

result = cursor.fetchone()
last_date = result[0]
ticker_count = result[1]

print("="*80)
print("LAST ACTUAL TRADING DATE")
print("="*80)
print(f"Date: {last_date}")
print(f"Day: {last_date.strftime('%A')}")
print(f"Tickers: {ticker_count:,}")

conn.close()

# Run predictions for this date
date_str = last_date.strftime('%Y-%m-%d')
print(f"\n[INFO] Running predictions for {date_str}...")
print("="*80)

# Modify the config to use this date
config_code = f"""
import os
os.environ['NSE_PREDICTION_DATE'] = '{date_str}'
"""

with open('temp_config.py', 'w') as f:
    f.write(config_code)

print(f"\n✓ Configured prediction date to {date_str}")
print(f"  Now run: python predict_nse_signals_v2.py")
print("\n" + "="*80)
