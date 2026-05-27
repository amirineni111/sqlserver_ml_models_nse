"""Check backfill status for May 25 and May 26."""
import pyodbc, os
from dotenv import load_dotenv
load_dotenv()

conn = pyodbc.connect(
    'DRIVER=' + os.getenv('SQL_DRIVER', 'ODBC Driver 17 for SQL Server') +
    ';SERVER=' + os.getenv('SQL_SERVER') +
    ';DATABASE=' + os.getenv('SQL_DATABASE') +
    ';UID=' + os.getenv('SQL_USERNAME') +
    ';PWD=' + os.getenv('SQL_PASSWORD')
)
cur = conn.cursor()

print("=== Predictions in DB (May 25-26) ===")
cur.execute("""
    SELECT trading_date, COUNT(*) as cnt,
           SUM(CASE WHEN predicted_signal='Buy' THEN 1 ELSE 0 END) as buys
    FROM ml_nse_trading_predictions
    WHERE trading_date IN ('2026-05-25','2026-05-26')
      AND model_name LIKE '%V2%'
    GROUP BY trading_date ORDER BY trading_date
""")
rows = cur.fetchall()
for r in rows:
    print(f"  {str(r[0])}: {r[1]} total, {r[2]} Buy")
if not rows:
    print("  None found")

print("\n=== Source data availability ===")
for d in ['2026-05-25', '2026-05-26']:
    cur.execute(f"SELECT COUNT(DISTINCT ticker) FROM nse_500_hist_data WHERE trading_date='{d}'")
    print(f"  nse_500_hist_data tickers on {d}: {cur.fetchone()[0]}")

conn.close()
