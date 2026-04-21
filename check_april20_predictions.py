"""Quick check for April 20th predictions"""
import pyodbc

conn_str = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=192.168.86.28\\MSSQLSERVER01;"
    "DATABASE=stockdata_db;"
    "UID=remote_user;"
    "PWD=YourStrongPassword123!;"
    "TrustServerCertificate=yes"
)

conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# Check predictions by date
query = """
SELECT 
    trading_date, 
    COUNT(*) as total_predictions,
    SUM(CASE WHEN predicted_signal='Buy' THEN 1 ELSE 0 END) as buy_signals,
    SUM(CASE WHEN predicted_signal='Sell' THEN 1 ELSE 0 END) as sell_signals,
    SUM(CASE WHEN high_confidence=1 THEN 1 ELSE 0 END) as high_confidence_count
FROM ml_nse_trading_predictions
WHERE trading_date >= '2026-04-15'
GROUP BY trading_date
ORDER BY trading_date DESC
"""

cursor.execute(query)
rows = cursor.fetchall()

print("\n" + "="*80)
print("NSE PREDICTIONS BY DATE")
print("="*80)
print(f"{'Date':<12} | {'Total':>6} | {'Buy':>5} | {'Sell':>5} | {'High Conf':>10}")
print("-"*80)

for row in rows:
    print(f"{str(row[0]):<12} | {row[1]:6d} | {row[2]:5d} | {row[3]:5d} | {row[4]:10d}")

# Check specific April 20 predictions
print("\n" + "="*80)
print("APRIL 20th SAMPLE PREDICTIONS (High Confidence)")
print("="*80)

query2 = """
SELECT TOP 10
    ticker,
    company,
    predicted_signal,
    confidence_percentage,
    CAST(close_price AS DECIMAL(10,2)) as close_price
FROM ml_nse_trading_predictions
WHERE trading_date = '2026-04-20'
    AND high_confidence = 1
ORDER BY confidence_percentage DESC
"""

cursor.execute(query2)
rows = cursor.fetchall()

print(f"{'Ticker':<15} | {'Signal':<6} | {'Conf %':>7} | {'Price':>10}")
print("-"*80)
for row in rows:
    print(f"{row[0]:<15} | {row[2]:<6} | {row[3]:6.1f}% | {row[4]:>10.2f}")

conn.close()
print("\n" + "="*80)
