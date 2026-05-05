"""
Quick check of May 4, 2026 predictions
"""

import pyodbc
import os
from dotenv import load_dotenv

load_dotenv()

# Connect to database
conn_str = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={os.getenv('SQL_SERVER')};"
    f"DATABASE={os.getenv('SQL_DATABASE')};"
    f"UID={os.getenv('SQL_USERNAME')};"
    f"PWD={os.getenv('SQL_PASSWORD')}"
)

conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# Check recent predictions
query = """
SELECT 
    trading_date,
    predicted_signal,
    COUNT(*) as count,
    AVG(confidence_percentage) as avg_confidence,
    AVG(buy_probability) as avg_buy_prob,
    AVG(sell_probability) as avg_sell_prob
FROM ml_nse_trading_predictions
WHERE trading_date >= '2026-05-01'
GROUP BY trading_date, predicted_signal
ORDER BY trading_date DESC, predicted_signal
"""

cursor.execute(query)
results = cursor.fetchall()

print("="*80)
print("RECENT NSE PREDICTIONS")
print("="*80)
for row in results:
    date, signal, count, avg_conf, avg_buy, avg_sell = row
    print(f"\n{date} - {signal}:")
    print(f"  Count: {count:,}")
    print(f"  Avg Confidence: {avg_conf:.1f}%")
    print(f"  Avg Buy Prob: {avg_buy:.2%}")
    print(f"  Avg Sell Prob: {avg_sell:.2%}")

# Check specific tickers with Buy signal
query2 = """
SELECT TOP 10 
    ticker, 
    predicted_signal, 
    confidence_percentage,
    buy_probability,
    sell_probability,
    rsi,
    sector
FROM ml_nse_trading_predictions
WHERE trading_date >= '2026-05-01'
  AND predicted_signal = 'Buy'
ORDER BY confidence_percentage DESC
"""

cursor.execute(query2)
results2 = cursor.fetchall()

print("\n" + "="*80)
print("TOP BUY SIGNALS")
print("="*80)
for row in results2:
    ticker, signal, conf, buy_prob, sell_prob, rsi, sector = row
    print(f"{ticker:12s} | {signal:4s} | Conf: {conf:5.1f}% | "
          f"Buy: {buy_prob:.2%} | Sell: {sell_prob:.2%} | "
          f"RSI: {rsi:.1f} | {sector}")

conn.close()
print("\n" + "="*80)
