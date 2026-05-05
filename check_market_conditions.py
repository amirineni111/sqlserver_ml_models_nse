"""
Check current market conditions to understand if predictions are justified
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

# Get recent market context
query = """
SELECT TOP 10
    trading_date,
    india_vix_close,
    vix_close,
    nifty50_return_1d,
    sp500_return_1d,
    dxy_close
FROM market_context_daily
ORDER BY trading_date DESC
"""

df = pd.read_sql(query, conn)
conn.close()

print("="*80)
print("RECENT MARKET CONDITIONS")
print("="*80)
print(df.to_string(index=False))

# Calculate averages
print("\n" + "="*80)
print("5-DAY AVERAGES")
print("="*80)
print(f"India VIX:     {df['india_vix_close'].head(5).mean():.2f}")
print(f"VIX:           {df['vix_close'].head(5).mean():.2f}")
print(f"NIFTY Return:  {df['nifty50_return_1d'].head(5).mean():.2%}")
print(f"S&P 500 Return: {df['sp500_return_1d'].head(5).mean():.2%}")
print(f"DXY:           {df['dxy_close'].head(5).mean():.2f}")

# Interpret
print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

india_vix = df['india_vix_close'].iloc[0]
nifty_return = df['nifty50_return_1d'].head(5).mean()

if india_vix > 20:
    print(f"⚠ India VIX {india_vix:.2f} is ELEVATED (>20) - market fear")
else:
    print(f"✓ India VIX {india_vix:.2f} is normal (<20)")

if nifty_return < -0.01:
    print(f"⚠ NIFTY 5-day avg return {nifty_return:.2%} is NEGATIVE - bearish trend")
elif nifty_return > 0.01:
    print(f"✓ NIFTY 5-day avg return {nifty_return:.2%} is POSITIVE - bullish trend")
else:
    print(f"○ NIFTY 5-day avg return {nifty_return:.2%} is FLAT - neutral")

print("\n" + "="*80)
print("VERDICT")
print("="*80)
if india_vix > 20 and nifty_return < -0.01:
    print("Market conditions are GENUINELY BEARISH")
    print("→ High Sell % could be justified by market conditions")
    print("→ BUT model should still find relative winners (Buy signals)")
    print("→ If Buy count < 50 (2.5%), model is OVER-REACTING to market")
else:
    print("Market conditions are NOT extreme bearish")
    print("→ 2061 Sell / 1 Buy ratio is NOT justified")
    print("→ Model has SEVERE BIAS - needs immediate retraining")

print("\n" + "="*80)
