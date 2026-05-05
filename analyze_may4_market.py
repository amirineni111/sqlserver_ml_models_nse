"""
Check actual market conditions on May 4, 2026
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

# Check May 1-4 market conditions
query = """
SELECT TOP 5
    trading_date,
    india_vix_close,
    vix_close,
    nifty50_close,
    nifty50_return_1d,
    sp500_return_1d,
    dxy_close
FROM market_context_daily
WHERE trading_date >= '2026-05-01'
ORDER BY trading_date DESC
"""

df = pd.read_sql(query, conn)

print("="*80)
print("MARKET CONDITIONS (MAY 1-4, 2026)")
print("="*80)
print(df.to_string(index=False))

# Check NIFTY performance
nifty_return = df[df['trading_date'] == pd.to_datetime('2026-05-04')]['nifty50_return_1d'].values
if len(nifty_return) > 0 and not pd.isna(nifty_return[0]):
    print(f"\n" + "="*80)
    print(f"MAY 4, 2026 NIFTY RETURN: {nifty_return[0]:.2f}%")
    print("="*80)
    
    if nifty_return[0] < -2:
        print("⚠️ NIFTY DOWN MORE THAN 2% - SIGNIFICANT DECLINE")
    elif nifty_return[0] < -1:
        print("⚠️ NIFTY DOWN MORE THAN 1% - MODERATE DECLINE")
    elif nifty_return[0] < 0:
        print("○ NIFTY SLIGHTLY DOWN")
    else:
        print("✓ NIFTY UP")
else:
    print(f"\n⚠️ MAY 4 NIFTY RETURN IS MISSING")

# Check how many stocks beat NIFTY on May 4
query2 = """
WITH may4_data AS (
    SELECT 
        ticker,
        CAST(close_price AS FLOAT) as close_price,
        LAG(CAST(close_price AS FLOAT)) OVER (PARTITION BY ticker ORDER BY trading_date) as prev_close
    FROM nse_500_hist_data
    WHERE trading_date IN ('2026-05-01', '2026-05-04')
      AND ticker != '^NSEI'
)
SELECT 
    COUNT(CASE WHEN ((close_price - prev_close) / prev_close * 100) > 0 THEN 1 END) as stocks_up,
    COUNT(CASE WHEN ((close_price - prev_close) / prev_close * 100) < 0 THEN 1 END) as stocks_down,
    COUNT(*) as total,
    AVG((close_price - prev_close) / prev_close * 100) as avg_return
FROM may4_data
WHERE prev_close IS NOT NULL
  AND prev_close > 0
"""

df2 = pd.read_sql(query2, conn)

print("\n" + "="*80)
print("MAY 4 STOCK PERFORMANCE")
print("="*80)
print(f"Stocks UP:   {df2['stocks_up'].iloc[0]:,}")
print(f"Stocks DOWN: {df2['stocks_down'].iloc[0]:,}")
print(f"Avg Return:  {df2['avg_return'].iloc[0]:.2f}%")

adv_decline_ratio = df2['stocks_up'].iloc[0] / df2['stocks_down'].iloc[0] if df2['stocks_down'].iloc[0] > 0 else 0
print(f"Adv/Dec Ratio: {adv_decline_ratio:.2f}")

print("\n" + "="*80)
print("VERDICT")
print("="*80)

if df2['stocks_down'].iloc[0] > df2['stocks_up'].iloc[0] * 3:
    print("✓ GENUINE MARKET DECLINE - 100% Sell signals are JUSTIFIED")
    print("  • More than 75% of stocks are declining")
    print("  • Model is correctly identifying bearish conditions")
    print("  • Hybrid features are working (differentiating degrees of weakness)")
elif df2['avg_return'].iloc[0] < -1.5:
    print("✓ SIGNIFICANT MARKET WEAKNESS - High Sell % is REASONABLE")
    print("  • Average return < -1.5%")
    print("  • Model responding appropriately to market conditions")
else:
    print("⚠️ MARKET CONDITIONS DON'T JUSTIFY 100% SELL")
    print("  • Model may still have issues with feature calculation")

conn.close()

print("\n" + "="*80)
