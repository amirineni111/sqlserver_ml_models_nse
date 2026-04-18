import pyodbc
import pandas as pd
from datetime import datetime

# Connect to database
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=192.168.86.28\\MSSQLSERVER01;'
    'DATABASE=stockdata_db;'
    'UID=remote_user;'
    'PWD=YourStrongPassword123!;'
    'TrustServerCertificate=yes'
)

print("="*80)
print("NSE MARKET REALITY CHECK - Why 99% Sell?")
print("="*80)

# Check 1: What was the ACTUAL market movement for stocks on April 17?
print("\n1. ACTUAL STOCK PERFORMANCE (April 17, 2026)")
print("-" * 80)

query = """
WITH DailyChanges AS (
    SELECT 
        ticker,
        trading_date,
        CAST(close_price AS FLOAT) as close_price,
        LAG(CAST(close_price AS FLOAT)) OVER (PARTITION BY ticker ORDER BY trading_date) as prev_close
    FROM nse_500_hist_data
    WHERE trading_date IN ('2026-04-17', '2026-04-16')
)
SELECT 
    COUNT(*) as total_stocks,
    SUM(CASE WHEN close_price > prev_close THEN 1 ELSE 0 END) as stocks_up,
    SUM(CASE WHEN close_price < prev_close THEN 1 ELSE 0 END) as stocks_down,
    SUM(CASE WHEN close_price = prev_close THEN 1 ELSE 0 END) as stocks_flat
FROM DailyChanges
WHERE trading_date = '2026-04-17' AND prev_close IS NOT NULL
"""

df = pd.read_sql(query, conn)
if not df.empty and df['total_stocks'].iloc[0] > 0:
    total = df['total_stocks'].iloc[0]
    up = df['stocks_up'].iloc[0] if df['stocks_up'].iloc[0] else 0
    down = df['stocks_down'].iloc[0] if df['stocks_down'].iloc[0] else 0
    flat = df['stocks_flat'].iloc[0] if df['stocks_flat'].iloc[0] else 0
    print(f"Total Stocks: {total}")
    print(f"Stocks Up (vs prev day): {up} ({up/total*100:.1f}%)")
    print(f"Stocks Down (vs prev day): {down} ({down/total*100:.1f}%)")
    print(f"Stocks Flat: {flat} ({flat/total*100:.1f}%)")
    print(f"\n⚠️ REALITY CHECK: {up/total*100:.1f}% of stocks actually went UP on April 17!")
    print(f"   But model predicted only 0.88% Buy signals")
else:
    print("No data available for April 17, 2026")

# Check 2: NIFTY 50 performance on April 17
print("\n2. NIFTY 50 INDEX (Market Benchmark)")
print("-" * 80)

query = """
SELECT TOP 5
    trading_date,
    nifty50_close,
    nifty50_return_1d,
    india_vix_close
FROM market_context_daily
WHERE trading_date <= '2026-04-17'
ORDER BY trading_date DESC
"""

df = pd.read_sql(query, conn)
if not df.empty:
    print(df.to_string(index=False))
    
    last_date = df['trading_date'].iloc[0]
    if pd.Timestamp(last_date) < pd.Timestamp('2026-04-17'):
        print(f"\n⚠️ WARNING: Latest market data is from {pd.Timestamp(last_date).strftime('%Y-%m-%d')}")
        print(f"   Gap: {(pd.Timestamp('2026-04-17') - pd.Timestamp(last_date)).days} days")
        print("   Model is using STALE market indicators!")
else:
    print("No market context data available")

# Check 3: Training period vs prediction date
print("\n3. TRAINING PERIOD vs PREDICTION DATE")
print("-" * 80)

print("Training Period: 2024-06-01 to 2026-02-28")
print("Prediction Date: 2026-04-17")
print(f"Gap: {(pd.to_datetime('2026-04-17') - pd.to_datetime('2026-02-28')).days} days")
print("\n⚠️ Model never saw data from March-April 2026!")
print("   If market deteriorated in Mar-Apr, model has no context")

# Check 4: What do the actual 5-day forward returns look like for recent data?
print("\n4. RECENT 5-DAY FORWARD RETURNS (Actual Future Performance)")
print("-" * 80)

query = """
WITH ForwardReturns AS (
    SELECT 
        h1.trading_date,
        h1.ticker,
        CAST(h1.close_price AS FLOAT) as close_price,
        CAST(h2.close_price AS FLOAT) as future_close
    FROM nse_500_hist_data h1
    LEFT JOIN nse_500_hist_data h2 
        ON h1.ticker = h2.ticker 
        AND h2.trading_date = DATEADD(day, 5, h1.trading_date)
    WHERE h1.trading_date BETWEEN '2026-03-01' AND '2026-04-12'
)
SELECT 
    trading_date,
    COUNT(*) as total_stocks,
    AVG(CASE 
        WHEN future_close > close_price THEN 1.0
        ELSE 0.0
    END) * 100 as pct_up_after_5d
FROM ForwardReturns
WHERE future_close IS NOT NULL
GROUP BY trading_date
ORDER BY trading_date DESC
"""

df = pd.read_sql(query, conn)
if not df.empty:
    print(df.head(15).to_string(index=False))
    
    avg_up_pct = df['pct_up_after_5d'].mean()
    print(f"\nAverage % stocks going up after 5 days (Mar-Apr 2026): {avg_up_pct:.1f}%")
    
    if avg_up_pct < 30:
        print("✅ Market was GENUINELY BEARISH in Mar-Apr 2026")
        print(f"   Only {avg_up_pct:.1f}% of stocks went up - model prediction makes sense!")
    elif avg_up_pct > 40:
        print(f"⚠️ Market had {avg_up_pct:.1f}% stocks going up after 5 days")
        print("   Model's 0.88% Buy signals seems WAY TOO CONSERVATIVE")
        print("   This suggests the model is BROKEN, not reflecting reality")
else:
    print("No forward return data available")

# Check 5: Compare V1 vs V2 predictions
print("\n5. V1 vs V2 PREDICTION COMPARISON")
print("-" * 80)

query = """
SELECT 
    model_name,
    predicted_signal,
    COUNT(*) as count,
    AVG(confidence) as avg_confidence
FROM ml_nse_trading_predictions
WHERE trading_date = '2026-04-17'
GROUP BY model_name, predicted_signal
ORDER BY model_name, predicted_signal
"""

df = pd.read_sql(query, conn)
if not df.empty:
    print(df.to_string(index=False))
else:
    print("No predictions found for April 17, 2026")

conn.close()

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
The 99% Sell bias could be due to:

1. STALE MARKET DATA (most likely)
   - Market context last updated: Feb 20, 2026
   - Prediction date: April 17, 2026 (58-day gap)
   - Model using outdated VIX, Nifty returns, etc.

2. OUT-OF-SAMPLE PERIOD
   - Training ended: Feb 28, 2026
   - Prediction date: April 17, 2026 (48 days later)
   - Model never saw March-April 2026 data

3. GENUINE MARKET CRASH?
   - If NSE actually crashed in March-April 2026
   - Model would correctly predict mostly Sell signals

4. FEATURE DOMINANCE
   - 8 of 20 features are market-wide (Nifty, VIX, etc.)
   - If these show bearish, model predicts Sell for ALL stocks

RECOMMENDATIONS:
1. Update market_context_daily table with April 2026 data
2. Extend training period to include March-April 2026
3. Check if Nifty 50 actually crashed in March-April
4. Consider reducing market-wide feature weight
""")
