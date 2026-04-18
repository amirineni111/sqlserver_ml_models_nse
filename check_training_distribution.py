"""Check NSE training data distribution from database"""
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from database.connection import SQLServerConnection
import pandas as pd
import numpy as np

print("=" * 70)
print("NSE TRAINING DATA DISTRIBUTION CHECK")
print("=" * 70)

db = SQLServerConnection()

# Query to match retrain_nse_model.py logic (last 730 days)
query = """
SELECT 
    h.ticker,
    h.trading_date,
    CAST(h.close_price AS FLOAT) as close_price
FROM dbo.nse_500_hist_data h
WHERE h.trading_date >= DATEADD(day, -730, CAST(GETDATE() AS DATE))
  AND h.trading_date < CAST(GETDATE() AS DATE)
ORDER BY h.ticker, h.trading_date
"""

print("\n[1] Loading historical data (last 730 days)...")
df = db.execute_query(query)
print(f"    Loaded {len(df):,} records")

# Calculate 5-day forward returns (matching training logic)
print("\n[2] Calculating 5-day forward returns...")
df = df.sort_values(['ticker', 'trading_date'])
df['next_5d_close'] = df.groupby('ticker')['close_price'].shift(-5)
df['next_5d_return'] = ((df['next_5d_close'] - df['close_price']) / df['close_price']) * 100
df['direction_5d'] = np.where(df['next_5d_return'] > 0, 'Up', 'Down')

# Remove rows without 5-day target
df_valid = df[df['next_5d_close'].notna()].copy()
print(f"    Valid samples (with 5-day target): {len(df_valid):,}")

# Distribution
print("\n[3] 5-DAY DIRECTION DISTRIBUTION (Training Data):")
print("    " + "=" * 60)
direction_counts = df_valid['direction_5d'].value_counts()
total = len(df_valid)
for direction, count in direction_counts.items():
    pct = (count / total) * 100
    print(f"    {direction:6s}: {count:8,} ({pct:5.1f}%)")

# Check imbalance
up_pct = direction_counts.get('Up', 0) / total
down_pct = direction_counts.get('Down', 0) / total
imbalance = abs(up_pct - down_pct)

print("    " + "=" * 60)
print(f"    IMBALANCE: {imbalance:.1%}")

if imbalance > 0.20:
    print("\n    ⚠️  WARNING: TRAINING DATA IS HEAVILY SKEWED!")
    print(f"    ⚠️  This explains why the model predicts {direction_counts.idxmax()} for most stocks")
else:
    print("\n    ✅ Training data is reasonably balanced")

# Date range
print(f"\n[4] Training period:")
print(f"    Start: {df_valid['trading_date'].min()}")
print(f"    End:   {df_valid['trading_date'].max()}")
print(f"    Days:  {(df_valid['trading_date'].max() - df_valid['trading_date'].min()).days}")

# Recent period check (last 90 days)
recent_cutoff = df_valid['trading_date'].max() - pd.Timedelta(days=90)
df_recent = df_valid[df_valid['trading_date'] >= recent_cutoff]
print(f"\n[5] RECENT 90 DAYS DISTRIBUTION:")
print("    " + "=" * 60)
recent_counts = df_recent['direction_5d'].value_counts()
recent_total = len(df_recent)
for direction, count in recent_counts.items():
    pct = (count / recent_total) * 100
    print(f"    {direction:6s}: {count:8,} ({pct:5.1f}%)")
print("    " + "=" * 60)

print("\n" + "=" * 70)
print("RECOMMENDATION:")
print("=" * 70)
if imbalance > 0.30:
    print("CRITICAL: Training data has >30% imbalance.")
    print("ACTION REQUIRED:")
    print("  1. Filter training period to exclude extreme bear/bull markets")
    print("  2. Or extend training window to 3+ years for more balance")
    print("  3. Verify class balancing (compute_sample_weight) is applied")
elif imbalance > 0.20:
    print("WARNING: Training data has 20-30% imbalance.")
    print("Class balancing should help, but consider filtering training period.")
else:
    print("Training data is balanced. Issue is likely in prediction or model code.")
print("=" * 70)
