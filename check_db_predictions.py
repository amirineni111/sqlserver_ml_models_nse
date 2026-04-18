"""Check actual prediction probabilities from database"""
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from database.connection import SQLServerConnection
import pandas as pd

print("=" * 70)
print("DATABASE PREDICTION PROBABILITY CHECK")
print("=" * 70)

db = SQLServerConnection()

# Query recent predictions
query = """
SELECT TOP 2000
    ticker,
    trading_date,
    predicted_signal,
    confidence_percentage,
    sector,
    sector_weightage
FROM ml_nse_trading_predictions
WHERE trading_date >= '2026-04-15'
ORDER BY trading_date DESC, ticker
"""

print("\n[1] Loading recent predictions from database...")
df = db.execute_query(query)
print(f"    Loaded {len(df)} predictions")

# Group by date
for date in df['trading_date'].unique()[:5]:  # Last 5 days
    date_df = df[df['trading_date'] == date]
    
    print(f"\n[{date}]:")
    print(f"    Total: {len(date_df)}")
    
    signal_counts = date_df['predicted_signal'].value_counts()
    for signal, count in signal_counts.items():
        pct = (count / len(date_df)) * 100
        print(f"    {signal:4s}: {count:4d} ({pct:5.1f}%)")
    
    # Check confidence distribution by signal
    for signal in ['Buy', 'Sell', 'Hold']:
        signal_df = date_df[date_df['predicted_signal'] == signal]
        if len(signal_df) > 0:
            avg_conf = signal_df['confidence_percentage'].mean()
            print(f"      {signal} avg confidence: {avg_conf:.1f}%")

# Check if there's a pattern by sector
print("\n" + "=" * 70)
print("SECTOR BREAKDOWN (latest date)")
print("=" * 70)
latest_date = df['trading_date'].max()
latest_df = df[df['trading_date'] == latest_date]

if 'sector' in latest_df.columns and latest_df['sector'].notna().any():
    for sector in latest_df['sector'].unique():
        if pd.isna(sector):
            continue
        sector_df = latest_df[latest_df['sector'] == sector]
        buy_count = (sector_df['predicted_signal'] == 'Buy').sum()
        sell_count = (sector_df['predicted_signal'] == 'Sell').sum()
        total = len(sector_df)
        print(f"  {sector:20s}: Buy={buy_count:3d}, Sell={sell_count:3d}, Total={total:3d}")
else:
    print("  Sector information not available in predictions")

print("\n" + "=" * 70)
print("CONFIDENCE DISTRIBUTION")
print("=" * 70)
print(f"All predictions confidence stats:")
print(f"  Mean: {df['confidence_percentage'].mean():.1f}%")
print(f"  Median: {df['confidence_percentage'].median():.1f}%")
print(f"  Min: {df['confidence_percentage'].min():.1f}%")
print(f"  Max: {df['confidence_percentage'].max():.1f}%")

print("\n" + "=" * 70)
