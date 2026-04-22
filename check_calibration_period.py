"""Check calibration set distribution for NSE V2 model"""
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from database.connection import SQLServerConnection
from datetime import datetime

conn = SQLServerConnection()

# Model trained on: 2024-06-01 to 2026-04-15
# 60/20/20 split means:
# Training: 2024-06-01 to early Dec 2025 (60%)
# Calibration: early Dec 2025 to early Feb 2026 (20%) 
# Test: early Feb 2026 to 2026-04-15 (20%)

print("\n" + "="*80)
print("NSE V2 CALIBRATION SET DISTRIBUTION ANALYSIS")
print("="*80)

# Query calibration period (approximately Dec 2025 - Feb 2026)
query = """
SELECT 
    COUNT(*) as total_samples,
    SUM(CASE WHEN future_return_5d > 0 THEN 1 ELSE 0 END) as up_count,
    SUM(CASE WHEN future_return_5d <= 0 THEN 1 ELSE 0 END) as down_count,
    CAST(SUM(CASE WHEN future_return_5d > 0 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100 AS DECIMAL(5,2)) as up_pct,
    CAST(SUM(CASE WHEN future_return_5d <= 0 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100 AS DECIMAL(5,2)) as down_pct,
    AVG(CAST(future_return_5d AS FLOAT)) as avg_return
FROM (
    SELECT 
        h.ticker,
        h.trading_date,
        CAST(h.close_price AS FLOAT) as close_price,
        LEAD(CAST(h.close_price AS FLOAT), 5) OVER (PARTITION BY h.ticker ORDER BY h.trading_date) as future_close_5d,
        (LEAD(CAST(h.close_price AS FLOAT), 5) OVER (PARTITION BY h.ticker ORDER BY h.trading_date) - CAST(h.close_price AS FLOAT)) / CAST(h.close_price AS FLOAT) * 100 as future_return_5d
    FROM nse_500_hist_data h
    WHERE h.trading_date >= '2025-12-01'
      AND h.trading_date <= '2026-02-28'
      AND CAST(h.close_price AS FLOAT) > 0
) t
WHERE future_return_5d IS NOT NULL
"""

print(f"\nCalibration Period (Approx): Dec 2025 - Feb 2026")
print(f"This is where isotonic calibration learned probability mappings")
print("-" * 80)

result = conn.execute_query(query)

if result is not None and not result.empty:
    row = result.iloc[0]
    print(f"\nCalibration Set Distribution (5-day targets):")
    print(f"  Total Samples: {row['total_samples']:,}")
    print(f"  UP (future_return > 0): {row['up_count']:,} ({row['up_pct']:.2f}%)")
    print(f"  DOWN (future_return <= 0): {row['down_count']:,} ({row['down_pct']:.2f}%)")
    print(f"  Average 5-day Return: {row['avg_return']:.4f}%")
    print(f"\n  Class Imbalance: {abs(row['up_pct'] - 50):.2f}% deviation from 50/50")
    
    if row['down_pct'] > 55:
        print(f"\n  ⚠️  [ROOT CAUSE FOUND]")
        print(f"  Calibration period was BEARISH ({row['down_pct']:.1f}% DOWN)")
        print(f"  Isotonic calibration learned to map probabilities to bearish outcomes")
        print(f"  Even with balanced training, calibration on imbalanced period causes bias!")
        print(f"\n  This is why predictions are 99.6% Sell despite balanced training data.")
    elif row['up_pct'] > 55:
        print(f"\n  Calibration period was bullish - not the root cause.")

# Also check training period for comparison
train_query = """
SELECT 
    COUNT(*) as total_samples,
    SUM(CASE WHEN future_return_5d > 0 THEN 1 ELSE 0 END) as up_count,
    SUM(CASE WHEN future_return_5d <= 0 THEN 1 ELSE 0 END) as down_count,
    CAST(SUM(CASE WHEN future_return_5d > 0 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100 AS DECIMAL(5,2)) as up_pct
FROM (
    SELECT 
        h.ticker,
        h.trading_date,
        LEAD(CAST(h.close_price AS FLOAT), 5) OVER (PARTITION BY h.ticker ORDER BY h.trading_date) as future_close_5d,
        (LEAD(CAST(h.close_price AS FLOAT), 5) OVER (PARTITION BY h.ticker ORDER BY h.trading_date) - CAST(h.close_price AS FLOAT)) / CAST(h.close_price AS FLOAT) * 100 as future_return_5d
    FROM nse_500_hist_data h
    WHERE h.trading_date >= '2024-06-01'
      AND h.trading_date < '2025-12-01'
      AND CAST(h.close_price AS FLOAT) > 0
) t
WHERE future_return_5d IS NOT NULL
"""

print(f"\n{'='*80}")
print(f"TRAINING PERIOD COMPARISON (Jun 2024 - Nov 2025)")
print(f"{'='*80}")

train_result = conn.execute_query(train_query)

if train_result is not None and not train_result.empty:
    row = train_result.iloc[0]
    print(f"  Total Samples: {row['total_samples']:,}")
    print(f"  UP: {row['up_count']:,} ({row['up_pct']:.2f}%)")
    print(f"  DOWN: {row['down_count']:,} ({100-row['up_pct']:.2f}%)")

print("\n" + "="*80)
print("SOLUTION:")
print("="*80)
print("Use STRATIFIED SPLIT for calibration set, not time-series split")
print("This ensures calibration set has balanced distribution like NASDAQ model")
print("="*80)
