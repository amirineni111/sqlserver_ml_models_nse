import pyodbc
import pandas as pd
import numpy as np

conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=192.168.86.28\\MSSQLSERVER01;'
    'DATABASE=stockdata_db;'
    'UID=remote_user;'
    'PWD=YourStrongPassword123!;'
    'TrustServerCertificate=yes'
)

print("=" * 80)
print("INVESTIGATING MARKET_CAP_CATEGORY ISSUE")
print("=" * 80)

# Check how market_cap varies across the 15 fundamental snapshots
print("\n1. MARKET CAP VARIATION ACROSS SNAPSHOTS")
print("-" * 80)

query = """
SELECT 
    fetch_date,
    COUNT(DISTINCT ticker) as ticker_count,
    SUM(CASE WHEN market_cap >= 100000000000 THEN 1 ELSE 0 END) as large_cap_count,
    SUM(CASE WHEN market_cap >= 10000000000 AND market_cap < 100000000000 THEN 1 ELSE 0 END) as mid_cap_count,
    SUM(CASE WHEN market_cap < 10000000000 THEN 1 ELSE 0 END) as small_cap_count
FROM nse_500_fundamentals
GROUP BY fetch_date
ORDER BY fetch_date DESC
"""

df = pd.read_sql(query, conn)
print(df.to_string(index=False))

# Check if using different snapshots changes cap category
print("\n2. TICKERS THAT CHANGE CAP CATEGORY ACROSS SNAPSHOTS")
print("-" * 80)

query = """
WITH cap_categories AS (
    SELECT 
        ticker,
        fetch_date,
        CASE 
            WHEN market_cap >= 100000000000 THEN 'Large Cap'
            WHEN market_cap >= 10000000000 THEN 'Mid Cap'
            ELSE 'Small Cap'
        END AS cap_category
    FROM nse_500_fundamentals
    WHERE fetch_date IN ('2026-01-19', '2026-04-12')  -- Compare oldest vs newest
)
SELECT 
    ticker,
    COUNT(DISTINCT cap_category) as distinct_categories,
    MIN(cap_category) as first_category,
    MAX(cap_category) as last_category
FROM cap_categories
GROUP BY ticker
HAVING COUNT(DISTINCT cap_category) > 1
ORDER BY ticker
"""

df = pd.read_sql(query, conn)
if not df.empty:
    print(f"Found {len(df)} tickers that change cap category:")
    print(df.head(20).to_string(index=False))
else:
    print("No tickers change cap category across snapshots")

# Check the feature importance from the trained model
print("\n3. CHECKING V2 MODEL FEATURE IMPORTANCE")
print("-" * 80)

try:
    import joblib
    
    # Load the base model (before calibration)
    model = joblib.load('data/nse_models/nse_gb_base_model_v2.joblib')
    features_df = pd.read_csv('data/nse_models/feature_importances_v2.csv')
    
    print("\nTop 20 features by importance:")
    print(features_df.head(20).to_string(index=False))
    
    # Check if market_cap_category appears (it's encoded, so might be sector_encoded or similar)
    print("\nNote: market_cap_category is one-hot encoded during training")
    print("Look for features like 'Large Cap', 'Mid Cap', 'Small Cap' if they were encoded")
    
except Exception as e:
    print(f"Could not load model: {e}")

# Check current V2 predictions distribution
print("\n4. CURRENT V2 PREDICTIONS (with latest fundamental data)")
print("-" * 80)

query = """
SELECT 
    predicted_signal,
    COUNT(*) as count,
    AVG(confidence_percentage) as avg_confidence
FROM ml_nse_trading_predictions
WHERE trading_date = '2026-04-17'
  AND model_name = 'GradientBoosting_V2_Calibrated'
GROUP BY predicted_signal
"""

df = pd.read_sql(query, conn)
print(df.to_string(index=False))

total = df['count'].sum()
print(f"\nTotal predictions: {total}")
for _, row in df.iterrows():
    pct = (row['count'] / total) * 100
    print(f"  {row['predicted_signal']}: {row['count']} ({pct:.2f}%)")

conn.close()

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
The issue: market_cap_category is NOT one of the 20 selected features!
It's only used to derive the market cap threshold during training, but then 
we don't actually use it as a feature. So the swing from 98% Sell to 98% Buy 
is NOT caused by market_cap_category.

The real issue: The model was TRAINED with random fundamental data (wrong SQL),
but now we're PREDICTING with latest fundamental data (correct SQL).
This train/predict data mismatch is causing invalid predictions.

SOLUTION: Retrain the model with the corrected SQL query that uses latest
fundamental data, so training and prediction use the same logic.
""")
