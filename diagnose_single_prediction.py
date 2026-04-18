import pandas as pd
import joblib
import pyodbc
import numpy as np

# Load model artifacts
model = joblib.load('data/nse_models/nse_gb_model_v2.joblib')
scaler = joblib.load('data/nse_models/nse_scaler_v2.joblib')
encoder = joblib.load('data/nse_models/nse_direction_encoder_v2.joblib')
with open('data/nse_models/selected_features_v2.json', 'r') as f:
    import json
    selected_features = json.load(f)

print("=" * 80)
print("DIAGNOSING PREDICTION FEATURE VALUES")
print("=" * 80)
print(f"\nModel encoder classes: {encoder.classes_}")
print(f"Down index: {list(encoder.classes_).index('Down')}")
print(f"Up index: {list(encoder.classes_).index('Up')}")

# Load training metadata to see feature ranges
metadata = json.load(open('data/nse_models/model_metadata_v2.json', 'r'))
print(f"\nModel trained on: {metadata.get('data_range', 'Unknown')}")
if 'training_samples' in metadata:
    print(f"Training samples: {metadata['training_samples']:,}")

# Now manually calculate features for one stock on April 17
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=192.168.86.28\\MSSQLSERVER01;'
    'DATABASE=stockdata_db;'
    'UID=remote_user;'
    'PWD=YourStrongPassword123!;'
    'TrustServerCertificate=yes'
)

# Use the EXACT same query as predict script
query = """
WITH latest_fundamentals AS (
    SELECT 
        ticker,
        market_cap,
        ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY fetch_date DESC) as rn
    FROM nse_500_fundamentals
),
price_data AS (
    SELECT 
        h.ticker,
        h.trading_date,
        CAST(h.close_price AS FLOAT) AS close_price,
        CAST(h.volume AS FLOAT) AS volume,
        f.market_cap
    FROM nse_500_hist_data h
    LEFT JOIN latest_fundamentals f ON h.ticker = f.ticker AND f.rn = 1
    WHERE h.ticker = 'RELIANCE.NS'
      AND h.trading_date <= '2026-04-17'
      AND h.trading_date >= DATEADD(day, -60, '2026-04-17')
      AND h.close_price IS NOT NULL
)
SELECT * FROM price_data
ORDER BY trading_date
"""

df_stock = pd.read_sql(query, conn)
print(f"\n1. RELIANCE.NS DATA (last 60 days)")
print(f"   Rows loaded: {len(df_stock)}")
print(f"   Date range: {df_stock['trading_date'].min()} to {df_stock['trading_date'].max()}")

# Calculate technical indicators manually
df = df_stock.copy()
df = df.sort_values('trading_date')

# Calculate returns
df['return_5d'] = df['close_price'].pct_change(5) * 100
df['return_10d'] = df['close_price'].pct_change(10) * 100

# Moving averages
df['sma_20'] = df['close_price'].rolling(window=20).mean()
df['price_to_sma20'] = (df['close_price'] / df['sma_20']) - 1

# Volume
df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()

# Bollinger Bands
df['bb_middle'] = df['close_price'].rolling(window=20).mean()
df['bb_std'] = df['close_price'].rolling(window=20).std()
df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
df['bb_position'] = (df['close_price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

# High 20d
df['high_20d'] = df['close_price'].rolling(window=20).max()

# Stochastic
df['low_14d'] = df['close_price'].rolling(window=14).min()
df['high_14d'] = df['close_price'].rolling(window=14).max()
df['stochastic_k'] = 100 * (df['close_price'] - df['low_14d']) / (df['high_14d'] - df['low_14d'])

# Get April 17 row
df['trading_date'] = pd.to_datetime(df['trading_date'])
target_date = pd.to_datetime('2026-04-17')
april17_rows = df[df['trading_date'] == target_date]
print(f"\n   Rows for April 17: {len(april17_rows)}")
if len(april17_rows) == 0:
    print(f"   Available dates after technical calc:")
    print(df[['trading_date', 'return_5d', 'bb_position']].tail(10))
    print("\n   ERROR: April 17 not in dataset after technical indicators!")
    exit(1)

april17 = april17_rows.iloc[0]

print(f"\n2. STOCK-SPECIFIC FEATURES ON APRIL 17")
print(f"   return_5d: {april17['return_5d']:.4f}")
print(f"   return_10d: {april17['return_10d']:.4f}")
print(f"   price_to_sma20: {april17['price_to_sma20']:.4f}")
print(f"   volume_ratio: {april17['volume_ratio']:.4f}")
print(f"   bb_position: {april17['bb_position']:.4f}")
print(f"   bb_upper: {april17['bb_upper']:.2f}")
print(f"   sma_20: {april17['sma_20']:.2f}")
print(f"   high_20d: {april17['high_20d']:.2f}")
print(f"   stochastic_k: {april17['stochastic_k']:.2f}")

# Get market context
query = """
SELECT *
FROM market_context_daily
WHERE trading_date = '2026-04-17'
"""
market = pd.read_sql(query, conn)

print(f"\n3. MARKET CONTEXT FEATURES ON APRIL 17")
print(f"   india_vix_close: {market.iloc[0]['india_vix_close']:.2f}")
print(f"   nifty50_close: {market.iloc[0]['nifty50_close']:.2f}")
print(f"   sp500_close: {market.iloc[0]['sp500_close']:.2f}")
print(f"   vix_close: {market.iloc[0]['vix_close']:.2f}")
print(f"   dxy_close: {market.iloc[0]['dxy_close']:.2f}")
print(f"   us_10y_yield_close: {market.iloc[0]['us_10y_yield_close']:.4f}")
print(f"   nifty50_return_1d: {market.iloc[0]['nifty50_return_1d']:.4f}")
print(f"   india_vix_change_pct: {market.iloc[0]['india_vix_change_pct']:.4f}")
print(f"   sp500_return_1d: {market.iloc[0]['sp500_return_1d']:.4f}")
print(f"   dxy_return_1d: {market.iloc[0]['dxy_return_1d']:.4f}")
print(f"   vix_change_pct: {market.iloc[0]['vix_change_pct']:.4f}")

# Build feature vector
feature_values = {}
for feat in selected_features:
    if feat in april17.index:
        feature_values[feat] = april17[feat]
    elif feat in market.columns:
        feature_values[feat] = market.iloc[0][feat]
    else:
        feature_values[feat] = np.nan
        print(f"   WARNING: Feature '{feat}' not found!")

X = pd.DataFrame([feature_values])[selected_features]

print(f"\n4. FEATURE VECTOR (all 20 features)")
for feat, val in zip(selected_features, X.values[0]):
    print(f"   {feat:25s}: {val:12.4f}")

# Check for NaNs
nan_count = X.isna().sum().sum()
if nan_count > 0:
    print(f"\n   ⚠️  WARNING: {nan_count} NaN values found!")
    print(X.isna().sum()[X.isna().sum() > 0])

# Scale and predict
X_scaled = scaler.transform(X)
y_proba = model.predict_proba(X_scaled)[0]

print(f"\n5. PREDICTION FOR RELIANCE.NS")
print(f"   Down (Sell) probability: {y_proba[0]:.4f} ({y_proba[0]*100:.2f}%)")
print(f"   Up (Buy) probability: {y_proba[1]:.4f} ({y_proba[1]*100:.2f}%)")
print(f"   Predicted: {'SELL' if y_proba[0] > y_proba[1] else 'BUY'}")

conn.close()

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)
print("""
If RELIANCE (a large stable stock) predicts 100% SELL, check:
1. Are NaN values being filled with 0 (extreme bearish signal)?
2. Is the VIX or market context triggering blanket SELL?
3. Are the scaled values out of training range?
""")
