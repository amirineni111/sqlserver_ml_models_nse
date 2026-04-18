"""
Debug NSE feature engineering during prediction
Compare training features vs prediction features
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load selected features
with open('data/nse_models/selected_features.json', 'r') as f:
    selected_features = json.load(f)

print("=" * 80)
print("SELECTED FEATURES (30 total)")
print("=" * 80)
for i, feat in enumerate(selected_features, 1):
    print(f"{i:2}. {feat}")

# Load latest predictions from database
import pyodbc
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=192.168.86.28\\MSSQLSERVER01;'
    'DATABASE=stockdata_db;'
    'UID=remote_user;'
    'PWD=YourStrongPassword123!;'
    'TrustServerCertificate=yes'
)

# Get prediction data for a sample stock
cursor = conn.cursor()
cursor.execute('''
    SELECT TOP 5
        ticker,
        buy_probability,
        sell_probability,
        confidence
    FROM ml_nse_trading_predictions
    WHERE trading_date = '2026-04-17'
    ORDER BY buy_probability DESC
''')

print("\n" + "=" * 80)
print("TOP 5 STOCKS BY BUY PROBABILITY")
print("=" * 80)
for row in cursor:
    ticker, buy_prob, sell_prob, conf = row
    print(f"{ticker:15} Buy={buy_prob:.3f} Sell={sell_prob:.3f} Conf={conf:.3f}")

cursor.close()
conn.close()

# Check if prediction script is loading features correctly
print("\n" + "=" * 80)
print("CHECKING FEATURE CATEGORIES")
print("=" * 80)

market_features = [f for f in selected_features if any(x in f for x in ['nifty50', 'vix', 'dxy', 'sp500', 'sector_index', 'us_10y'])]
stock_features = [f for f in selected_features if f not in market_features]

print(f"Market-wide features ({len(market_features)}):")
for f in market_features:
    print(f"  - {f}")

print(f"\nStock-specific features ({len(stock_features)}):")
for f in stock_features:
    print(f"  - {f}")

# Check for potential issues
print("\n" + "=" * 80)
print("POTENTIAL ISSUES TO INVESTIGATE")
print("=" * 80)

issues = []

# Check for forward-looking features
forward_looking = [f for f in selected_features if 'next' in f or 'future' in f]
if forward_looking:
    issues.append(f"⚠️  Forward-looking features found: {forward_looking}")

# Check for complex derived features
derived = [f for f in selected_features if any(x in f for x in ['regime', 'trend', 'momentum', 'signal'])]
if derived:
    print(f"✓ Derived features ({len(derived)}): May need careful calculation during prediction")
    for f in derived:
        print(f"    - {f}")

# Check for encoded features
encoded = [f for f in selected_features if 'encoded' in f or 'signal' in f]
if encoded:
    print(f"\n✓ Encoded/signal features ({len(encoded)}): Check encoding consistency")
    for f in encoded:
        print(f"    - {f}")

if issues:
    for issue in issues:
        print(issue)
else:
    print("✓ No obvious forward-looking features")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("""
The NSE model predicts 99.9% Sell while NASDAQ predicts 64% Buy for the same date.

This suggests feature engineering during prediction is broken. Likely causes:
1. Market-wide features (nifty50, VIX, etc.) missing or set to 0
2. Signal encoding (sr_signal_strength, sma_cross_signal) calculated differently
3. Regime/trend features using different windows

Next steps:
1. Add debug logging to predict_nse_signals.py to print feature values
2. Compare feature distributions: training vs prediction
3. Check if market-wide features are being loaded from correct tables
""")
