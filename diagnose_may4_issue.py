"""
Diagnose May 4, 2026 prediction issue
Check: model, features, data, probabilities
"""

import os
import sys
import numpy as np
import pandas as pd
import pyodbc
import joblib
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

print("="*80)
print("NSE PREDICTION DIAGNOSTIC - MAY 4, 2026")
print("="*80)

# Step 1: Check model files
print("\n[1] CHECKING MODEL FILES...")
model_dir = Path('data/nse_models')
model_file = model_dir / 'nse_gb_model_v2.joblib'
features_file = model_dir / 'selected_features_v2.json'

if model_file.exists():
    print(f"  ✓ Model file exists: {model_file}")
    model = joblib.load(model_file)
    print(f"  ✓ Model loaded: {type(model).__name__}")
    print(f"  ✓ Model classes: {model.classes_}")
else:
    print(f"  ✗ Model file NOT FOUND: {model_file}")
    sys.exit(1)

if features_file.exists():
    print(f"  ✓ Features file exists: {features_file}")
    with open(features_file, 'r') as f:
        selected_features = json.load(f)
    print(f"  ✓ Number of features: {len(selected_features)}")
    print(f"  ✓ Features: {selected_features}")
else:
    print(f"  ✗ Features file NOT FOUND: {features_file}")
    sys.exit(1)

# Step 2: Check database connection and data
print("\n[2] CHECKING DATABASE AND DATA...")
conn_str = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={os.getenv('SQL_SERVER')};"
    f"DATABASE={os.getenv('SQL_DATABASE')};"
    f"UID={os.getenv('SQL_USERNAME')};"
    f"PWD={os.getenv('SQL_PASSWORD')}"
)

try:
    conn = pyodbc.connect(conn_str)
    print("  ✓ Database connection successful")
    
    # Check latest trading date
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(trading_date) FROM nse_500_hist_data")
    latest_date = cursor.fetchone()[0]
    print(f"  ✓ Latest trading date in DB: {latest_date}")
    
    # Check prediction counts
    cursor.execute("""
        SELECT 
            predicted_signal,
            COUNT(*) as count,
            AVG(CAST(buy_probability AS FLOAT)) as avg_buy_prob,
            AVG(CAST(sell_probability AS FLOAT)) as avg_sell_prob
        FROM ml_nse_trading_predictions
        WHERE trading_date = (SELECT MAX(trading_date) FROM ml_nse_trading_predictions)
        GROUP BY predicted_signal
    """)
    rows = cursor.fetchall()
    
    if rows:
        pred_date = None
        print("\n  Current predictions in database:")
        for row in rows:
            signal, count, avg_buy, avg_sell = row
            print(f"    {signal}: {count:,} signals (Buy prob: {avg_buy:.2%}, Sell prob: {avg_sell:.2%})")
    else:
        print("  ⚠ No predictions found in database")
    
    conn.close()
    
except Exception as e:
    print(f"  ✗ Database error: {e}")
    sys.exit(1)

# Step 3: Check for hybrid features
print("\n[3] CHECKING HYBRID FEATURES...")
market_neutral = ['stock_return_vs_nifty', 'rsi_vs_sector_avg', 'beta_adjusted_return', 
                  'relative_strength_20d', 'beta_adjusted_return']
interaction = ['outperformance_in_fear', 'contrarian_strength']

found_neutral = [f for f in market_neutral if f in selected_features]
found_interaction = [f for f in interaction if f in selected_features]

print(f"  Market-neutral features found: {len(found_neutral)}")
for f in found_neutral:
    print(f"    ✓ {f}")

print(f"  Interaction features found: {len(found_interaction)}")
for f in found_interaction:
    print(f"    ✓ {f}")

if len(found_neutral) == 0 or len(found_interaction) == 0:
    print("\n  ⚠ WARNING: Hybrid features missing or incomplete!")
    print("    This could explain the bearish bias.")

# Step 4: Check model metadata
print("\n[4] CHECKING MODEL METADATA...")
metadata_file = model_dir / 'model_metadata_v2.json'
if metadata_file.exists():
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    print(f"  ✓ Training date: {metadata.get('training_date', 'N/A')}")
    print(f"  ✓ Train samples: {metadata.get('train_samples', 'N/A')}")
    print(f"  ✓ Cal samples: {metadata.get('cal_samples', 'N/A')}")
    print(f"  ✓ Test samples: {metadata.get('test_samples', 'N/A')}")
    print(f"  ✓ Test accuracy: {metadata.get('test_accuracy', 'N/A')}")
    
    if 'class_distribution' in metadata:
        print(f"  ✓ Class distribution in training:")
        for cls, count in metadata['class_distribution'].items():
            print(f"    {cls}: {count}")
else:
    print("  ⚠ Metadata file not found")

# Step 5: Final diagnosis
print("\n" + "="*80)
print("DIAGNOSIS SUMMARY")
print("="*80)

issues = []
if len(found_neutral) < 2:
    issues.append("Missing market-neutral features - model relies too much on market context")
if len(found_interaction) < 1:
    issues.append("Missing interaction features - model can't learn 'strong stock in weak market'")

if issues:
    print("\n⚠ ISSUES DETECTED:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    print("\nRECOMMENDED FIX:")
    print("  Run: python retrain_nse_model_v2.py")
    print("  This will regenerate the model with proper hybrid features")
else:
    print("\n✓ MODEL AND FEATURES LOOK CORRECT")
    print("\nPOSSIBLE CAUSES:")
    print("  1. Model needs retraining with fresh data")
    print("  2. Recent market conditions are truly bearish")
    print("  3. Calibration needs adjustment")
    print("\nRECOMMENDED ACTIONS:")
    print("  1. Check market conditions (VIX, India VIX, NIFTY trends)")
    print("  2. Retrain model: python retrain_nse_model_v2.py")
    print("  3. Compare with NASDAQ predictions for validation")

print("\n" + "="*80)
