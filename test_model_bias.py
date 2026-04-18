"""Test NSE model predictions to see actual output"""
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from database.connection import SQLServerConnection

print("=" * 70)
print("NSE MODEL PREDICTION TEST")
print("=" * 70)

# Load models
models_dir = Path('data/nse_models')
encoder = joblib.load(models_dir / 'nse_direction_encoder.joblib')
scaler = joblib.load(models_dir / 'nse_scaler.joblib')
best_model = joblib.load(models_dir / 'nse_clf_ensemble.joblib')

print(f"\n[1] Encoder classes: {encoder.classes_}")
print(f"    Down = {list(encoder.classes_).index('Down')}")
print(f"    Up = {list(encoder.classes_).index('Up')}")

# Get some recent data
db = SQLServerConnection()
query = """
SELECT TOP 100
    ticker,
    CAST(close_price AS FLOAT) as close_price,
    CAST(open_price AS FLOAT) as open_price,
    CAST(high_price AS FLOAT) as high_price,
    CAST(low_price AS FLOAT) as low_price,
    CAST(volume AS FLOAT) as volume
FROM dbo.nse_500_hist_data
WHERE trading_date = (SELECT MAX(trading_date) FROM dbo.nse_500_hist_data)
"""
df = db.execute_query(query)
print(f"\n[2] Loaded {len(df)} recent records")

# Create dummy features (just to test model output)
# In real prediction, these would be calculated properly
n_features = 30  # from metadata
X_test = np.random.randn(len(df), n_features)
X_scaled = scaler.transform(X_test)

# Get model predictions
print(f"\n[3] Running model predictions...")
predictions_int = best_model.predict(X_scaled)
probabilities = best_model.predict_proba(X_scaled)

print(f"\n[4] Raw model output (first 10 samples):")
print(f"    predictions_int: {predictions_int[:10]}")
print(f"    probabilities shape: {probabilities.shape}")
print(f"    probabilities[:5, :] =")
for i in range(min(5, len(probabilities))):
    print(f"      {probabilities[i]}")

# Convert to labels
predicted_labels = encoder.inverse_transform(predictions_int)
print(f"\n[5] After inverse_transform (first 10):")
print(f"    {predicted_labels[:10]}")

# Count predictions
unique, counts = np.unique(predicted_labels, return_counts=True)
print(f"\n[6] PREDICTION DISTRIBUTION (with random features):")
for label, count in zip(unique, counts):
    pct = (count / len(predicted_labels)) * 100
    signal = 'Buy' if label == 'Up' else 'Sell'
    print(f"    {label:6s} → {signal:4s}: {count:3d} ({pct:5.1f}%)")

# Check probability distribution
prob_down = probabilities[:, 0]  # P(Down) since Down is class 0
prob_up = probabilities[:, 1]    # P(Up) since Up is class 1

print(f"\n[7] PROBABILITY STATISTICS:")
print(f"    P(Down) - mean: {prob_down.mean():.3f}, median: {np.median(prob_down):.3f}")
print(f"    P(Up)   - mean: {prob_up.mean():.3f}, median: {np.median(prob_up):.3f}")
print(f"    P(Down) range: [{prob_down.min():.3f}, {prob_down.max():.3f}]")
print(f"    P(Up)   range: [{prob_up.min():.3f}, {prob_up.max():.3f}]")

# Check if probabilities are systematically biased
high_down_prob = (prob_down > 0.6).sum()
high_up_prob = (prob_up > 0.6).sum()
print(f"\n[8] HIGH CONFIDENCE COUNTS (>60%):")
print(f"    P(Down) > 0.6: {high_down_prob} ({high_down_prob/len(df)*100:.1f}%)")
print(f"    P(Up)   > 0.6: {high_up_prob} ({high_up_prob/len(df)*100:.1f}%)")

if high_down_prob > high_up_prob * 2:
    print(f"\n❌ MODEL IS BIASED TOWARD DOWN/SELL!")
    print(f"   Even with random features, model predicts Down {high_down_prob/high_up_prob:.1f}x more often")
    print(f"   This indicates the model learned to predict Down regardless of input")
    print(f"   ROOT CAUSE: Class balancing was NOT applied during training!")
elif high_up_prob > high_down_prob * 2:
    print(f"\n❌ MODEL IS BIASED TOWARD UP/BUY!")
else:
    print(f"\n✅ Model predictions are balanced with random features")
    print(f"   Issue is likely in feature engineering, not model bias")

print("\n" + "=" * 70)
