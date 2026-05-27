"""Debug model predictions for today - identify root cause of 99.9% Sell."""
import sys, json
import numpy as np
sys.path.insert(0, '.')

# Import the module so IsotonicCalibratedModel is in __main__
import retrain_nse_model_v2
from retrain_nse_model_v2 import IsotonicCalibratedModel

# Make it available in __main__ (needed for joblib deserialization)
import __main__
__main__.IsotonicCalibratedModel = IsotonicCalibratedModel

import joblib
model = joblib.load('data/nse_models/nse_gb_model_v2.joblib')
print('Model loaded:', type(model).__name__)

with open('data/nse_models/selected_features_v2.json') as f:
    feats = json.load(f)

print('\nFeatures (20):')
for i, f in enumerate(feats):
    print(f'  {i:2d}: {f}')

print('\n=== SENSITIVITY TESTS ===\n')

# All-zero test
X_zeros = np.zeros((1, 20))
p = model.predict_proba(X_zeros)
print(f'All zeros:        Buy={p[0][1]:.4f}  Sell={p[0][0]:.4f}')

# Neutral baseline: ratio features = 1.0, position/RSI in middle, returns = 0
# [nifty_vs200d, sp500_vs200d, vix_vs60d, india_vix_vs60d, dxy_vs60d,
#  price_sma20, bb_pos, risk_adj_mom, return10d, price_sma50,
#  high_20d, low_20d, rel_str_20d, sector_rsi_avg, sector_ret5d,
#  stock_vs_nifty, stock_vs_nifty_5d, contrarian, defensive, outperf_fear]
X_neutral = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 1.0,
                        1.0, 0.9, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
p = model.predict_proba(X_neutral)
print(f'Neutral (all 1.0 for ratios): Buy={p[0][1]:.4f}  Sell={p[0][0]:.4f}')

# Today's market context with NULL filled as 1.0
# nifty_vs200d=1.0 (NULL!), sp500_vs200d=1.1036, vix_vs60d=0.802, india_vix_vs60d=1.0(NULL), dxy_vs60d=1.001
X_today_mkt = np.array([[1.0, 1.1036, 0.802, 1.0, 1.001, 1.0, 0.5, 0.0, 0.0, 1.0,
                           1.0, 0.9, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
p = model.predict_proba(X_today_mkt)
print(f'Today market ctx (NIFTY/IVIX NULL->1.0): Buy={p[0][1]:.4f}  Sell={p[0][0]:.4f}')

# Today with REAL nifty value (0.9613 = 24031/24998), real india_vix (0.851 = 16.7/19.64)
X_today_real = np.array([[0.9613, 1.1036, 0.802, 0.851, 1.001, 1.0, 0.5, 0.0, 0.0, 1.0,
                            1.0, 0.9, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
p = model.predict_proba(X_today_real)
print(f'Today market ctx (REAL nifty/ivix):       Buy={p[0][1]:.4f}  Sell={p[0][0]:.4f}')

# Sweep nifty50_vs_200d from 0.8 to 1.2 with bullish sp500/vix
print('\n=== nifty50_vs_200d sweep (other features = bullish) ===')
for nifty_v in [0.80, 0.85, 0.90, 0.95, 0.9613, 1.0, 1.05, 1.10, 1.15, 1.20]:
    X = np.array([[nifty_v, 1.1036, 0.802, 0.851, 1.001, 1.0, 0.5, 0.0, 0.0, 1.0,
                   1.0, 0.9, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    p = model.predict_proba(X)
    print(f'  nifty_vs_200d={nifty_v:.4f}: Buy={p[0][1]:.4f}  Sell={p[0][0]:.4f}')

# Sweep sp500_vs_200d
print('\n=== sp500_vs_200d sweep (nifty=0.96, vix=0.80, ivix=0.85) ===')
for sp_v in [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.1036, 1.15, 1.20]:
    X = np.array([[0.9613, sp_v, 0.802, 0.851, 1.001, 1.0, 0.5, 0.0, 0.0, 1.0,
                   1.0, 0.9, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    p = model.predict_proba(X)
    print(f'  sp500_vs_200d={sp_v:.4f}: Buy={p[0][1]:.4f}  Sell={p[0][0]:.4f}')

# What if scaler is expected?
print('\n=== Checking if scaler is needed ===')
scaler = joblib.load('data/nse_models/nse_scaler_v2.joblib')
X_scaled = scaler.transform(X_today_real)
p_scaled = model.predict_proba(X_scaled)
print(f'Scaled today (real nifty): Buy={p_scaled[0][1]:.4f}  Sell={p_scaled[0][0]:.4f}')

X_mkt_scaled = scaler.transform(X_today_mkt)
p_mkt = model.predict_proba(X_mkt_scaled)
print(f'Scaled today (NULL->1.0): Buy={p_mkt[0][1]:.4f}  Sell={p_mkt[0][0]:.4f}')

print('\n=== Scaler mean/scale for first 5 features ===')
for i, (f, mean, scale) in enumerate(zip(feats[:5], scaler.mean_[:5], scaler.scale_[:5])):
    print(f'  {f}: mean={mean:.4f} scale={scale:.4f}')
