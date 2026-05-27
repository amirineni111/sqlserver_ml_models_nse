"""Run full prediction pipeline for 5 stocks and print feature values."""
import sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import retrain_nse_model_v2
from retrain_nse_model_v2 import IsotonicCalibratedModel
import __main__
__main__.IsotonicCalibratedModel = IsotonicCalibratedModel

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os, pyodbc

load_dotenv()
conn_str = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={os.getenv('SQL_SERVER')};"
    f"DATABASE={os.getenv('SQL_DATABASE')};"
    f"UID={os.getenv('SQL_USERNAME')};"
    f"PWD={os.getenv('SQL_PASSWORD')};"
    "TrustServerCertificate=yes"
)
conn = pyodbc.connect(conn_str)

model = joblib.load('data/nse_models/nse_gb_model_v2.joblib')
scaler = joblib.load('data/nse_models/nse_scaler_v2.joblib')
with open('data/nse_models/selected_features_v2.json') as f:
    selected_features = json.load(f)

# ---- run the actual prediction pipeline for a few stocks ----
# import the function from predict_nse_signals_v2
import importlib.util
spec = importlib.util.spec_from_file_location("predict", "predict_nse_signals_v2.py")
predict_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict_mod)

pred_date = pd.read_sql("SELECT MAX(trading_date) as d FROM nse_500_hist_data", conn)
prediction_date = pd.to_datetime(pred_date.iloc[0]['d'])
print(f"Prediction date: {prediction_date.date()}\n")

# Load a small subset (5 tickers)
df_full = predict_mod.load_prediction_data(conn, prediction_date)
print(f"Loaded {len(df_full)} total rows\n")

# Filter to just today
df_today = df_full[df_full['trading_date'] == prediction_date].copy()
print(f"Today's rows: {len(df_today)}\n")

# Check each feature stats for today
print("=== FEATURE STATS FOR TODAY ===")
print(f"{'Feature':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'#NaN':>6}")
for feat in selected_features:
    if feat in df_today.columns:
        col = df_today[feat]
        print(f"{feat:<30} {col.mean():>10.4f} {col.std():>10.4f} {col.min():>10.4f} {col.max():>10.4f} {col.isna().sum():>6}")
    else:
        print(f"{feat:<30} MISSING COLUMN")

print("\n=== SCALED FEATURE STATS ===")
X = df_today[selected_features].values
X_scaled = scaler.transform(X)

print(f"{'Feature':<30} {'ScaledMean':>12} {'ScaledStd':>12} {'ScaledMin':>12} {'ScaledMax':>12}")
for i, feat in enumerate(selected_features):
    col = X_scaled[:, i]
    print(f"{feat:<30} {col.mean():>12.4f} {col.std():>12.4f} {col.min():>12.4f} {col.max():>12.4f}")

print("\n=== SAMPLE PREDICTIONS (first 20 stocks) ===")
sample_X = X_scaled[:20]
# Handle NaN - replace with 0 like production would
X_scaled_clean = np.nan_to_num(X_scaled, nan=0.0)
sample_X_clean = X_scaled_clean[:20]
probas = model.predict_proba(sample_X_clean)
tickers_sample = df_today['ticker'].iloc[:20].tolist()
for t, p in zip(tickers_sample, probas):
    print(f"  {t}: Buy={p[1]:.4f} Sell={p[0]:.4f}")

# Distribution
all_probas = model.predict_proba(X_scaled_clean)
buy_pct = (all_probas[:, 1] >= 0.5).mean() * 100
print(f"\nTotal: {len(X_scaled_clean)} stocks, Buy% with NaN->0: {buy_pct:.1f}%")

# Also test with NaN -> scaler mean (i.e., pre-scale substitute to mean)
X_nanmean = X.copy()
for i in range(X.shape[1]):
    col_mean = np.nanmean(X[:, i])
    X_nanmean[np.isnan(X_nanmean[:, i]), i] = col_mean
X_nanmean_scaled = scaler.transform(X_nanmean)
probas2 = model.predict_proba(X_nanmean_scaled)
buy_pct2 = (probas2[:, 1] >= 0.5).mean() * 100
print(f"Total: {len(X_nanmean_scaled)} stocks, Buy% with NaN->column_mean: {buy_pct2:.1f}%")

conn.close()
