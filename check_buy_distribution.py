"""Check Buy probability distribution for today's stocks."""
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

import importlib.util
spec = importlib.util.spec_from_file_location("predict", "predict_nse_signals_v2.py")
predict_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict_mod)

pred_date = pd.read_sql("SELECT MAX(trading_date) as d FROM nse_500_hist_data", conn)
prediction_date = pd.to_datetime(pred_date.iloc[0]['d'])

model = joblib.load('data/nse_models/nse_gb_model_v2.joblib')
scaler = joblib.load('data/nse_models/nse_scaler_v2.joblib')
with open('data/nse_models/selected_features_v2.json') as f:
    selected_features = json.load(f)

df = predict_mod.load_prediction_data(conn, prediction_date)
df_today = df[df['trading_date'] == prediction_date].copy()

X = df_today[selected_features].fillna(0)
X_scaled = scaler.transform(X)
X_scaled_clean = np.nan_to_num(X_scaled, nan=0.0)
probas = model.predict_proba(X_scaled_clean)

buy_proba = probas[:, 1]  # Up = Buy

print(f"Buy probability distribution for {len(buy_proba)} stocks on {prediction_date.date()}:")
print(f"  Mean:     {buy_proba.mean():.4f} ({buy_proba.mean()*100:.1f}%)")
print(f"  Median:   {np.median(buy_proba):.4f} ({np.median(buy_proba)*100:.1f}%)")
print(f"  Std:      {buy_proba.std():.4f}")
print(f"  Min:      {buy_proba.min():.4f}")
print(f"  Max:      {buy_proba.max():.4f}")
print(f"  P25:      {np.percentile(buy_proba, 25):.4f}")
print(f"  P75:      {np.percentile(buy_proba, 75):.4f}")
print()

# Distribution buckets
thresholds = [0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]
print("Cumulative distribution (% stocks with Buy prob ≥ threshold):")
for t in thresholds:
    pct = (buy_proba >= t).mean() * 100
    print(f"  >= {t:.2f}: {pct:.1f}% ({(buy_proba >= t).sum()} stocks)")

print()
# Top 20 stocks by Buy probability
df_today['buy_proba'] = buy_proba
top_buys = df_today.nlargest(20, 'buy_proba')[['ticker', 'sector', 'buy_proba', 'return_10d', 'price_to_sma20']].copy()
print("Top 20 Buy candidates:")
print(top_buys.to_string(index=False))

conn.close()
