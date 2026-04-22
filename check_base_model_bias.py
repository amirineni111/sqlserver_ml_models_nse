"""
Check base model predictions BEFORE calibration to isolate the bias source
"""

import joblib
import pyodbc
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

load_dotenv()

def get_connection():
    """Create database connection"""
    sql_server = os.getenv('SQL_SERVER')
    sql_database = os.getenv('SQL_DATABASE')
    sql_username = os.getenv('SQL_USERNAME')
    sql_password = os.getenv('SQL_PASSWORD')
    sql_driver = os.getenv('SQL_DRIVER', 'ODBC Driver 17 for SQL Server')
    
    conn_str = (
        f"DRIVER={{{sql_driver}}};"
        f"SERVER={sql_server};"
        f"DATABASE={sql_database};"
        f"UID={sql_username};"
        f"PWD={sql_password};"
        f"TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str)

def check_base_model_bias():
    """Check if base GB model (before calibration) is biased"""
    
    print("="*80)
    print("CHECKING BASE MODEL BIAS (BEFORE CALIBRATION)")
    print("="*80)
    
    # Load model artifacts
    print("\n[INFO] Loading model artifacts...")
    base_model = joblib.load('data/nse_models/nse_gb_base_model_v2.joblib')
    scaler = joblib.load('data/nse_models/nse_scaler_v2.joblib')
    encoder = joblib.load('data/nse_models/nse_direction_encoder_v2.joblib')
    
    # Load selected features
    import json
    with open('data/nse_models/selected_features_v2.json', 'r') as f:
        feature_cols = json.load(f)
    
    print(f"[SUCCESS] Loaded base model: {type(base_model).__name__}")
    print(f"[SUCCESS] Loaded {len(feature_cols)} features")
    
    # Fetch recent data (same as predictions)
    print("\n[INFO] Fetching April 21, 2026 data...")
    conn = get_connection()
    
    query = """
    SELECT 
        h.ticker,
        h.trading_date,
        h.close_price,
        h.open_price,
        h.high_price,
        h.low_price,
        h.volume,
        h.rsi,
        h.macd,
        h.macd_signal,
        h.macd_histogram,
        h.bb_upper,
        h.bb_middle,
        h.bb_lower,
        h.atr,
        f.market_cap_category,
        f.sector,
        f.industry
    FROM nse_500_hist_data h
    LEFT JOIN nse_500_fundamentals f ON h.ticker = f.ticker
    WHERE h.trading_date = '2026-04-21'
    AND h.close_price IS NOT NULL
    ORDER BY h.ticker
    """
    
    df = pd.read_sql(query, conn)
    print(f"[SUCCESS] Fetched {len(df)} records for April 21, 2026")
    
    # Calculate features (simplified - just the basics)
    print("\n[INFO] Calculating features...")
    
    # Ensure all feature columns exist (fill missing with 0)
    X = pd.DataFrame(0, index=df.index, columns=feature_cols)
    
    # Fill available features
    for col in feature_cols:
        if col in df.columns:
            X[col] = df[col]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Get BASE model predictions (before calibration)
    print("\n[INFO] Getting BASE model predictions...")
    base_proba = base_model.predict_proba(X_scaled)
    base_pred = base_model.predict(X_scaled)
    
    # Decode predictions
    base_pred_labels = encoder.inverse_transform(base_pred)
    
    # Analyze distribution
    print("\n" + "="*80)
    print("BASE MODEL PREDICTIONS (BEFORE CALIBRATION)")
    print("="*80)
    
    buy_count = (base_pred_labels == 'Buy').sum()
    sell_count = (base_pred_labels == 'Sell').sum()
    
    print(f"\nPrediction Distribution:")
    print(f"  Buy:  {buy_count:4d} ({buy_count/len(df)*100:5.1f}%)")
    print(f"  Sell: {sell_count:4d} ({sell_count/len(df)*100:5.1f}%)")
    
    # Get class indices
    buy_idx = list(encoder.classes_).index('Buy')
    sell_idx = list(encoder.classes_).index('Sell')
    
    print(f"\nProbability Distribution (BASE MODEL):")
    print(f"  Mean P(Buy):  {base_proba[:, buy_idx].mean():.4f}")
    print(f"  Mean P(Sell): {base_proba[:, sell_idx].mean():.4f}")
    print(f"  Median P(Buy):  {np.median(base_proba[:, buy_idx]):.4f}")
    print(f"  Std P(Buy):     {base_proba[:, buy_idx].std():.4f}")
    
    # Load calibrated model for comparison
    print("\n[INFO] Loading calibrated model for comparison...")
    calibrated_model = joblib.load('data/nse_models/nse_gb_model_v2.joblib')
    cal_proba = calibrated_model.predict_proba(X_scaled)
    cal_pred = calibrated_model.predict(X_scaled)
    cal_pred_labels = encoder.inverse_transform(cal_pred)
    
    print("\n" + "="*80)
    print("CALIBRATED MODEL PREDICTIONS (AFTER CALIBRATION)")
    print("="*80)
    
    buy_count_cal = (cal_pred_labels == 'Buy').sum()
    sell_count_cal = (cal_pred_labels == 'Sell').sum()
    
    print(f"\nPrediction Distribution:")
    print(f"  Buy:  {buy_count_cal:4d} ({buy_count_cal/len(df)*100:5.1f}%)")
    print(f"  Sell: {sell_count_cal:4d} ({sell_count_cal/len(df)*100:5.1f}%)")
    
    print(f"\nProbability Distribution (CALIBRATED):")
    print(f"  Mean P(Buy):  {cal_proba[:, buy_idx].mean():.4f}")
    print(f"  Mean P(Sell): {cal_proba[:, sell_idx].mean():.4f}")
    print(f"  Median P(Buy):  {np.median(cal_proba[:, buy_idx]):.4f}")
    print(f"  Std P(Buy):     {cal_proba[:, buy_idx].std():.4f}")
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON: BASE vs CALIBRATED")
    print("="*80)
    
    print(f"\nBuy Signal Count:")
    print(f"  Base model: {buy_count} ({buy_count/len(df)*100:.1f}%)")
    print(f"  Calibrated: {buy_count_cal} ({buy_count_cal/len(df)*100:.1f}%)")
    print(f"  Difference: {buy_count - buy_count_cal:+d}")
    
    print(f"\nMean P(Buy):")
    print(f"  Base model: {base_proba[:, buy_idx].mean():.4f}")
    print(f"  Calibrated: {cal_proba[:, buy_idx].mean():.4f}")
    print(f"  Difference: {base_proba[:, buy_idx].mean() - cal_proba[:, buy_idx].mean():+.4f}")
    
    # Probability shift analysis
    prob_shift = cal_proba[:, buy_idx] - base_proba[:, buy_idx]
    print(f"\nProbability Shift (Calibrated - Base):")
    print(f"  Mean shift: {prob_shift.mean():+.4f}")
    print(f"  Median shift: {np.median(prob_shift):+.4f}")
    print(f"  Max negative shift: {prob_shift.min():.4f}")
    print(f"  Max positive shift: {prob_shift.max():.4f}")
    
    if prob_shift.mean() < -0.01:
        print(f"\n⚠️  CALIBRATION IS MAKING THE BIAS WORSE!")
        print(f"    Calibration reduces Buy probability by {-prob_shift.mean():.4f} on average")
    elif base_proba[:, buy_idx].mean() < 0.45:
        print(f"\n⚠️  BASE MODEL IS ALREADY BIASED!")
        print(f"    Base model has mean P(Buy) = {base_proba[:, buy_idx].mean():.4f} (should be ~0.50)")
    
    conn.close()

if __name__ == "__main__":
    check_base_model_bias()
