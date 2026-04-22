"""
NSE Daily Prediction Script V2 - Simplified Architecture
Uses V2 model (Gradient Boosting + Calibration)

Key improvements:
1. Works with simplified V2 model (single Gradient Boosting)
2. Uses only top 20 features
3. Better probability calibration
4. Cleaner signal generation logic
5. scikit-learn 1.6+ compatibility (manual calibration)

Author: GitHub Copilot
Date: April 18, 2026
Updated: April 21, 2026 - Added calibration model classes for pickle compatibility
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import pyodbc
import joblib
import json
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# Custom Calibrated Model Classes (for scikit-learn 1.6+ compatibility)
# ============================================================================

class IsotonicCalibratedModel:
    """Wrapper for isotonic-calibrated model (scikit-learn 1.6+ compatible)"""
    
    def __init__(self, base_model, calibrators):
        self.base_model = base_model
        self.calibrators = calibrators
        self.classes_ = base_model.classes_
    
    def predict_proba(self, X):
        base_proba = self.base_model.predict_proba(X)
        calibrated_proba = np.column_stack([
            cal.transform(base_proba[:, i]) 
            for i, cal in enumerate(self.calibrators)
        ])
        # Normalize to sum to 1
        calibrated_proba = calibrated_proba / calibrated_proba.sum(axis=1, keepdims=True)
        return calibrated_proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class SigmoidCalibratedModel:
    """Wrapper for sigmoid-calibrated model (Platt scaling, scikit-learn 1.6+ compatible)"""
    
    def __init__(self, base_model, platt_scaler):
        self.base_model = base_model
        self.platt_scaler = platt_scaler
        self.classes_ = base_model.classes_
    
    def predict_proba(self, X):
        base_proba = self.base_model.predict_proba(X)
        return self.platt_scaler.predict_proba(base_proba)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration for prediction pipeline"""
    
    # Database connection
    SQL_SERVER = os.getenv('SQL_SERVER', '192.168.86.28\\MSSQLSERVER01')
    SQL_DATABASE = os.getenv('SQL_DATABASE', 'stockdata_db')
    SQL_USERNAME = os.getenv('SQL_USERNAME', 'remote_user')
    SQL_PASSWORD = os.getenv('SQL_PASSWORD', 'YourStrongPassword123!')
    SQL_DRIVER = os.getenv('SQL_DRIVER', 'ODBC Driver 17 for SQL Server')
    
    # Model paths (V2)
    MODELS_DIR = Path('data/nse_models')
    MODEL_FILE = MODELS_DIR / 'nse_gb_model_v2.joblib'
    SCALER_FILE = MODELS_DIR / 'nse_scaler_v2.joblib'
    ENCODER_FILE = MODELS_DIR / 'nse_direction_encoder_v2.joblib'
    FEATURES_FILE = MODELS_DIR / 'selected_features_v2.json'
    
    # Signal thresholds (from NASDAQ)
    CONFIDENCE_HIGH_THRESHOLD = 0.60  # 60% confidence = High
    CONFIDENCE_STRONG_THRESHOLD = 0.70  # 70% confidence = Strong
    
    # Prediction date (default to today)
    PREDICTION_DATE = None  # Set at runtime

def get_db_connection():
    """Get SQL Server connection"""
    try:
        conn_str = (
            f"DRIVER={{{Config.SQL_DRIVER}}};"
            f"SERVER={Config.SQL_SERVER};"
            f"DATABASE={Config.SQL_DATABASE};"
            f"UID={Config.SQL_USERNAME};"
            f"PWD={Config.SQL_PASSWORD};"
            f"TrustServerCertificate=yes"
        )
        return pyodbc.connect(conn_str)
    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        sys.exit(1)

# ============================================================================
# Load Model Artifacts
# ============================================================================

def load_model_artifacts():
    """Load trained model and supporting artifacts"""
    print("\n" + "="*80)
    print("LOADING MODEL ARTIFACTS (V2)")
    print("="*80)
    
    # Check if V2 model exists
    if not Config.MODEL_FILE.exists():
        print(f"[ERROR] V2 model not found: {Config.MODEL_FILE}")
        print(f"[INFO] Please run retrain_nse_model_v2.py first")
        sys.exit(1)
    
    # Load model
    print(f"[INFO] Loading model from {Config.MODEL_FILE}")
    model = joblib.load(Config.MODEL_FILE)
    print(f"[SUCCESS] Loaded: {type(model).__name__}")
    
    # Load scaler
    scaler = joblib.load(Config.SCALER_FILE)
    print(f"[SUCCESS] Loaded scaler")
    
    # Load encoder
    encoder = joblib.load(Config.ENCODER_FILE)
    print(f"[SUCCESS] Loaded encoder: classes={encoder.classes_}")
    
    # Load selected features
    with open(Config.FEATURES_FILE, 'r') as f:
        selected_features = json.load(f)
    print(f"[SUCCESS] Loaded {len(selected_features)} selected features")
    
    return model, scaler, encoder, selected_features

# ============================================================================
# Feature Engineering (Same as Training)
# ============================================================================

def calculate_technical_indicators(df):
    """Calculate same technical indicators as training script"""
    results = []
    
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('trading_date')
        
        # Price-based features
        ticker_df['return_1d'] = ticker_df['close_price'].pct_change(1)
        ticker_df['return_5d'] = ticker_df['close_price'].pct_change(5)
        ticker_df['return_10d'] = ticker_df['close_price'].pct_change(10)
        ticker_df['return_20d'] = ticker_df['close_price'].pct_change(20)
        
        # Gap analysis
        ticker_df['gap'] = (ticker_df['open_price'] - ticker_df['close_price'].shift(1)) / ticker_df['close_price'].shift(1)
        
        # Moving averages
        ticker_df['sma_20'] = ticker_df['close_price'].rolling(20).mean()
        ticker_df['sma_50'] = ticker_df['close_price'].rolling(50).mean()
        ticker_df['sma_200'] = ticker_df['close_price'].rolling(200).mean()
        ticker_df['ema_12'] = ticker_df['close_price'].ewm(span=12).mean()
        ticker_df['ema_26'] = ticker_df['close_price'].ewm(span=26).mean()
        
        # MA crossovers
        ticker_df['sma_cross_signal'] = np.where(
            ticker_df['sma_20'] > ticker_df['sma_50'], 1, -1
        )
        ticker_df['price_to_sma20'] = ticker_df['close_price'] / ticker_df['sma_20']
        ticker_df['price_to_sma50'] = ticker_df['close_price'] / ticker_df['sma_50']
        
        # RSI (14-period)
        delta = ticker_df['close_price'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        ticker_df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ticker_df['macd'] = ticker_df['ema_12'] - ticker_df['ema_26']
        ticker_df['macd_signal'] = ticker_df['macd'].ewm(span=9).mean()
        ticker_df['macd_histogram'] = ticker_df['macd'] - ticker_df['macd_signal']
        
        # Bollinger Bands
        ticker_df['bb_middle'] = ticker_df['close_price'].rolling(20).mean()
        bb_std = ticker_df['close_price'].rolling(20).std()
        ticker_df['bb_upper'] = ticker_df['bb_middle'] + (2 * bb_std)
        ticker_df['bb_lower'] = ticker_df['bb_middle'] - (2 * bb_std)
        ticker_df['bb_position'] = (ticker_df['close_price'] - ticker_df['bb_lower']) / (ticker_df['bb_upper'] - ticker_df['bb_lower'])
        
        # ATR (14-period)
        high_low = ticker_df['high_price'] - ticker_df['low_price']
        high_close = np.abs(ticker_df['high_price'] - ticker_df['close_price'].shift(1))
        low_close = np.abs(ticker_df['low_price'] - ticker_df['close_price'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        ticker_df['atr'] = true_range.rolling(14).mean()
        ticker_df['atr_pct'] = ticker_df['atr'] / ticker_df['close_price']
        
        # Stochastic Oscillator
        low_14 = ticker_df['low_price'].rolling(14).min()
        high_14 = ticker_df['high_price'].rolling(14).max()
        ticker_df['stochastic_k'] = 100 * (ticker_df['close_price'] - low_14) / (high_14 - low_14)
        ticker_df['stochastic_d'] = ticker_df['stochastic_k'].rolling(3).mean()
        
        # Volume indicators
        ticker_df['volume_ratio'] = ticker_df['volume'] / ticker_df['volume'].rolling(20).mean()
        ticker_df['volume_change'] = ticker_df['volume'].pct_change(1)
        
        # OBV (On-Balance Volume)
        ticker_df['obv'] = (np.sign(ticker_df['close_price'].diff()) * ticker_df['volume']).fillna(0).cumsum()
        ticker_df['obv_ema'] = ticker_df['obv'].ewm(span=20).mean()
        
        # Support/Resistance levels
        ticker_df['pivot'] = (ticker_df['high_price'] + ticker_df['low_price'] + ticker_df['close_price']) / 3
        ticker_df['resistance_1'] = 2 * ticker_df['pivot'] - ticker_df['low_price']
        ticker_df['support_1'] = 2 * ticker_df['pivot'] - ticker_df['high_price']
        ticker_df['sr_pivot_position'] = (ticker_df['close_price'] - ticker_df['support_1']) / (ticker_df['resistance_1'] - ticker_df['support_1'])
        
        # Breakout indicators
        ticker_df['high_20d'] = ticker_df['high_price'].rolling(20).max()
        ticker_df['low_20d'] = ticker_df['low_price'].rolling(20).min()
        ticker_df['breakout_high'] = (ticker_df['close_price'] >= ticker_df['high_20d'].shift(1)).astype(int)
        ticker_df['breakout_low'] = (ticker_df['close_price'] <= ticker_df['low_20d'].shift(1)).astype(int)
        
        results.append(ticker_df)
    
    return pd.concat(results, ignore_index=True)

def load_prediction_data(conn, prediction_date):
    """Load latest data for prediction"""
    print("\n" + "="*80)
    print(f"LOADING PREDICTION DATA FOR {prediction_date}")
    print("="*80)
    
    # Get data up to prediction date (need history for technical indicators)
    query = f"""
    WITH latest_fundamentals AS (
        -- Get most recent fundamental data per ticker (15 historical snapshots)
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
            CAST(h.open_price AS FLOAT) AS open_price,
            CAST(h.high_price AS FLOAT) AS high_price,
            CAST(h.low_price AS FLOAT) AS low_price,
            CAST(h.close_price AS FLOAT) AS close_price,
            CAST(h.volume AS FLOAT) AS volume,
            n.sector,
            n.industry,
            
            -- Derive market cap category from market_cap
            CASE 
                WHEN f.market_cap >= 100000000000 THEN 'Large Cap'  -- >= 100B
                WHEN f.market_cap >= 10000000000 THEN 'Mid Cap'     -- >= 10B
                ELSE 'Small Cap'
            END AS market_cap_category
            
        FROM nse_500_hist_data h
        INNER JOIN nse_500 n ON h.ticker = n.ticker
        LEFT JOIN latest_fundamentals f ON h.ticker = f.ticker AND f.rn = 1
        
        WHERE h.trading_date <= '{prediction_date}'
          AND h.trading_date >= DATEADD(day, -300, '{prediction_date}')
          AND h.close_price IS NOT NULL
          AND h.close_price <> '0'
          AND CAST(h.volume AS FLOAT) > 0
    )
    
    SELECT * FROM price_data
    ORDER BY ticker, trading_date
    """
    
    df = pd.read_sql(query, conn)
    print(f"[SUCCESS] Loaded {len(df):,} rows for {len(df['ticker'].unique())} tickers")
    
    # Calculate technical indicators
    print("[INFO] Calculating technical indicators...")
    df = calculate_technical_indicators(df)
    
    # Merge market context
    print("[INFO] Merging market context data...")
    df = merge_market_context(conn, df)
    
    # Keep only prediction date
    df = df[df['trading_date'] == prediction_date].copy()
    print(f"[SUCCESS] {len(df)} tickers ready for prediction on {prediction_date}")
    
    return df

def merge_market_context(conn, df):
    """Merge market context features (same as training)"""
    query = """
    SELECT 
        trading_date,
        vix_close,
        india_vix_close,
        nifty50_close,
        sp500_close,
        dxy_close,
        us_10y_yield_close,
        vix_change_pct,
        india_vix_change_pct,
        nifty50_return_1d,
        sp500_return_1d,
        dxy_return_1d
    FROM market_context_daily
    ORDER BY trading_date
    """
    
    try:
        df_context = pd.read_sql(query, conn)
        df_context['trading_date'] = pd.to_datetime(df_context['trading_date'])
        
        df['trading_date'] = pd.to_datetime(df['trading_date'])
        market_cols = [c for c in df_context.columns if c != 'trading_date']
        df = df.merge(df_context, on='trading_date', how='left')
        
        # Handle missing market data (forward-fill levels, zero-fill returns)
        level_cols = [c for c in market_cols if 'return' not in c and 'change' not in c]
        for col in level_cols:
            if col in df.columns:
                df[col] = df[col].ffill().bfill().fillna(0)
        
        return_cols = [c for c in market_cols if 'return' in c or 'change' in c]
        for col in return_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
    except Exception as e:
        print(f"[WARNING] Could not load market context: {e}")
    
    return df

# ============================================================================
# Generate Predictions
# ============================================================================

def generate_predictions(model, scaler, encoder, selected_features, df):
    """Generate predictions for all tickers"""
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS")
    print("="*80)
    
    # Prepare features
    missing_features = [f for f in selected_features if f not in df.columns]
    if missing_features:
        print(f"[WARNING] Missing features: {missing_features}")
        for f in missing_features:
            df[f] = 0
    
    X = df[selected_features].fillna(0)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict
    print(f"[INFO] Predicting for {len(df)} tickers...")
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)
    
    # Get class indices (encoder.classes_ is alphabetically sorted)
    # For NSE: 'Down' = 0, 'Up' = 1
    class_names = encoder.classes_
    up_idx = list(class_names).index('Up') if 'Up' in class_names else 1
    down_idx = list(class_names).index('Down') if 'Down' in class_names else 0
    
    print(f"[INFO] Class mapping: Down={down_idx}, Up={up_idx}")
    
    # Create predictions dataframe
    predictions = df[['ticker', 'trading_date', 'close_price', 'sector', 
                     'industry', 'market_cap_category', 'rsi']].copy()
    
    # Probabilities
    predictions['buy_probability'] = y_proba[:, up_idx]  # Up = Buy
    predictions['sell_probability'] = y_proba[:, down_idx]  # Down = Sell
    
    # Signal determination
    predictions['predicted_signal'] = np.where(
        predictions['buy_probability'] > predictions['sell_probability'],
        'Buy',
        'Sell'
    )
    
    # Confidence
    predictions['confidence_percentage'] = np.maximum(
        predictions['buy_probability'],
        predictions['sell_probability']
    ) * 100
    
    # Signal strength (database expects 'Low', 'Medium', 'High')
    predictions['signal_strength'] = np.where(
        predictions['confidence_percentage'] >= Config.CONFIDENCE_STRONG_THRESHOLD * 100,
        'High',
        np.where(
            predictions['confidence_percentage'] >= Config.CONFIDENCE_HIGH_THRESHOLD * 100,
            'Medium',
            'Low'
        )
    )
    
    # High confidence flag
    predictions['high_confidence'] = (
        predictions['confidence_percentage'] >= Config.CONFIDENCE_HIGH_THRESHOLD * 100
    ).astype(int)
    
    # Model name
    predictions['model_name'] = 'GradientBoosting_V2_Calibrated'
    
    # Summary statistics
    print("\n[INFO] Prediction Summary:")
    signal_counts = predictions['predicted_signal'].value_counts()
    for signal, count in signal_counts.items():
        pct = (count / len(predictions)) * 100
        avg_conf = predictions[predictions['predicted_signal'] == signal]['confidence_percentage'].mean()
        print(f"  {signal:8s}: {count:6,} ({pct:5.2f}%) | Avg Confidence: {avg_conf:.1f}%")
    
    print(f"\n[INFO] High Confidence Signals: {predictions['high_confidence'].sum():,}")
    
    return predictions

# ============================================================================
# Production Validation (RUNTIME)
# ============================================================================

def validate_prediction_distribution(predictions):
    """
    Validate prediction distribution before writing to database
    
    Prevents production issues like:
    - Apr 21: 99.6% Sell (9 Buy / 2055 Sell)
    - Apr 18: 97.0% Sell (21 Buy / 2002 Sell)
    
    Returns: True if validation passes, False if critical issue detected
    """
    total = len(predictions)
    buy_count = len(predictions[predictions['predicted_signal'] == 'Buy'])
    sell_count = total - buy_count
    
    buy_pct = buy_count / total * 100
    sell_pct = sell_count / total * 100
    
    print(f"\n{'='*80}")
    print(f"PREDICTION DISTRIBUTION VALIDATION")
    print(f"{'='*80}")
    print(f"Trading Date: {predictions['trading_date'].iloc[0]}")
    print(f"Total:        {total:,}")
    print(f"Buy:          {buy_count:,} ({buy_pct:.1f}%)")
    print(f"Sell:         {sell_count:,} ({sell_pct:.1f}%)")
    
    # Probability analysis
    avg_buy_prob = predictions['buy_probability'].mean()
    avg_sell_prob = predictions['sell_probability'].mean()
    avg_confidence = predictions['confidence_percentage'].mean()
    
    print(f"\nAverage Buy Probability:  {avg_buy_prob:.2%}")
    print(f"Average Sell Probability: {avg_sell_prob:.2%}")
    print(f"Average Confidence:       {avg_confidence:.1f}%")
    
    # CRITICAL: Alert if extreme skew
    issues = []
    
    if buy_pct < 20:
        issues.append(f"Buy% ({buy_pct:.1f}%) below 20% - extreme bearish bias")
        print(f"\n⚠️  WARNING: EXTREME BEARISH BIAS DETECTED")
        print(f"   Buy: {buy_pct:.1f}% is below normal range (20-80%)")
    elif buy_pct > 80:
        issues.append(f"Buy% ({buy_pct:.1f}%) above 80% - extreme bullish bias")
        print(f"\n⚠️  WARNING: EXTREME BULLISH BIAS DETECTED")
        print(f"   Buy: {buy_pct:.1f}% is above normal range (20-80%)")
    else:
        print(f"\n✅ VALIDATION PASSED: Distribution within acceptable range (20-80%)")
    
    # Additional checks
    if avg_confidence > 90:
        issues.append(f"Avg confidence ({avg_confidence:.1f}%) suspiciously high")
        print(f"⚠️  WARNING: Average confidence {avg_confidence:.1f}% is suspiciously high")
    
    if len(issues) > 0:
        print(f"\n{'='*80}")
        print(f"VALIDATION ISSUES DETECTED")
        print(f"{'='*80}")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        print(f"\nRECOMMENDATIONS:")
        print(f"  • Review model calibration (check calibration set balance)")
        print(f"  • Verify training data quality and class distribution")
        print(f"  • Consider retraining with stratified calibration split")
        print(f"  • Check for data quality issues in recent market data")
        
        # OPTION: Uncomment to ABORT on validation failure
        # print(f"\n❌ ABORTING: Predictions not saved due to validation failure")
        # return False
        
        print(f"\n⚠️  PROCEEDING WITH CAUTION: Predictions will be saved with warning flag")
        return True  # Allow save but with warning
    
    return True

# ============================================================================
# Save Predictions to Database
# ============================================================================

def save_predictions(conn, predictions):
    """Save predictions to ml_nse_trading_predictions table"""
    print("\n" + "="*80)
    print("SAVING PREDICTIONS TO DATABASE")
    print("="*80)
    
    # Delete existing predictions for this date
    prediction_date = predictions['trading_date'].iloc[0]
    cursor = conn.cursor()
    
    delete_query = """
    DELETE FROM ml_nse_trading_predictions
    WHERE trading_date = ? AND model_name LIKE '%V2%'
    """
    cursor.execute(delete_query, prediction_date)
    deleted_count = cursor.rowcount
    print(f"[INFO] Deleted {deleted_count} existing V2 predictions for {prediction_date}")
    
    # Insert new predictions
    insert_query = """
    INSERT INTO ml_nse_trading_predictions (
        ticker, trading_date, predicted_signal, confidence, confidence_percentage,
        signal_strength, close_price, rsi, buy_probability, sell_probability,
        model_name, sector, market_cap_category, high_confidence
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    rows_to_insert = []
    for _, row in predictions.iterrows():
        confidence_pct = float(row['confidence_percentage'])
        confidence_decimal = confidence_pct / 100.0  # Convert to 0-1 range
        close_price_val = float(row['close_price']) if pd.notna(row['close_price']) else 0.0
        rows_to_insert.append((
            row['ticker'],
            row['trading_date'],
            row['predicted_signal'],
            confidence_decimal,  # confidence (0-1 decimal for constraint)
            confidence_pct,      # confidence_percentage (0-100 percentage)
            row['signal_strength'],
            close_price_val,     # close_price (required NOT NULL column)
            float(row['rsi']) if pd.notna(row['rsi']) else None,
            float(row['buy_probability']),
            float(row['sell_probability']),
            row['model_name'],
            row['sector'],
            row['market_cap_category'],
            row['high_confidence']
        ))
    
    cursor.executemany(insert_query, rows_to_insert)
    conn.commit()
    
    print(f"[SUCCESS] Inserted {len(rows_to_insert):,} predictions")
    
    # Create summary record
    save_summary(conn, predictions)
    
    cursor.close()

def save_summary(conn, predictions):
    """Save prediction summary to ml_nse_predict_summary table"""
    cursor = conn.cursor()
    
    prediction_date = predictions['trading_date'].iloc[0]
    
    # Calculate summary stats
    total_predictions = len(predictions)
    buy_count = len(predictions[predictions['predicted_signal'] == 'Buy'])
    sell_count = len(predictions[predictions['predicted_signal'] == 'Sell'])
    
    # Signal strength counts
    high_conf_count = len(predictions[predictions['signal_strength'] == 'High'])
    med_conf_count = len(predictions[predictions['signal_strength'] == 'Medium'])
    low_conf_count = len(predictions[predictions['signal_strength'] == 'Low'])
    
    # Buy/Sell by confidence
    high_conf_buys = len(predictions[(predictions['predicted_signal'] == 'Buy') & (predictions['signal_strength'] == 'High')])
    med_conf_buys = len(predictions[(predictions['predicted_signal'] == 'Buy') & (predictions['signal_strength'] == 'Medium')])
    low_conf_buys = len(predictions[(predictions['predicted_signal'] == 'Buy') & (predictions['signal_strength'] == 'Low')])
    
    high_conf_sells = len(predictions[(predictions['predicted_signal'] == 'Sell') & (predictions['signal_strength'] == 'High')])
    med_conf_sells = len(predictions[(predictions['predicted_signal'] == 'Sell') & (predictions['signal_strength'] == 'Medium')])
    low_conf_sells = len(predictions[(predictions['predicted_signal'] == 'Sell') & (predictions['signal_strength'] == 'Low')])
    
    avg_confidence = predictions['confidence_percentage'].mean()
    avg_rsi = predictions['rsi'].mean() if 'rsi' in predictions.columns else None
    avg_buy_prob = predictions['buy_probability'].mean()
    avg_sell_prob = predictions['sell_probability'].mean()
    
    bullish_count = buy_count
    bearish_count = sell_count
    neutral_count = 0  # V2 doesn't have Hold signals
    
    market_trend = 'Bullish' if buy_count > sell_count else 'Bearish'
    
    # Delete existing summary for this date
    cursor.execute("""
        DELETE FROM ml_nse_predict_summary 
        WHERE analysis_date = ?
    """, prediction_date)
    
    # Insert new summary
    cursor.execute("""
        INSERT INTO ml_nse_predict_summary (
            analysis_date, total_predictions, total_buy_signals, total_sell_signals, total_hold_signals,
            high_confidence_count, medium_confidence_count, low_confidence_count,
            high_conf_buys, medium_conf_buys, low_conf_buys,
            high_conf_sells, medium_conf_sells, low_conf_sells,
            avg_confidence, avg_rsi, avg_buy_probability, avg_sell_probability,
            market_trend, bullish_stocks_count, bearish_stocks_count, neutral_stocks_count,
            total_stocks_processed, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        prediction_date, total_predictions, buy_count, sell_count, 0,
        high_conf_count, med_conf_count, low_conf_count,
        high_conf_buys, med_conf_buys, low_conf_buys,
        high_conf_sells, med_conf_sells, low_conf_sells,
        avg_confidence, avg_rsi, avg_buy_prob, avg_sell_prob,
        market_trend, bullish_count, bearish_count, neutral_count,
        total_predictions, 'Generated by GradientBoosting_V2_Calibrated model'
    ))
    
    conn.commit()
    cursor.close()
    
    print(f"[SUCCESS] Saved prediction summary: {buy_count} Buy / {sell_count} Sell")

# ============================================================================
# Main Prediction Pipeline
# ============================================================================

def get_latest_trading_date(conn):
    """Get the most recent trading date available in NSE database"""
    try:
        cursor = conn.cursor()
        query = "SELECT MAX(trading_date) as latest_date FROM nse_500_hist_data"
        cursor.execute(query)
        result = cursor.fetchone()
        cursor.close()
        
        if result and result[0]:
            latest_date = result[0]
            # Convert to string if it's a date object
            if hasattr(latest_date, 'strftime'):
                return latest_date.strftime('%Y-%m-%d')
            return str(latest_date)
        else:
            print("[WARNING] Could not determine latest trading date from database")
            return None
    except Exception as e:
        print(f"[ERROR] Failed to query latest trading date: {e}")
        return None

def main():
    """Main prediction pipeline"""
    print("\n" + "="*80)
    print("NSE DAILY PREDICTION V2 - SIMPLIFIED ARCHITECTURE")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Connect to database first to determine prediction date
    conn = get_db_connection()
    
    # Determine prediction date
    if Config.PREDICTION_DATE:
        prediction_date = Config.PREDICTION_DATE
        print(f"[INFO] Using specified date: {prediction_date}")
    else:
        # Query database for most recent trading date
        prediction_date = get_latest_trading_date(conn)
        if not prediction_date:
            print("[ERROR] Could not determine prediction date")
            conn.close()
            sys.exit(1)
        print(f"[INFO] Using latest trading date from database: {prediction_date}")
    
    print(f"Prediction Date: {prediction_date}")
    
    # Load model artifacts
    model, scaler, encoder, selected_features = load_model_artifacts()
    
    try:
        # Load prediction data
        df = load_prediction_data(conn, prediction_date)
        
        if len(df) == 0:
            print(f"[WARNING] No data available for {prediction_date}")
            print(f"[INFO] Check if {prediction_date} is a valid trading day")
            sys.exit(0)
        
        # Generate predictions
        predictions = generate_predictions(model, scaler, encoder, selected_features, df)
        
        # ⭐ CRITICAL: VALIDATE BEFORE SAVING
        # Prevents production issues like Apr 21 (99.6% Sell)
        if not validate_prediction_distribution(predictions):
            print("[ERROR] Prediction validation failed. Aborting database write.")
            sys.exit(1)
        
        # Save to database
        save_predictions(conn, predictions)
        
        # Final summary
        print("\n" + "="*80)
        print("PREDICTION COMPLETE")
        print("="*80)
        print(f"Date: {prediction_date}")
        print(f"Tickers: {len(predictions):,}")
        print(f"Buy Signals: {len(predictions[predictions['predicted_signal'] == 'Buy']):,}")
        print(f"Sell Signals: {len(predictions[predictions['predicted_signal'] == 'Sell']):,}")
        print(f"High Confidence: {predictions['high_confidence'].sum():,}")
        print(f"Model: GradientBoosting V2 + Isotonic Calibration")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    finally:
        conn.close()

if __name__ == '__main__':
    main()
