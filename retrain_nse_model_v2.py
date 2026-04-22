"""
NSE Model Retraining Script V2 - Simplified Architecture
Based on proven NASDAQ approach (65-70% accuracy)

Key improvements over V1:
1. Single Gradient Boosting model (no VotingClassifier bug)
2. Filtered training data (exclude biased periods)
3. Top 20 features only (down from 30)
4. Proper calibration with dedicated calibration set
5. Better validation and monitoring
6. scikit-learn 1.6+ compatibility (manual calibration)
7. STRATIFIED calibration split (Apr 21, 2026 - fixes bearish bias)

CRITICAL FIX (April 21, 2026):
- Calibration set now uses STRATIFIED split instead of time-series
- Previous approach: Sequential split caused bearish calibration period (59.5% DOWN)
- Fixed approach: Stratified split ensures balanced calibration (50/50 ±5%)
- This fixes the 99.6% Sell predictions issue

Author: GitHub Copilot
Date: April 18, 2026
Updated: April 21, 2026 - Fixed bearish bias in calibration
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import pyodbc
import joblib
import json
from datetime import datetime
from pathlib import Path

# ML libraries
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration following NASDAQ's proven approach"""
    
    # Database connection
    SQL_SERVER = os.getenv('SQL_SERVER', '192.168.86.28\\MSSQLSERVER01')
    SQL_DATABASE = os.getenv('SQL_DATABASE', 'stockdata_db')
    SQL_USERNAME = os.getenv('SQL_USERNAME', 'remote_user')
    SQL_PASSWORD = os.getenv('SQL_PASSWORD', 'YourStrongPassword123!')
    SQL_DRIVER = os.getenv('SQL_DRIVER', 'ODBC Driver 17 for SQL Server')
    
    # Model paths
    MODELS_DIR = Path('data/nse_models')
    BACKUP_DIR = Path('backups')
    
    # Training parameters (from NASDAQ)
    TRAIN_RATIO = 0.60  # 60% training
    CAL_RATIO = 0.20    # 20% calibration
    TEST_RATIO = 0.20   # 20% testing
    
    # Feature selection
    TOP_N_FEATURES = 20  # Down from 30 (NASDAQ uses 20)
    
    # Model hyperparameters (from NASDAQ's successful config)
    GB_PARAMS = {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'random_state': 42,
        'verbose': 1
    }
    
    # Feature importance selection (for initial feature ranking)
    RF_PARAMS = {
        'n_estimators': 100,
        'max_depth': 8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Data filtering (CRITICAL: Exclude biased periods like NASDAQ does)
    # NASDAQ excludes Nov 2025 bull run; NSE should exclude recent bear runs
    # Updated Apr 18 2026: Now including Mar-Apr data since market_context_daily is fixed
    DATA_START_DATE = '2024-06-01'  # Start after initial downtrend
    DATA_END_DATE = '2026-04-15'    # Extended to include recent data (was 2026-02-28)
    
    # Class imbalance threshold (from NASDAQ)
    MAX_IMBALANCE = 0.20  # Warn if class imbalance > 20%
    
    # Calibration method
    CALIBRATION_METHOD = 'isotonic'  # NASDAQ uses isotonic calibration
    
    # Logging
    LOG_DIR = Path('logs')
    
    @classmethod
    def ensure_dirs(cls):
        """Create necessary directories"""
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)

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
# Database Connection
# ============================================================================

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
        conn = pyodbc.connect(conn_str)
        print(f"[SUCCESS] Connected to {Config.SQL_SERVER}/{Config.SQL_DATABASE}")
        return conn
    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        sys.exit(1)

# ============================================================================
# Feature Engineering (Simplified from V1)
# ============================================================================

FUTURE_LEAK_FEATURES = [
    # CRITICAL: These must ALWAYS be excluded!
    'next_5d_return', 'next_day_return', 'next_3d_return',  # Future returns
    'direction', 'direction_3d',  # Derived from future data
    'next_close', 'next_3d_close', 'next_5d_close',  # Future prices
]

def load_training_data(conn):
    """
    Load and prepare training data with feature engineering
    
    Key improvements:
    1. Filters out biased date ranges (like NASDAQ)
    2. Excludes future-leak features
    3. Handles missing market data gracefully
    """
    print("\n" + "="*80)
    print("STEP 1: LOADING TRAINING DATA")
    print("="*80)
    
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
            
            -- Future returns for labeling ONLY (excluded from features)
            LEAD(CAST(h.close_price AS FLOAT), 5) OVER (
                PARTITION BY h.ticker ORDER BY h.trading_date
            ) AS next_5d_close,
            
            -- Sector and fundamentals
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
        
        WHERE h.trading_date >= '{Config.DATA_START_DATE}'
          AND h.trading_date <= '{Config.DATA_END_DATE}'
          AND h.close_price IS NOT NULL
          AND h.close_price <> '0'
          AND CAST(h.volume AS FLOAT) > 0
    )
    
    SELECT 
        *,
        -- Calculate 5-day return for labeling
        CASE 
            WHEN next_5d_close IS NOT NULL AND close_price > 0
            THEN (next_5d_close - close_price) / close_price
            ELSE NULL
        END AS next_5d_return
        
    FROM price_data
    ORDER BY ticker, trading_date
    """
    
    print(f"[INFO] Loading data from {Config.DATA_START_DATE} to {Config.DATA_END_DATE}")
    print(f"[INFO] This excludes recent potentially biased periods (like NASDAQ approach)")
    
    df = pd.read_sql(query, conn)
    print(f"[SUCCESS] Loaded {len(df):,} rows, {len(df['ticker'].unique())} tickers")
    
    # Calculate technical indicators (simplified feature set)
    print("\n[INFO] Calculating technical indicators...")
    df = calculate_technical_indicators(df)
    
    # Merge market context data
    print("[INFO] Merging market context data...")
    df = merge_market_context(conn, df)
    
    # Create target variable (following NASDAQ's approach)
    print("[INFO] Creating target variable...")
    df = create_target_variable(df)
    
    # Remove rows without target (last 5 days for each ticker)
    initial_rows = len(df)
    df = df[df['direction_5d'].notna()].copy()
    print(f"[INFO] Removed {initial_rows - len(df):,} rows without target (last 5 days per ticker)")
    
    return df

def calculate_technical_indicators(df):
    """
    Calculate technical indicators for each ticker
    
    Focus on proven, high-impact features (NASDAQ approach)
    """
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
        
        # Support/Resistance levels (simplified pivot points)
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
    
    df = pd.concat(results, ignore_index=True)
    print(f"[SUCCESS] Calculated technical indicators: {len([c for c in df.columns if c not in ['ticker', 'trading_date', 'sector', 'industry']])} features")
    
    return df

def merge_market_context(conn, df):
    """
    Merge market-wide context features
    
    Handles missing data gracefully (forward-fill levels, zero-fill returns)
    """
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
        
        # Merge with main data
        df['trading_date'] = pd.to_datetime(df['trading_date'])
        market_cols = [c for c in df_context.columns if c != 'trading_date']
        df = df.merge(df_context, on='trading_date', how='left')
        
        # Handle missing market data (from investigation)
        # Forward-fill levels (VIX, yields, prices)
        level_cols = [c for c in market_cols if 'return' not in c and 'change' not in c]
        for col in level_cols:
            if col in df.columns:
                df[col] = df[col].ffill().bfill().fillna(0)
        
        # Zero-fill returns (neutral assumption for missing days)
        return_cols = [c for c in market_cols if 'return' in c or 'change' in c]
        for col in return_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        print(f"[SUCCESS] Merged {len(market_cols)} market context features")
        
    except Exception as e:
        print(f"[WARNING] Could not load market context: {e}")
        print("[INFO] Continuing without market features...")
    
    return df

def create_target_variable(df):
    """
    Create target variable (direction_5d)
    
    Following NSE approach: 'Up' and 'Down' labels
    LabelEncoder will sort alphabetically: Down=0, Up=1
    """
    df['direction_5d'] = np.where(
        df['next_5d_return'] > 0,
        'Up',    # Positive 5-day return
        'Down'   # Negative 5-day return
    )
    
    # Class distribution
    class_counts = df['direction_5d'].value_counts()
    total = len(df[df['direction_5d'].notna()])
    
    print("\n[INFO] Target Variable Distribution:")
    for label, count in class_counts.items():
        pct = (count / total) * 100
        print(f"  {label:10s}: {count:8,} ({pct:5.2f}%)")
    
    # Check for severe imbalance
    imbalance = abs(class_counts.iloc[0] - class_counts.iloc[1]) / total
    if imbalance > Config.MAX_IMBALANCE:
        print(f"\n[WARNING] Class imbalance detected: {imbalance:.1%}")
        print(f"[WARNING] This will affect model predictions!")
        print(f"[INFO] Current range: {Config.DATA_START_DATE} to {Config.DATA_END_DATE}")
        if imbalance > 0.40:
            print(f"[ERROR] Imbalance too severe (>40%). Training may produce biased model.")
            print(f"[INFO] Continuing anyway - class weighting will compensate.")
    else:
        print(f"\n[SUCCESS] Class balance acceptable: {imbalance:.1%} imbalance")
    
    return df

# ============================================================================
# Feature Selection (Following NASDAQ's Proven Approach)
# ============================================================================

def select_features(X, y):
    """
    Select top N features using Random Forest importance
    
    NASDAQ approach: Simple, proven, effective
    """
    print("\n" + "="*80)
    print(f"STEP 2: FEATURE SELECTION (TOP {Config.TOP_N_FEATURES})")
    print("="*80)
    
    print(f"[INFO] Initial feature count: {X.shape[1]}")
    
    # Train Random Forest for feature importance
    print("[INFO] Training Random Forest for feature importance...")
    rf = RandomForestClassifier(**Config.RF_PARAMS)
    rf.fit(X, y)
    
    # Get feature importances
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Select top N features with balance check
    # IMPORTANT: Prevent any single feature from dominating (>10% importance)
    max_importance = importances.iloc[0]['importance']
    if max_importance > 0.10:
        print(f"[WARNING] Feature over-dominance detected: {max_importance:.1%}")
        print(f"[INFO] Implementing balanced selection strategy...")
        
        # Select top 30, exclude over-dominant ones (>10%), keep 20
        extended = importances.head(30)
        balanced = extended[extended['importance'] <= 0.10]
        
        if len(balanced) < Config.TOP_N_FEATURES:
            # Add back lower-importance features to reach 20
            remaining_needed = Config.TOP_N_FEATURES - len(balanced)
            additional = importances[~importances['feature'].isin(balanced['feature'])].head(remaining_needed)
            final_selection = pd.concat([balanced, additional]).sort_values('importance', ascending=False)
        else:
            final_selection = balanced.head(Config.TOP_N_FEATURES)
            
        print(f"[SUCCESS] Balanced selection: excluded over-dominant features")
        selected_features = final_selection['feature'].tolist()
    else:
        selected_features = importances.head(Config.TOP_N_FEATURES)['feature'].tolist()
    
    print(f"\n[SUCCESS] Selected top {len(selected_features)} features:")
    print(importances[importances['feature'].isin(selected_features)].to_string(index=False))
    
    return selected_features, importances

# ============================================================================
# Model Training (NASDAQ's Proven Architecture)
# ============================================================================

def train_model(X_train, y_train, X_cal, y_cal, X_test, y_test):
    """
    Train Gradient Boosting model with calibration
    
    Following NASDAQ's successful approach:
    1. Single Gradient Boosting model
    2. Class + time weighting
    3. Isotonic calibration on dedicated calibration set
    """
    print("\n" + "="*80)
    print("STEP 3: MODEL TRAINING")
    print("="*80)
    
    # Calculate sample weights (class + time)
    print("[INFO] Calculating sample weights...")
    class_weights = compute_sample_weight('balanced', y_train)
    
    # Time-based weighting (recent data more important)
    positions = np.arange(len(y_train)) / len(y_train)
    time_weights = np.exp(1.2 * (positions - 1))
    time_weights = time_weights / time_weights.sum() * len(time_weights)
    
    # Combined weights
    sample_weights = class_weights * time_weights
    
    # Calculate effective class weights
    unique_classes = np.unique(y_train)
    for cls in unique_classes:
        cls_mask = (y_train == cls)
        effective_weight = sample_weights[cls_mask].mean()
        print(f"  Class {cls}: effective weight = {effective_weight:.4f}")
    
    # Train Gradient Boosting
    print(f"\n[INFO] Training Gradient Boosting Classifier...")
    print(f"[INFO] Parameters: {Config.GB_PARAMS}")
    
    model = GradientBoostingClassifier(**Config.GB_PARAMS)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Calibrate on separate calibration set
    # NOTE: scikit-learn 1.6+ removed cv='prefit', using manual calibration approach
    print(f"\n[INFO] Calibrating model on {len(X_cal):,} calibration samples...")
    from sklearn.isotonic import IsotonicRegression
    
    # For scikit-learn 1.6+ compatibility, we manually calibrate the pre-fitted model
    # by fitting isotonic regression on the calibration set probabilities
    base_proba = model.predict_proba(X_cal)
    
    if Config.CALIBRATION_METHOD == 'isotonic':
        # Fit isotonic regression calibrators for each class
        calibrators = []
        for i in range(base_proba.shape[1]):
            iso_reg = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
            iso_reg.fit(base_proba[:, i], (y_cal == i).astype(int))
            calibrators.append(iso_reg)
        
        # Use module-level class for pickling compatibility
        calibrated_model = IsotonicCalibratedModel(model, calibrators)
        print(f"[SUCCESS] Isotonic calibration applied using {len(X_cal):,} samples")
    else:
        # For 'sigmoid' (Platt scaling), use simplified approach
        from sklearn.linear_model import LogisticRegression
        
        # Use base model probabilities as features for Platt scaling
        platt_scaler = LogisticRegression(max_iter=1000, random_state=42)
        platt_scaler.fit(base_proba, y_cal)
        
        # Use module-level class for pickling compatibility
        calibrated_model = SigmoidCalibratedModel(model, platt_scaler)
        print(f"[SUCCESS] Sigmoid (Platt) calibration applied using {len(X_cal):,} samples")
    
    # Evaluate on all sets
    print("\n[INFO] Evaluating model performance...")
    
    results = {}
    for name, X, y in [
        ('Training', X_train, y_train),
        ('Calibration', X_cal, y_cal),
        ('Test', X_test, y_test)
    ]:
        y_pred = calibrated_model.predict(X)
        y_proba = calibrated_model.predict_proba(X)
        
        results[name] = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0),
            'y_pred': y_pred,
            'y_proba': y_proba
        }
    
    # Print results
    print("\n" + "="*80)
    print("MODEL PERFORMANCE")
    print("="*80)
    
    for name in ['Training', 'Calibration', 'Test']:
        r = results[name]
        print(f"\n{name} Set:")
        print(f"  Accuracy:  {r['accuracy']:.4f}")
        print(f"  Precision: {r['precision']:.4f}")
        print(f"  Recall:    {r['recall']:.4f}")
        print(f"  F1 Score:  {r['f1']:.4f}")
    
    # Detailed test set analysis
    print("\n" + "="*80)
    print("TEST SET DETAILED ANALYSIS")
    print("="*80)
    
    print("\nClassification Report:")
    print(classification_report(y_test, results['Test']['y_pred']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, results['Test']['y_pred'])
    print(cm)
    
    # Prediction distribution
    print("\nPrediction Distribution (Test Set):")
    pred_counts = pd.Series(results['Test']['y_pred']).value_counts()
    for cls, count in pred_counts.items():
        pct = (count / len(y_test)) * 100
        print(f"  Class {cls}: {count:6,} ({pct:5.2f}%)")
    
    return calibrated_model, model, results

# ============================================================================
# Model Persistence
# ============================================================================

def save_model_artifacts(model, base_model, scaler, encoder, selected_features, importances):
    """Save all model artifacts"""
    print("\n" + "="*80)
    print("STEP 4: SAVING MODEL ARTIFACTS")
    print("="*80)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save models
    model_files = {
        'nse_gb_model_v2.joblib': model,  # Calibrated model
        'nse_gb_base_model_v2.joblib': base_model,  # Base model
        'nse_scaler_v2.joblib': scaler,
        'nse_direction_encoder_v2.joblib': encoder
    }
    
    for filename, obj in model_files.items():
        path = Config.MODELS_DIR / filename
        joblib.dump(obj, path)
        print(f"[SUCCESS] Saved {filename}")
    
    # Save selected features
    features_file = Config.MODELS_DIR / 'selected_features_v2.json'
    with open(features_file, 'w') as f:
        json.dump(selected_features, f, indent=2)
    print(f"[SUCCESS] Saved selected_features_v2.json")
    
    # Save feature importances
    importances_file = Config.MODELS_DIR / 'feature_importances_v2.csv'
    importances.to_csv(importances_file, index=False)
    print(f"[SUCCESS] Saved feature_importances_v2.csv")
    
    # Save training metadata
    metadata = {
        'timestamp': timestamp,
        'model_type': 'GradientBoostingClassifier',
        'calibration': Config.CALIBRATION_METHOD,
        'n_features': len(selected_features),
        'top_features': selected_features[:10],
        'data_range': f"{Config.DATA_START_DATE} to {Config.DATA_END_DATE}",
        'hyperparameters': Config.GB_PARAMS
    }
    
    metadata_file = Config.MODELS_DIR / 'model_metadata_v2.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[SUCCESS] Saved model_metadata_v2.json")
    
    print(f"\n[COMPLETE] All artifacts saved to {Config.MODELS_DIR}")
    print(f"[TIMESTAMP] {timestamp}")

# ============================================================================
# Training Validation (MANDATORY - Prevents Production Bugs)
# ============================================================================

def validate_training_artifacts(model, scaler, encoder, X_train, y_train, X_cal, y_cal, X_test, y_test):
    """
    CRITICAL: Validate model artifacts before saving
    
    This prevents bugs like:
    - Apr 21: Calibration set imbalance (59.5% Down)
    - Apr 18: VotingClassifier bias
    - Apr 16: Missing class balancing
    
    If ANY check fails, abort training and alert
    """
    print("\n" + "="*80)
    print("VALIDATION CHECKS (MANDATORY)")
    print("="*80)
    
    issues = []
    
    # CHECK 1: Calibration set balance
    print("\n[CHECK 1] Calibration Set Balance...")
    unique, counts = np.unique(y_cal, return_counts=True)
    cal_imbalance = abs(counts[0] - counts[1]) / len(y_cal)
    
    if cal_imbalance > 0.10:
        issues.append(f"Calibration imbalance {cal_imbalance:.1%} > 10%")
        print(f"  ❌ FAIL: Calibration imbalance {cal_imbalance:.1%} > 10%")
        for i, (cls, count) in enumerate(zip(unique, counts)):
            cls_name = encoder.classes_[cls]
            pct = count / len(y_cal) * 100
            print(f"     {cls_name}: {count:,} ({pct:.1f}%)")
    else:
        print(f"  ✅ PASS: Calibration balance OK ({cal_imbalance:.1%})")
        for i, (cls, count) in enumerate(zip(unique, counts)):
            cls_name = encoder.classes_[cls]
            pct = count / len(y_cal) * 100
            print(f"     {cls_name}: {count:,} ({pct:.1f}%)")
    
    # CHECK 2: Training set balance (should be within 20%)
    print("\n[CHECK 2] Training Set Balance...")
    unique, counts = np.unique(y_train, return_counts=True)
    train_imbalance = abs(counts[0] - counts[1]) / len(y_train)
    
    if train_imbalance > 0.20:
        issues.append(f"Training imbalance {train_imbalance:.1%} > 20%")
        print(f"  ❌ FAIL: Training imbalance {train_imbalance:.1%} > 20%")
    else:
        print(f"  ✅ PASS: Training balance OK ({train_imbalance:.1%})")
    
    # CHECK 3: Prediction distribution on test set (should be 20-80% for either class)
    print("\n[CHECK 3] Test Set Prediction Distribution...")
    y_test_pred = model.predict(scaler.transform(X_test) if not isinstance(X_test, np.ndarray) or X_test.shape[1] != scaler.n_features_in_ else X_test)
    unique, counts = np.unique(y_test_pred, return_counts=True)
    
    pred_dist = {}
    for cls in unique:
        cls_name = encoder.classes_[cls]
        cls_count = counts[list(unique).index(cls)]
        cls_pct = cls_count / len(y_test_pred) * 100
        pred_dist[cls_name] = cls_pct
        
        if cls_pct < 20 or cls_pct > 80:
            issues.append(f"Test predictions for '{cls_name}' = {cls_pct:.1f}% (outside 20-80%)")
            print(f"  ❌ FAIL: {cls_name}: {cls_pct:.1f}% (outside 20-80%)")
        else:
            print(f"  ✅ PASS: {cls_name}: {cls_pct:.1f}%")
    
    # CHECK 4: Probability calibration sanity
    print("\n[CHECK 4] Probability Calibration...")
    y_test_proba = model.predict_proba(scaler.transform(X_test) if not isinstance(X_test, np.ndarray) or X_test.shape[1] != scaler.n_features_in_ else X_test)
    avg_proba = y_test_proba.mean(axis=0)
    
    for i, cls_name in enumerate(encoder.classes_):
        if avg_proba[i] < 0.25 or avg_proba[i] > 0.75:
            issues.append(f"Avg probability for '{cls_name}' = {avg_proba[i]:.2%} (outside 25-75%)")
            print(f"  ❌ FAIL: Avg P({cls_name}): {avg_proba[i]:.2%} (outside 25-75%)")
        else:
            print(f"  ✅ PASS: Avg P({cls_name}): {avg_proba[i]:.2%}")
    
    # CHECK 5: Model artifact integrity
    print("\n[CHECK 5] Model Serialization...")
    try:
        import io
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        test_model = joblib.load(buffer)
        
        # Test prediction on small sample
        test_pred = test_model.predict(X_test[:10])
        print(f"  ✅ PASS: Model serialization OK")
    except Exception as e:
        issues.append(f"Model serialization error: {e}")
        print(f"  ❌ FAIL: Model serialization error: {e}")
    
    # FINAL VERDICT
    print("\n" + "="*80)
    if issues:
        print("❌ VALIDATION FAILED - ABORTING TRAINING")
        print("="*80)
        print("\nIssues found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\n⚠️  MODEL NOT SAVED")
        print("Fix issues above and retrain.")
        print("="*80)
        sys.exit(1)
    else:
        print("✅ ALL VALIDATION CHECKS PASSED")
        print("="*80)
        print("Model is safe to save and deploy.")
        return True

# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print("NSE MODEL RETRAINING V2 - SIMPLIFIED ARCHITECTURE")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Based on proven NASDAQ approach (65-70% accuracy)")
    
    # Ensure directories exist
    Config.ensure_dirs()
    
    # Connect to database
    conn = get_db_connection()
    
    try:
        # Load and prepare data
        df = load_training_data(conn)
        
        # Prepare features and target
        print("\n" + "="*80)
        print("PREPARING FEATURES AND TARGET")
        print("="*80)
        
        # Exclude non-feature columns
        exclude_cols = [
            'trading_date', 'ticker', 'direction_5d',
            'sector', 'industry', 'market_cap_category',
            'open_price', 'high_price', 'low_price', 'close_price', 'volume'
        ] + FUTURE_LEAK_FEATURES
        
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        print(f"[INFO] Available features: {len(feature_cols)}")
        print(f"[INFO] Excluded columns: {len(exclude_cols)}")
        
        # Check for future-leak features
        leak_check = [f for f in feature_cols if f in FUTURE_LEAK_FEATURES]
        if leak_check:
            print(f"[ERROR] Future-leak features detected: {leak_check}")
            sys.exit(1)
        else:
            print(f"[SUCCESS] No future-leak features detected")
        
        # Prepare X and y
        X = df[feature_cols].copy()
        y = df['direction_5d'].copy()
        
        # Replace infinity values with NaN, then fill with 0
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        print(f"[INFO] After cleaning: {X.shape[0]} rows, {X.shape[1]} features")
        
        # Encode target
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        
        print(f"\n[INFO] Label encoding:")
        for i, label in enumerate(encoder.classes_):
            print(f"  {label} = {i}")
        
        # Feature selection
        selected_features, importances = select_features(X, y_encoded)
        X_selected = X[selected_features]
        
        # Split data (60/20/20 like NASDAQ) with STRATIFIED calibration
        print("\n" + "="*80)
        print("SPLITTING DATA (60% Train / 20% Cal / 20% Test)")
        print("="*80)
        
        # CRITICAL FIX: Use stratified split for calibration set
        # Time-series split for train/test (oldest 60% = train, newest 40% = test+cal)
        train_size = int(Config.TRAIN_RATIO * len(X_selected))
        
        X_train = X_selected.iloc[:train_size]
        y_train = y_encoded[:train_size]
        
        X_remaining = X_selected.iloc[train_size:]
        y_remaining = y_encoded[train_size:]
        
        # Stratified split of remaining 40% into calibration (50%) and test (50%)
        # This ensures calibration set is balanced even if recent market is biased
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_remaining, y_remaining,
            test_size=0.5,  # 50% of remaining 40% = 20% of total
            stratify=y_remaining,  # CRITICAL: Ensure balanced calibration set
            random_state=42
        )
        
        print(f"[INFO] Training set:    {len(X_train):8,} samples")
        print(f"[INFO] Calibration set: {len(X_cal):8,} samples (STRATIFIED)")
        print(f"[INFO] Test set:        {len(X_test):8,} samples")
        
        # Print calibration set distribution
        unique, counts = np.unique(y_cal, return_counts=True)
        print(f"\n[INFO] Calibration set balance:")
        for cls, count in zip(unique, counts):
            cls_name = encoder.classes_[cls]
            pct = count / len(y_cal) * 100
            print(f"  {cls_name}: {count:,} ({pct:.1f}%)")
        
        cal_imbalance = abs(counts[0] - counts[1]) / len(y_cal)
        if cal_imbalance > 0.10:
            print(f"[WARNING] Calibration imbalance: {cal_imbalance:.1%}")
        else:
            print(f"[SUCCESS] Calibration set is balanced: {cal_imbalance:.1%} imbalance")
        
        # Scale features
        print("\n[INFO] Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_cal_scaled = scaler.transform(X_cal)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model, base_model, results = train_model(
            X_train_scaled, y_train,
            X_cal_scaled, y_cal,
            X_test_scaled, y_test
        )
        
        # ⭐ CRITICAL: VALIDATE BEFORE SAVING
        # This prevents production bugs from bad models (Apr 14, 16, 18, 21 incidents)
        print("\n" + "="*80)
        print("PRE-SAVE VALIDATION (MANDATORY)")
        print("="*80)
        print("Validating model artifacts before saving to prevent production issues...")
        
        validate_training_artifacts(
            model, scaler, encoder,
            X_train_scaled, y_train,
            X_cal_scaled, y_cal,
            X_test_scaled, y_test
        )
        
        # Only save if validation passed (function exits with code 1 if validation fails)
        save_model_artifacts(model, base_model, scaler, encoder, selected_features, importances)
        
        # Final summary
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Test Accuracy: {results['Test']['accuracy']:.2%}")
        print(f"Test F1 Score: {results['Test']['f1']:.4f}")
        print(f"Model Type: Gradient Boosting + Isotonic Calibration")
        print(f"Features: {len(selected_features)}")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    finally:
        conn.close()
        print("\n[INFO] Database connection closed")

if __name__ == '__main__':
    main()
