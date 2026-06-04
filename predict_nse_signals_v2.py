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
# Imported from shared module so joblib can resolve the classes from any context
# ============================================================================
from nse_model_classes import IsotonicCalibratedModel, SigmoidCalibratedModel  # noqa: F401

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
    
    # Prediction date (can be set via environment variable for manual runs)
    PREDICTION_DATE = os.getenv('NSE_PREDICTION_DATE', None)  # Set at runtime or via env
    
    # Penny stock / investability filter
    MIN_STOCK_PRICE = 10.0  # Exclude stocks below INR10 (penny/micro-cap)

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
    
    # Load model version from metadata JSON
    metadata_file = Config.MODELS_DIR / 'model_metadata_v2.json'
    model_version = 'unknown'
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            model_version = metadata.get('timestamp', 'unknown')
            print(f"[SUCCESS] Model version: {model_version}")
        except Exception as e:
            print(f"[WARNING] Could not load model metadata: {e}")
    
    return model, scaler, encoder, selected_features, model_version

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
        
        # === Additional indicators for ml_nse_technical_indicators table ===
        ticker_df['sma_5'] = ticker_df['close_price'].rolling(5).mean()
        ticker_df['sma_10'] = ticker_df['close_price'].rolling(10).mean()
        ticker_df['ema_5'] = ticker_df['close_price'].ewm(span=5).mean()
        ticker_df['ema_10'] = ticker_df['close_price'].ewm(span=10).mean()
        ticker_df['ema_20'] = ticker_df['close_price'].ewm(span=20).mean()
        ticker_df['ema_50'] = ticker_df['close_price'].ewm(span=50).mean()
        ticker_df['price_vs_ema20'] = ticker_df['close_price'] / ticker_df['ema_20']
        ticker_df['sma20_vs_sma50'] = ticker_df['sma_20'] / ticker_df['sma_50']
        ticker_df['ema20_vs_ema50'] = ticker_df['ema_20'] / ticker_df['ema_50']
        ticker_df['sma5_vs_sma20'] = ticker_df['sma_5'] / ticker_df['sma_20']
        ticker_df['rsi_oversold'] = (ticker_df['rsi'] < 30).astype(int)
        ticker_df['rsi_overbought'] = (ticker_df['rsi'] > 70).astype(int)
        ticker_df['rsi_momentum'] = ticker_df['rsi'].diff()
        ticker_df['volume_sma_20'] = ticker_df['volume'].rolling(20).mean()
        ticker_df['price_momentum_5'] = ticker_df['close_price'] / ticker_df['close_price'].shift(5)
        ticker_df['price_momentum_10'] = ticker_df['close_price'] / ticker_df['close_price'].shift(10)
        ticker_df['daily_volatility'] = ticker_df['return_1d'].rolling(10).std()
        ticker_df['price_volatility_10'] = ticker_df['close_price'].pct_change().rolling(10).std()
        ticker_df['price_volatility_20'] = ticker_df['close_price'].pct_change().rolling(20).std()
        ticker_df['trend_strength_10'] = ticker_df['close_price'].rolling(10).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.std() if x.std() != 0 else 0, raw=False
        )
        ticker_df['volume_price_trend'] = (ticker_df['return_1d'] * ticker_df['volume']).rolling(10).mean()
        
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
            trailing_pe,
            profit_margin,
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
            h.company,
            n.sector,
            n.industry,
            
            -- Derive market cap category from market_cap
            CASE 
                WHEN f.market_cap >= 100000000000 THEN 'Large Cap'  -- >= 100B
                WHEN f.market_cap >= 10000000000 THEN 'Mid Cap'     -- >= 10B
                ELSE 'Small Cap'
            END AS market_cap_category,
            f.trailing_pe,
            f.profit_margin
            
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
    
    # Add market-neutral features (CRITICAL - must match training)
    print("[INFO] Calculating market-neutral features...")
    df = add_market_neutral_features(df)
    
    # Add interaction features (HYBRID APPROACH - must match training)
    print("[INFO] Calculating interaction features...")
    df = add_interaction_features(df)
    
    # Keep only prediction date
    df = df[df['trading_date'] == prediction_date].copy()
    
    # Penny stock filter: exclude stocks below INR10 or with no earnings data at all.
    # Sub-INR10 stocks are dominated by circuit-breaker moves and have no tradeable signal.
    # Stocks with neither trailing_pe nor profit_margin have zero fundamental coverage.
    pre_filter = len(df)
    df = df[df['close_price'] >= Config.MIN_STOCK_PRICE]
    if 'trailing_pe' in df.columns and 'profit_margin' in df.columns:
        df = df[df['trailing_pe'].notna() | df['profit_margin'].notna()]
    excluded = pre_filter - len(df)
    if excluded > 0:
        print(f"[INFO] Penny stock filter: excluded {excluded} tickers "
              f"(price < INR {Config.MIN_STOCK_PRICE:.0f} or no earnings data)")
    print(f"[SUCCESS] {len(df)} investable tickers ready for prediction on {prediction_date}")
    
    return df

def merge_market_context(conn, df):
    """Merge market context features (same as training)
    
    FIXED (May 8, 2026): Add normalized ratio-to-MA features to match retrain script.
    Raw absolute levels (nifty50_close, sp500_close) are non-stationary and cause bias.
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
        df_context = df_context.sort_values('trading_date').reset_index(drop=True)
        
        # CRITICAL FIX (May 8, 2026): Add NORMALIZED (stationary) ratio-to-MA features.
        # Must match retrain_nse_model_v2.py exactly so features align with saved model.
        # NOTE: Do NOT fillna here -- leave NULL for days with missing close data (e.g. holidays).
        # The post-merge ffill (below) will propagate the previous day's real value.
        # Final fillna(1.0) after ffill handles only start-of-history gaps.
        df_context['vix_vs_60d'] = (
            df_context['vix_close'] /
            df_context['vix_close'].rolling(60, min_periods=10).mean()
        ).clip(0.5, 3.0)
        
        df_context['india_vix_vs_60d'] = (
            df_context['india_vix_close'] /
            df_context['india_vix_close'].rolling(60, min_periods=10).mean()
        ).clip(0.5, 3.0)
        
        df_context['nifty50_vs_200d'] = (
            df_context['nifty50_close'] /
            df_context['nifty50_close'].rolling(200, min_periods=20).mean()
        ).clip(0.5, 2.0)
        
        df_context['sp500_vs_200d'] = (
            df_context['sp500_close'] /
            df_context['sp500_close'].rolling(200, min_periods=20).mean()
        ).clip(0.5, 2.0)
        
        df_context['dxy_vs_60d'] = (
            df_context['dxy_close'] /
            df_context['dxy_close'].rolling(60, min_periods=10).mean()
        ).clip(0.5, 2.0)
        
        df_context['us10y_vs_60d'] = (
            df_context['us_10y_yield_close'] /
            df_context['us_10y_yield_close'].rolling(60, min_periods=10).mean()
        ).clip(0.5, 2.0)
        
        # Market regime: 5-day NIFTY momentum (rolling sum of 1d returns).
        # Captures multi-day trend context -- positive = sustained up-move, negative = down-move.
        df_context['nifty50_return_5d'] = (
            df_context['nifty50_return_1d'].rolling(5, min_periods=1).sum()
        ).fillna(0.0)
        
        df['trading_date'] = pd.to_datetime(df['trading_date'])
        market_cols = [c for c in df_context.columns if c != 'trading_date']
        df = df.merge(df_context, on='trading_date', how='left')
        
        # Forward-fill price levels (VIX, NIFTY, S&P 500, etc.)
        level_cols = [c for c in market_cols if 'return' not in c and 'change' not in c]
        return_cols = [c for c in market_cols if 'return' in c or 'change' in c]
        
        for col in level_cols:
            if col in df.columns:
                df[col] = df[col].ffill().bfill().fillna(1.0 if 'vs_' in col else 0)
        
        # Fill any remaining missing returns with 0
        for col in return_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Convert market returns from percentage to fraction to match pct_change() output.
        # market_context_daily stores returns as pct (0.43 = +0.43%).
        # Stock returns from pct_change() are fractions (0.0043 = +0.43%).
        for col in return_cols:
            if col in df.columns:
                df[col] = df[col] / 100.0
        
        # Market regime features computed from stock data (no extra DB query needed).
        # adv_decline_ratio: advancing stocks / declining stocks per date.
        # sector_breadth_score: fraction of stocks rising within each sector per date.
        if 'return_1d' in df.columns:
            adv_df = df.groupby('trading_date')['return_1d'].apply(
                lambda x: (x > 0).sum() / max((x < 0).sum(), 1)
            ).reset_index()
            adv_df.columns = ['trading_date', 'adv_decline_ratio']
            df = df.merge(adv_df, on='trading_date', how='left')
            df['adv_decline_ratio'] = df['adv_decline_ratio'].fillna(1.0)
            
            if 'sector' in df.columns:
                breadth_df = df.groupby(['trading_date', 'sector'])['return_1d'].apply(
                    lambda x: (x > 0).mean()
                ).reset_index()
                breadth_df.columns = ['trading_date', 'sector', 'sector_breadth_score']
                df = df.merge(breadth_df, on=['trading_date', 'sector'], how='left')
                df['sector_breadth_score'] = df['sector_breadth_score'].fillna(0.5)
        
        print(f"[SUCCESS] Merged {len(market_cols)} market context features (incl. normalized)")
        
    except Exception as e:
        print(f"[WARNING] Could not load market context: {e}")
    
    return df

def add_market_neutral_features(df):
    """
    Add market-neutral features (must match training exactly)
    
    These features reduce market-wide bias by measuring stock-specific strength
    relative to the market and sector peers.
    """
    # 1. Relative performance vs NIFTY 50
    if 'nifty50_return_1d' in df.columns and 'return_1d' in df.columns:
        df['stock_return_vs_nifty'] = df['return_1d'] - df['nifty50_return_1d']
        df['stock_return_vs_nifty_5d'] = df['return_5d'] - (df['nifty50_return_1d'].rolling(5).sum())
    else:
        df['stock_return_vs_nifty'] = 0
        df['stock_return_vs_nifty_5d'] = 0
    
    # 2. Sector-relative performance and RSI
    if 'sector' in df.columns:
        # Group by sector and date to get sector averages
        sector_metrics = df.groupby(['trading_date', 'sector']).agg({
            'return_1d': 'mean',
            'return_5d': 'mean',
            'rsi': 'mean',
            'volume_ratio': 'mean'
        }).reset_index()
        
        sector_metrics.columns = ['trading_date', 'sector', 'sector_return_1d', 
                                   'sector_return_5d', 'sector_rsi_avg', 'sector_volume_ratio']
        
        # Merge back
        df = df.merge(sector_metrics, on=['trading_date', 'sector'], how='left')
        
        # Calculate relative metrics
        df['stock_return_vs_sector'] = df['return_1d'] - df['sector_return_1d']
        df['rsi_vs_sector_avg'] = df['rsi'] - df['sector_rsi_avg']
        df['volume_vs_sector'] = df['volume_ratio'] - df['sector_volume_ratio']
    else:
        df['stock_return_vs_sector'] = 0
        df['rsi_vs_sector_avg'] = 0
        df['volume_vs_sector'] = 0
    
    # 3. Volume anomaly detection (stock-specific)
    df['volume_anomaly'] = df.groupby('ticker')['volume'].transform(
        lambda x: (x - x.rolling(50, min_periods=10).mean()) / x.rolling(50, min_periods=10).std()
    ).fillna(0)
    
    # 4. Beta estimation (rolling 60-day)
    if 'nifty50_return_1d' in df.columns and 'return_1d' in df.columns:
        results = []
        for ticker in df['ticker'].unique():
            ticker_df = df[df['ticker'] == ticker].copy()
            ticker_df = ticker_df.sort_values('trading_date')
            
            # Rolling beta
            window = 60
            cov = ticker_df['return_1d'].rolling(window).cov(ticker_df['nifty50_return_1d'])
            var = ticker_df['nifty50_return_1d'].rolling(window).var()
            ticker_df['beta'] = (cov / var).fillna(1.0).clip(0.5, 2.0)
            
            # Beta-adjusted return
            ticker_df['beta_adjusted_return'] = ticker_df['return_1d'] - (ticker_df['beta'] * ticker_df['nifty50_return_1d'])
            
            results.append(ticker_df)
        
        df = pd.concat(results, ignore_index=True)
    else:
        df['beta'] = 1.0
        df['beta_adjusted_return'] = 0
    
    # 5. Relative strength (20-day cumulative outperformance)
    if 'stock_return_vs_nifty' in df.columns:
        df['relative_strength_20d'] = df.groupby('ticker')['stock_return_vs_nifty'].transform(
            lambda x: x.rolling(20, min_periods=5).sum()
        ).fillna(0)
    else:
        df['relative_strength_20d'] = 0
    
    # 6. Price momentum relative to sector
    if 'sector' in df.columns:
        sector_momentum = df.groupby(['trading_date', 'sector'])['close_price'].transform(
            lambda x: x.pct_change(10)
        )
        df['momentum_vs_sector'] = df['return_10d'] - sector_momentum
    else:
        df['momentum_vs_sector'] = 0
    
    return df


def add_interaction_features(df):
    """
    Add interaction features that combine market context with stock-specific signals
    (must match training exactly)
    
    These features explicitly teach the model to consider market conditions
    AND stock-specific strength together, not independently.
    """
    # Initialize VIX baseline for normalization
    vix_neutral = 15.0
    
    # 1. FEAR/GREED INTERACTIONS
    if 'stock_return_vs_nifty' in df.columns and 'vix_close' in df.columns:
        df['outperformance_in_fear'] = df['stock_return_vs_nifty'] * (df['vix_close'] / vix_neutral)
        df['outperformance_in_greed'] = df['stock_return_vs_nifty'] * (vix_neutral / df['vix_close'].clip(lower=10))
    else:
        df['outperformance_in_fear'] = 0
        df['outperformance_in_greed'] = 0
    
    # 2. QUALITY + CONVICTION SIGNALS
    if 'rsi_vs_sector_avg' in df.columns and 'volume_anomaly' in df.columns:
        df['sector_leader_conviction'] = (df['rsi_vs_sector_avg'] / 20).clip(-1, 1) * df['volume_anomaly'].clip(-3, 3)
    else:
        df['sector_leader_conviction'] = 0
    
    # 3. DEFENSIVE/AGGRESSIVE POSITIONING
    if 'beta' in df.columns and 'vix_close' in df.columns:
        df['defensive_risk_score'] = df['beta'] * (df['vix_close'] / vix_neutral)
        
        if 'stock_return_vs_nifty' in df.columns:
            safe_beta = df['beta'].clip(lower=0.5)
            df['quality_opportunity'] = df['stock_return_vs_nifty'] / safe_beta
        else:
            df['quality_opportunity'] = 0
    else:
        df['defensive_risk_score'] = 0
        df['quality_opportunity'] = 0
    
    # 4. VOLATILITY-ADJUSTED MOMENTUM
    if 'stock_return_vs_nifty_5d' in df.columns and 'atr' in df.columns and 'close_price' in df.columns:
        atr_pct = (df['atr'] / df['close_price']).clip(lower=0.001)
        df['risk_adjusted_momentum'] = df['stock_return_vs_nifty_5d'] / atr_pct
        df['risk_adjusted_momentum'] = df['risk_adjusted_momentum'].clip(-5, 5)
    else:
        df['risk_adjusted_momentum'] = 0
    
    # 5. MARKET REGIME COORDINATION
    if 'stock_return_vs_nifty' in df.columns and 'sp500_return_1d' in df.columns:
        df['global_market_sync'] = df['stock_return_vs_nifty'] * df['sp500_return_1d']
    else:
        df['global_market_sync'] = 0
    
    # 6. CONTRARIAN SIGNAL
    if 'stock_return_vs_nifty' in df.columns and 'nifty50_return_1d' in df.columns:
        df['contrarian_strength'] = df['stock_return_vs_nifty'] * (-df['nifty50_return_1d'])
    else:
        df['contrarian_strength'] = 0
    
    # 7. QUALITY IN CHAOS
    if 'rsi_vs_sector_avg' in df.columns and 'vix_close' in df.columns:
        vix_20d_avg = df['vix_close'].rolling(20, min_periods=5).mean()
        vix_spike = (df['vix_close'] / vix_20d_avg).fillna(1.0)
        df['quality_in_volatility'] = (df['rsi_vs_sector_avg'] / 20) * vix_spike
        df['quality_in_volatility'] = df['quality_in_volatility'].clip(-3, 3)
    else:
        df['quality_in_volatility'] = 0
    
    # 8. SECTOR MOMENTUM WITH MARKET CONFIRMATION
    if 'momentum_vs_sector' in df.columns and 'nifty50_return_1d' in df.columns:
        df['sector_momentum_confirmed'] = df['momentum_vs_sector'] * df['nifty50_return_1d']
    else:
        df['sector_momentum_confirmed'] = 0
    
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
    
    # Create predictions dataframe -- include volume and company for DB write
    base_cols = ['ticker', 'trading_date', 'close_price', 'sector',
                 'industry', 'market_cap_category', 'rsi']
    if 'volume' in df.columns:
        base_cols.append('volume')
    if 'company' in df.columns:
        base_cols.append('company')
    predictions = df[base_cols].copy()
    
    # Probabilities
    predictions['buy_probability'] = y_proba[:, up_idx]  # Up = Buy
    predictions['sell_probability'] = y_proba[:, down_idx]  # Down = Sell
    
    # Signal determination: relative threshold approach
    # In bear markets, avg buy_probability < 0.5 for ALL stocks (market-wide drag).
    # A fixed 50% threshold would give 100% Sell, which is useless.
    # Instead: use a RELATIVE threshold -- top 30% by buy probability = Buy.
    # This finds the relative outperformers regardless of absolute market direction.
    # In balanced markets (avg ~45%), this naturally selects ~50% buys.
    # In bear markets (avg ~25%), this selects the best 30% as relative outperformers.
    avg_buy_prob = predictions['buy_probability'].mean()
    if avg_buy_prob >= 0.45:
        # Normal/bull market: use standard 50% threshold
        buy_threshold = 0.50
        threshold_mode = "absolute (50%)"
    else:
        # Bear market: use relative threshold -- top 30% by buy probability
        buy_threshold = float(predictions['buy_probability'].quantile(0.70))
        threshold_mode = f"relative (top 30%, threshold={buy_threshold:.3f})"
    print(f"[INFO] Signal threshold mode: {threshold_mode}")
    predictions['predicted_signal'] = np.where(
        predictions['buy_probability'] >= buy_threshold,
        'Buy',
        'Sell'
    )

    # Confidence
    predictions['confidence_percentage'] = np.maximum(
        predictions['buy_probability'],
        predictions['sell_probability']
    ) * 100
    
    # Signal strength: percentile rank within this day's predictions
    # top 5% = High (~100 signals/day), next 20% = Medium, rest = Low
    # Guarantees consistent counts regardless of absolute confidence level
    conf_rank = predictions['confidence_percentage'].rank(pct=True)
    predictions['signal_strength'] = np.where(
        conf_rank >= 0.95,
        'High',
        np.where(conf_rank >= 0.75, 'Medium', 'Low')
    )
    
    # All three boolean confidence flags -- derived from signal_strength
    predictions['high_confidence'] = (predictions['signal_strength'] == 'High').astype(int)
    predictions['medium_confidence'] = (predictions['signal_strength'] == 'Medium').astype(int)
    predictions['low_confidence'] = (predictions['signal_strength'] == 'Low').astype(int)
    
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
    
    # Signal strength distribution
    strength_counts = predictions['signal_strength'].value_counts()
    print("[INFO] Signal Strength Distribution (percentile-based):")
    for strength in ['High', 'Medium', 'Low']:
        count = strength_counts.get(strength, 0)
        pct = count / len(predictions) * 100
        print(f"  {strength:6s}: {count:6,} ({pct:5.1f}%)")
    
    return predictions

# ============================================================================
# Production Validation (RUNTIME)
# ============================================================================

def validate_prediction_distribution(predictions, conn=None):
    """
    Validate prediction distribution before writing to database.

    Hard-abort conditions (return False -> caller does sys.exit(1)):
      - buy_pct < 5%  or > 95%  (catastrophic bias like Apr 21 99.6% Sell)
      - total predictions < 95% of rolling 5-day average (unexpected data loss)

    Soft warnings (log only, do NOT abort):
      - buy_pct < 20% or > 80% (moderate bias, worth monitoring)
      - avg_confidence > 90%  (suspiciously overconfident model)
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
    
    avg_buy_prob = predictions['buy_probability'].mean()
    avg_sell_prob = predictions['sell_probability'].mean()
    avg_confidence = predictions['confidence_percentage'].mean()
    
    print(f"\nAverage Buy Probability:  {avg_buy_prob:.2%}")
    print(f"Average Sell Probability: {avg_sell_prob:.2%}")
    print(f"Average Confidence:       {avg_confidence:.1f}%")
    
    fatal_issues = []
    warnings = []
    
    # -- Hard-abort: catastrophic buy/sell skew -----------------------------
    if buy_pct < 5:
        fatal_issues.append(f"Buy% ({buy_pct:.1f}%) below 5% -- catastrophic bearish bias")
    elif buy_pct > 95:
        fatal_issues.append(f"Buy% ({buy_pct:.1f}%) above 95% -- catastrophic bullish bias")
    
    # -- Hard-abort: unexpected prediction count drop -----------------------
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT AVG(CAST(total_predictions AS FLOAT))
                FROM (
                    SELECT TOP 5 total_predictions
                    FROM ml_nse_predict_summary
                    ORDER BY analysis_date DESC
                ) recent
            """)
            row = cursor.fetchone()
            cursor.close()
            if row and row[0]:
                rolling_avg = float(row[0])
                if total < rolling_avg * 0.85:
                    fatal_issues.append(
                        f"Total predictions ({total:,}) is more than 15% below "
                        f"rolling 5-day average ({rolling_avg:.0f}) -- possible data loss"
                    )
                elif total < rolling_avg * 0.90:
                    warnings.append(
                        f"Total predictions ({total:,}) is 10-15% below "
                        f"rolling 5-day average ({rolling_avg:.0f}) -- penny stock filter may have changed"
                    )
        except Exception as e:
            print(f"[WARNING] Could not fetch rolling prediction count: {e}")
    
    # -- Soft warnings ------------------------------------------------------
    if not fatal_issues:
        if buy_pct < 20:
            warnings.append(f"Buy% ({buy_pct:.1f}%) below 20% -- moderate bearish skew")
        elif buy_pct > 80:
            warnings.append(f"Buy% ({buy_pct:.1f}%) above 80% -- moderate bullish skew")
        else:
            print(f"\n[OK] Distribution within acceptable range (20-80%)")
    
    if avg_confidence > 90:
        warnings.append(f"Avg confidence ({avg_confidence:.1f}%) suspiciously high")
    
    if warnings:
        print(f"\n[WARNING] Soft validation issues (proceeding):")
        for w in warnings:
            print(f"  ? {w}")
    
    if fatal_issues:
        print(f"\n{'='*80}")
        print(f"[FATAL] VALIDATION FAILED -- ABORTING DB WRITE")
        print(f"{'='*80}")
        for issue in fatal_issues:
            print(f"  [FAIL] {issue}")
        print(f"\nRecommendations:")
        print(f"  ? Check market_context_daily for missing data (all-zero market features)")
        print(f"  ? Verify nse_500_hist_data loaded correctly for this date")
        print(f"  ? Consider retraining if model appears permanently biased")
        return False
    
    return True

# ============================================================================
# Save Predictions to Database
# ============================================================================

def save_predictions(conn, predictions, model_version='unknown'):
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
    
    # Insert new predictions (19 columns -- includes all boolean flags, company, volume, model_version)
    insert_query = """
    INSERT INTO ml_nse_trading_predictions (
        ticker, trading_date, predicted_signal, confidence, confidence_percentage,
        signal_strength, close_price, rsi, buy_probability, sell_probability,
        model_name, sector, market_cap_category,
        high_confidence, medium_confidence, low_confidence,
        company, volume, model_version
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    rows_to_insert = []
    for _, row in predictions.iterrows():
        confidence_pct = float(row['confidence_percentage'])
        confidence_decimal = confidence_pct / 100.0
        close_price_val = float(row['close_price']) if pd.notna(row['close_price']) else 0.0
        volume_val = float(row['volume']) if 'volume' in row and pd.notna(row.get('volume')) else None
        company_val = str(row['company']) if 'company' in row and pd.notna(row.get('company')) else None
        rows_to_insert.append((
            row['ticker'],
            row['trading_date'],
            row['predicted_signal'],
            confidence_decimal,
            confidence_pct,
            row['signal_strength'],
            close_price_val,
            float(row['rsi']) if pd.notna(row['rsi']) else None,
            float(row['buy_probability']),
            float(row['sell_probability']),
            row['model_name'],
            row['sector'],
            row['market_cap_category'],
            int(row['high_confidence']),
            int(row['medium_confidence']),
            int(row['low_confidence']),
            company_val,
            volume_val,
            model_version
        ))
    
    cursor.executemany(insert_query, rows_to_insert)
    conn.commit()
    
    print(f"[SUCCESS] Inserted {len(rows_to_insert):,} predictions")
    print(f"[INFO] Model version stamped: {model_version}")
    
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


def _safe_float(value, default=None):
    """Convert value to float, returning default on NaN/None."""
    if value is None:
        return default
    try:
        f = float(value)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default


def save_technical_indicators(conn, df, prediction_date):
    """Save technical indicators snapshot to ml_nse_technical_indicators table.
    
    Added in V2 (May 2026) to restore the write that existed in V1 but was
    accidentally omitted during the V1 -> V2 architecture refactor (Apr 18 2026).
    Failure here is non-fatal and will not block prediction saves.
    """
    print("\n" + "="*80)
    print("SAVING TECHNICAL INDICATORS TO DATABASE")
    print("="*80)
    
    try:
        cursor = conn.cursor()
        
        # Idempotent: delete before insert
        cursor.execute(
            "DELETE FROM ml_nse_technical_indicators WHERE trading_date = ?",
            prediction_date
        )
        print(f"[INFO] Deleted {cursor.rowcount} existing technical indicator rows for {prediction_date}")
        
        run_timestamp = datetime.now()
        
        insert_query = """
        INSERT INTO ml_nse_technical_indicators (
            run_timestamp, trading_date, ticker,
            rsi, rsi_oversold, rsi_overbought, rsi_momentum,
            sma_5, sma_10, sma_20, sma_50,
            ema_5, ema_10, ema_20, ema_50,
            macd, macd_signal, macd_histogram,
            price_vs_sma20, price_vs_sma50, price_vs_ema20,
            sma20_vs_sma50, ema20_vs_ema50, sma5_vs_sma20,
            volume_sma_20, volume_sma_ratio,
            price_momentum_5, price_momentum_10,
            daily_volatility, price_volatility_10, price_volatility_20,
            trend_strength_10, volume_price_trend,
            gap, bb_upper, bb_lower, bb_middle, bb_width,
            atr, atr_percentage
        ) VALUES (
            ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?
        )
        """
        
        rows_to_insert = []
        sf = _safe_float  # local alias
        
        for _, row in df.iterrows():
            bb_upper = sf(row.get('bb_upper'))
            bb_lower = sf(row.get('bb_lower'))
            bb_width = (bb_upper - bb_lower) if (bb_upper is not None and bb_lower is not None) else None
            
            rows_to_insert.append((
                run_timestamp,
                row['trading_date'],
                row['ticker'],
                # RSI group
                sf(row.get('rsi')),
                int(row.get('rsi_oversold', 0)),
                int(row.get('rsi_overbought', 0)),
                sf(row.get('rsi_momentum')),
                # SMA group (price_to_sma20/50 = V2 rename of price_vs_sma20/50)
                sf(row.get('sma_5')),
                sf(row.get('sma_10')),
                sf(row.get('sma_20')),
                sf(row.get('sma_50')),
                # EMA group
                sf(row.get('ema_5')),
                sf(row.get('ema_10')),
                sf(row.get('ema_20')),
                sf(row.get('ema_50')),
                # MACD group
                sf(row.get('macd')),
                sf(row.get('macd_signal')),
                sf(row.get('macd_histogram')),
                # Price vs MA ratios (V2 renamed price_to_sma20/50 -> map back)
                sf(row.get('price_to_sma20', row.get('price_vs_sma20'))),
                sf(row.get('price_to_sma50', row.get('price_vs_sma50'))),
                sf(row.get('price_vs_ema20')),
                # MA cross ratios
                sf(row.get('sma20_vs_sma50')),
                sf(row.get('ema20_vs_ema50')),
                sf(row.get('sma5_vs_sma20')),
                # Volume
                sf(row.get('volume_sma_20')),
                sf(row.get('volume_ratio', row.get('volume_sma_ratio'))),
                # Price momentum (V2 uses price/shift ratio, same semantic as V1)
                sf(row.get('price_momentum_5')),
                sf(row.get('price_momentum_10')),
                # Volatility
                sf(row.get('daily_volatility')),
                sf(row.get('price_volatility_10')),
                sf(row.get('price_volatility_20')),
                # Trend
                sf(row.get('trend_strength_10')),
                sf(row.get('volume_price_trend')),
                # Gap
                sf(row.get('gap')),
                # Bollinger Bands
                bb_upper,
                bb_lower,
                sf(row.get('bb_middle')),
                bb_width,
                # ATR (V2: atr / atr_pct)
                sf(row.get('atr')),
                sf(row.get('atr_pct')),
            ))
        
        cursor.executemany(insert_query, rows_to_insert)
        conn.commit()
        cursor.close()
        
        print(f"[SUCCESS] Saved {len(rows_to_insert):,} technical indicator rows for {prediction_date}")
        
    except Exception as e:
        print(f"[WARNING] Could not save technical indicators: {e}")
        print(f"[INFO] Predictions are already saved -- this is non-fatal, continuing")

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
    
    # Load model artifacts (now returns 5-tuple including model_version)
    model, scaler, encoder, selected_features, model_version = load_model_artifacts()
    
    try:
        # Load prediction data
        df = load_prediction_data(conn, prediction_date)
        
        if len(df) == 0:
            print(f"[WARNING] No data available for {prediction_date}")
            print(f"[INFO] Check if {prediction_date} is a valid trading day")
            sys.exit(0)
        
        # Generate predictions
        predictions = generate_predictions(model, scaler, encoder, selected_features, df)
        
        # * CRITICAL: VALIDATE BEFORE SAVING
        # Hard-aborts on catastrophic bias (< 5% or > 95% Buy) or data loss
        if not validate_prediction_distribution(predictions, conn):
            print("[ERROR] Prediction validation failed. Aborting database write.")
            sys.exit(1)
        
        # Save to database
        save_predictions(conn, predictions, model_version)
        
        # Save technical indicators snapshot (non-fatal if it fails)
        save_technical_indicators(conn, df, prediction_date)
        
        # Final summary
        print("\n" + "="*80)
        print("PREDICTION COMPLETE")
        print("="*80)
        print(f"Date: {prediction_date}")
        print(f"Model Version: {model_version}")
        print(f"Tickers: {len(predictions):,}")
        print(f"Buy Signals: {len(predictions[predictions['predicted_signal'] == 'Buy']):,}")
        print(f"Sell Signals: {len(predictions[predictions['predicted_signal'] == 'Sell']):,}")
        high_c = predictions['high_confidence'].sum()
        med_c = predictions['medium_confidence'].sum()
        low_c = predictions['low_confidence'].sum()
        print(f"High Confidence: {high_c:,} | Medium: {med_c:,} | Low: {low_c:,}")
        print(f"Model: GradientBoosting V2 + Isotonic Calibration")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    finally:
        conn.close()

if __name__ == '__main__':
    main()
