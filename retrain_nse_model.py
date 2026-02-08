"""
NSE 500 Model Retraining Script

This script trains ML models specifically on NSE 500 data to predict:
1. Next-day price direction (classification: Up/Down)
2. Next-day price change magnitude (regression)

Key improvements over the previous approach:
- Trains on NSE data (not NASDAQ) for proper market alignment
- Uses proper target: next-day price direction instead of RSI signals
- Enriches features with Bollinger Bands, Stochastic, ATR, MACD from database views
- Trains ensemble of models (Random Forest, Gradient Boosting, Extra Trees, Ridge)
- Uses walk-forward time-series validation
- Saves NSE-specific model artifacts

Usage:
    python retrain_nse_model.py                    # Full retrain with latest data
    python retrain_nse_model.py --quick            # Quick retrain (skip EDA)
    python retrain_nse_model.py --backup-old       # Backup current model before retrain
    python retrain_nse_model.py --days-back 365    # Train on last N days of data
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import warnings
import joblib
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Configure UTF-8 encoding for Windows console compatibility
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

# ML imports
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, RandomForestRegressor,
    GradientBoostingRegressor, ExtraTreesRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.calibration import CalibratedClassifierCV

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection

warnings.filterwarnings('ignore')


class NSEModelRetrainer:
    """NSE 500 specific model retraining system"""
    
    def __init__(self, backup_old=False, quick_mode=False, days_back=None):
        self.backup_old = backup_old
        self.quick_mode = quick_mode
        self.days_back = days_back or 730  # Default: 2 years of data
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Database connection
        self.db = SQLServerConnection()
        
        # Paths - NSE specific
        self.data_dir = Path('data')
        self.nse_model_dir = Path('data/nse_models')
        self.reports_dir = Path('reports')
        self.backup_dir = Path('data/nse_backups') if backup_old else None
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.nse_model_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        if self.backup_dir:
            self.backup_dir.mkdir(exist_ok=True)
        
        # Feature columns for the final model (set during training)
        self.feature_columns = []
        
        # Classification model feature columns (all technical data points)
        self.clf_feature_columns = [
            # --- Raw OHLCV ---
            'open_price', 'high_price', 'low_price', 'close_price', 'volume',
            # --- RSI (from nse_500_RSI_calculation) ---
            'RSI', 'rsi_oversold', 'rsi_overbought', 'rsi_momentum',
            # --- Calculated price features ---
            'daily_volatility', 'daily_return', 'volume_millions',
            'price_range', 'price_position', 'gap', 'volume_price_trend',
            # --- Moving Averages (calculated) ---
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_5', 'ema_10', 'ema_20', 'ema_50',
            # --- MACD (from nse_500_macd + calculated) ---
            'macd', 'macd_signal', 'macd_histogram',
            # --- Price vs MA ratios ---
            'price_vs_sma20', 'price_vs_sma50', 'price_vs_ema20',
            'sma20_vs_sma50', 'ema20_vs_ema50', 'sma5_vs_sma20',
            # --- Volume indicators ---
            'volume_sma_20', 'volume_sma_ratio', 'vol_change_ratio',
            # --- Momentum ---
            'price_momentum_5', 'price_momentum_10',
            # --- Volatility ---
            'price_volatility_10', 'price_volatility_20', 'trend_strength_10',
            # --- Calendar ---
            'day_of_week', 'month',
            # --- Bollinger Bands (from nse_500_bollingerband) ---
            'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
            # --- Stochastic (from nse_500_stochastic) ---
            'stoch_k', 'stoch_d', 'stoch_momentum',
            # --- ATR (from nse_500_atr) ---
            'atr_14', 'atr_pct',
            # --- Multi-day returns ---
            'return_1d', 'return_3d', 'return_5d',
            # --- Candlestick shape ---
            'high_low_ratio', 'close_open_ratio', 'upper_shadow', 'lower_shadow',
            # --- Fibonacci (from nse_500_fibonacci) ---
            'fib_distance_pct', 'fib_signal_strength',
            # --- Support/Resistance (from nse_500_support_resistance) ---
            'sr_distance_to_support_pct', 'sr_distance_to_resistance_pct',
            'sr_pivot_position', 'sr_signal_strength',
            # --- Candlestick Patterns (from nse_500_patterns) ---
            'pattern_signal_strength',
            'has_doji', 'has_hammer', 'has_shooting_star',
            'has_bullish_engulfing', 'has_bearish_engulfing',
            'has_morning_star', 'has_evening_star',
            # --- SMA Signals (from nse_500_sma_signals) ---
            'sma_200', 'sma_100',
            'price_vs_sma100', 'price_vs_sma200',
            'sma_200_flag', 'sma_100_flag', 'sma_50_flag', 'sma_20_flag',
            # --- MACD Signal (from nse_500_macd_signals) ---
            'macd_signal_strength',
        ]
        
        # Signal encoding maps (categorical -> numeric)
        self.signal_strength_map = {
            'STRONG_BUY': 2, 'BUY': 1, 'BUY_FIB_500': 1, 'BUY_FIB_382': 1,
            'BUY_FIB_236': 1, 'BUY_BULLISH_CROSS': 1,
            'NEUTRAL': 0, 'NEUTRAL_WAIT': 0, 'No Signal': 0,
            'SELL': -1, 'SELL_BEARISH_CROSS': -1,
            'STRONG_SELL': -2,
            'NEAR_SUPPORT_BUY': 1, 'BULLISH_ZONE': 1,
            'NEAR_RESISTANCE_SELL': -1, 'BEARISH_ZONE': -1,
            'Bullish Crossover': 1, 'Bearish Crossover': -1,
        }
        
        self.position_map = {
            'ABOVE_PIVOT': 1, 'BELOW_PIVOT': -1,
            'Above': 1, 'Below': -1,
            'BULLISH': 1, 'BEARISH': -1,
        }
    
    def backup_existing_model(self):
        """Backup existing NSE model files before retraining"""
        if not self.backup_old:
            return
        
        print("[BACKUP] Backing up existing NSE model artifacts...")
        
        nse_model_files = list(self.nse_model_dir.glob('*.joblib')) + list(self.nse_model_dir.glob('*.pkl'))
        
        backup_count = 0
        for src_path in nse_model_files:
            backup_path = self.backup_dir / f"{self.timestamp}_{src_path.name}"
            shutil.copy2(src_path, backup_path)
            backup_count += 1
            print(f"  [OK] Backed up {src_path.name}")
        
        print(f"[BACKUP] Complete: {backup_count} files backed up to {self.backup_dir}")
    
    def load_nse_training_data(self):
        """Load NSE 500 historical data with enriched features from database views
        
        TIMEZONE CONTEXT:
        - This system runs in EST, NSE operates in IST (IST = EST + 10.5 hours)
        - NSE closes at 3:30 PM IST = ~5:00 AM EST
        - When this runs at 7 AM EST on weekdays, today's NSE data is already available
        - Training includes all available data up to the current date
        """
        print("[DATA] Loading NSE 500 training data from SQL Server...")
        
        # Main historical data query - loads from NSE tables (NOT NASDAQ!)
        # GETDATE() is the SQL Server time - data includes up to the latest available trading day
        hist_query = f"""
        SELECT 
            h.trading_date,
            h.ticker,
            h.company,
            CAST(h.open_price AS FLOAT) as open_price,
            CAST(h.high_price AS FLOAT) as high_price,
            CAST(h.low_price AS FLOAT) as low_price,
            CAST(h.close_price AS FLOAT) as close_price,
            CAST(h.volume AS BIGINT) as volume
        FROM dbo.nse_500_hist_data h
        WHERE h.trading_date >= DATEADD(day, -{self.days_back}, CAST(GETDATE() AS DATE))
            AND h.trading_date <= CAST(GETDATE() AS DATE)
            AND ISNUMERIC(h.close_price) = 1
            AND ISNUMERIC(h.open_price) = 1
            AND CAST(h.close_price AS FLOAT) > 0
            AND CAST(h.volume AS BIGINT) > 0
        ORDER BY h.ticker, h.trading_date
        """
        
        try:
            df = self.db.execute_query(hist_query)
            print(f"[OK] Historical data loaded: {df.shape[0]:,} records, "
                  f"{df['ticker'].nunique()} tickers, "
                  f"{df['trading_date'].min()} to {df['trading_date'].max()}")
            
            if df.empty:
                raise ValueError("No NSE historical data found")
            
            return df
            
        except Exception as e:
            print(f"[ERROR] Error loading NSE data: {e}")
            raise
    
    def load_enriched_features(self):
        """Load enriched technical features from database views"""
        print("[DATA] Loading enriched features from database views...")
        
        enriched_data = {}
        
        # Load RSI data
        try:
            rsi_query = f"""
            SELECT ticker, trading_date, RSI
            FROM dbo.nse_500_RSI_calculation
            WHERE trading_date >= DATEADD(day, -{self.days_back}, CAST(GETDATE() AS DATE))
            """
            enriched_data['rsi'] = self.db.execute_query(rsi_query)
            print(f"  [OK] RSI: {len(enriched_data['rsi']):,} records")
        except Exception as e:
            print(f"  [WARN] RSI load failed: {e}")
        
        # Load Bollinger Bands
        try:
            bb_query = f"""
            SELECT ticker, trading_date, 
                   CAST(close_price AS FLOAT) as bb_close,
                   CAST(SMA_20 AS FLOAT) as bb_sma20,
                   CAST(Upper_Band AS FLOAT) as bb_upper,
                   CAST(Lower_Band AS FLOAT) as bb_lower
            FROM dbo.nse_500_bollingerband
            WHERE trading_date >= DATEADD(day, -{self.days_back}, CAST(GETDATE() AS DATE))
            """
            enriched_data['bb'] = self.db.execute_query(bb_query)
            print(f"  [OK] Bollinger Bands: {len(enriched_data['bb']):,} records")
        except Exception as e:
            print(f"  [WARN] Bollinger Bands load failed: {e}")
        
        # Load Stochastic
        try:
            stoch_query = f"""
            SELECT ticker, trading_date,
                   CAST(stoch_14d_k AS FLOAT) as stoch_k,
                   CAST(stoch_14d_d AS FLOAT) as stoch_d,
                   CAST(momentum_strength AS FLOAT) as stoch_momentum
            FROM dbo.nse_500_stochastic
            WHERE trading_date >= DATEADD(day, -{self.days_back}, CAST(GETDATE() AS DATE))
            """
            enriched_data['stoch'] = self.db.execute_query(stoch_query)
            print(f"  [OK] Stochastic: {len(enriched_data['stoch']):,} records")
        except Exception as e:
            print(f"  [WARN] Stochastic load failed: {e}")
        
        # Load ATR
        try:
            atr_query = f"""
            SELECT ticker, trading_date,
                   CAST(ATR_14 AS FLOAT) as atr_14
            FROM dbo.nse_500_atr
            WHERE trading_date >= DATEADD(day, -{self.days_back}, CAST(GETDATE() AS DATE))
            """
            enriched_data['atr'] = self.db.execute_query(atr_query)
            print(f"  [OK] ATR: {len(enriched_data['atr']):,} records")
        except Exception as e:
            print(f"  [WARN] ATR load failed: {e}")
        
        # Load MACD from database view
        try:
            macd_query = f"""
            SELECT ticker, trading_date,
                   CAST(MACD AS FLOAT) as db_macd,
                   CAST(Signal_Line AS FLOAT) as db_macd_signal
            FROM dbo.nse_500_macd
            WHERE trading_date >= DATEADD(day, -{self.days_back}, CAST(GETDATE() AS DATE))
            """
            enriched_data['macd'] = self.db.execute_query(macd_query)
            print(f"  [OK] MACD: {len(enriched_data['macd']):,} records")
        except Exception as e:
            print(f"  [WARN] MACD load failed: {e}")
        
        # Load Fibonacci levels
        try:
            fib_query = f"""
            SELECT ticker, trading_date,
                   CAST(distance_to_nearest_fib_pct AS FLOAT) as fib_distance_pct,
                   fib_trade_signal
            FROM dbo.nse_500_fibonacci
            WHERE trading_date >= DATEADD(day, -{self.days_back}, CAST(GETDATE() AS DATE))
            """
            enriched_data['fibonacci'] = self.db.execute_query(fib_query)
            print(f"  [OK] Fibonacci: {len(enriched_data['fibonacci']):,} records")
        except Exception as e:
            print(f"  [WARN] Fibonacci load failed: {e}")
        
        # Load Support/Resistance
        try:
            sr_query = f"""
            SELECT ticker, trading_date,
                   CAST(distance_to_s1_pct AS FLOAT) as sr_distance_to_support_pct,
                   CAST(distance_to_r1_pct AS FLOAT) as sr_distance_to_resistance_pct,
                   pivot_status,
                   sr_trade_signal
            FROM dbo.nse_500_support_resistance
            WHERE trading_date >= DATEADD(day, -{self.days_back}, CAST(GETDATE() AS DATE))
            """
            enriched_data['support_resistance'] = self.db.execute_query(sr_query)
            print(f"  [OK] Support/Resistance: {len(enriched_data['support_resistance']):,} records")
        except Exception as e:
            print(f"  [WARN] Support/Resistance load failed: {e}")
        
        # Load Candlestick Patterns
        try:
            pattern_query = f"""
            SELECT ticker, trading_date,
                   pattern_signal,
                   CASE WHEN doji IS NOT NULL THEN 1 ELSE 0 END as has_doji,
                   CASE WHEN hammer IS NOT NULL THEN 1 ELSE 0 END as has_hammer,
                   CASE WHEN shooting_star IS NOT NULL THEN 1 ELSE 0 END as has_shooting_star,
                   CASE WHEN bullish_engulfing IS NOT NULL THEN 1 ELSE 0 END as has_bullish_engulfing,
                   CASE WHEN bearish_engulfing IS NOT NULL THEN 1 ELSE 0 END as has_bearish_engulfing,
                   CASE WHEN morning_star IS NOT NULL THEN 1 ELSE 0 END as has_morning_star,
                   CASE WHEN evening_star IS NOT NULL THEN 1 ELSE 0 END as has_evening_star
            FROM dbo.nse_500_patterns
            WHERE trading_date >= DATEADD(day, -{self.days_back}, CAST(GETDATE() AS DATE))
            """
            enriched_data['patterns'] = self.db.execute_query(pattern_query)
            print(f"  [OK] Patterns: {len(enriched_data['patterns']):,} records")
        except Exception as e:
            print(f"  [WARN] Patterns load failed: {e}")
        
        # Load SMA Signals (includes SMA_200, SMA_100, and Above/Below flags)
        try:
            sma_sig_query = f"""
            SELECT ticker, trading_date,
                   CAST(SMA_200 AS FLOAT) as sma_200,
                   CAST(SMA_100 AS FLOAT) as sma_100,
                   SMA_200_Flag, SMA_100_Flag, SMA_50_Flag, SMA_20_Flag
            FROM dbo.nse_500_sma_signals
            WHERE trading_date >= DATEADD(day, -{self.days_back}, CAST(GETDATE() AS DATE))
            """
            enriched_data['sma_signals'] = self.db.execute_query(sma_sig_query)
            print(f"  [OK] SMA Signals: {len(enriched_data['sma_signals']):,} records")
        except Exception as e:
            print(f"  [WARN] SMA Signals load failed: {e}")
        
        # Load MACD Signals (crossover signals)
        try:
            macd_sig_query = f"""
            SELECT ticker, trading_date,
                   MACD_Signal as macd_crossover_signal
            FROM dbo.nse_500_macd_signals
            WHERE trading_date >= DATEADD(day, -{self.days_back}, CAST(GETDATE() AS DATE))
            """
            enriched_data['macd_signals'] = self.db.execute_query(macd_sig_query)
            print(f"  [OK] MACD Signals: {len(enriched_data['macd_signals']):,} records")
        except Exception as e:
            print(f"  [WARN] MACD Signals load failed: {e}")
        
        return enriched_data
    
    def merge_enriched_features(self, df, enriched_data):
        """Merge enriched features from database views into the main dataframe"""
        print("[MERGE] Merging enriched features...")
        
        df_merged = df.copy()
        df_merged['trading_date'] = pd.to_datetime(df_merged['trading_date'])
        
        # CRITICAL: Remove duplicate ticker-date rows from base data first
        # (source table nse_500_hist_data may have duplicates from overlapping yfinance fetches)
        before_dedup = len(df_merged)
        df_merged = df_merged.drop_duplicates(subset=['ticker', 'trading_date'], keep='last')
        if before_dedup > len(df_merged):
            print(f"  [DEDUP] Removed {before_dedup - len(df_merged):,} duplicate rows from base data")
        
        for name, enriched_df in enriched_data.items():
            if enriched_df is not None and not enriched_df.empty:
                enriched_df = enriched_df.copy()
                enriched_df['trading_date'] = pd.to_datetime(enriched_df['trading_date'])
                
                # Remove duplicates from enriched view (views inherit source table duplicates)
                enriched_df = enriched_df.drop_duplicates(subset=['ticker', 'trading_date'], keep='last')
                
                merge_cols = [c for c in enriched_df.columns if c not in ['ticker', 'trading_date']]
                
                df_merged = df_merged.merge(
                    enriched_df[['ticker', 'trading_date'] + merge_cols],
                    on=['ticker', 'trading_date'],
                    how='left'
                )
                print(f"  [OK] Merged {name}: {len(merge_cols)} features")
        
        print(f"[OK] Merged dataset: {df_merged.shape[0]:,} records, {df_merged.shape[1]} columns")
        return df_merged
    
    def encode_signal_features(self, df):
        """Encode categorical signal columns from DB views into numeric features"""
        print("[ENCODE] Encoding categorical signals to numeric features...")
        
        df_enc = df.copy()
        
        # --- Fibonacci signal ---
        if 'fib_trade_signal' in df_enc.columns:
            df_enc['fib_signal_strength'] = df_enc['fib_trade_signal'].map(
                self.signal_strength_map
            ).fillna(0).astype(float)
            df_enc.drop(columns=['fib_trade_signal'], inplace=True, errors='ignore')
        else:
            df_enc['fib_signal_strength'] = 0
        
        if 'fib_distance_pct' not in df_enc.columns:
            df_enc['fib_distance_pct'] = 0
        
        # --- Support/Resistance signals ---
        if 'pivot_status' in df_enc.columns:
            df_enc['sr_pivot_position'] = df_enc['pivot_status'].map(
                self.position_map
            ).fillna(0).astype(float)
            df_enc.drop(columns=['pivot_status'], inplace=True, errors='ignore')
        else:
            df_enc['sr_pivot_position'] = 0
        
        if 'sr_trade_signal' in df_enc.columns:
            df_enc['sr_signal_strength'] = df_enc['sr_trade_signal'].map(
                self.signal_strength_map
            ).fillna(0).astype(float)
            df_enc.drop(columns=['sr_trade_signal'], inplace=True, errors='ignore')
        else:
            df_enc['sr_signal_strength'] = 0
        
        for col in ['sr_distance_to_support_pct', 'sr_distance_to_resistance_pct']:
            if col not in df_enc.columns:
                df_enc[col] = 0
        
        # --- Pattern signals ---
        if 'pattern_signal' in df_enc.columns:
            df_enc['pattern_signal_strength'] = df_enc['pattern_signal'].map(
                self.signal_strength_map
            ).fillna(0).astype(float)
            df_enc.drop(columns=['pattern_signal'], inplace=True, errors='ignore')
        else:
            df_enc['pattern_signal_strength'] = 0
        
        # Pattern binary flags (already 0/1 from SQL CASE)
        for col in ['has_doji', 'has_hammer', 'has_shooting_star',
                     'has_bullish_engulfing', 'has_bearish_engulfing',
                     'has_morning_star', 'has_evening_star']:
            if col not in df_enc.columns:
                df_enc[col] = 0
            else:
                df_enc[col] = df_enc[col].fillna(0).astype(float)
        
        # --- SMA signal flags ---
        if 'sma_200' in df_enc.columns:
            # Price vs SMA 100/200 ratios
            df_enc['price_vs_sma100'] = df_enc['close_price'] / df_enc['sma_100'].replace(0, np.nan)
            df_enc['price_vs_sma200'] = df_enc['close_price'] / df_enc['sma_200'].replace(0, np.nan)
        else:
            df_enc['sma_200'] = 0
            df_enc['sma_100'] = 0
            df_enc['price_vs_sma100'] = 1.0
            df_enc['price_vs_sma200'] = 1.0
        
        for flag_col in ['SMA_200_Flag', 'SMA_100_Flag', 'SMA_50_Flag', 'SMA_20_Flag']:
            target_col = flag_col.lower().replace('_flag', '_flag')  # sma_200_flag etc.
            if flag_col in df_enc.columns:
                df_enc[target_col] = df_enc[flag_col].map(
                    self.position_map
                ).fillna(0).astype(float)
                df_enc.drop(columns=[flag_col], inplace=True, errors='ignore')
            else:
                df_enc[target_col] = 0
        
        # --- MACD crossover signal ---
        if 'macd_crossover_signal' in df_enc.columns:
            df_enc['macd_signal_strength'] = df_enc['macd_crossover_signal'].map(
                self.signal_strength_map
            ).fillna(0).astype(float)
            df_enc.drop(columns=['macd_crossover_signal'], inplace=True, errors='ignore')
        else:
            df_enc['macd_signal_strength'] = 0
        
        # Drop any remaining non-numeric columns from DB views
        drop_cols = ['bb_close', 'bb_sma20', 'db_macd', 'db_macd_signal',
                     'patterns_detected', 'rsi_trade_signal']
        df_enc.drop(columns=[c for c in drop_cols if c in df_enc.columns],
                    inplace=True, errors='ignore')
        
        encoded_count = sum(1 for c in ['fib_signal_strength', 'sr_pivot_position',
                                         'sr_signal_strength', 'pattern_signal_strength',
                                         'macd_signal_strength', 'sma_200_flag',
                                         'sma_100_flag', 'sma_50_flag', 'sma_20_flag']
                           if c in df_enc.columns)
        print(f"  [OK] Encoded {encoded_count} signal features + 7 pattern flags")
        
        return df_enc
    
    def create_target_variable(self, df):
        """
        Create proper target variable: next-day price direction
        
        Instead of using RSI-based signals (which only fire at extremes),
        we create a target that directly measures what we want to predict:
        - 'Up' = next day's close > today's close
        - 'Down' = next day's close <= today's close
        
        Also creates regression target: next-day percentage change
        """
        print("[TARGET] Creating target variables...")
        
        df_target = df.copy()
        df_target = df_target.sort_values(['ticker', 'trading_date'])
        
        # Next-day close price (shifted by -1 within each ticker group)
        df_target['next_close'] = df_target.groupby('ticker')['close_price'].shift(-1)
        
        # Next-day return (percentage change)
        df_target['next_day_return'] = (
            (df_target['next_close'] - df_target['close_price']) / df_target['close_price'] * 100
        )
        
        # Classification target: direction
        df_target['direction'] = np.where(df_target['next_day_return'] > 0, 'Up', 'Down')
        
        # Also create 3-day and 5-day targets for multi-horizon
        df_target['next_3d_close'] = df_target.groupby('ticker')['close_price'].shift(-3)
        df_target['next_3d_return'] = (
            (df_target['next_3d_close'] - df_target['close_price']) / df_target['close_price'] * 100
        )
        df_target['direction_3d'] = np.where(df_target['next_3d_return'] > 0, 'Up', 'Down')
        
        df_target['next_5d_close'] = df_target.groupby('ticker')['close_price'].shift(-5)
        df_target['next_5d_return'] = (
            (df_target['next_5d_close'] - df_target['close_price']) / df_target['close_price'] * 100
        )
        df_target['direction_5d'] = np.where(df_target['next_5d_return'] > 0, 'Up', 'Down')
        
        # Remove rows without target (last row per ticker)
        valid_mask = df_target['next_close'].notna()
        df_target = df_target[valid_mask]
        
        # Report target distribution
        direction_dist = df_target['direction'].value_counts()
        print(f"  [OK] Target distribution (1-day):")
        for direction, count in direction_dist.items():
            pct = (count / len(df_target)) * 100
            print(f"    {direction}: {count:,} ({pct:.1f}%)")
        
        print(f"  [OK] Valid training samples: {len(df_target):,}")
        
        return df_target
    
    def engineer_features(self, df):
        """Apply comprehensive feature engineering for NSE data"""
        print("[FEATURES] Performing feature engineering...")
        
        df_features = df.copy()
        df_features = df_features.sort_values(['ticker', 'trading_date'])
        
        # Process per ticker for time-series features
        grouped_results = []
        ticker_count = 0
        
        for ticker, group in df_features.groupby('ticker'):
            group = group.sort_values('trading_date').reset_index(drop=True)
            
            # Skip tickers with insufficient data (need at least 60 days for indicators)
            if len(group) < 60:
                continue
            
            ticker_count += 1
            
            # === Basic Price Features ===
            group['daily_return'] = group['close_price'].pct_change() * 100
            group['daily_volatility'] = group['daily_return'].rolling(window=10).std()
            group['volume_millions'] = group['volume'] / 1_000_000
            group['price_range'] = group['high_price'] - group['low_price']
            
            # Safe division for price_position
            range_nonzero = group['price_range'].replace(0, np.nan)
            group['price_position'] = (group['close_price'] - group['low_price']) / range_nonzero
            
            group['gap'] = (group['open_price'] - group['close_price'].shift(1)) / group['close_price'].shift(1) * 100
            group['volume_price_trend'] = (group['daily_return'] * group['volume']).rolling(window=10).mean()
            
            # === RSI Features (calculate if not from DB) ===
            if 'RSI' not in group.columns or group['RSI'].isna().all():
                delta = group['close_price'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                group['RSI'] = 100 - (100 / (1 + rs))
            
            group['rsi_oversold'] = (group['RSI'] < 30).astype(int)
            group['rsi_overbought'] = (group['RSI'] > 70).astype(int)
            group['rsi_momentum'] = group['RSI'].diff()
            
            # === Moving Averages ===
            for period in [5, 10, 20, 50]:
                group[f'sma_{period}'] = group['close_price'].rolling(window=period).mean()
                group[f'ema_{period}'] = group['close_price'].ewm(span=period, adjust=False).mean()
            
            # === MACD (calculated for consistency) ===
            ema_12 = group['close_price'].ewm(span=12, adjust=False).mean()
            ema_26 = group['close_price'].ewm(span=26, adjust=False).mean()
            group['macd'] = ema_12 - ema_26
            group['macd_signal'] = group['macd'].ewm(span=9, adjust=False).mean()
            group['macd_histogram'] = group['macd'] - group['macd_signal']
            
            # === Price vs MA Ratios (normalized features - proven high impact) ===
            group['price_vs_sma20'] = group['close_price'] / group['sma_20']
            group['price_vs_sma50'] = group['close_price'] / group['sma_50']
            group['price_vs_ema20'] = group['close_price'] / group['ema_20']
            
            # === MA Crossover Ratios ===
            group['sma20_vs_sma50'] = group['sma_20'] / group['sma_50']
            group['ema20_vs_ema50'] = group['ema_20'] / group['ema_50']
            group['sma5_vs_sma20'] = group['sma_5'] / group['sma_20']
            
            # === Volume Indicators ===
            group['volume_sma_20'] = group['volume'].rolling(window=20).mean()
            vol_sma_nonzero = group['volume_sma_20'].replace(0, np.nan)
            group['volume_sma_ratio'] = group['volume'] / vol_sma_nonzero
            group['vol_change_ratio'] = group['volume'].pct_change()
            
            # === Price Momentum ===
            group['price_momentum_5'] = group['close_price'] / group['close_price'].shift(5)
            group['price_momentum_10'] = group['close_price'] / group['close_price'].shift(10)
            
            # === Volatility Features ===
            group['price_volatility_10'] = group['close_price'].pct_change().rolling(window=10).std()
            group['price_volatility_20'] = group['close_price'].pct_change().rolling(window=20).std()
            
            # === Trend Strength ===
            group['trend_strength_10'] = group['close_price'].rolling(window=10).apply(
                lambda x: (x.iloc[-1] - x.iloc[0]) / x.std() if x.std() != 0 else 0, raw=False
            )
            
            # === Multi-day Returns (lagged, so no look-ahead) ===
            group['return_1d'] = group['close_price'].pct_change() * 100
            group['return_3d'] = group['close_price'].pct_change(3) * 100
            group['return_5d'] = group['close_price'].pct_change(5) * 100
            
            # === Candlestick Features ===
            group['high_low_ratio'] = group['high_price'] / group['low_price']
            group['close_open_ratio'] = group['close_price'] / group['open_price']
            group['upper_shadow'] = (group['high_price'] - np.maximum(group['open_price'], group['close_price'])) / group['price_range'].replace(0, np.nan)
            group['lower_shadow'] = (np.minimum(group['open_price'], group['close_price']) - group['low_price']) / group['price_range'].replace(0, np.nan)
            
            # === Bollinger Band Features (use from DB if available, else calculate) ===
            if 'bb_upper' not in group.columns or group['bb_upper'].isna().all():
                bb_sma = group['close_price'].rolling(window=20).mean()
                bb_std = group['close_price'].rolling(window=20).std()
                group['bb_upper'] = bb_sma + (2 * bb_std)
                group['bb_lower'] = bb_sma - (2 * bb_std)
            
            bb_range = (group['bb_upper'] - group['bb_lower']).replace(0, np.nan)
            group['bb_width'] = bb_range / group['close_price']
            group['bb_position'] = (group['close_price'] - group['bb_lower']) / bb_range
            
            # === Stochastic Features (use from DB if available, else calculate) ===
            if 'stoch_k' not in group.columns or group['stoch_k'].isna().all():
                low_14 = group['low_price'].rolling(window=14).min()
                high_14 = group['high_price'].rolling(window=14).max()
                stoch_range = (high_14 - low_14).replace(0, np.nan)
                group['stoch_k'] = ((group['close_price'] - low_14) / stoch_range) * 100
                group['stoch_d'] = group['stoch_k'].rolling(window=3).mean()
                group['stoch_momentum'] = group['stoch_k'] - group['stoch_d']
            
            # === ATR Features (use from DB if available, else calculate) ===
            if 'atr_14' not in group.columns or group['atr_14'].isna().all():
                high_low = group['high_price'] - group['low_price']
                high_close = abs(group['high_price'] - group['close_price'].shift(1))
                low_close = abs(group['low_price'] - group['close_price'].shift(1))
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                group['atr_14'] = true_range.rolling(window=14).mean()
            
            group['atr_pct'] = group['atr_14'] / group['close_price'] * 100
            
            # === Date Features ===
            trading_dates = pd.to_datetime(group['trading_date'])
            group['day_of_week'] = trading_dates.dt.dayofweek
            group['month'] = trading_dates.dt.month
            
            grouped_results.append(group)
        
        if not grouped_results:
            raise ValueError("No valid data after feature engineering")
        
        result_df = pd.concat(grouped_results, ignore_index=True)
        
        # Handle infinite values and NaN
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        result_df = result_df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        print(f"[OK] Feature engineering complete: {result_df.shape[1]} columns, "
              f"{ticker_count} tickers, {len(result_df):,} records")
        
        return result_df
    
    def perform_eda(self, df):
        """Perform exploratory data analysis on NSE data"""
        if self.quick_mode:
            print("[SKIP] Skipping detailed EDA (quick mode)")
            return {'data_shape': df.shape}
        
        print("[EDA] Performing exploratory data analysis on NSE data...")
        
        print(f"  Data shape: {df.shape}")
        print(f"  Date range: {df['trading_date'].min()} to {df['trading_date'].max()}")
        print(f"  Unique tickers: {df['ticker'].nunique()}")
        
        # Target distribution
        if 'direction' in df.columns:
            dist = df['direction'].value_counts()
            print(f"  Direction target distribution:")
            for val, cnt in dist.items():
                print(f"    {val}: {cnt:,} ({cnt/len(df)*100:.1f}%)")
        
        # Feature statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"  Numeric features: {len(numeric_cols)}")
        
        # Check for NaN
        nan_counts = df[numeric_cols].isna().sum()
        nan_features = nan_counts[nan_counts > 0]
        if len(nan_features) > 0:
            print(f"  Features with NaN: {len(nan_features)}")
        else:
            print(f"  [OK] No NaN values")
        
        return {'data_shape': df.shape, 'unique_tickers': df['ticker'].nunique()}
    
    def prepare_ml_dataset(self, df):
        """Prepare dataset for ML training with proper feature selection"""
        print("[PREP] Preparing ML dataset...")
        
        # Determine available features
        available_features = [col for col in self.clf_feature_columns if col in df.columns]
        missing_features = [col for col in self.clf_feature_columns if col not in df.columns]
        
        if missing_features:
            print(f"  [WARN] Missing features ({len(missing_features)}): {missing_features[:5]}...")
        
        self.feature_columns = available_features
        
        # Prepare X and y for classification
        X = df[available_features].copy()
        y_direction = df['direction'].copy()
        
        # Also prepare regression target
        y_return = df['next_day_return'].copy()
        
        # Remove any remaining invalid rows
        valid_mask = X.notna().all(axis=1) & y_direction.notna() & y_return.notna()
        X = X[valid_mask]
        y_direction = y_direction[valid_mask]
        y_return = y_return[valid_mask]
        
        # Encode classification target
        direction_encoder = LabelEncoder()
        y_direction_encoded = direction_encoder.fit_transform(y_direction)
        
        print(f"[OK] ML dataset prepared:")
        print(f"  Features: {X.shape[1]}")
        print(f"  Samples: {X.shape[0]:,}")
        print(f"  Direction classes: {list(direction_encoder.classes_)}")
        print(f"  Direction balance: {dict(zip(*np.unique(y_direction_encoded, return_counts=True)))}")
        
        return X, y_direction_encoded, y_return, direction_encoder
    
    def train_classification_models(self, X, y, feature_cols):
        """Train ensemble of classification models for direction prediction"""
        print("[TRAIN] Training classification models (direction prediction)...")
        
        # Time-series aware split (80/20)
        split_idx = int(0.8 * len(X))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        print(f"  Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models with optimized hyperparameters
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                class_weight='balanced',
                C=0.1,
                solver='liblinear',
                max_iter=2000,
                random_state=42
            )
        }
        
        # Train and evaluate each model
        model_results = {}
        trained_models = {}
        cv_splitter = TimeSeriesSplit(n_splits=3)
        
        for model_name, model in models.items():
            print(f"  Training {model_name}...", flush=True)
            
            try:
                # Train
                model.fit(X_train_scaled, y_train)
                print(f"    [fit done]", flush=True)
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Direction accuracy (this is what ai_prediction_history tracks)
                direction_accuracy = accuracy
                
                print(f"    [eval done] Acc={accuracy:.3f}", flush=True)
                
                # Cross-validation (uses a fresh clone each fold)
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train,
                    cv=cv_splitter, scoring='accuracy'
                )
                
                model_results[model_name] = {
                    'accuracy': accuracy,
                    'direction_accuracy': direction_accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                }
                
                trained_models[model_name] = model
                
                print(f"    Accuracy: {accuracy:.3f}, F1: {f1:.3f}, "
                      f"CV: {cv_scores.mean():.3f} (+/-{cv_scores.std():.3f})")
                
            except Exception as e:
                print(f"    [ERROR] {model_name}: {e}")
        
        # Create calibrated ensemble (Voting Classifier)
        print("  Training Ensemble (Voting Classifier)...")
        try:
            ensemble_estimators = [
                (name.lower().replace(' ', '_'), model) 
                for name, model in trained_models.items()
            ]
            
            ensemble = VotingClassifier(
                estimators=ensemble_estimators,
                voting='soft',
                n_jobs=-1
            )
            ensemble.fit(X_train_scaled, y_train)
            
            y_pred_ensemble = ensemble.predict(X_test_scaled)
            ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
            ensemble_f1 = f1_score(y_test, y_pred_ensemble, average='weighted')
            
            model_results['Ensemble'] = {
                'accuracy': ensemble_accuracy,
                'direction_accuracy': ensemble_accuracy,
                'f1_score': ensemble_f1,
                'cv_mean': 0,
                'cv_std': 0,
            }
            trained_models['Ensemble'] = ensemble
            
            print(f"    Ensemble Accuracy: {ensemble_accuracy:.3f}, F1: {ensemble_f1:.3f}")
            
        except Exception as e:
            print(f"    [ERROR] Ensemble: {e}")
        
        # Find best model
        best_model_name = max(model_results.keys(),
                             key=lambda k: model_results[k]['f1_score'])
        
        print(f"\n[BEST] Best classification model: {best_model_name} "
              f"(F1: {model_results[best_model_name]['f1_score']:.3f}, "
              f"Direction Accuracy: {model_results[best_model_name]['accuracy']:.1%})")
        
        return {
            'model_results': model_results,
            'trained_models': trained_models,
            'best_model_name': best_model_name,
            'best_model': trained_models[best_model_name],
            'scaler': scaler,
            'feature_columns': feature_cols,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
        }
    
    def train_regression_models(self, X, y_return, feature_cols):
        """Train regression models for price change prediction"""
        print("[TRAIN] Training regression models (price change prediction)...")
        
        # Time-series aware split
        split_idx = int(0.8 * len(X))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y_return.iloc[:split_idx]
        y_test = y_return.iloc[split_idx:]
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Clip extreme returns for training stability
        y_train_clipped = np.clip(y_train.values, -10, 10)
        
        # Define regression models
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42
            ),
            'Ridge': Ridge(
                alpha=1.0,
                random_state=42
            ),
        }
        
        model_results = {}
        trained_models = {}
        
        for model_name, model in models.items():
            print(f"  Training {model_name} regressor...")
            
            try:
                model.fit(X_train_scaled, y_train_clipped)
                y_pred = model.predict(X_test_scaled)
                
                # Metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                # Direction accuracy from regression predictions
                pred_direction = (y_pred > 0).astype(int)
                actual_direction = (y_test.values > 0).astype(int)
                direction_accuracy = accuracy_score(actual_direction, pred_direction)
                
                model_results[model_name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'direction_accuracy': direction_accuracy,
                }
                
                trained_models[model_name] = model
                
                print(f"    MAE: {mae:.4f}%, RMSE: {rmse:.4f}%, "
                      f"Direction Accuracy: {direction_accuracy:.1%}")
                
            except Exception as e:
                print(f"    [ERROR] {model_name}: {e}")
        
        # Create regression ensemble
        print("  Training Regression Ensemble...")
        try:
            reg_estimators = [
                (name.lower().replace(' ', '_'), model)
                for name, model in trained_models.items()
            ]
            
            reg_ensemble = VotingRegressor(
                estimators=reg_estimators,
                n_jobs=-1
            )
            reg_ensemble.fit(X_train_scaled, y_train_clipped)
            
            y_pred_ensemble = reg_ensemble.predict(X_test_scaled)
            ens_mae = mean_absolute_error(y_test, y_pred_ensemble)
            ens_direction = accuracy_score(
                (y_test.values > 0).astype(int),
                (y_pred_ensemble > 0).astype(int)
            )
            
            model_results['Ensemble'] = {
                'mae': ens_mae,
                'direction_accuracy': ens_direction,
            }
            trained_models['Ensemble'] = reg_ensemble
            
            print(f"    Ensemble MAE: {ens_mae:.4f}%, Direction Accuracy: {ens_direction:.1%}")
            
        except Exception as e:
            print(f"    [ERROR] Ensemble: {e}")
        
        # Find best regression model by direction accuracy
        best_model_name = max(model_results.keys(),
                             key=lambda k: model_results[k]['direction_accuracy'])
        
        print(f"\n[BEST] Best regression model: {best_model_name} "
              f"(Direction Accuracy: {model_results[best_model_name]['direction_accuracy']:.1%})")
        
        return {
            'model_results': model_results,
            'trained_models': trained_models,
            'best_model_name': best_model_name,
            'best_model': trained_models[best_model_name],
            'scaler': scaler,
        }
    
    def save_model_artifacts(self, clf_results, reg_results, direction_encoder):
        """Save all NSE model artifacts"""
        print("[SAVE] Saving NSE model artifacts...")
        
        # Save classification models
        for name, model in clf_results['trained_models'].items():
            safe_name = name.lower().replace(' ', '_')
            path = self.nse_model_dir / f'nse_clf_{safe_name}.joblib'
            joblib.dump(model, path)
            print(f"  [OK] Saved: {path}")
        
        # Save best classification model separately
        best_clf_path = self.nse_model_dir / 'nse_best_classifier.joblib'
        joblib.dump(clf_results['best_model'], best_clf_path)
        
        # Save regression models
        for name, model in reg_results['trained_models'].items():
            safe_name = name.lower().replace(' ', '_')
            path = self.nse_model_dir / f'nse_reg_{safe_name}.joblib'
            joblib.dump(model, path)
            print(f"  [OK] Saved: {path}")
        
        # Save best regression model separately
        best_reg_path = self.nse_model_dir / 'nse_best_regressor.joblib'
        joblib.dump(reg_results['best_model'], best_reg_path)
        
        # Save preprocessing artifacts
        joblib.dump(clf_results['scaler'], self.nse_model_dir / 'nse_scaler.joblib')
        joblib.dump(direction_encoder, self.nse_model_dir / 'nse_direction_encoder.joblib')
        
        # Save regression scaler separately (same features, same scaler)
        joblib.dump(reg_results['scaler'], self.nse_model_dir / 'nse_reg_scaler.joblib')
        
        # Save training metadata (includes encoding maps for prediction consistency)
        metadata = {
            'training_timestamp': self.timestamp,
            'feature_columns': self.feature_columns,
            'signal_strength_map': self.signal_strength_map,
            'position_map': self.position_map,
            'clf_results': {
                name: {k: v for k, v in res.items() if k != 'predictions' and k != 'probabilities'}
                for name, res in clf_results['model_results'].items()
            },
            'reg_results': {
                name: {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in res.items()}
                for name, res in reg_results['model_results'].items()
            },
            'best_clf_model': clf_results['best_model_name'],
            'best_reg_model': reg_results['best_model_name'],
            'direction_classes': list(direction_encoder.classes_),
            'training_samples': len(clf_results['X_train']),
            'test_samples': len(clf_results['X_test']),
            'days_back': self.days_back,
        }
        
        with open(self.nse_model_dir / 'nse_training_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\n[OK] All NSE model artifacts saved to {self.nse_model_dir}/")
        print(f"  Best Classifier: {clf_results['best_model_name']}")
        print(f"  Best Regressor: {reg_results['best_model_name']}")
    
    def run_full_retrain(self):
        """Execute complete NSE model retraining pipeline"""
        print("=" * 80)
        print(f"[START] NSE 500 Model Retraining - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Mode: {'Quick' if self.quick_mode else 'Full'}")
        print(f"  Backup: {'Yes' if self.backup_old else 'No'}")
        print(f"  Training window: {self.days_back} days")
        print("=" * 80)
        
        try:
            # Step 1: Backup
            if self.backup_old:
                self.backup_existing_model()
            
            # Step 2: Load NSE data
            df = self.load_nse_training_data()
            
            # Step 3: Load enriched features from DB views
            enriched_data = self.load_enriched_features()
            
            # Step 4: Merge enriched features
            df = self.merge_enriched_features(df, enriched_data)
            
            # Step 5: Encode categorical signals to numeric
            df = self.encode_signal_features(df)
            
            # Step 6: Feature engineering
            df_features = self.engineer_features(df)
            
            # Step 7: Create target variable (next-day direction)
            df_features = self.create_target_variable(df_features)
            
            # Step 8: EDA
            self.perform_eda(df_features)
            
            # Step 9: Prepare ML dataset
            X, y_direction, y_return, direction_encoder = self.prepare_ml_dataset(df_features)
            
            # Step 10: Train classification models
            clf_results = self.train_classification_models(X, y_direction, self.feature_columns)
            
            # Step 11: Train regression models
            reg_results = self.train_regression_models(X, y_return, self.feature_columns)
            
            # Step 12: Save artifacts
            self.save_model_artifacts(clf_results, reg_results, direction_encoder)
            
            print("\n" + "=" * 80)
            print("[COMPLETE] NSE MODEL RETRAINING COMPLETE!")
            print("=" * 80)
            print(f"  Best Classifier: {clf_results['best_model_name']} "
                  f"(Direction Accuracy: {clf_results['model_results'][clf_results['best_model_name']]['accuracy']:.1%})")
            print(f"  Best Regressor: {reg_results['best_model_name']} "
                  f"(Direction Accuracy: {reg_results['model_results'][reg_results['best_model_name']]['direction_accuracy']:.1%})")
            print(f"  Models saved to: data/nse_models/")
            print(f"  Timestamp: {self.timestamp}")
            
            return True
            
        except Exception as e:
            print(f"\n[ERROR] NSE retraining failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description='NSE 500 ML Model Retraining System')
    parser.add_argument('--quick', action='store_true', help='Quick retrain (skip detailed EDA)')
    parser.add_argument('--backup-old', action='store_true', help='Backup existing model files')
    parser.add_argument('--days-back', type=int, default=730, help='Training data window in days (default: 730)')
    
    args = parser.parse_args()
    
    retrainer = NSEModelRetrainer(
        backup_old=args.backup_old,
        quick_mode=args.quick,
        days_back=args.days_back
    )
    
    success = retrainer.run_full_retrain()
    
    if success:
        print("\n[NEXT STEPS]")
        print("1. Test predictions: python predict_nse_signals.py --all-nse")
        print("2. Review model metrics in data/nse_models/nse_training_metadata.pkl")
        print("3. Run daily automation: python daily_nse_automation.py")
    else:
        print("\n[ERROR] Retraining failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
