"""
NSE 500 Trading Signal Prediction Script

This script generates trading signals for NSE 500 stocks using ML models
trained specifically on NSE data (not NASDAQ).

Key improvements:
- Uses NSE-trained models from data/nse_models/ (not NASDAQ-trained models)
- Ensemble of multiple models for more robust predictions
- Enriched features from database views (Bollinger Bands, Stochastic, ATR, MACD)
- Both classification (direction) and regression (price change) predictions
- Proper confidence calibration
- Tracks model name/version for ai_prediction_history

Usage:
    python predict_nse_signals.py                  # Process all NSE 500 stocks
    python predict_nse_signals.py --ticker RELIANCE.NS  # Single ticker
    python predict_nse_signals.py --check-only     # Check data availability only
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
import logging

# Configure UTF-8 encoding for Windows console compatibility
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection

warnings.filterwarnings('ignore')


def safe_print(text):
    """Print text with safe encoding handling for Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        emoji_replacements = {
            '\u2705': '[SUCCESS]', '\u274c': '[ERROR]', '\U0001f4ca': '[DATA]',
            '\U0001f1ee\U0001f1f3': '[NSE]', '\U0001f4c8': '[PREDICTION]',
            '\u26a0\ufe0f': '[WARN]', '\U0001f3af': '[TARGET]',
            '\U0001f504': '[PROCESSING]', '\U0001f4be': '[SAVE]',
            '\U0001f389': '[COMPLETE]'
        }
        for emoji, replacement in emoji_replacements.items():
            text = text.replace(emoji, replacement)
        print(text)


class NSETradingSignalPredictor:
    """NSE 500 trading signal prediction system using NSE-trained models"""
    
    def __init__(self):
        """Initialize the NSE predictor with NSE-specific model artifacts"""
        
        self.nse_model_dir = Path('data/nse_models')
        self.legacy_model_dir = Path('data')
        
        # Database connection
        self.db = SQLServerConnection()
        
        # Load model artifacts
        self.models_loaded = False
        self.use_nse_models = False
        self.load_model_artifacts()
    
    def load_model_artifacts(self):
        """Load NSE-specific models, falling back to legacy models if needed"""
        
        # Try loading NSE-specific models first
        if self._load_nse_models():
            self.use_nse_models = True
            safe_print("[SUCCESS] NSE-specific models loaded successfully")
            return
        
        # Fallback to legacy models with STRONG warning
        safe_print("")
        safe_print("=" * 70)
        safe_print("[WARN] NSE-SPECIFIC MODELS NOT FOUND!")
        safe_print("[WARN] Falling back to legacy models trained on NASDAQ data.")
        safe_print("[WARN] These models are NOT trained on NSE data and will produce")
        safe_print("[WARN] UNRELIABLE predictions for NSE stocks (expected ~50% accuracy).")
        safe_print("[WARN]")
        safe_print("[WARN] To fix this, run:")
        safe_print("[WARN]   python retrain_nse_model.py")
        safe_print("[WARN] This trains models specifically on NSE 500 historical data.")
        safe_print("=" * 70)
        safe_print("")
        self._load_legacy_models()
    
    def _load_nse_models(self):
        """Load NSE-trained models from data/nse_models/"""
        try:
            metadata_path = self.nse_model_dir / 'nse_training_metadata.pkl'
            if not metadata_path.exists():
                return False
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            self.feature_columns = self.metadata['feature_columns']
            
            # Load encoding maps from training metadata for prediction consistency
            self._signal_map = self.metadata.get('signal_strength_map', {})
            self._position_map = self.metadata.get('position_map', {})
            
            # Load classification models (all available)
            self.clf_models = {}
            for model_file in self.nse_model_dir.glob('nse_clf_*.joblib'):
                model_name = model_file.stem.replace('nse_clf_', '').replace('_', ' ').title()
                self.clf_models[model_name] = joblib.load(model_file)
                safe_print(f"  [OK] Loaded classifier: {model_name}")
            
            # Load best classifier
            best_clf_path = self.nse_model_dir / 'nse_best_classifier.joblib'
            if best_clf_path.exists():
                self.best_classifier = joblib.load(best_clf_path)
            
            # Load regression models (all available)
            self.reg_models = {}
            for model_file in self.nse_model_dir.glob('nse_reg_*.joblib'):
                model_name = model_file.stem.replace('nse_reg_', '').replace('_', ' ').title()
                self.reg_models[model_name] = joblib.load(model_file)
                safe_print(f"  [OK] Loaded regressor: {model_name}")
            
            # Load best regressor
            best_reg_path = self.nse_model_dir / 'nse_best_regressor.joblib'
            if best_reg_path.exists():
                self.best_regressor = joblib.load(best_reg_path)
            
            # Load preprocessing artifacts
            self.clf_scaler = joblib.load(self.nse_model_dir / 'nse_scaler.joblib')
            self.direction_encoder = joblib.load(self.nse_model_dir / 'nse_direction_encoder.joblib')
            self.reg_scaler = joblib.load(self.nse_model_dir / 'nse_reg_scaler.joblib')
            
            safe_print(f"  [OK] {len(self.clf_models)} classifiers, {len(self.reg_models)} regressors loaded")
            safe_print(f"  [OK] Best classifier: {self.metadata.get('best_clf_model', 'Unknown')}")
            safe_print(f"  [OK] Best regressor: {self.metadata.get('best_reg_model', 'Unknown')}")
            safe_print(f"  [OK] Training date: {self.metadata.get('training_timestamp', 'Unknown')}")
            
            self.models_loaded = True
            return True
            
        except Exception as e:
            safe_print(f"  [WARN] Error loading NSE models: {e}")
            return False
    
    def _load_legacy_models(self):
        """Load legacy models (backward compatibility)"""
        try:
            model_path = self.legacy_model_dir / 'best_model_extra_trees.joblib'
            scaler_path = self.legacy_model_dir / 'scaler.joblib'
            encoder_path = self.legacy_model_dir / 'target_encoder.joblib'
            
            self.best_classifier = joblib.load(model_path)
            self.clf_scaler = joblib.load(scaler_path)
            self.direction_encoder = joblib.load(encoder_path)
            
            # Legacy feature columns
            self.feature_columns = [
                'open_price', 'high_price', 'low_price', 'close_price', 'volume',
                'RSI', 'daily_volatility', 'daily_return', 'volume_millions',
                'price_range', 'price_position', 'gap', 'volume_price_trend',
                'rsi_oversold', 'rsi_overbought', 'rsi_momentum',
                'sma_5', 'sma_10', 'sma_20', 'sma_50',
                'ema_5', 'ema_10', 'ema_20', 'ema_50',
                'macd', 'macd_signal', 'macd_histogram',
                'price_vs_sma20', 'price_vs_sma50', 'price_vs_ema20',
                'sma20_vs_sma50', 'ema20_vs_ema50', 'sma5_vs_sma20',
                'volume_sma_20', 'volume_sma_ratio',
                'price_momentum_5', 'price_momentum_10',
                'price_volatility_10', 'price_volatility_20',
                'trend_strength_10',
                'day_of_week', 'month'
            ]
            
            self.clf_models = {'Extra Trees': self.best_classifier}
            self.reg_models = {}
            self.reg_scaler = self.clf_scaler
            self.best_regressor = None
            self.metadata = {'best_clf_model': 'Extra Trees (Legacy)', 'training_timestamp': 'Unknown'}
            
            self.models_loaded = True
            safe_print("[OK] Legacy models loaded")
            
        except FileNotFoundError as e:
            safe_print(f"[ERROR] No models found: {e}")
            safe_print("[ERROR] Run 'python retrain_nse_model.py' first to train NSE models")
            sys.exit(1)
    
    def get_nse_data(self, ticker=None, days_back=90):
        """Fetch NSE stock data for prediction (increased window for better indicators)"""
        
        ticker_filter = f"AND h.ticker = '{ticker}'" if ticker else ""
        
        query = f"""
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
        WHERE h.trading_date >= DATEADD(day, -{days_back}, CAST(GETDATE() AS DATE))
            AND ISNUMERIC(h.close_price) = 1
            AND ISNUMERIC(h.open_price) = 1
            AND CAST(h.close_price AS FLOAT) > 0
            {ticker_filter}
        ORDER BY h.ticker, h.trading_date
        """
        
        try:
            df = self.db.execute_query(query)
            if df.empty:
                safe_print(f"[WARN] No NSE data found" + (f" for ticker: {ticker}" if ticker else ""))
                return None
            safe_print(f"[DATA] Loaded {len(df):,} records for {df['ticker'].nunique()} stocks")
            return df
        except Exception as e:
            safe_print(f"[ERROR] Error fetching NSE data: {e}")
            return None
    
    def load_enriched_features_for_prediction(self, tickers=None):
        """Load enriched features from database views for prediction"""
        enriched = {}
        
        ticker_filter = ""
        if tickers:
            # Use batches if too many tickers
            ticker_list = "','".join(tickers[:500])
            ticker_filter = f"AND ticker IN ('{ticker_list}')"
        
        # RSI
        try:
            rsi_df = self.db.execute_query(f"""
                SELECT ticker, trading_date, RSI
                FROM dbo.nse_500_RSI_calculation
                WHERE trading_date >= DATEADD(day, -90, CAST(GETDATE() AS DATE))
                {ticker_filter}
            """)
            enriched['rsi'] = rsi_df
        except Exception:
            pass
        
        # Bollinger Bands
        try:
            bb_df = self.db.execute_query(f"""
                SELECT ticker, trading_date,
                       CAST(Upper_Band AS FLOAT) as bb_upper,
                       CAST(Lower_Band AS FLOAT) as bb_lower
                FROM dbo.nse_500_bollingerband
                WHERE trading_date >= DATEADD(day, -90, CAST(GETDATE() AS DATE))
                {ticker_filter}
            """)
            enriched['bb'] = bb_df
        except Exception:
            pass
        
        # Stochastic
        try:
            stoch_df = self.db.execute_query(f"""
                SELECT ticker, trading_date,
                       CAST(stoch_14d_k AS FLOAT) as stoch_k,
                       CAST(stoch_14d_d AS FLOAT) as stoch_d,
                       CAST(momentum_strength AS FLOAT) as stoch_momentum
                FROM dbo.nse_500_stochastic
                WHERE trading_date >= DATEADD(day, -90, CAST(GETDATE() AS DATE))
                {ticker_filter}
            """)
            enriched['stoch'] = stoch_df
        except Exception:
            pass
        
        # ATR
        try:
            atr_df = self.db.execute_query(f"""
                SELECT ticker, trading_date,
                       CAST(ATR_14 AS FLOAT) as atr_14
                FROM dbo.nse_500_atr
                WHERE trading_date >= DATEADD(day, -90, CAST(GETDATE() AS DATE))
                {ticker_filter}
            """)
            enriched['atr'] = atr_df
        except Exception:
            pass
        
        # Fibonacci
        try:
            fib_df = self.db.execute_query(f"""
                SELECT ticker, trading_date,
                       CAST(distance_to_nearest_fib_pct AS FLOAT) as fib_distance_pct,
                       fib_trade_signal
                FROM dbo.nse_500_fibonacci
                WHERE trading_date >= DATEADD(day, -90, CAST(GETDATE() AS DATE))
                {ticker_filter}
            """)
            enriched['fibonacci'] = fib_df
        except Exception:
            pass
        
        # Support/Resistance
        try:
            sr_df = self.db.execute_query(f"""
                SELECT ticker, trading_date,
                       CAST(distance_to_s1_pct AS FLOAT) as sr_distance_to_support_pct,
                       CAST(distance_to_r1_pct AS FLOAT) as sr_distance_to_resistance_pct,
                       pivot_status,
                       sr_trade_signal
                FROM dbo.nse_500_support_resistance
                WHERE trading_date >= DATEADD(day, -90, CAST(GETDATE() AS DATE))
                {ticker_filter}
            """)
            enriched['support_resistance'] = sr_df
        except Exception:
            pass
        
        # Candlestick Patterns
        try:
            pat_df = self.db.execute_query(f"""
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
                WHERE trading_date >= DATEADD(day, -90, CAST(GETDATE() AS DATE))
                {ticker_filter}
            """)
            enriched['patterns'] = pat_df
        except Exception:
            pass
        
        # SMA Signals (SMA 200, 100 + Above/Below flags)
        try:
            sma_df = self.db.execute_query(f"""
                SELECT ticker, trading_date,
                       CAST(SMA_200 AS FLOAT) as sma_200,
                       CAST(SMA_100 AS FLOAT) as sma_100,
                       SMA_200_Flag, SMA_100_Flag, SMA_50_Flag, SMA_20_Flag
                FROM dbo.nse_500_sma_signals
                WHERE trading_date >= DATEADD(day, -90, CAST(GETDATE() AS DATE))
                {ticker_filter}
            """)
            enriched['sma_signals'] = sma_df
        except Exception:
            pass
        
        # MACD Signals (crossover)
        try:
            macd_sig_df = self.db.execute_query(f"""
                SELECT ticker, trading_date,
                       MACD_Signal as macd_crossover_signal
                FROM dbo.nse_500_macd_signals
                WHERE trading_date >= DATEADD(day, -90, CAST(GETDATE() AS DATE))
                {ticker_filter}
            """)
            enriched['macd_signals'] = macd_sig_df
        except Exception:
            pass
        
        return enriched
    
    def encode_signal_features(self, df):
        """Encode categorical signal columns from DB views into numeric features.
        Uses encoding maps from training metadata for consistency."""
        
        # Get encoding maps (from trained model metadata or defaults)
        signal_map = getattr(self, '_signal_map', {
            'STRONG_BUY': 2, 'BUY': 1, 'BUY_FIB_500': 1, 'BUY_FIB_382': 1,
            'BUY_FIB_236': 1, 'BUY_BULLISH_CROSS': 1,
            'NEUTRAL': 0, 'NEUTRAL_WAIT': 0, 'No Signal': 0,
            'SELL': -1, 'SELL_BEARISH_CROSS': -1,
            'STRONG_SELL': -2,
            'NEAR_SUPPORT_BUY': 1, 'BULLISH_ZONE': 1,
            'NEAR_RESISTANCE_SELL': -1, 'BEARISH_ZONE': -1,
            'Bullish Crossover': 1, 'Bearish Crossover': -1,
        })
        
        position_map = getattr(self, '_position_map', {
            'ABOVE_PIVOT': 1, 'BELOW_PIVOT': -1,
            'Above': 1, 'Below': -1,
            'BULLISH': 1, 'BEARISH': -1,
        })
        
        df_enc = df.copy()
        
        # Fibonacci
        if 'fib_trade_signal' in df_enc.columns:
            df_enc['fib_signal_strength'] = df_enc['fib_trade_signal'].map(signal_map).fillna(0)
            df_enc.drop(columns=['fib_trade_signal'], inplace=True, errors='ignore')
        else:
            df_enc['fib_signal_strength'] = 0
        if 'fib_distance_pct' not in df_enc.columns:
            df_enc['fib_distance_pct'] = 0
        
        # Support/Resistance
        if 'pivot_status' in df_enc.columns:
            df_enc['sr_pivot_position'] = df_enc['pivot_status'].map(position_map).fillna(0)
            df_enc.drop(columns=['pivot_status'], inplace=True, errors='ignore')
        else:
            df_enc['sr_pivot_position'] = 0
        
        if 'sr_trade_signal' in df_enc.columns:
            df_enc['sr_signal_strength'] = df_enc['sr_trade_signal'].map(signal_map).fillna(0)
            df_enc.drop(columns=['sr_trade_signal'], inplace=True, errors='ignore')
        else:
            df_enc['sr_signal_strength'] = 0
        
        for col in ['sr_distance_to_support_pct', 'sr_distance_to_resistance_pct']:
            if col not in df_enc.columns:
                df_enc[col] = 0
        
        # Patterns
        if 'pattern_signal' in df_enc.columns:
            df_enc['pattern_signal_strength'] = df_enc['pattern_signal'].map(signal_map).fillna(0)
            df_enc.drop(columns=['pattern_signal'], inplace=True, errors='ignore')
        else:
            df_enc['pattern_signal_strength'] = 0
        
        for col in ['has_doji', 'has_hammer', 'has_shooting_star',
                     'has_bullish_engulfing', 'has_bearish_engulfing',
                     'has_morning_star', 'has_evening_star']:
            if col not in df_enc.columns:
                df_enc[col] = 0
            else:
                df_enc[col] = df_enc[col].fillna(0).astype(float)
        
        # SMA signals
        if 'sma_200' in df_enc.columns:
            df_enc['price_vs_sma100'] = df_enc['close_price'] / df_enc['sma_100'].replace(0, np.nan)
            df_enc['price_vs_sma200'] = df_enc['close_price'] / df_enc['sma_200'].replace(0, np.nan)
        else:
            df_enc['sma_200'] = 0
            df_enc['sma_100'] = 0
            df_enc['price_vs_sma100'] = 1.0
            df_enc['price_vs_sma200'] = 1.0
        
        for flag_col in ['SMA_200_Flag', 'SMA_100_Flag', 'SMA_50_Flag', 'SMA_20_Flag']:
            target_col = flag_col.lower().replace('_flag', '_flag')
            if flag_col in df_enc.columns:
                df_enc[target_col] = df_enc[flag_col].map(position_map).fillna(0)
                df_enc.drop(columns=[flag_col], inplace=True, errors='ignore')
            else:
                df_enc[target_col] = 0
        
        # MACD signal
        if 'macd_crossover_signal' in df_enc.columns:
            df_enc['macd_signal_strength'] = df_enc['macd_crossover_signal'].map(signal_map).fillna(0)
            df_enc.drop(columns=['macd_crossover_signal'], inplace=True, errors='ignore')
        else:
            df_enc['macd_signal_strength'] = 0
        
        # Clean up leftover non-numeric columns
        drop_cols = ['bb_close', 'bb_sma20', 'db_macd', 'db_macd_signal',
                     'patterns_detected', 'rsi_trade_signal']
        df_enc.drop(columns=[c for c in drop_cols if c in df_enc.columns],
                    inplace=True, errors='ignore')
        
        return df_enc
    
    def calculate_technical_indicators(self, df, enriched_data=None):
        """Calculate technical indicators for NSE data - consistent with training"""
        if df is None or df.empty:
            return None
        
        df_features = df.copy()
        df_features = df_features.sort_values(['ticker', 'trading_date'])
        df_features['trading_date'] = pd.to_datetime(df_features['trading_date'])
        
        # CRITICAL: Remove duplicate ticker-date rows from base data
        # (source table may have duplicates from overlapping yfinance fetches)
        df_features = df_features.drop_duplicates(subset=['ticker', 'trading_date'], keep='last')
        
        # Merge enriched features if available
        if enriched_data:
            for name, enriched_df in enriched_data.items():
                if enriched_df is not None and not enriched_df.empty:
                    enriched_df = enriched_df.copy()
                    enriched_df['trading_date'] = pd.to_datetime(enriched_df['trading_date'])
                    # Remove duplicates from enriched view
                    enriched_df = enriched_df.drop_duplicates(subset=['ticker', 'trading_date'], keep='last')
                    merge_cols = [c for c in enriched_df.columns if c not in ['ticker', 'trading_date']]
                    df_features = df_features.merge(
                        enriched_df[['ticker', 'trading_date'] + merge_cols],
                        on=['ticker', 'trading_date'],
                        how='left'
                    )
        
        # Encode categorical signals to numeric (fibonacci, S/R, patterns, SMA flags, MACD signal)
        df_features = self.encode_signal_features(df_features)
        
        safe_print(f"[PROCESSING] Calculating indicators for {df_features['ticker'].nunique()} stocks...")
        
        grouped_results = []
        
        for ticker, group in df_features.groupby('ticker'):
            group = group.sort_values('trading_date').reset_index(drop=True)
            
            if len(group) < 30:
                continue
            
            # === Basic Price Features ===
            group['daily_return'] = group['close_price'].pct_change() * 100
            group['daily_volatility'] = group['daily_return'].rolling(window=10).std()
            group['volume_millions'] = group['volume'] / 1_000_000
            group['price_range'] = group['high_price'] - group['low_price']
            
            range_nonzero = group['price_range'].replace(0, np.nan)
            group['price_position'] = (group['close_price'] - group['low_price']) / range_nonzero
            
            group['gap'] = (group['open_price'] - group['close_price'].shift(1)) / group['close_price'].shift(1) * 100
            group['volume_price_trend'] = (group['daily_return'] * group['volume']).rolling(window=10).mean()
            
            # === RSI ===
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
            
            # === MACD ===
            ema_12 = group['close_price'].ewm(span=12, adjust=False).mean()
            ema_26 = group['close_price'].ewm(span=26, adjust=False).mean()
            group['macd'] = ema_12 - ema_26
            group['macd_signal'] = group['macd'].ewm(span=9, adjust=False).mean()
            group['macd_histogram'] = group['macd'] - group['macd_signal']
            
            # === Price vs MA Ratios ===
            group['price_vs_sma20'] = group['close_price'] / group['sma_20']
            group['price_vs_sma50'] = group['close_price'] / group['sma_50']
            group['price_vs_ema20'] = group['close_price'] / group['ema_20']
            
            # === MA Crossover Ratios ===
            group['sma20_vs_sma50'] = group['sma_20'] / group['sma_50']
            group['ema20_vs_ema50'] = group['ema_20'] / group['ema_50']
            group['sma5_vs_sma20'] = group['sma_5'] / group['sma_20']
            
            # === Volume ===
            group['volume_sma_20'] = group['volume'].rolling(window=20).mean()
            vol_sma_nonzero = group['volume_sma_20'].replace(0, np.nan)
            group['volume_sma_ratio'] = group['volume'] / vol_sma_nonzero
            group['vol_change_ratio'] = group['volume'].pct_change()
            
            # === Price Momentum ===
            group['price_momentum_5'] = group['close_price'] / group['close_price'].shift(5)
            group['price_momentum_10'] = group['close_price'] / group['close_price'].shift(10)
            
            # === Volatility ===
            group['price_volatility_10'] = group['close_price'].pct_change().rolling(window=10).std()
            group['price_volatility_20'] = group['close_price'].pct_change().rolling(window=20).std()
            
            # === Trend Strength ===
            group['trend_strength_10'] = group['close_price'].rolling(window=10).apply(
                lambda x: (x.iloc[-1] - x.iloc[0]) / x.std() if x.std() != 0 else 0, raw=False
            )
            
            # === Multi-day Returns ===
            group['return_1d'] = group['close_price'].pct_change() * 100
            group['return_3d'] = group['close_price'].pct_change(3) * 100
            group['return_5d'] = group['close_price'].pct_change(5) * 100
            
            # === Candlestick Features ===
            group['high_low_ratio'] = group['high_price'] / group['low_price']
            group['close_open_ratio'] = group['close_price'] / group['open_price']
            group['upper_shadow'] = (group['high_price'] - np.maximum(group['open_price'], group['close_price'])) / group['price_range'].replace(0, np.nan)
            group['lower_shadow'] = (np.minimum(group['open_price'], group['close_price']) - group['low_price']) / group['price_range'].replace(0, np.nan)
            
            # === Bollinger Bands ===
            if 'bb_upper' not in group.columns or group['bb_upper'].isna().all():
                bb_sma = group['close_price'].rolling(window=20).mean()
                bb_std = group['close_price'].rolling(window=20).std()
                group['bb_upper'] = bb_sma + (2 * bb_std)
                group['bb_lower'] = bb_sma - (2 * bb_std)
            
            bb_range = (group['bb_upper'] - group['bb_lower']).replace(0, np.nan)
            group['bb_width'] = bb_range / group['close_price']
            group['bb_position'] = (group['close_price'] - group['bb_lower']) / bb_range
            
            # === Stochastic ===
            if 'stoch_k' not in group.columns or group['stoch_k'].isna().all():
                low_14 = group['low_price'].rolling(window=14).min()
                high_14 = group['high_price'].rolling(window=14).max()
                stoch_range = (high_14 - low_14).replace(0, np.nan)
                group['stoch_k'] = ((group['close_price'] - low_14) / stoch_range) * 100
                group['stoch_d'] = group['stoch_k'].rolling(window=3).mean()
                group['stoch_momentum'] = group['stoch_k'] - group['stoch_d']
            
            # === ATR ===
            if 'atr_14' not in group.columns or group['atr_14'].isna().all():
                high_low = group['high_price'] - group['low_price']
                high_close = abs(group['high_price'] - group['close_price'].shift(1))
                low_close = abs(group['low_price'] - group['close_price'].shift(1))
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                group['atr_14'] = true_range.rolling(window=14).mean()
            
            group['atr_pct'] = group['atr_14'] / group['close_price'] * 100
            
            # === Market Regime Detection ===
            sma20 = group['close_price'].rolling(window=20).mean()
            group['regime_sma20_slope'] = sma20.pct_change(5) * 100
            
            up_move = group['high_price'].diff()
            down_move = -group['low_price'].diff()
            pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            pos_dm_smooth = pd.Series(pos_dm, index=group.index).rolling(14).mean()
            neg_dm_smooth = pd.Series(neg_dm, index=group.index).rolling(14).mean()
            dm_sum = pos_dm_smooth + neg_dm_smooth
            dx = np.where(dm_sum > 0, np.abs(pos_dm_smooth - neg_dm_smooth) / dm_sum * 100, 0)
            group['regime_adx'] = pd.Series(dx, index=group.index).rolling(14).mean()
            
            vol_short = group['close_price'].pct_change().rolling(10).std()
            vol_long = group['close_price'].pct_change().rolling(60).std()
            group['regime_vol_ratio'] = np.where(vol_long > 0, vol_short / vol_long, 1.0)
            
            sma50 = group['close_price'].rolling(window=50).mean()
            group['regime_mean_reversion'] = np.where(
                group['atr_14'] > 0,
                (group['close_price'] - sma50) / group['atr_14'],
                0
            )
            
            overall_dir = np.sign(group['close_price'].diff(20))
            daily_dirs = np.sign(group['close_price'].diff(1))
            consistent = (daily_dirs == overall_dir).astype(float)
            group['regime_trend_consistency'] = consistent.rolling(20).mean()
            
            # === Date Features ===
            group['day_of_week'] = pd.to_datetime(group['trading_date']).dt.dayofweek
            group['month'] = pd.to_datetime(group['trading_date']).dt.month
            
            grouped_results.append(group)
        
        if not grouped_results:
            safe_print("[WARN] No valid data after indicator calculation")
            return None
        
        result_df = pd.concat(grouped_results, ignore_index=True)
        
        # Handle infinities and NaN
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        result_df = result_df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        safe_print(f"[SUCCESS] Indicators calculated for {len(result_df):,} records")
        return result_df
    
    def make_predictions(self, df):
        """Make trading signal predictions using ensemble of NSE models"""
        if df is None or df.empty:
            return None
        
        safe_print("[TARGET] Making ML predictions...")
        
        # Get latest data for each ticker
        latest_data = df.groupby('ticker').tail(1).copy()
        
        # Determine the actual trading date being predicted from
        # (NSE data is available same day after market close at 3:30 PM IST)
        self.latest_trading_date = pd.to_datetime(latest_data['trading_date']).max()
        safe_print(f"  [DATA] Latest trading date in data: {self.latest_trading_date.strftime('%Y-%m-%d')}")
        safe_print(f"  [DATA] Predictions are for: NEXT trading day after {self.latest_trading_date.strftime('%Y-%m-%d')}")
        
        # Prepare features (only use features that exist)
        available_features = [col for col in self.feature_columns if col in latest_data.columns]
        missing_features = [col for col in self.feature_columns if col not in latest_data.columns]
        
        if missing_features:
            safe_print(f"  [WARN] Missing {len(missing_features)} features, filling with 0")
            for feat in missing_features:
                latest_data[feat] = 0
            available_features = self.feature_columns
        
        feature_data = latest_data[available_features].copy()
        feature_data = feature_data.fillna(0)
        
        # Scale features for classification
        clf_scaled = self.clf_scaler.transform(feature_data)
        
        if self.use_nse_models:
            # === F1-Weighted Ensemble Classification Predictions ===
            # Exclude the 'Ensemble' (VotingClassifier) model to avoid double-counting
            # since it's already a combination of the base models.
            # Weight each base model's probabilities by its training F1 score.
            clf_results = self.metadata.get('clf_results', {})
            
            all_probabilities = []
            model_weights = []
            model_names_used = []
            
            for model_name, model in self.clf_models.items():
                # Skip the redundant VotingClassifier ensemble
                if model_name.lower() == 'ensemble':
                    continue
                try:
                    proba = model.predict_proba(clf_scaled)
                    all_probabilities.append(proba)
                    model_names_used.append(model_name)
                    # Get F1 weight from training metadata (default 1.0 if not found)
                    f1 = clf_results.get(model_name, {}).get('f1_score', 1.0)
                    model_weights.append(f1)
                except Exception as e:
                    safe_print(f"  [WARN] {model_name} prediction failed: {e}")
            
            if not all_probabilities:
                safe_print("[ERROR] All classification models failed")
                return None
            
            # F1-weighted average of probabilities (better models contribute more)
            weights = np.array(model_weights)
            weights = weights / weights.sum()  # Normalize to sum to 1
            avg_probabilities = np.average(all_probabilities, axis=0, weights=weights)
            ensemble_predictions = np.argmax(avg_probabilities, axis=1)
            
            # Decode predictions
            predicted_labels = self.direction_encoder.inverse_transform(ensemble_predictions)
            
            # Map direction labels to trading signals
            signal_map = {'Up': 'Buy', 'Down': 'Sell'}
            latest_data['predicted_signal'] = [signal_map.get(label, 'Hold') for label in predicted_labels]
            
            # Probabilities
            class_names = self.direction_encoder.classes_
            up_idx = list(class_names).index('Up') if 'Up' in class_names else 0
            down_idx = list(class_names).index('Down') if 'Down' in class_names else 1
            
            latest_data['buy_probability'] = avg_probabilities[:, up_idx]
            latest_data['sell_probability'] = avg_probabilities[:, down_idx]
            latest_data['hold_probability'] = 0.0  # Binary classification, no hold class
            
            # Confidence = max probability (calibrated from ensemble)
            latest_data['confidence'] = np.max(avg_probabilities, axis=1)
            latest_data['confidence_percentage'] = latest_data['confidence'] * 100
            
            # Model info
            latest_data['model_name'] = 'NSE_Ensemble'
            latest_data['model_version'] = self.metadata.get('training_timestamp', 'unknown')
            
            # === Regression Predictions (predicted price change) ===
            if self.reg_models:
                try:
                    reg_scaled = self.reg_scaler.transform(feature_data)
                    
                    reg_predictions = []
                    for model_name, model in self.reg_models.items():
                        try:
                            pred = model.predict(reg_scaled)
                            reg_predictions.append(pred)
                        except Exception:
                            pass
                    
                    if reg_predictions:
                        # Average regression predictions
                        avg_reg_pred = np.mean(reg_predictions, axis=0)
                        latest_data['predicted_change_pct'] = avg_reg_pred
                        latest_data['predicted_price'] = latest_data['close_price'] * (1 + avg_reg_pred / 100)
                except Exception as e:
                    safe_print(f"  [WARN] Regression prediction failed: {e}")
            
            weight_info = ', '.join([f"{n}({w:.2f})" for n, w in zip(model_names_used, weights)])
            safe_print(f"  [OK] F1-weighted ensemble of {len(model_names_used)} models: {weight_info}")
            
        else:
            # Legacy single model prediction
            predictions = self.best_classifier.predict(clf_scaled)
            probabilities = self.best_classifier.predict_proba(clf_scaled)
            
            # Map legacy labels to signals
            predicted_labels = self.direction_encoder.inverse_transform(predictions)
            
            # Legacy model uses 'Overbought (Sell)' / 'Oversold (Buy)'
            signal_map = {
                'Overbought (Sell)': 'Sell',
                'Oversold (Buy)': 'Buy',
                'Up': 'Buy',
                'Down': 'Sell'
            }
            latest_data['predicted_signal'] = [signal_map.get(label, 'Hold') for label in predicted_labels]
            
            latest_data['buy_probability'] = probabilities[:, 0] if probabilities.shape[1] > 0 else 0
            latest_data['sell_probability'] = probabilities[:, 1] if probabilities.shape[1] > 1 else 0
            latest_data['hold_probability'] = 0.0
            
            latest_data['confidence'] = np.max(probabilities, axis=1)
            latest_data['confidence_percentage'] = latest_data['confidence'] * 100
            
            latest_data['model_name'] = 'ExtraTreesClassifier (Legacy)'
            latest_data['model_version'] = 'legacy'
        
        # Confidence levels
        latest_data['high_confidence'] = (latest_data['confidence'] >= 0.7).astype(int)
        latest_data['medium_confidence'] = ((latest_data['confidence'] >= 0.55) & (latest_data['confidence'] < 0.7)).astype(int)
        latest_data['low_confidence'] = (latest_data['confidence'] < 0.55).astype(int)
        
        # Signal strength
        latest_data['signal_strength'] = latest_data.apply(
            lambda x: 'High' if x['high_confidence'] else ('Medium' if x['medium_confidence'] else 'Low'),
            axis=1
        )
        
        # RSI category
        latest_data['rsi_category'] = latest_data['RSI'].apply(
            lambda x: 'Overbought' if x > 70 else ('Oversold' if x < 30 else 'Neutral')
        )
        
        # Summary stats
        buy_count = (latest_data['predicted_signal'] == 'Buy').sum()
        sell_count = (latest_data['predicted_signal'] == 'Sell').sum()
        high_conf = latest_data['high_confidence'].sum()
        
        safe_print(f"[SUCCESS] Predictions for {len(latest_data)} stocks: "
                   f"{buy_count} Buy, {sell_count} Sell, "
                   f"{high_conf} High Confidence")
        
        return latest_data
    
    def save_predictions_to_db(self, predictions_df):
        """Save predictions to NSE database tables"""
        if predictions_df is None or predictions_df.empty:
            return False
        
        safe_print("[SAVE] Saving predictions to database...")
        
        try:
            run_timestamp = datetime.now()
            
            # IMPORTANT: Use the actual trading date from the data, NOT today's date.
            #
            # TIMEZONE CONTEXT (EST/IST):
            # - NSE closes at 3:30 PM IST = ~5:00 AM EST
            # - This system runs at ~7 AM EST
            # - On weekdays: today's NSE data is already available (NSE closed 2 hours ago)
            #   e.g., Monday 7 AM EST -> NSE Monday data (Feb 9) is in the DB
            # - On weekends: latest data is from Friday
            #   e.g., Saturday 7 AM EST -> latest data is Friday (Feb 6)
            # - The trading_date in predictions = the date of the data used
            # - The prediction is for the NEXT trading day after that date
            #
            actual_trading_date = pd.to_datetime(predictions_df['trading_date']).max()
            trading_date_str = actual_trading_date.strftime('%Y-%m-%d')
            today_str = run_timestamp.strftime('%Y-%m-%d')
            
            safe_print(f"  [DATA] NSE trading date (data from): {trading_date_str}")
            safe_print(f"  [DATA] Run date (EST): {today_str}")
            
            if trading_date_str == today_str:
                safe_print(f"  [INFO] Using today's NSE data (NSE closed at ~5 AM EST)")
            elif trading_date_str != today_str:
                day_of_week = run_timestamp.weekday()
                if day_of_week in [5, 6]:  # Saturday or Sunday
                    safe_print(f"  [INFO] Weekend run - using Friday's NSE data ({trading_date_str})")
                else:
                    safe_print(f"  [INFO] Data from {trading_date_str} (run date: {today_str})")
            
            # Prepare prediction records
            prediction_records = []
            technical_records = []
            
            for _, row in predictions_df.iterrows():
                pred_record = {
                    'run_timestamp': run_timestamp,
                    'trading_date': row['trading_date'],
                    'ticker': row['ticker'],
                    'company': row.get('company', ''),
                    'predicted_signal': row['predicted_signal'],
                    'confidence': float(row['confidence']),
                    'confidence_percentage': float(row['confidence_percentage']),
                    'signal_strength': row['signal_strength'],
                    'close_price': float(row['close_price']),
                    'volume': int(row['volume']),
                    'rsi': float(row.get('RSI', 0)),
                    'rsi_category': row.get('rsi_category', 'Neutral'),
                    'high_confidence': int(row['high_confidence']),
                    'medium_confidence': int(row['medium_confidence']),
                    'low_confidence': int(row['low_confidence']),
                    'sell_probability': float(row.get('sell_probability', 0)),
                    'buy_probability': float(row.get('buy_probability', 0)),
                    'hold_probability': float(row.get('hold_probability', 0)),
                    'model_name': row.get('model_name', 'Unknown'),
                    'model_version': row.get('model_version', 'unknown'),
                }
                prediction_records.append(pred_record)
                
                # Technical indicators record
                tech_record = {
                    'run_timestamp': run_timestamp,
                    'trading_date': row['trading_date'],
                    'ticker': row['ticker'],
                    'rsi': float(row.get('RSI', 0)),
                    'rsi_oversold': int(row.get('rsi_oversold', 0)),
                    'rsi_overbought': int(row.get('rsi_overbought', 0)),
                    'rsi_momentum': float(row.get('rsi_momentum', 0)),
                    'sma_5': float(row.get('sma_5', 0)),
                    'sma_10': float(row.get('sma_10', 0)),
                    'sma_20': float(row.get('sma_20', 0)),
                    'sma_50': float(row.get('sma_50', 0)),
                    'ema_5': float(row.get('ema_5', 0)),
                    'ema_10': float(row.get('ema_10', 0)),
                    'ema_20': float(row.get('ema_20', 0)),
                    'ema_50': float(row.get('ema_50', 0)),
                    'macd': float(row.get('macd', 0)),
                    'macd_signal': float(row.get('macd_signal', 0)),
                    'macd_histogram': float(row.get('macd_histogram', 0)),
                    'price_vs_sma20': float(row.get('price_vs_sma20', 0)),
                    'price_vs_sma50': float(row.get('price_vs_sma50', 0)),
                    'price_vs_ema20': float(row.get('price_vs_ema20', 0)),
                    'sma20_vs_sma50': float(row.get('sma20_vs_sma50', 0)),
                    'ema20_vs_ema50': float(row.get('ema20_vs_ema50', 0)),
                    'sma5_vs_sma20': float(row.get('sma5_vs_sma20', 0)),
                    'volume_sma_20': float(row.get('volume_sma_20', 0)),
                    'volume_sma_ratio': float(row.get('volume_sma_ratio', 0)),
                    'price_momentum_5': float(row.get('price_momentum_5', 0)),
                    'price_momentum_10': float(row.get('price_momentum_10', 0)),
                    'daily_volatility': float(row.get('daily_volatility', 0)),
                    'price_volatility_10': float(row.get('price_volatility_10', 0)),
                    'price_volatility_20': float(row.get('price_volatility_20', 0)),
                    'trend_strength_10': float(row.get('trend_strength_10', 0)),
                    'volume_price_trend': float(row.get('volume_price_trend', 0)),
                    'gap': float(row.get('gap', 0)),
                }
                
                # Add Bollinger Band columns if available
                if 'bb_upper' in row and pd.notna(row.get('bb_upper')):
                    tech_record['bb_upper'] = float(row['bb_upper'])
                    tech_record['bb_lower'] = float(row.get('bb_lower', 0))
                    tech_record['bb_middle'] = float(row.get('sma_20', 0))
                    tech_record['bb_width'] = float(row.get('bb_width', 0))
                
                # Add ATR if available
                if 'atr_14' in row and pd.notna(row.get('atr_14')):
                    tech_record['atr'] = float(row['atr_14'])
                    tech_record['atr_percentage'] = float(row.get('atr_pct', 0))
                
                technical_records.append(tech_record)
            
            # Clear existing data for the ACTUAL trading date (not today's date)
            # This prevents duplicates when re-running for the same trading day
            try:
                self.db.execute_query(
                    f"DELETE FROM ml_nse_trading_predictions WHERE trading_date = '{trading_date_str}'"
                )
                safe_print(f"  [OK] Cleared old predictions for {trading_date_str}")
            except Exception:
                pass
            
            try:
                self.db.execute_query(
                    f"DELETE FROM ml_nse_technical_indicators WHERE trading_date = '{trading_date_str}'"
                )
                safe_print(f"  [OK] Cleared old technical indicators for {trading_date_str}")
            except Exception:
                pass
            
            # Insert predictions
            engine = self.db.get_sqlalchemy_engine()
            
            pred_df = pd.DataFrame(prediction_records)
            pred_df.to_sql('ml_nse_trading_predictions', engine, if_exists='append', index=False, schema='dbo')
            
            tech_df = pd.DataFrame(technical_records)
            tech_df.to_sql('ml_nse_technical_indicators', engine, if_exists='append', index=False, schema='dbo')
            
            safe_print(f"[SUCCESS] Saved {len(prediction_records)} predictions and {len(technical_records)} technical indicators")
            
            # Create summary
            self.create_prediction_summary(predictions_df, run_timestamp)
            
            return True
            
        except Exception as e:
            safe_print(f"[ERROR] Error saving to database: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_prediction_summary(self, predictions_df, run_timestamp):
        """Create daily prediction summary"""
        try:
            # Use actual trading date from data for the analysis_date
            # This ensures summary aligns with the predictions' trading_date
            actual_trading_date = pd.to_datetime(predictions_df['trading_date']).max()
            analysis_date = actual_trading_date.strftime('%Y-%m-%d')
            
            total_predictions = len(predictions_df)
            buy_signals = predictions_df[predictions_df['predicted_signal'] == 'Buy']
            sell_signals = predictions_df[predictions_df['predicted_signal'] == 'Sell']
            hold_signals = predictions_df[predictions_df['predicted_signal'] == 'Hold']
            
            high_conf = predictions_df[predictions_df['high_confidence'] == 1]
            medium_conf = predictions_df[predictions_df['medium_confidence'] == 1]
            low_conf = predictions_df[predictions_df['low_confidence'] == 1]
            
            # Determine market trend from buy/sell ratio
            buy_ratio = len(buy_signals) / total_predictions if total_predictions > 0 else 0
            if buy_ratio > 0.6:
                market_trend = 'Bullish'
            elif buy_ratio < 0.4:
                market_trend = 'Bearish'
            else:
                market_trend = 'Neutral'
            
            # Model accuracy from metadata
            model_accuracy = None
            if self.use_nse_models and self.metadata:
                best_clf = self.metadata.get('best_clf_model', '')
                clf_results = self.metadata.get('clf_results', {})
                if best_clf in clf_results:
                    model_accuracy = clf_results[best_clf].get('accuracy', None)
            
            # Top signals
            top_buys = buy_signals.nlargest(5, 'confidence')[['ticker', 'confidence_percentage']].to_dict('records')
            top_sells = sell_signals.nlargest(5, 'confidence')[['ticker', 'confidence_percentage']].to_dict('records')
            
            import json
            
            summary_data = {
                'run_timestamp': run_timestamp,
                'analysis_date': analysis_date,
                'total_predictions': total_predictions,
                'total_buy_signals': len(buy_signals),
                'total_sell_signals': len(sell_signals),
                'total_hold_signals': len(hold_signals),
                'high_confidence_count': int(high_conf.shape[0]),
                'medium_confidence_count': int(medium_conf.shape[0]),
                'low_confidence_count': int(low_conf.shape[0]),
                'high_conf_buys': int(len(buy_signals[buy_signals['high_confidence'] == 1])),
                'medium_conf_buys': int(len(buy_signals[buy_signals['medium_confidence'] == 1])),
                'low_conf_buys': int(len(buy_signals[buy_signals['low_confidence'] == 1])),
                'high_conf_sells': int(len(sell_signals[sell_signals['high_confidence'] == 1])),
                'medium_conf_sells': int(len(sell_signals[sell_signals['medium_confidence'] == 1])),
                'low_conf_sells': int(len(sell_signals[sell_signals['low_confidence'] == 1])),
                'avg_confidence': float(predictions_df['confidence'].mean()),
                'avg_rsi': float(predictions_df['RSI'].mean()),
                'avg_buy_probability': float(predictions_df['buy_probability'].mean()),
                'avg_sell_probability': float(predictions_df['sell_probability'].mean()),
                'top_buy_signals': json.dumps(top_buys, default=str),
                'top_sell_signals': json.dumps(top_sells, default=str),
                'market_trend': market_trend,
                'bullish_stocks_count': int((predictions_df.get('price_vs_sma20', pd.Series([1])) > 1).sum()),
                'bearish_stocks_count': int((predictions_df.get('price_vs_sma20', pd.Series([1])) <= 1).sum()),
                'model_accuracy': model_accuracy,
                'total_stocks_processed': total_predictions,
            }
            
            # Clear existing summary for the actual trading date
            try:
                self.db.execute_query(f"DELETE FROM ml_nse_predict_summary WHERE analysis_date = '{analysis_date}'")
            except Exception:
                pass
            
            engine = self.db.get_sqlalchemy_engine()
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_sql('ml_nse_predict_summary', engine, if_exists='append', index=False, schema='dbo')
            
            safe_print(f"[SUCCESS] Summary: {total_predictions} predictions, "
                       f"{len(buy_signals)} buys, {len(sell_signals)} sells, "
                       f"Market: {market_trend}")
            
        except Exception as e:
            safe_print(f"[ERROR] Error creating summary: {e}")
            import traceback
            traceback.print_exc()
    
    def run_prediction(self, ticker=None):
        """Run complete prediction workflow"""
        safe_print("[NSE] Starting NSE 500 prediction workflow...")
        safe_print(f"  Models: {'NSE-specific ensemble' if self.use_nse_models else 'Legacy (NASDAQ-trained - UNRELIABLE)'}")
        safe_print(f"  Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get data (90 days for indicator calculation, predictions use latest day)
        df = self.get_nse_data(ticker, days_back=90)
        if df is None:
            return False
        
        # Show data timing info
        # TIMEZONE CONTEXT: This system runs in EST. NSE closes at 3:30 PM IST = ~5:00 AM EST.
        # So when this runs at 7 AM EST on a weekday, today's NSE data is already available.
        # Example: Monday 7 AM EST -> NSE closed Monday at 5 AM EST -> Feb 9 data is available.
        # On weekends: Saturday/Sunday -> latest data is from Friday.
        latest_date = pd.to_datetime(df['trading_date']).max()
        today = datetime.now().date()
        data_age = (today - latest_date.date()).days
        
        safe_print(f"  [DATA] Latest NSE trading date in DB: {latest_date.strftime('%Y-%m-%d')}")
        safe_print(f"  [DATA] Current date (EST): {today}")
        safe_print(f"  [DATA] Data age: {data_age} day(s)")
        
        if data_age > 3:
            safe_print(f"  [WARN] Data is {data_age} days old - check if NSE data feed is running")
        elif data_age == 0:
            safe_print(f"  [INFO] Today's NSE data is available (NSE closes at 3:30 PM IST = ~5 AM EST)")
        elif data_age <= 2 and today.weekday() in [5, 6]:  # Saturday=5, Sunday=6
            safe_print(f"  [INFO] Weekend - using Friday's data (normal)")
        else:
            safe_print(f"  [INFO] Using most recent trading day data ({data_age} day(s) ago)")
        
        safe_print(f"  [INFO] Prediction target: NEXT trading day after {latest_date.strftime('%Y-%m-%d')}")
        
        # Load enriched features from DB views
        tickers = df['ticker'].unique().tolist() if len(df['ticker'].unique()) < 500 else None
        enriched_data = self.load_enriched_features_for_prediction(tickers)
        
        # Calculate technical indicators
        df_with_indicators = self.calculate_technical_indicators(df, enriched_data)
        if df_with_indicators is None:
            return False
        
        # Make predictions
        predictions = self.make_predictions(df_with_indicators)
        if predictions is None:
            return False
        
        # Save to database
        success = self.save_predictions_to_db(predictions)
        
        if success:
            safe_print("[COMPLETE] NSE prediction workflow completed successfully!")
            safe_print(f"  [INFO] Predictions stored for trading_date = {latest_date.strftime('%Y-%m-%d')}")
            safe_print(f"  [INFO] These predict the NEXT trading day's price direction after {latest_date.strftime('%Y-%m-%d')}")
            if data_age == 0:
                safe_print(f"  [INFO] Fresh data: using today's NSE close (NSE already closed by 5 AM EST)")
        else:
            safe_print("[ERROR] NSE prediction workflow failed")
        
        return success


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='NSE 500 Trading Signal Predictions')
    parser.add_argument('--ticker', help='Specific NSE ticker to predict (e.g., RELIANCE.NS)')
    parser.add_argument('--all-nse', action='store_true', help='Process all NSE 500 stocks')
    parser.add_argument('--check-only', action='store_true', help='Only check data availability')
    
    args = parser.parse_args()
    
    try:
        predictor = NSETradingSignalPredictor()
        
        if args.check_only:
            safe_print("[CHECK] Checking NSE data availability...")
            df = predictor.get_nse_data()
            if df is not None:
                safe_print(f"[SUCCESS] Found data for {df['ticker'].nunique()} NSE stocks")
                safe_print(f"[DATA] Date range: {df['trading_date'].min()} to {df['trading_date'].max()}")
                safe_print(f"[DATA] Using models: {'NSE-specific' if predictor.use_nse_models else 'Legacy'}")
            return
        
        if args.ticker:
            safe_print(f"[NSE] Processing ticker: {args.ticker}")
            success = predictor.run_prediction(args.ticker)
        else:
            safe_print("[NSE] Processing all NSE 500 stocks...")
            success = predictor.run_prediction()
        
        if success:
            safe_print("\n[SUCCESS] NSE prediction completed successfully!")
            safe_print("[SAVE] Results saved to:")
            safe_print("  - ml_nse_trading_predictions")
            safe_print("  - ml_nse_technical_indicators")
            safe_print("  - ml_nse_predict_summary")
        else:
            safe_print("[ERROR] NSE prediction failed")
    
    except Exception as e:
        safe_print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
