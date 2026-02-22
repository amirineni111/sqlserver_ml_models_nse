"""
NSE 500 Model Retraining Script

This script trains ML models specifically on NSE 500 data to predict:
1. 5-day price direction (classification: Up/Down) -- primary target
2. 5-day price change magnitude (regression)

Key improvements over the previous approach:
- Uses 5-day direction target instead of 1-day (less noise, more predictable)
- 3-way data split: 60% train / 20% calibration / 20% test (prevents data leakage)
- Isotonic probability calibration on dedicated calibration set
- Trains on NSE data (not NASDAQ) for proper market alignment
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
import json
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
from sklearn.feature_selection import mutual_info_classif


class PurgedTimeSeriesSplit:
    """
    Time series cross-validation with purge gap to prevent data leakage.
    With 5-day prediction targets, we need at least 5 samples gap.
    """
    
    def __init__(self, n_splits=5, purge_gap=5):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        min_train_size = max(50, n_samples // (self.n_splits + 2))
        fold_size = max(20, (n_samples - min_train_size) // self.n_splits)
        
        for i in range(self.n_splits):
            train_end = min_train_size + i * fold_size
            val_start = train_end + self.purge_gap
            val_end = min(val_start + fold_size, n_samples)
            
            if val_start >= n_samples or val_end <= val_start:
                continue
            
            train_indices = np.arange(0, train_end)
            val_indices = np.arange(val_start, val_end)
            
            yield train_indices, val_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

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
        
        # Sector encoder (fitted during feature engineering, saved with model)
        self.sector_encoder = None
        
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
            # --- Market Regime Detection ---
            'regime_sma20_slope', 'regime_adx', 'regime_vol_ratio',
            'regime_mean_reversion', 'regime_trend_consistency',
            # --- Fundamental Features (from nse_500_fundamentals) ---
            'profit_margin',
            'price_vs_52wk_high', 'price_vs_52wk_low', 'price_vs_200d_avg',
            'sector_encoded',
            # --- Market Context Features (from market_context_daily) ---
            'vix_close', 'vix_change_pct',
            'india_vix_close', 'india_vix_change_pct',
            'nifty50_return_1d', 'sp500_return_1d',
            'dxy_return_1d', 'us_10y_yield_close', 'us_10y_yield_change',
            'sector_index_return_1d',  # Mapped from sector → NIFTY sector index
            # --- Calendar Features (from market_calendar) ---
            'is_pre_holiday', 'is_post_holiday', 'is_short_week',
            'trading_days_in_week', 'is_month_end', 'is_month_start',
            'is_quarter_end', 'is_options_expiry',
            'days_until_next_holiday', 'days_since_last_holiday',
            'other_market_closed',
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
            
            # Load fundamentals separately and merge (avoids slow SQL JOINs)
            df = self._merge_fundamentals(df)
            
            # Load market context (VIX, NIFTY indices, sector returns, treasury) and merge
            df = self._merge_market_context(df)
            
            # Load calendar features (holidays, short weeks, expiry) and merge
            df = self._merge_calendar_features(df, market='NSE')
            
            return df
            
        except Exception as e:
            print(f"[ERROR] Error loading NSE data: {e}")
            raise
    
    def _merge_fundamentals(self, df):
        """Load NSE 500 fundamentals & sector data separately and merge into main DataFrame.
        
        Based on NASDAQ feature selection results, only these fundamental features
        survived as top-20 important:
        - price_vs_52wk_high (#2 most important)
        - price_vs_52wk_low (#3 most important)
        - profit_margin (#12 most important)
        
        We load the raw data here and compute derived features in engineer_features().
        """
        print("[DATA] Loading NSE 500 fundamental data...")
        
        try:
            # Load latest fundamentals per ticker (most recent fetch_date per ticker)
            fund_query = """
            SELECT f1.ticker, f1.profit_margin, f1.revenue_growth, f1.earnings_growth,
                   f1.return_on_equity, f1.debt_to_equity, f1.current_ratio,
                   f1.dividend_yield, f1.beta,
                   f1.fifty_two_week_high, f1.fifty_two_week_low,
                   f1.two_hundred_day_avg
            FROM dbo.nse_500_fundamentals f1
            INNER JOIN (
                SELECT ticker, MAX(fetch_date) as max_date
                FROM dbo.nse_500_fundamentals
                GROUP BY ticker
            ) f2 ON f1.ticker = f2.ticker AND f1.fetch_date = f2.max_date
            """
            df_fund = self.db.execute_query(fund_query)
            print(f"  Fundamentals loaded: {len(df_fund)} tickers")
            
            # Merge fundamentals on ticker
            if not df_fund.empty:
                df = df.merge(df_fund, on='ticker', how='left')
        except Exception as e:
            print(f"  [WARN] Could not load fundamentals: {e}")
        
        try:
            # Load sector data from nse_500 master table
            sector_query = "SELECT ticker, sector FROM dbo.nse_500"
            df_sector = self.db.execute_query(sector_query)
            print(f"  Sectors loaded: {len(df_sector)} tickers")
            
            # Merge sector on ticker
            if not df_sector.empty:
                df = df.merge(df_sector, on='ticker', how='left')
        except Exception as e:
            print(f"  [WARN] Could not load sector data: {e}")
        
        return df
    
    def _merge_market_context(self, df):
        """Load market context data (VIX, NIFTY indices, India sector returns, treasury) and merge.
        
        Uses India-specific columns (nifty50, india_vix, nifty sector indices) plus global
        context (VIX, DXY, US 10Y yield) from the shared market_context_daily table.
        """
        print("[DATA] Loading market context data...")
        
        try:
            context_query = """
            SELECT trading_date,
                   vix_close, vix_change_pct,
                   india_vix_close, india_vix_change_pct,
                   nifty50_close, nifty50_return_1d,
                   sp500_return_1d,
                   dxy_close, dxy_return_1d,
                   us_10y_yield_close, us_10y_yield_change,
                   nifty_it_return_1d, nifty_bank_return_1d,
                   nifty_pharma_return_1d, nifty_auto_return_1d,
                   nifty_fmcg_return_1d
            FROM dbo.market_context_daily
            ORDER BY trading_date
            """
            df_context = self.db.execute_query(context_query)
            
            if not df_context.empty:
                df['trading_date'] = pd.to_datetime(df['trading_date'])
                df_context['trading_date'] = pd.to_datetime(df_context['trading_date'])
                
                df = df.merge(df_context, on='trading_date', how='left')
                matched = df['vix_close'].notna().sum()
                print(f"  Market context merged: {len(df_context)} dates, {matched} matched rows")
            else:
                print("  [WARN] No market context data found — run get_market_context_daily.py --backfill")
        except Exception as e:
            print(f"  [WARN] Could not load market context: {e}")
        
        return df
    
    def _merge_calendar_features(self, df, market='NSE'):
        """Load calendar features (holidays, short weeks, expiry) and merge on trading_date.
        
        Adds pre/post holiday flags, short week indicators, options expiry,
        and cross-market holiday awareness from the shared market_calendar table.
        Each row broadcasts to all tickers for the same date (market-wide features).
        """
        print("[DATA] Loading market calendar features...")
        
        try:
            cal_query = f"""
            SELECT calendar_date,
                   is_pre_holiday, is_post_holiday, is_short_week,
                   trading_days_in_week, is_month_end, is_month_start,
                   is_quarter_end, is_options_expiry,
                   days_until_next_holiday, days_since_last_holiday,
                   other_market_closed
            FROM dbo.vw_market_calendar_features
            WHERE market = '{market}'
            """
            df_cal = self.db.execute_query(cal_query)
            
            if not df_cal.empty:
                df['trading_date'] = pd.to_datetime(df['trading_date'])
                df_cal['calendar_date'] = pd.to_datetime(df_cal['calendar_date'])
                df = df.merge(df_cal, left_on='trading_date', right_on='calendar_date', how='left')
                df = df.drop(columns=['calendar_date'], errors='ignore')
                matched = df['is_pre_holiday'].notna().sum()
                print(f"  Calendar features merged: {len(df_cal)} dates, {matched} matched rows")
            else:
                print("  [WARN] No calendar data found — run sql/create_market_calendar.sql")
        except Exception as e:
            print(f"  [WARN] Could not load calendar features: {e}")
        
        return df
    
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
        Create target variable: 5-day price direction (primary) + 1-day (secondary)
        
        Using 5-day forward returns instead of 1-day because:
        - 1-day direction is essentially random noise (~50% accuracy)
        - 5-day direction captures meaningful trends with less noise
        - NSE 5-day backfill accuracy was already 53.8% vs 50.5% for 1-day
        - More actionable for swing trading (hold 5 days vs day-trade)
        
        Classification target: 5-day direction ('Up'/'Down')
        Regression target: 5-day percentage change
        """
        print("[TARGET] Creating target variables (5-day horizon)...")
        
        df_target = df.copy()
        df_target = df_target.sort_values(['ticker', 'trading_date'])
        
        # Next-day targets (kept for reference/comparison)
        df_target['next_close'] = df_target.groupby('ticker')['close_price'].shift(-1)
        df_target['next_day_return'] = (
            (df_target['next_close'] - df_target['close_price']) / df_target['close_price'] * 100
        )
        df_target['direction'] = np.where(df_target['next_day_return'] > 0, 'Up', 'Down')
        
        # 3-day targets (kept for reference)
        df_target['next_3d_close'] = df_target.groupby('ticker')['close_price'].shift(-3)
        df_target['next_3d_return'] = (
            (df_target['next_3d_close'] - df_target['close_price']) / df_target['close_price'] * 100
        )
        df_target['direction_3d'] = np.where(df_target['next_3d_return'] > 0, 'Up', 'Down')
        
        # 5-day targets (PRIMARY -- used for training)
        df_target['next_5d_close'] = df_target.groupby('ticker')['close_price'].shift(-5)
        df_target['next_5d_return'] = (
            (df_target['next_5d_close'] - df_target['close_price']) / df_target['close_price'] * 100
        )
        df_target['direction_5d'] = np.where(df_target['next_5d_return'] > 0, 'Up', 'Down')
        
        # Remove rows without 5-day target (last 5 rows per ticker)
        valid_mask = df_target['next_5d_close'].notna()
        df_target = df_target[valid_mask]
        
        # Report target distribution
        direction_dist_5d = df_target['direction_5d'].value_counts()
        print(f"  [OK] Primary target distribution (5-day):")
        for direction, count in direction_dist_5d.items():
            pct = (count / len(df_target)) * 100
            print(f"    {direction}: {count:,} ({pct:.1f}%)")
        
        direction_dist_1d = df_target['direction'].value_counts()
        print(f"  [INFO] Secondary reference (1-day):")
        for direction, count in direction_dist_1d.items():
            pct = (count / len(df_target)) * 100
            print(f"    {direction}: {count:,} ({pct:.1f}%)")
        
        print(f"  [OK] Valid training samples: {len(df_target):,}")
        
        return df_target
    
    def engineer_features(self, df):
        """VECTORIZED feature engineering for NSE data (matching NASDAQ approach for speed)"""
        print("[FEATURES] Engineering features (vectorized)...")

        df = df.copy()
        df = df.sort_values(['ticker', 'trading_date']).reset_index(drop=True)
        df['trading_date'] = pd.to_datetime(df['trading_date'])

        price_col = 'close_price'
        high_col = 'high_price'
        low_col = 'low_price'
        volume_col = 'volume'

        # === Basic Price Features ===
        df['daily_volatility'] = ((df[high_col] - df[low_col]) / df[price_col]) * 100
        df['daily_return'] = ((df[price_col] - df['open_price']) / df['open_price']) * 100
        df['volume_millions'] = df[volume_col] / 1_000_000
        df['price_range'] = df[high_col] - df[low_col]
        df['price_position'] = np.where(
            df['price_range'] > 0,
            (df[price_col] - df[low_col]) / df['price_range'],
            0.5
        )

        # Gap (normalized)
        gap_raw = (df.groupby('ticker')['open_price'].transform(lambda x: x.diff())
                   - df.groupby('ticker')[price_col].transform(lambda x: x.shift(1)))
        df['gap'] = np.where(df[price_col] > 0, gap_raw / df[price_col] * 100, 0)

        # Volume-price trend
        df['volume_price_trend'] = df[volume_col] * df['daily_return']

        # === RSI Features ===
        if 'RSI' not in df.columns or df['RSI'].isna().all():
            # Calculate RSI if not from DB
            delta = df.groupby('ticker')[price_col].transform(lambda x: x.diff())
            gain = delta.where(delta > 0, 0).groupby(df['ticker']).transform(lambda x: x.rolling(14, min_periods=1).mean())
            loss = (-delta.where(delta < 0, 0)).groupby(df['ticker']).transform(lambda x: x.rolling(14, min_periods=1).mean())
            rs = gain / loss.replace(0, np.nan)
            df['RSI'] = 100 - (100 / (1 + rs))
        df['rsi_oversold'] = (df['RSI'] < 30).astype(int)
        df['rsi_overbought'] = (df['RSI'] > 70).astype(int)
        df['rsi_momentum'] = df.groupby('ticker')['RSI'].transform(lambda x: x.diff())

        # === Time Features ===
        df['day_of_week'] = df['trading_date'].dt.dayofweek
        df['month'] = df['trading_date'].dt.month

        # === Moving Averages (vectorized) ===
        print("[FEATURES] Adding technical indicators (vectorized)...")
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df.groupby('ticker')[price_col].transform(
                lambda x: x.rolling(window=period, min_periods=1).mean()
            )
            df[f'ema_{period}'] = df.groupby('ticker')[price_col].transform(
                lambda x: x.ewm(span=period, min_periods=1).mean()
            )

        # === MACD (vectorized) ===
        ema_12 = df.groupby('ticker')[price_col].transform(lambda x: x.ewm(span=12, min_periods=1).mean())
        ema_26 = df.groupby('ticker')[price_col].transform(lambda x: x.ewm(span=26, min_periods=1).mean())
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df.groupby('ticker')['macd'].transform(lambda x: x.ewm(span=9, min_periods=1).mean())
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # === Price vs MA Ratios (safe division) ===
        df['price_vs_sma20'] = np.where(df['sma_20'] > 0, df[price_col] / df['sma_20'], 1.0)
        df['price_vs_sma50'] = np.where(df['sma_50'] > 0, df[price_col] / df['sma_50'], 1.0)
        df['price_vs_ema20'] = np.where(df['ema_20'] > 0, df[price_col] / df['ema_20'], 1.0)
        df['sma20_vs_sma50'] = np.where(df['sma_50'] > 0, df['sma_20'] / df['sma_50'], 1.0)
        df['ema20_vs_ema50'] = np.where(df['ema_50'] > 0, df['ema_20'] / df['ema_50'], 1.0)
        df['sma5_vs_sma20'] = np.where(df['sma_20'] > 0, df['sma_5'] / df['sma_20'], 1.0)

        # === SMA 100/200 from DB (or calculate fallback) ===
        if 'sma_200' not in df.columns or df['sma_200'].isna().all():
            df['sma_200'] = df.groupby('ticker')[price_col].transform(
                lambda x: x.rolling(window=200, min_periods=1).mean()
            )
        if 'sma_100' not in df.columns or df['sma_100'].isna().all():
            df['sma_100'] = df.groupby('ticker')[price_col].transform(
                lambda x: x.rolling(window=100, min_periods=1).mean()
            )
        df['price_vs_sma100'] = np.where(df['sma_100'] > 0, df[price_col] / df['sma_100'], 1.0)
        df['price_vs_sma200'] = np.where(df['sma_200'] > 0, df[price_col] / df['sma_200'], 1.0)

        # === SMA Flag encoding (Above/Below -> 1/-1) ===
        sma_flag_map = {'Above': 1, 'Below': -1, 'BULLISH': 1, 'BEARISH': -1}
        for flag_col in ['SMA_200_Flag', 'SMA_100_Flag', 'SMA_50_Flag', 'SMA_20_Flag']:
            target_col = flag_col.lower()
            if flag_col in df.columns:
                df[target_col] = df[flag_col].map(sma_flag_map).fillna(0).astype(float)
                df.drop(columns=[flag_col], inplace=True, errors='ignore')
            else:
                df[target_col] = 0

        # === MACD crossover signal encoding ===
        if 'macd_crossover_signal' in df.columns:
            df['macd_signal_strength'] = df['macd_crossover_signal'].map(
                self.signal_strength_map
            ).fillna(0).astype(float)
            df.drop(columns=['macd_crossover_signal'], inplace=True, errors='ignore')
        else:
            df['macd_signal_strength'] = 0

        # === Volume Indicators ===
        df['volume_sma_20'] = df.groupby('ticker')[volume_col].transform(
            lambda x: x.rolling(window=20, min_periods=1).mean()
        )
        df['volume_sma_ratio'] = np.where(df['volume_sma_20'] > 0, df[volume_col] / df['volume_sma_20'], 1.0)
        df['vol_change_ratio'] = df.groupby('ticker')[volume_col].transform(lambda x: x.pct_change())

        # === Momentum ===
        df['price_momentum_5'] = df.groupby('ticker')[price_col].transform(lambda x: x / x.shift(5))
        df['price_momentum_10'] = df.groupby('ticker')[price_col].transform(lambda x: x / x.shift(10))

        # === Volatility ===
        df['price_volatility_10'] = df.groupby('ticker')[price_col].transform(
            lambda x: x.pct_change().rolling(window=10, min_periods=1).std()
        )
        df['price_volatility_20'] = df.groupby('ticker')[price_col].transform(
            lambda x: x.pct_change().rolling(window=20, min_periods=1).std()
        )

        # === Trend Strength ===
        df['trend_strength_10'] = df.groupby('ticker')[price_col].transform(
            lambda x: x.rolling(window=10, min_periods=1).apply(
                lambda y: (y.iloc[-1] - y.iloc[0]) / y.std() if len(y) > 1 and y.std() != 0 else 0
            )
        )

        # === Bollinger Bands (use DB if available, else calculate) ===
        if 'bb_upper' in df.columns and df['bb_upper'].notna().any():
            bb_upper = df['bb_upper']
            bb_lower = df['bb_lower']
        else:
            bb_sma = df.groupby('ticker')[price_col].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
            bb_std = df.groupby('ticker')[price_col].transform(lambda x: x.rolling(window=20, min_periods=1).std())
            bb_upper = bb_sma + 2 * bb_std
            bb_lower = bb_sma - 2 * bb_std
        bb_range = bb_upper - bb_lower
        df['bb_width'] = np.where(df[price_col] > 0, bb_range / df[price_col], 0)
        df['bb_position'] = np.where(bb_range > 0, (df[price_col] - bb_lower) / bb_range, 0.5)
        # Drop raw BB columns
        df.drop(columns=['bb_upper', 'bb_lower', 'bb_sma20', 'bb_close'], inplace=True, errors='ignore')

        # === Stochastic Oscillator (use DB if available, else calculate) ===
        if 'stoch_k' in df.columns and df['stoch_k'].notna().any():
            low_14 = df.groupby('ticker')[low_col].transform(lambda x: x.rolling(window=14, min_periods=1).min())
            high_14 = df.groupby('ticker')[high_col].transform(lambda x: x.rolling(window=14, min_periods=1).max())
            stoch_range = high_14 - low_14
            calc_k = np.where(stoch_range > 0, (df[price_col] - low_14) / stoch_range * 100, 50)
            df['stoch_k'] = df['stoch_k'].fillna(pd.Series(calc_k, index=df.index))
            calc_d = df.groupby('ticker')['stoch_k'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
            df['stoch_d'] = df['stoch_d'].fillna(calc_d) if 'stoch_d' in df.columns else calc_d
            if 'stoch_momentum' not in df.columns or df['stoch_momentum'].isna().all():
                df['stoch_momentum'] = df['stoch_k'] - df['stoch_d']
        else:
            low_14 = df.groupby('ticker')[low_col].transform(lambda x: x.rolling(window=14, min_periods=1).min())
            high_14 = df.groupby('ticker')[high_col].transform(lambda x: x.rolling(window=14, min_periods=1).max())
            stoch_range = high_14 - low_14
            df['stoch_k'] = np.where(stoch_range > 0, (df[price_col] - low_14) / stoch_range * 100, 50)
            df['stoch_d'] = df.groupby('ticker')['stoch_k'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            df['stoch_momentum'] = df['stoch_k'] - df['stoch_d']

        # === ATR (use DB if available, else calculate) ===
        prev_close = df.groupby('ticker')[price_col].transform(lambda x: x.shift(1))
        high_low = df[high_col] - df[low_col]
        high_close = (df[high_col] - prev_close).abs()
        low_close = (df[low_col] - prev_close).abs()
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        if 'atr_14' in df.columns and df['atr_14'].notna().any():
            df['atr_14'] = df['atr_14'].fillna(
                pd.Series(true_range).groupby(df['ticker']).transform(
                    lambda x: x.rolling(window=14, min_periods=1).mean()
                )
            )
        else:
            df['_true_range'] = true_range
            df['atr_14'] = df.groupby('ticker')['_true_range'].transform(
                lambda x: x.rolling(window=14, min_periods=1).mean()
            )
            df.drop(columns=['_true_range'], inplace=True, errors='ignore')
        df['atr_pct'] = np.where(df[price_col] > 0, df['atr_14'] / df[price_col] * 100, 0)

        # === Normalized MACD ===
        df['macd_normalized'] = np.where(df[price_col] > 0, df['macd'] / df[price_col] * 100, 0)
        df['macd_signal_normalized'] = np.where(df[price_col] > 0, df['macd_signal'] / df[price_col] * 100, 0)
        df['macd_histogram_normalized'] = np.where(df[price_col] > 0, df['macd_histogram'] / df[price_col] * 100, 0)

        # === Lagged Returns ===
        for period in [1, 2, 3, 5, 10]:
            df[f'return_{period}d'] = df.groupby('ticker')[price_col].transform(lambda x: x.pct_change(period))

        # === RSI-Price Divergence ===
        price_dir_5 = df.groupby('ticker')[price_col].transform(lambda x: np.sign(x.pct_change(5)))
        rsi_dir_5 = df.groupby('ticker')['RSI'].transform(lambda x: np.sign(x.diff(5)))
        df['rsi_price_divergence'] = (price_dir_5 != rsi_dir_5).astype(int)

        # === Candlestick Features ===
        df['high_low_ratio'] = np.where(df[low_col] > 0, df[high_col] / df[low_col], 1.0)
        df['close_open_ratio'] = np.where(df['open_price'] > 0, df[price_col] / df['open_price'], 1.0)
        df['upper_shadow'] = np.where(
            df['price_range'] > 0,
            (df[high_col] - np.maximum(df['open_price'], df[price_col])) / df['price_range'],
            0
        )
        df['lower_shadow'] = np.where(
            df['price_range'] > 0,
            (np.minimum(df['open_price'], df[price_col]) - df[low_col]) / df['price_range'],
            0
        )

        # ================================================================
        # ENRICHED DB FEATURE ENCODING (Fibonacci, S/R, Patterns)
        # ================================================================
        print("[FEATURES] Encoding enriched DB signals...")

        # --- Fibonacci signal ---
        if 'fib_trade_signal' in df.columns:
            df['fib_signal_strength'] = df['fib_trade_signal'].map(
                self.signal_strength_map
            ).fillna(0).astype(float)
            df.drop(columns=['fib_trade_signal'], inplace=True, errors='ignore')
        else:
            df['fib_signal_strength'] = 0
        if 'fib_distance_pct' not in df.columns:
            df['fib_distance_pct'] = 0

        # --- Support/Resistance signals ---
        if 'pivot_status' in df.columns:
            df['sr_pivot_position'] = df['pivot_status'].map(
                self.position_map
            ).fillna(0).astype(float)
            df.drop(columns=['pivot_status'], inplace=True, errors='ignore')
        else:
            df['sr_pivot_position'] = 0

        if 'sr_trade_signal' in df.columns:
            df['sr_signal_strength'] = df['sr_trade_signal'].map(
                self.signal_strength_map
            ).fillna(0).astype(float)
            df.drop(columns=['sr_trade_signal'], inplace=True, errors='ignore')
        else:
            df['sr_signal_strength'] = 0

        for col in ['sr_distance_to_support_pct', 'sr_distance_to_resistance_pct']:
            if col not in df.columns:
                df[col] = 0

        # --- Pattern signals ---
        if 'pattern_signal' in df.columns:
            df['pattern_signal_strength'] = df['pattern_signal'].map(
                self.signal_strength_map
            ).fillna(0).astype(float)
            df.drop(columns=['pattern_signal'], inplace=True, errors='ignore')
        else:
            df['pattern_signal_strength'] = 0

        # Pattern binary flags (already 0/1 from SQL CASE)
        for col in ['has_doji', 'has_hammer', 'has_shooting_star',
                     'has_bullish_engulfing', 'has_bearish_engulfing',
                     'has_morning_star', 'has_evening_star']:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = df[col].fillna(0).astype(float)

        # ================================================================
        # MARKET REGIME DETECTION
        # ================================================================
        print("[FEATURES] Adding market regime features...")

        # SMA trend direction
        df['regime_sma20_slope'] = df.groupby('ticker')[price_col].transform(
            lambda x: x.rolling(window=20, min_periods=1).mean().pct_change(5) * 100
        )

        # ADX-like trend strength
        up_move = df.groupby('ticker')[high_col].transform(lambda x: x.diff())
        down_move = -df.groupby('ticker')[low_col].transform(lambda x: x.diff())
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        df['_pos_dm'] = pos_dm
        df['_neg_dm'] = neg_dm
        pos_dm_smooth = df.groupby('ticker')['_pos_dm'].transform(lambda x: x.rolling(14, min_periods=1).mean())
        neg_dm_smooth = df.groupby('ticker')['_neg_dm'].transform(lambda x: x.rolling(14, min_periods=1).mean())
        dm_sum = pos_dm_smooth + neg_dm_smooth
        dx = np.where(dm_sum > 0, np.abs(pos_dm_smooth - neg_dm_smooth) / dm_sum * 100, 0)
        df['_dx'] = dx
        df['regime_adx'] = df.groupby('ticker')['_dx'].transform(lambda x: x.rolling(14, min_periods=1).mean())
        df.drop(columns=['_pos_dm', '_neg_dm', '_dx'], inplace=True, errors='ignore')

        # Volatility regime
        vol_short = df.groupby('ticker')[price_col].transform(lambda x: x.pct_change().rolling(10, min_periods=1).std())
        vol_long = df.groupby('ticker')[price_col].transform(lambda x: x.pct_change().rolling(60, min_periods=1).std())
        df['regime_vol_ratio'] = np.where(vol_long > 0, vol_short / vol_long, 1.0)

        # Mean reversion indicator
        sma50 = df.groupby('ticker')[price_col].transform(lambda x: x.rolling(window=50, min_periods=1).mean())
        df['regime_mean_reversion'] = np.where(
            df['atr_14'] > 0,
            (df[price_col] - sma50) / df['atr_14'],
            0
        )

        # Trend consistency
        overall_dir = df.groupby('ticker')[price_col].transform(lambda x: np.sign(x.diff(20)))
        daily_dirs = df.groupby('ticker')[price_col].transform(lambda x: np.sign(x.diff(1)))
        consistent = (daily_dirs == overall_dir).astype(float)
        df['regime_trend_consistency'] = consistent.groupby(df['ticker']).transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )

        # ================================================================
        # FUNDAMENTAL FEATURES
        # ================================================================
        print("[FEATURES] Adding fundamental features...")

        if 'fifty_two_week_high' in df.columns:
            df['price_vs_52wk_high'] = np.where(
                df['fifty_two_week_high'] > 0, df[price_col] / df['fifty_two_week_high'], 0
            )
        else:
            df['price_vs_52wk_high'] = 0

        if 'fifty_two_week_low' in df.columns:
            df['price_vs_52wk_low'] = np.where(
                df['fifty_two_week_low'] > 0, df[price_col] / df['fifty_two_week_low'], 0
            )
        else:
            df['price_vs_52wk_low'] = 0

        if 'two_hundred_day_avg' in df.columns:
            df['price_vs_200d_avg'] = np.where(
                df['two_hundred_day_avg'] > 0, df[price_col] / df['two_hundred_day_avg'], 0
            )
        else:
            df['price_vs_200d_avg'] = 0

        # Sector encoding
        if 'sector' in df.columns:
            self.sector_encoder = LabelEncoder()
            df['sector_encoded'] = self.sector_encoder.fit_transform(
                df['sector'].fillna('Unknown')
            )
            print(f"  Sectors found: {len(self.sector_encoder.classes_)} unique")
        else:
            df['sector_encoded'] = 0

        # === Market Context: Sector-specific NIFTY index return mapping ===
        NSE_SECTOR_TO_INDEX = {
            'Information Technology': 'nifty_it_return_1d',
            'Financial Services': 'nifty_bank_return_1d',
            'Banks': 'nifty_bank_return_1d',
            'Healthcare': 'nifty_pharma_return_1d',
            'Pharmaceuticals': 'nifty_pharma_return_1d',
            'Automobile': 'nifty_auto_return_1d',
            'Auto Components': 'nifty_auto_return_1d',
            'Fast Moving Consumer Goods': 'nifty_fmcg_return_1d',
            'Consumer Staples': 'nifty_fmcg_return_1d',
        }
        if 'sector' in df.columns and 'nifty_it_return_1d' in df.columns:
            df['sector_index_return_1d'] = df['sector'].map(
                lambda s: NSE_SECTOR_TO_INDEX.get(s, 'nifty50_return_1d')
            )
            df['sector_index_return_1d'] = df.apply(
                lambda row: row.get(row['sector_index_return_1d'], 0) if pd.notna(row.get('sector_index_return_1d')) else 0,
                axis=1
            )
        else:
            df['sector_index_return_1d'] = 0

        # Ensure fundamental feature defaults if columns missing
        for fund_col in ['price_vs_52wk_high', 'price_vs_52wk_low', 'price_vs_200d_avg', 'profit_margin']:
            if fund_col not in df.columns:
                df[fund_col] = 0

        # Drop raw DB/merge columns not needed as features
        drop_cols = ['db_macd', 'db_macd_signal', 'company', 'rsi_trade_signal',
                     'next_5d_close', 'next_3d_close', 'next_close', 'sector',
                     'fifty_two_week_high', 'fifty_two_week_low', 'two_hundred_day_avg',
                     'nifty50_close', 'dxy_close',
                     # Individual NIFTY sector index columns (we use the mapped one)
                     'nifty_it_return_1d', 'nifty_bank_return_1d',
                     'nifty_pharma_return_1d', 'nifty_auto_return_1d',
                     'nifty_fmcg_return_1d',
                     # Raw pattern/signal columns already encoded
                     'patterns_detected', 'bb_close', 'bb_sma20',
                     ]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')

        # Handle infinite values and NaN — forward fill only (no bfill to prevent leakage)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(0)

        print(f"[FEATURES] Complete — {df.shape[1]} columns, {df.shape[0]:,} rows")
        return df
    
    def perform_eda(self, df):
        """Perform exploratory data analysis on NSE data"""
        if self.quick_mode:
            print("[SKIP] Skipping detailed EDA (quick mode)")
            return {'data_shape': df.shape}
        
        print("[EDA] Performing exploratory data analysis on NSE data...")
        
        print(f"  Data shape: {df.shape}")
        print(f"  Date range: {df['trading_date'].min()} to {df['trading_date'].max()}")
        print(f"  Unique tickers: {df['ticker'].nunique()}")
        
        # Target distribution (5-day primary, 1-day reference)
        if 'direction_5d' in df.columns:
            dist = df['direction_5d'].value_counts()
            print(f"  5-day direction target distribution (PRIMARY):")
            for val, cnt in dist.items():
                print(f"    {val}: {cnt:,} ({cnt/len(df)*100:.1f}%)")
        if 'direction' in df.columns:
            dist = df['direction'].value_counts()
            print(f"  1-day direction distribution (reference):")
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
        """Prepare dataset for ML training with dynamic feature selection (mutual_info_classif)"""
        print("[PREP] Preparing ML dataset...")

        target_column = 'direction_5d'

        # Exclude non-feature columns — let mutual_info_classif decide what's useful
        exclude_cols = ['trading_date', 'ticker', target_column, 'next_5d_return',
                        'open_price', 'high_price', 'low_price', 'close_price', 'volume',
                        'sector', 'industry', 'company_name']

        feature_cols = [col for col in df.columns
                        if col not in exclude_cols
                        and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]

        # Prepare X and y for classification (5-day direction target)
        X = df[feature_cols].copy()
        y_direction = df[target_column].copy()

        # Regression target: 5-day return (matches classification horizon)
        y_return = df['next_5d_return'].copy()

        # Remove any remaining invalid rows
        valid_mask = X.notna().all(axis=1) & y_direction.notna() & y_return.notna()
        X = X[valid_mask]
        y_direction = y_direction[valid_mask]
        y_return = y_return[valid_mask]

        # Encode classification target
        direction_encoder = LabelEncoder()
        y_direction_encoded = direction_encoder.fit_transform(y_direction)

        print(f"[PREP] Dataset ready:")
        print(f"  Candidate features: {X.shape[1]}")
        print(f"  Samples: {X.shape[0]:,}")
        print(f"  Direction classes: {list(direction_encoder.classes_)}")
        print(f"  Balance: {dict(zip(*np.unique(y_direction_encoded, return_counts=True)))}")

        # ================================================================
        # STRATIFIED FEATURE SELECTION (mutual_info_classif)
        # Split features into market-wide (identical for all stocks on same day)
        # and stock-specific, then select top from each category to avoid
        # prediction bias where all stocks get identical signals.
        # ================================================================
        MARKET_WIDE_PATTERNS = [
            'vix', 'india_vix', 'sp500', 'nifty50', 'nifty_', 'dxy', 'us_10y',
            'sector_index_return', 'sector_etf',
            'is_pre_holiday', 'is_post_holiday', 'is_short_week',
            'trading_days_in_week', 'is_month_end', 'is_month_start',
            'is_quarter_end', 'is_options_expiry',
            'days_until_next_holiday', 'days_since_last_holiday',
            'other_market_closed',
        ]

        print("[PREP] Running STRATIFIED feature selection (mutual_info_classif)...")
        try:
            mi_scores = mutual_info_classif(X, y_direction_encoded, random_state=42)
            mi_ranking = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)

            # Classify features as market-wide vs stock-specific
            market_features = []
            stock_features = []
            for feat in feature_cols:
                is_market = any(pat in feat.lower() for pat in MARKET_WIDE_PATTERNS)
                if is_market:
                    market_features.append(feat)
                else:
                    stock_features.append(feat)

            print(f"  Feature split: {len(market_features)} market-wide, {len(stock_features)} stock-specific")

            # Select top from each category
            n_market = min(8, len(market_features))
            n_stock = min(22, len(stock_features))

            market_ranking = mi_ranking[mi_ranking.index.isin(market_features)]
            stock_ranking = mi_ranking[mi_ranking.index.isin(stock_features)]

            selected_market = market_ranking.head(n_market).index.tolist()
            selected_stock = stock_ranking.head(n_stock).index.tolist()
            selected_features = selected_stock + selected_market

            print(f"  Selected: {len(selected_stock)} stock-specific + {len(selected_market)} market-wide = {len(selected_features)} total")
            print(f"  Stock-specific features:")
            for i, feat in enumerate(selected_stock):
                print(f"    {i+1:2d}. {feat} (MI={mi_ranking[feat]:.4f})")
            print(f"  Market-wide features:")
            for i, feat in enumerate(selected_market):
                print(f"    {i+1:2d}. {feat} (MI={mi_ranking[feat]:.4f})")

            # Save selected features for prediction consistency
            features_path = self.nse_model_dir / 'selected_features.json'
            with open(features_path, 'w') as f:
                json.dump(selected_features, f, indent=2)
            print(f"  Saved to {features_path}")

            X = X[selected_features]
            feature_cols = selected_features
        except Exception as e:
            print(f"  [WARN] Feature selection failed, using all features: {e}")

        self.feature_columns = feature_cols

        return X, y_direction_encoded, y_return, direction_encoder, feature_cols
    
    def train_classification_models(self, X, y, feature_cols):
        """Train ensemble of classification models for direction prediction"""
        print("[TRAIN] Training classification models (direction prediction)...")
        
        # Time-series aware 3-way split: train (60%) / calibration (20%) / test (20%)
        # Calibrating on a separate set prevents overfitting the probability estimates
        train_end = int(0.60 * len(X))
        cal_end = int(0.80 * len(X))
        X_train = X.iloc[:train_end]
        X_cal = X.iloc[train_end:cal_end]
        X_test = X.iloc[cal_end:]
        y_train = y[:train_end]
        y_cal = y[train_end:cal_end]
        y_test = y[cal_end:]
        
        print(f"  Split: Train={len(X_train):,}, Calibration={len(X_cal):,}, Test={len(X_test):,}")
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_cal_scaled = scaler.transform(X_cal)
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
        
        # Compute time-weighted sample weights (recent data is more relevant)
        n_train = len(y_train)
        time_positions = np.arange(n_train) / n_train  # 0 to ~1 (oldest to newest)
        decay_rate = 1.2  # Higher = more emphasis on recent data
        time_weights = np.exp(decay_rate * (time_positions - 1))  # ~0.3 to 1.0
        time_weights = time_weights / time_weights.mean()  # Normalize to mean=1
        
        print(f"  Time-weighted training: oldest weight={time_weights[0]:.3f}, "
              f"newest={time_weights[-1]:.3f}, ratio={time_weights[-1]/time_weights[0]:.1f}x")
        
        # Train and evaluate each model
        model_results = {}
        trained_models = {}
        # Purged CV: 5-sample gap prevents 5-day target leakage between folds
        cv_splitter = PurgedTimeSeriesSplit(n_splits=3, purge_gap=5)
        
        for model_name, model in models.items():
            print(f"  Training {model_name}...", flush=True)
            
            try:
                # Train with time-weighted sample weights
                try:
                    model.fit(X_train_scaled, y_train, sample_weight=time_weights)
                except TypeError:
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
                n_jobs=1  # n_jobs=-1 can deadlock on Windows; use sequential
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
        best_model = trained_models[best_model_name]
        
        print(f"\n[BEST] Best classification model: {best_model_name} "
              f"(F1: {model_results[best_model_name]['f1_score']:.3f}, "
              f"Direction Accuracy: {model_results[best_model_name]['accuracy']:.1%})")
        
        # Calibrate probabilities using dedicated calibration set (NOT test set)
        # Using isotonic regression for more flexible calibration curve
        print("[CONFIG] Calibrating model probabilities on held-out calibration set...")
        try:
            calibrated_model = CalibratedClassifierCV(
                estimator=best_model, cv='prefit', method='isotonic'
            )
            calibrated_model.fit(X_cal_scaled, y_cal)
            
            # Verify calibration: high-confidence should be more accurate than low
            cal_probs = calibrated_model.predict_proba(X_test_scaled)
            cal_preds = calibrated_model.predict(X_test_scaled)
            cal_accuracy = accuracy_score(y_test, cal_preds)
            uncal_accuracy = model_results[best_model_name]['accuracy']
            
            print(f"  Pre-calibration test accuracy:  {uncal_accuracy:.3f}")
            print(f"  Post-calibration test accuracy: {cal_accuracy:.3f}")
            
            # Check if calibration helps: high-confidence should be more accurate
            max_probs = cal_probs.max(axis=1)
            high_mask = max_probs >= 0.65
            if high_mask.sum() > 10:
                high_acc = accuracy_score(y_test[high_mask], cal_preds[high_mask])
                low_acc = accuracy_score(y_test[~high_mask], cal_preds[~high_mask]) if (~high_mask).sum() > 0 else 0
                print(f"  High-confidence (>=65%) accuracy: {high_acc:.3f} ({high_mask.sum()} samples)")
                print(f"  Low-confidence (<65%) accuracy:   {low_acc:.3f} ({(~high_mask).sum()} samples)")
                
                if high_acc > low_acc:
                    print("[SUCCESS] Calibration confirmed: high confidence = higher accuracy")
                else:
                    print("[WARN] Calibration check: high confidence NOT more accurate - review needed")
            
            best_model = calibrated_model
            print("[SUCCESS] Probability calibration applied successfully")
        except Exception as e:
            print(f"[WARN] Calibration skipped: {e}")
        
        return {
            'model_results': model_results,
            'trained_models': trained_models,
            'best_model_name': best_model_name,
            'best_model': best_model,
            'scaler': scaler,
            'feature_columns': feature_cols,
            'X_train': X_train,
            'X_cal': X_cal,
            'X_test': X_test,
            'y_train': y_train,
            'y_cal': y_cal,
            'y_test': y_test,
        }
    
    def train_regression_models(self, X, y_return, feature_cols):
        """Train regression models for 5-day price change prediction"""
        print("[TRAIN] Training regression models (5-day price change prediction)...")
        
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
        
        # Time-weighted sample weights for regression too
        n_reg_train = len(y_train)
        reg_time_pos = np.arange(n_reg_train) / n_reg_train
        reg_time_weights = np.exp(1.2 * (reg_time_pos - 1))
        reg_time_weights = reg_time_weights / reg_time_weights.mean()
        
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
                try:
                    model.fit(X_train_scaled, y_train_clipped, sample_weight=reg_time_weights)
                except TypeError:
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
        
        # Create regression ensemble using pre-fitted models
        # (Avoids re-training timeout and pickle issues with VotingRegressor)
        print("  Creating Pre-Fitted Regression Ensemble...")
        try:
            class PreFittedEnsemble:
                """Averages predictions from already-trained models."""
                def __init__(self, models_dict):
                    self.models = models_dict
                def predict(self, X):
                    preds = np.column_stack([
                        m.predict(X) for m in self.models.values()
                    ])
                    return preds.mean(axis=1)

            reg_ensemble = PreFittedEnsemble(trained_models)
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
        
        # Save regression models (try/except for unpicklable PreFittedEnsemble)
        for name, model in reg_results['trained_models'].items():
            safe_name = name.lower().replace(' ', '_')
            path = self.nse_model_dir / f'nse_reg_{safe_name}.joblib'
            try:
                joblib.dump(model, path)
                print(f"  [OK] Saved: {path}")
            except Exception as e:
                print(f"  [SKIP] Cannot pickle {name}: {e}")
        
        # Save best regression model separately
        best_reg_path = self.nse_model_dir / 'nse_best_regressor.joblib'
        try:
            joblib.dump(reg_results['best_model'], best_reg_path)
        except Exception as e:
            print(f"  [SKIP] Cannot pickle best regressor: {e}")
        
        # Save preprocessing artifacts
        joblib.dump(clf_results['scaler'], self.nse_model_dir / 'nse_scaler.joblib')
        joblib.dump(direction_encoder, self.nse_model_dir / 'nse_direction_encoder.joblib')
        
        # Save sector encoder (for consistent encoding during prediction)
        if self.sector_encoder is not None:
            joblib.dump(self.sector_encoder, self.nse_model_dir / 'nse_sector_encoder.joblib')
            print(f"  [OK] Saved: {self.nse_model_dir / 'nse_sector_encoder.joblib'}")
        
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
            
            # Step 5: Feature engineering (includes signal encoding — vectorized)
            # Note: encode_signal_features() is now handled inside engineer_features()
            df_features = self.engineer_features(df)
            
            # Step 6: Create target variable (5-day direction)
            df_features = self.create_target_variable(df_features)
            
            # Step 7: EDA
            self.perform_eda(df_features)
            
            # Step 8: Prepare ML dataset (dynamic feature selection)
            X, y_direction, y_return, direction_encoder, feature_cols = self.prepare_ml_dataset(df_features)
            
            # Step 9: Train classification models
            clf_results = self.train_classification_models(X, y_direction, self.feature_columns)
            
            # Step 10: Train regression models
            reg_results = self.train_regression_models(X, y_return, self.feature_columns)
            
            # Step 11: Save artifacts
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
