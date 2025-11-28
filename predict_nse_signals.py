"""
NSE 500 Trading Signal Prediction Deployment Script

This script provides a production-ready interface for making trading signal predictions
using the trained machine learning models for NSE 500 stocks.

Features:
- Reads data from NSE 500 historical data table
- Calculates technical indicators (RSI, SMA, EMA, MACD, BB, ATR)
- Generates ML predictions for Buy/Sell/Hold signals
- Stores results in NSE-specific result tables
- Supports both single ticker and batch processing

Usage:
    python predict_nse_signals.py --ticker RELIANCE.NS --date 2025-11-28
    python predict_nse_signals.py --batch --file nse_tickers.csv
    python predict_nse_signals.py --all-nse  # Process all NSE 500 stocks
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime, timedelta
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def safe_print(text):
    """Print text with safe encoding handling for Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace emoji with text equivalents for Windows console
        emoji_replacements = {
            '✅': '[SUCCESS]',
            '❌': '[ERROR]',
            '📊': '[DATA]',
            '💡': '[TIP]',
            '🎯': '[TARGET]',
            '📈': '[PREDICTION]',
            '⚠️': '[WARN]',
            '🔍': '[INFO]',
            '📋': '[RESULTS]',
            '📁': '[FILE]',
            '🚀': '[START]',
            '🔄': '[PROCESSING]',
            '💾': '[SAVE]',
            '🟢': '[BUY]',
            '🔴': '[SELL]',
            '🟡': '[HOLD]',
            '🇮🇳': '[NSE]'
        }
        for emoji, replacement in emoji_replacements.items():
            text = text.replace(emoji, replacement)
        print(text)

class NSETradingSignalPredictor:
    """NSE 500 trading signal prediction system"""
    
    def __init__(self, model_path=None, scaler_path=None, encoder_path=None):
        """Initialize the NSE predictor with saved model artifacts"""
        
        # Default paths (use the same trained models but adapt for NSE data)
        self.model_path = model_path or 'data/best_model_extra_trees.joblib'
        self.scaler_path = scaler_path or 'data/scaler.joblib'
        self.encoder_path = encoder_path or 'data/target_encoder.joblib'
        
        # Load model artifacts
        self.load_model_artifacts()
        
        # Database connection
        self.db = SQLServerConnection()
        
        # Feature columns (adapted for NSE data)
        self.feature_columns = [
            'open_price', 'high_price', 'low_price', 'close_price', 'volume', 
            'RSI', 'daily_volatility', 'daily_return', 'volume_millions',
            'price_range', 'price_position', 'gap', 'volume_price_trend',
            'rsi_oversold', 'rsi_overbought', 'rsi_momentum',
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_5', 'ema_10', 'ema_20', 'ema_50',
            'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'atr', 'atr_percentage',
            'price_vs_sma20', 'price_vs_sma50', 'price_vs_ema20',
            'sma20_vs_sma50', 'ema20_vs_ema50', 'sma5_vs_sma20',
            'volume_sma_20', 'volume_sma_ratio',
            'price_momentum_5', 'price_momentum_10',
            'price_volatility_10', 'price_volatility_20',
            'trend_strength_10',
            'day_of_week', 'month'
        ]
    
    def load_model_artifacts(self):
        """Load the trained model, scaler, and encoder"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.target_encoder = joblib.load(self.encoder_path)
            safe_print("✅ Model artifacts loaded successfully for NSE prediction")
            
            # Get class names
            self.class_names = self.target_encoder.classes_
            safe_print(f"📊 Target classes: {list(self.class_names)}")
            
        except FileNotFoundError as e:
            safe_print(f"❌ Error loading model artifacts: {e}")
            safe_print("Please ensure the model has been trained and saved.")
            sys.exit(1)
    
    def get_nse_latest_data(self, ticker=None, days_back=5):
        """Fetch latest NSE stock data for prediction"""
        
        if ticker:
            # Remove .NS suffix if present and add it back consistently
            ticker_clean = ticker.replace('.NS', '')
            ticker_filter = f"AND h.ticker = '{ticker_clean}.NS'"
        else:
            ticker_filter = ""
        
        # Query NSE 500 historical data
        query = f"""
        SELECT TOP 1000
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
            {ticker_filter}
        ORDER BY h.trading_date DESC, h.ticker
        """
        
        try:
            df = self.db.execute_query(query)
            if df.empty:
                safe_print(f"⚠️  No NSE data found for ticker: {ticker}")
                return None
            
            safe_print(f"🇮🇳 Retrieved {len(df)} NSE records")
            return df
            
        except Exception as e:
            safe_print(f"❌ Error fetching NSE data: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for NSE data"""
        if df is None or df.empty:
            return None
        
        df_calc = df.copy()
        df_calc = df_calc.sort_values(['ticker', 'trading_date']).reset_index(drop=True)
        
        # Group by ticker to calculate indicators
        result_dfs = []
        
        for ticker in df_calc['ticker'].unique():
            ticker_df = df_calc[df_calc['ticker'] == ticker].copy()
            
            if len(ticker_df) < 50:  # Need sufficient data for indicators
                continue
                
            # Calculate RSI
            ticker_df['rsi'] = self.calculate_rsi(ticker_df['close_price'])
            
            # Calculate SMAs
            ticker_df['sma_5'] = ticker_df['close_price'].rolling(window=5).mean()
            ticker_df['sma_10'] = ticker_df['close_price'].rolling(window=10).mean()
            ticker_df['sma_20'] = ticker_df['close_price'].rolling(window=20).mean()
            ticker_df['sma_50'] = ticker_df['close_price'].rolling(window=50).mean()
            
            # Calculate EMAs
            ticker_df['ema_5'] = ticker_df['close_price'].ewm(span=5).mean()
            ticker_df['ema_10'] = ticker_df['close_price'].ewm(span=10).mean()
            ticker_df['ema_20'] = ticker_df['close_price'].ewm(span=20).mean()
            ticker_df['ema_50'] = ticker_df['close_price'].ewm(span=50).mean()
            
            # Calculate MACD
            exp1 = ticker_df['close_price'].ewm(span=12).mean()
            exp2 = ticker_df['close_price'].ewm(span=26).mean()
            ticker_df['macd'] = exp1 - exp2
            ticker_df['macd_signal'] = ticker_df['macd'].ewm(span=9).mean()
            ticker_df['macd_histogram'] = ticker_df['macd'] - ticker_df['macd_signal']
            
            # Calculate Bollinger Bands
            ticker_df['bb_middle'] = ticker_df['close_price'].rolling(window=20).mean()
            bb_std = ticker_df['close_price'].rolling(window=20).std()
            ticker_df['bb_upper'] = ticker_df['bb_middle'] + (bb_std * 2)
            ticker_df['bb_lower'] = ticker_df['bb_middle'] - (bb_std * 2)
            ticker_df['bb_width'] = ticker_df['bb_upper'] - ticker_df['bb_lower']
            
            # Calculate ATR (Average True Range)
            ticker_df['tr1'] = ticker_df['high_price'] - ticker_df['low_price']
            ticker_df['tr2'] = abs(ticker_df['high_price'] - ticker_df['close_price'].shift(1))
            ticker_df['tr3'] = abs(ticker_df['low_price'] - ticker_df['close_price'].shift(1))
            ticker_df['tr'] = ticker_df[['tr1', 'tr2', 'tr3']].max(axis=1)
            ticker_df['atr'] = ticker_df['tr'].rolling(window=14).mean()
            ticker_df['atr_percentage'] = (ticker_df['atr'] / ticker_df['close_price']) * 100
            
            # Remove temporary columns
            ticker_df = ticker_df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1)
            
            result_dfs.append(ticker_df)
        
        if result_dfs:
            return pd.concat(result_dfs, ignore_index=True)
        else:
            return None
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def engineer_features(self, df):
        """Apply feature engineering to NSE data"""
        if df is None or df.empty:
            return None
        
        df_features = df.copy()
        
        # Calculate additional features
        df_features['daily_return'] = df_features.groupby('ticker')['close_price'].pct_change()
        df_features['daily_volatility'] = df_features.groupby('ticker')['daily_return'].rolling(10).std().reset_index(0, drop=True)
        
        # Volume features
        df_features['volume_millions'] = df_features['volume'] / 1_000_000
        df_features['volume_sma_20'] = df_features.groupby('ticker')['volume'].rolling(20).mean().reset_index(0, drop=True)
        df_features['volume_sma_ratio'] = df_features['volume'] / df_features['volume_sma_20']
        
        # Price features
        df_features['price_range'] = df_features['high_price'] - df_features['low_price']
        df_features['price_position'] = (df_features['close_price'] - df_features['low_price']) / df_features['price_range']
        df_features['gap'] = df_features['open_price'] - df_features.groupby('ticker')['close_price'].shift(1)
        
        # Technical indicator features
        df_features['rsi_oversold'] = (df_features['rsi'] < 30).astype(int)
        df_features['rsi_overbought'] = (df_features['rsi'] > 70).astype(int)
        df_features['rsi_momentum'] = df_features.groupby('ticker')['rsi'].diff()
        
        # Price vs indicators
        df_features['price_vs_sma20'] = df_features['close_price'] / df_features['sma_20']
        df_features['price_vs_sma50'] = df_features['close_price'] / df_features['sma_50']
        df_features['price_vs_ema20'] = df_features['close_price'] / df_features['ema_20']
        
        # Indicator crossovers
        df_features['sma20_vs_sma50'] = df_features['sma_20'] / df_features['sma_50']
        df_features['ema20_vs_ema50'] = df_features['ema_20'] / df_features['ema_50']
        df_features['sma5_vs_sma20'] = df_features['sma_5'] / df_features['sma_20']
        
        # Momentum features
        df_features['price_momentum_5'] = df_features['close_price'] / df_features.groupby('ticker')['close_price'].shift(5)
        df_features['price_momentum_10'] = df_features['close_price'] / df_features.groupby('ticker')['close_price'].shift(10)
        
        # Volatility features
        df_features['price_volatility_10'] = df_features.groupby('ticker')['close_price'].rolling(10).std().reset_index(0, drop=True)
        df_features['price_volatility_20'] = df_features.groupby('ticker')['close_price'].rolling(20).std().reset_index(0, drop=True)
        
        # Trend strength
        df_features['trend_strength_10'] = abs(df_features['close_price'] - df_features.groupby('ticker')['close_price'].shift(10)) / df_features['close_price']
        
        # Volume price trend
        df_features['volume_price_trend'] = df_features['volume_sma_ratio'] * df_features['daily_return']
        
        # Date features
        df_features['trading_date'] = pd.to_datetime(df_features['trading_date'])
        df_features['day_of_week'] = df_features['trading_date'].dt.dayofweek
        df_features['month'] = df_features['trading_date'].dt.month
        
        # Fill NaN values
        df_features = df_features.fillna(method='ffill')
        df_features = df_features.fillna(0)
        
        return df_features
    
    def predict_signals(self, df):
        """Make trading signal predictions for NSE stocks"""
        if df is None or df.empty:
            return None
        
        # Prepare features for prediction
        features_df = df[self.feature_columns].copy()
        
        # Handle missing columns by setting to 0
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Make predictions
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        # Create results dataframe
        results = df[['trading_date', 'ticker', 'company', 'close_price', 'volume', 'rsi']].copy()
        
        # Map predictions back to original labels
        results['predicted_signal'] = [self.class_names[pred] for pred in predictions]
        
        # Add probabilities
        for i, class_name in enumerate(self.class_names):
            results[f'{class_name.lower()}_probability'] = probabilities[:, i]
        
        # Calculate confidence
        results['confidence'] = np.max(probabilities, axis=1)
        results['confidence_percentage'] = results['confidence'] * 100
        
        # Determine signal strength
        results['signal_strength'] = pd.cut(results['confidence'], 
                                          bins=[0, 0.6, 0.8, 1.0], 
                                          labels=['Low', 'Medium', 'High'])
        
        # Set confidence flags
        results['high_confidence'] = (results['confidence'] >= 0.8).astype(int)
        results['medium_confidence'] = ((results['confidence'] >= 0.6) & (results['confidence'] < 0.8)).astype(int)
        results['low_confidence'] = (results['confidence'] < 0.6).astype(int)
        
        # RSI category
        results['rsi_category'] = pd.cut(results['rsi'], 
                                       bins=[0, 30, 70, 100], 
                                       labels=['Oversold', 'Neutral', 'Overbought'])
        
        return results
    
    def save_technical_indicators(self, df):
        """Save technical indicators to ml_nse_technical_indicators table"""
        if df is None or df.empty:
            return False
        
        try:
            # Prepare data for insertion
            indicators_data = []
            
            for _, row in df.iterrows():
                indicators_data.append({
                    'trading_date': row['trading_date'],
                    'ticker': row['ticker'],
                    'sma_5': row.get('sma_5'),
                    'sma_10': row.get('sma_10'),
                    'sma_20': row.get('sma_20'),
                    'sma_50': row.get('sma_50'),
                    'ema_5': row.get('ema_5'),
                    'ema_10': row.get('ema_10'),
                    'ema_20': row.get('ema_20'),
                    'ema_50': row.get('ema_50'),
                    'macd': row.get('macd'),
                    'macd_signal': row.get('macd_signal'),
                    'macd_histogram': row.get('macd_histogram'),
                    'rsi': row.get('rsi'),
                    'rsi_oversold': row.get('rsi_oversold'),
                    'rsi_overbought': row.get('rsi_overbought'),
                    'rsi_momentum': row.get('rsi_momentum'),
                    'bb_upper': row.get('bb_upper'),
                    'bb_middle': row.get('bb_middle'),
                    'bb_lower': row.get('bb_lower'),
                    'bb_width': row.get('bb_width'),
                    'atr': row.get('atr'),
                    'atr_percentage': row.get('atr_percentage'),
                    'price_vs_sma20': row.get('price_vs_sma20'),
                    'price_vs_sma50': row.get('price_vs_sma50'),
                    'price_vs_ema20': row.get('price_vs_ema20'),
                    'sma20_vs_sma50': row.get('sma20_vs_sma50'),
                    'ema20_vs_ema50': row.get('ema20_vs_ema50'),
                    'sma5_vs_sma20': row.get('sma5_vs_sma20'),
                    'trend_direction': self.get_trend_direction(row),
                    'volume_sma_20': row.get('volume_sma_20'),
                    'volume_sma_ratio': row.get('volume_sma_ratio'),
                    'price_momentum_5': row.get('price_momentum_5'),
                    'price_momentum_10': row.get('price_momentum_10'),
                    'daily_volatility': row.get('daily_volatility'),
                    'price_volatility_10': row.get('price_volatility_10'),
                    'price_volatility_20': row.get('price_volatility_20'),
                    'trend_strength_10': row.get('trend_strength_10'),
                    'volume_price_trend': row.get('volume_price_trend'),
                    'gap': row.get('gap')
                })
            
            # Convert to DataFrame and save
            indicators_df = pd.DataFrame(indicators_data)
            
            # Use SQLAlchemy to insert data
            engine = self.db.get_sqlalchemy_engine()
            indicators_df.to_sql('ml_nse_technical_indicators', engine, if_exists='append', index=False)
            
            safe_print(f"💾 Saved {len(indicators_df)} technical indicator records to database")
            return True
            
        except Exception as e:
            safe_print(f"❌ Error saving technical indicators: {e}")
            return False
    
    def save_predictions(self, df):
        """Save predictions to ml_nse_trading_predictions table"""
        if df is None or df.empty:
            return False
        
        try:
            # Prepare data for insertion
            predictions_data = []
            
            for _, row in df.iterrows():
                predictions_data.append({
                    'trading_date': row['trading_date'],
                    'ticker': row['ticker'],
                    'company': row.get('company'),
                    'predicted_signal': row['predicted_signal'],
                    'confidence': row['confidence'],
                    'confidence_percentage': row['confidence_percentage'],
                    'signal_strength': str(row['signal_strength']),
                    'close_price': row['close_price'],
                    'volume': row.get('volume'),
                    'rsi': row.get('rsi'),
                    'rsi_category': str(row.get('rsi_category')),
                    'high_confidence': row['high_confidence'],
                    'medium_confidence': row['medium_confidence'],
                    'low_confidence': row['low_confidence'],
                    'sell_probability': row.get('sell_probability', row.get('Sell_probability')),
                    'buy_probability': row.get('buy_probability', row.get('Buy_probability')),
                    'hold_probability': row.get('hold_probability', row.get('Hold_probability'))
                })
            
            # Convert to DataFrame and save
            predictions_df = pd.DataFrame(predictions_data)
            
            # Use SQLAlchemy to insert data
            engine = self.db.get_sqlalchemy_engine()
            predictions_df.to_sql('ml_nse_trading_predictions', engine, if_exists='append', index=False)
            
            safe_print(f"💾 Saved {len(predictions_df)} prediction records to database")
            return True
            
        except Exception as e:
            safe_print(f"❌ Error saving predictions: {e}")
            return False
    
    def get_trend_direction(self, row):
        """Determine trend direction based on technical indicators"""
        try:
            sma20_vs_sma50 = row.get('sma20_vs_sma50', 1)
            price_vs_sma20 = row.get('price_vs_sma20', 1)
            
            if sma20_vs_sma50 > 1.02 and price_vs_sma20 > 1.02:
                return 'Uptrend'
            elif sma20_vs_sma50 < 0.98 and price_vs_sma20 < 0.98:
                return 'Downtrend'
            else:
                return 'Sideways'
        except:
            return 'Unknown'
    
    def generate_summary(self, predictions_df):
        """Generate summary statistics and save to ml_nse_predict_summary table"""
        if predictions_df is None or predictions_df.empty:
            return None
        
        try:
            analysis_date = predictions_df['trading_date'].max()
            
            summary_data = {
                'analysis_date': analysis_date,
                'total_predictions': len(predictions_df),
                'total_buy_signals': len(predictions_df[predictions_df['predicted_signal'] == 'Buy']),
                'total_sell_signals': len(predictions_df[predictions_df['predicted_signal'] == 'Sell']),
                'total_hold_signals': len(predictions_df[predictions_df['predicted_signal'] == 'Hold']),
                'high_confidence_count': len(predictions_df[predictions_df['high_confidence'] == 1]),
                'medium_confidence_count': len(predictions_df[predictions_df['medium_confidence'] == 1]),
                'low_confidence_count': len(predictions_df[predictions_df['low_confidence'] == 1]),
                'avg_confidence': predictions_df['confidence'].mean(),
                'avg_rsi': predictions_df['rsi'].mean() if 'rsi' in predictions_df.columns else None,
                'avg_buy_probability': predictions_df.get('buy_probability', predictions_df.get('Buy_probability', pd.Series([None]))).mean(),
                'avg_sell_probability': predictions_df.get('sell_probability', predictions_df.get('Sell_probability', pd.Series([None]))).mean(),
                'processing_time_seconds': None,  # To be filled by caller
                'total_stocks_processed': len(predictions_df['ticker'].unique()),
                'failed_predictions': 0,  # To be updated based on errors
                'notes': f'NSE 500 analysis for {analysis_date}'
            }
            
            # Convert to DataFrame and save
            summary_df = pd.DataFrame([summary_data])
            
            engine = self.db.get_sqlalchemy_engine()
            summary_df.to_sql('ml_nse_predict_summary', engine, if_exists='append', index=False)
            
            safe_print(f"📋 Saved summary for {summary_data['total_predictions']} predictions")
            return summary_data
            
        except Exception as e:
            safe_print(f"❌ Error generating summary: {e}")
            return None

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='NSE 500 Trading Signal Prediction')
    parser.add_argument('--ticker', help='NSE ticker symbol (e.g., RELIANCE.NS)')
    parser.add_argument('--date', help='Trading date (YYYY-MM-DD)')
    parser.add_argument('--batch', action='store_true', help='Batch processing mode')
    parser.add_argument('--file', help='CSV file with tickers for batch processing')
    parser.add_argument('--all-nse', action='store_true', help='Process all NSE 500 stocks')
    parser.add_argument('--days-back', type=int, default=60, help='Number of days to look back for data')
    
    args = parser.parse_args()
    
    # Initialize predictor
    safe_print("🇮🇳 Initializing NSE 500 Trading Signal Predictor...")
    predictor = NSETradingSignalPredictor()
    
    start_time = datetime.now()
    
    try:
        if args.all_nse:
            # Process all NSE 500 stocks
            safe_print("🚀 Processing all NSE 500 stocks...")
            data = predictor.get_nse_latest_data(days_back=args.days_back)
            
        elif args.ticker:
            # Process single ticker
            safe_print(f"🚀 Processing single NSE ticker: {args.ticker}")
            data = predictor.get_nse_latest_data(args.ticker, days_back=args.days_back)
            
        elif args.batch and args.file:
            # Process batch from file
            safe_print(f"🚀 Processing batch from file: {args.file}")
            # TODO: Implement batch processing from file
            safe_print("❌ Batch processing from file not implemented yet")
            return
            
        else:
            safe_print("❌ Please specify --ticker, --all-nse, or --batch with --file")
            return
        
        if data is None or data.empty:
            safe_print("❌ No data retrieved. Exiting.")
            return
        
        # Calculate technical indicators
        safe_print("🔄 Calculating technical indicators...")
        data_with_indicators = predictor.calculate_technical_indicators(data)
        
        if data_with_indicators is None:
            safe_print("❌ Failed to calculate technical indicators")
            return
        
        # Engineer features
        safe_print("🔄 Engineering features...")
        features_df = predictor.engineer_features(data_with_indicators)
        
        if features_df is None:
            safe_print("❌ Failed to engineer features")
            return
        
        # Make predictions
        safe_print("📈 Making predictions...")
        predictions = predictor.predict_signals(features_df)
        
        if predictions is None:
            safe_print("❌ Failed to make predictions")
            return
        
        # Get latest predictions only (most recent date per ticker)
        latest_predictions = predictions.loc[predictions.groupby('ticker')['trading_date'].idxmax()]
        
        safe_print(f"📋 Generated {len(latest_predictions)} predictions")
        
        # Save technical indicators
        safe_print("💾 Saving technical indicators...")
        predictor.save_technical_indicators(features_df.loc[features_df.groupby('ticker')['trading_date'].idxmax()])
        
        # Save predictions
        safe_print("💾 Saving predictions...")
        predictor.save_predictions(latest_predictions)
        
        # Generate and save summary
        safe_print("📊 Generating summary...")
        processing_time = (datetime.now() - start_time).total_seconds()
        summary = predictor.generate_summary(latest_predictions)
        
        # Print results
        safe_print("\n" + "="*60)
        safe_print("🇮🇳 NSE 500 TRADING SIGNAL PREDICTION RESULTS")
        safe_print("="*60)
        safe_print(f"📅 Analysis Date: {latest_predictions['trading_date'].iloc[0]}")
        safe_print(f"📊 Total Predictions: {len(latest_predictions)}")
        safe_print(f"🟢 Buy Signals: {len(latest_predictions[latest_predictions['predicted_signal'] == 'Buy'])}")
        safe_print(f"🔴 Sell Signals: {len(latest_predictions[latest_predictions['predicted_signal'] == 'Sell'])}")
        safe_print(f"🟡 Hold Signals: {len(latest_predictions[latest_predictions['predicted_signal'] == 'Hold'])}")
        safe_print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
        
        # Show high confidence signals
        high_conf = latest_predictions[latest_predictions['high_confidence'] == 1]
        if not high_conf.empty:
            safe_print(f"\n🎯 High Confidence Signals ({len(high_conf)}):")
            for _, row in high_conf.head(10).iterrows():
                safe_print(f"  {row['ticker']}: {row['predicted_signal']} ({row['confidence_percentage']:.1f}%)")
        
        safe_print("\n✅ NSE 500 prediction completed successfully!")
        
    except Exception as e:
        safe_print(f"❌ Error during prediction: {e}")
        logger.error(f"Prediction error: {e}", exc_info=True)

if __name__ == "__main__":
    main()