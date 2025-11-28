"""
NSE 500 Trading Signal Prediction Script

This script generates trading signals for NSE 500 stocks using machine learning models.
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

def safe_print(text):
    """Print text with safe encoding handling for Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        emoji_replacements = {
            '✅': '[SUCCESS]',
            '❌': '[ERROR]',
            '📊': '[DATA]',
            '🇮🇳': '[NSE]',
            '📈': '[PREDICTION]',
            '⚠️': '[WARN]',
            '🎯': '[TARGET]',
            '🔄': '[PROCESSING]'
        }
        for emoji, replacement in emoji_replacements.items():
            text = text.replace(emoji, replacement)
        print(text)

class NSETradingSignalPredictor:
    """NSE 500 trading signal prediction system"""
    
    def __init__(self, model_path=None, scaler_path=None, encoder_path=None):
        """Initialize the NSE predictor with saved model artifacts"""
        
        # Default paths
        self.model_path = model_path or 'data/best_model_extra_trees.joblib'
        self.scaler_path = scaler_path or 'data/scaler.joblib'
        self.encoder_path = encoder_path or 'data/target_encoder.joblib'
        
        # Load model artifacts
        self.load_model_artifacts()
        
        # Database connection
        self.db = SQLServerConnection()
        
        # Feature columns (must match training)
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
    
    def load_model_artifacts(self):
        """Load the trained model, scaler, and encoder"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.target_encoder = joblib.load(self.encoder_path)
            safe_print("✅ Model artifacts loaded successfully")
            
            # Get class names
            self.class_names = self.target_encoder.classes_
            safe_print(f"📊 Target classes: {list(self.class_names)}")
            
        except FileNotFoundError as e:
            safe_print(f"❌ Error loading model artifacts: {e}")
            sys.exit(1)
    
    def get_nse_data(self, ticker=None, days_back=60):
        """Fetch NSE stock data for prediction"""
        
        if ticker:
            ticker_filter = f"AND h.ticker = '{ticker}'"
        else:
            ticker_filter = ""
        
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
            {ticker_filter}
        ORDER BY h.ticker, h.trading_date
        """
        
        try:
            df = self.db.execute_query(query)
            if df.empty:
                safe_print(f"⚠️  No NSE data found for ticker: {ticker}")
                return None
            return df
        except Exception as e:
            safe_print(f"❌ Error fetching NSE data: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for NSE data"""
        if df is None or df.empty:
            return None
        
        df_features = df.copy()
        df_features = df_features.sort_values(['ticker', 'trading_date'])
        
        safe_print(f"🔄 Calculating technical indicators for {df_features['ticker'].nunique()} stocks...")
        
        # Group by ticker for technical calculations
        grouped_results = []
        
        for ticker, group in df_features.groupby('ticker'):
            group = group.sort_values('trading_date').reset_index(drop=True)
            
            # Skip if insufficient data
            if len(group) < 30:
                continue
            
            # Basic features
            group['daily_return'] = group['close_price'].pct_change()
            group['daily_volatility'] = group['daily_return'].rolling(window=10).std()
            group['volume_millions'] = group['volume'] / 1_000_000
            group['price_range'] = group['high_price'] - group['low_price']
            group['price_position'] = (group['close_price'] - group['low_price']) / (group['high_price'] - group['low_price'])
            group['gap'] = (group['open_price'] - group['close_price'].shift(1)) / group['close_price'].shift(1)
            
            # RSI calculation
            delta = group['close_price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            group['RSI'] = 100 - (100 / (1 + rs))
            
            # RSI features
            group['rsi_oversold'] = (group['RSI'] < 30).astype(int)
            group['rsi_overbought'] = (group['RSI'] > 70).astype(int)
            group['rsi_momentum'] = group['RSI'].diff()
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                group[f'sma_{period}'] = group['close_price'].rolling(window=period).mean()
                
            # Exponential moving averages
            for period in [5, 10, 20, 50]:
                group[f'ema_{period}'] = group['close_price'].ewm(span=period).mean()
            
            # MACD
            ema_12 = group['close_price'].ewm(span=12).mean()
            ema_26 = group['close_price'].ewm(span=26).mean()
            group['macd'] = ema_12 - ema_26
            group['macd_signal'] = group['macd'].ewm(span=9).mean()
            group['macd_histogram'] = group['macd'] - group['macd_signal']
            
            # Price vs MA ratios
            group['price_vs_sma20'] = group['close_price'] / group['sma_20']
            group['price_vs_sma50'] = group['close_price'] / group['sma_50']
            group['price_vs_ema20'] = group['close_price'] / group['ema_20']
            
            # MA crossovers
            group['sma20_vs_sma50'] = group['sma_20'] / group['sma_50']
            group['ema20_vs_ema50'] = group['ema_20'] / group['ema_50']
            group['sma5_vs_sma20'] = group['sma_5'] / group['sma_20']
            
            # Volume indicators
            group['volume_sma_20'] = group['volume'].rolling(window=20).mean()
            group['volume_sma_ratio'] = group['volume'] / group['volume_sma_20']
            group['volume_price_trend'] = (group['close_price'].pct_change() * group['volume']).rolling(window=10).mean()
            
            # Price momentum
            group['price_momentum_5'] = group['close_price'] / group['close_price'].shift(5)
            group['price_momentum_10'] = group['close_price'] / group['close_price'].shift(10)
            
            # Volatility
            group['price_volatility_10'] = group['close_price'].rolling(window=10).std() / group['close_price'].rolling(window=10).mean()
            group['price_volatility_20'] = group['close_price'].rolling(window=20).std() / group['close_price'].rolling(window=20).mean()
            
            # Trend strength
            group['trend_strength_10'] = abs(group['close_price'].rolling(window=10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]))
            
            # Date features
            group['day_of_week'] = pd.to_datetime(group['trading_date']).dt.dayofweek
            group['month'] = pd.to_datetime(group['trading_date']).dt.month
            
            grouped_results.append(group)
        
        if not grouped_results:
            safe_print("⚠️  No valid data after technical indicator calculation")
            return None
        
        result_df = pd.concat(grouped_results, ignore_index=True)
        safe_print(f"✅ Technical indicators calculated for {len(result_df)} records")
        
        return result_df
    
    def make_predictions(self, df):
        """Make trading signal predictions"""
        if df is None or df.empty:
            return None
        
        safe_print("🎯 Making ML predictions...")
        
        # Get the latest data for each ticker
        latest_data = df.groupby('ticker').tail(1).copy()
        
        # Prepare features
        feature_data = latest_data[self.feature_columns].copy()
        
        # Handle missing values
        feature_data = feature_data.fillna(0)
        
        # Scale features
        scaled_features = self.scaler.transform(feature_data)
        
        # Make predictions
        predictions = self.model.predict(scaled_features)
        probabilities = self.model.predict_proba(scaled_features)
        
        # Add predictions to dataframe
        latest_data['predicted_signal'] = predictions
        latest_data['buy_probability'] = probabilities[:, 0] if probabilities.shape[1] > 0 else 0
        latest_data['sell_probability'] = probabilities[:, 1] if probabilities.shape[1] > 1 else 0
        
        # Calculate confidence
        latest_data['confidence'] = np.max(probabilities, axis=1)
        latest_data['confidence_percentage'] = latest_data['confidence'] * 100
        
        # Determine confidence levels
        latest_data['high_confidence'] = (latest_data['confidence'] >= 0.8).astype(int)
        latest_data['medium_confidence'] = ((latest_data['confidence'] >= 0.6) & (latest_data['confidence'] < 0.8)).astype(int)
        latest_data['low_confidence'] = (latest_data['confidence'] < 0.6).astype(int)
        
        # Signal strength
        latest_data['signal_strength'] = latest_data.apply(
            lambda x: 'High' if x['high_confidence'] else ('Medium' if x['medium_confidence'] else 'Low'), axis=1
        )
        
        safe_print(f"✅ Predictions completed for {len(latest_data)} stocks")
        
        return latest_data
    
    def save_predictions_to_db(self, predictions_df):
        """Save predictions to NSE database tables"""
        if predictions_df is None or predictions_df.empty:
            return False
        
        safe_print("💾 Saving predictions to database...")
        
        try:
            # Prepare prediction records
            prediction_records = []
            technical_records = []
            
            for _, row in predictions_df.iterrows():
                # Prediction record
                pred_record = {
                    'trading_date': row['trading_date'],
                    'ticker': row['ticker'],
                    'company': row['company'],
                    'predicted_signal': row['predicted_signal'],
                    'confidence': row['confidence'],
                    'confidence_percentage': row['confidence_percentage'],
                    'signal_strength': row['signal_strength'],
                    'close_price': row['close_price'],
                    'volume': row['volume'],
                    'rsi': row['RSI'],
                    'high_confidence': row['high_confidence'],
                    'medium_confidence': row['medium_confidence'],
                    'low_confidence': row['low_confidence'],
                    'sell_probability': row['sell_probability'],
                    'buy_probability': row['buy_probability']
                }
                prediction_records.append(pred_record)
                
                # Technical indicators record
                tech_record = {
                    'trading_date': row['trading_date'],
                    'ticker': row['ticker'],
                    'rsi': row['RSI'],
                    'sma_5': row['sma_5'],
                    'sma_10': row['sma_10'],
                    'sma_20': row['sma_20'],
                    'sma_50': row['sma_50'],
                    'ema_5': row['ema_5'],
                    'ema_10': row['ema_10'],
                    'ema_20': row['ema_20'],
                    'ema_50': row['ema_50'],
                    'macd': row['macd'],
                    'macd_signal': row['macd_signal'],
                    'macd_histogram': row['macd_histogram'],
                    'price_vs_sma20': row['price_vs_sma20'],
                    'price_vs_sma50': row['price_vs_sma50'],
                    'price_vs_ema20': row['price_vs_ema20'],
                    'sma20_vs_sma50': row['sma20_vs_sma50'],
                    'ema20_vs_ema50': row['ema20_vs_ema50'],
                    'sma5_vs_sma20': row['sma5_vs_sma20'],
                    'volume_sma_20': row['volume_sma_20'],
                    'volume_sma_ratio': row['volume_sma_ratio'],
                    'price_momentum_5': row['price_momentum_5'],
                    'price_momentum_10': row['price_momentum_10'],
                    'daily_volatility': row['daily_volatility'],
                    'rsi_oversold': row['rsi_oversold'],
                    'rsi_overbought': row['rsi_overbought'],
                    'rsi_momentum': row['rsi_momentum']
                }
                technical_records.append(tech_record)
            
            # Clear existing data for today
            today = datetime.now().strftime('%Y-%m-%d')
            
            clear_predictions = f"""
            DELETE FROM ml_nse_trading_predictions 
            WHERE trading_date = '{today}'
            """
            
            clear_technical = f"""
            DELETE FROM ml_nse_technical_indicators 
            WHERE trading_date = '{today}'
            """
            
            self.db.execute_query(clear_predictions)
            self.db.execute_query(clear_technical)
            
            # Insert new predictions
            pred_df = pd.DataFrame(prediction_records)
            tech_df = pd.DataFrame(technical_records)
            
            # Use pandas to_sql to insert data
            engine = self.db.get_sqlalchemy_engine()
            
            pred_df.to_sql('ml_nse_trading_predictions', engine, if_exists='append', index=False, schema='dbo')
            tech_df.to_sql('ml_nse_technical_indicators', engine, if_exists='append', index=False, schema='dbo')
            
            safe_print(f"✅ Saved {len(prediction_records)} predictions and {len(technical_records)} technical indicators")
            
            # Create summary
            self.create_prediction_summary(predictions_df)
            
            return True
            
        except Exception as e:
            safe_print(f"❌ Error saving to database: {e}")
            return False
    
    def create_prediction_summary(self, predictions_df):
        """Create daily prediction summary"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            # Calculate summary statistics
            total_predictions = len(predictions_df)
            total_buy_signals = len(predictions_df[predictions_df['predicted_signal'] == 'Buy'])
            total_sell_signals = len(predictions_df[predictions_df['predicted_signal'] == 'Sell'])
            total_hold_signals = len(predictions_df[predictions_df['predicted_signal'] == 'Hold'])
            
            high_confidence_count = predictions_df['high_confidence'].sum()
            medium_confidence_count = predictions_df['medium_confidence'].sum()
            low_confidence_count = predictions_df['low_confidence'].sum()
            
            high_conf_buys = len(predictions_df[(predictions_df['predicted_signal'] == 'Buy') & (predictions_df['high_confidence'] == 1)])
            high_conf_sells = len(predictions_df[(predictions_df['predicted_signal'] == 'Sell') & (predictions_df['high_confidence'] == 1)])
            
            avg_confidence = predictions_df['confidence'].mean()
            avg_rsi = predictions_df['RSI'].mean()
            
            # Clear existing summary for today
            clear_summary = f"DELETE FROM ml_nse_predict_summary WHERE analysis_date = '{today}'"
            self.db.execute_query(clear_summary)
            
            # Insert summary
            summary_data = {
                'analysis_date': today,
                'total_predictions': total_predictions,
                'total_buy_signals': total_buy_signals,
                'total_sell_signals': total_sell_signals,
                'total_hold_signals': total_hold_signals,
                'high_confidence_count': high_confidence_count,
                'medium_confidence_count': medium_confidence_count,
                'low_confidence_count': low_confidence_count,
                'high_conf_buys': high_conf_buys,
                'high_conf_sells': high_conf_sells,
                'avg_confidence': avg_confidence,
                'avg_rsi': avg_rsi,
                'total_stocks_processed': total_predictions
            }
            
            summary_df = pd.DataFrame([summary_data])
            engine = self.db.get_sqlalchemy_engine()
            summary_df.to_sql('ml_nse_predict_summary', engine, if_exists='append', index=False, schema='dbo')
            
            safe_print(f"✅ Summary created: {total_predictions} predictions, {high_conf_buys} high-conf buys, {high_conf_sells} high-conf sells")
            
        except Exception as e:
            safe_print(f"❌ Error creating summary: {e}")
    
    def run_prediction(self, ticker=None):
        """Run complete prediction workflow"""
        safe_print("🇮🇳 Starting NSE 500 prediction workflow...")
        
        # Get data
        df = self.get_nse_data(ticker)
        if df is None:
            return False
        
        # Calculate technical indicators
        df_with_indicators = self.calculate_technical_indicators(df)
        if df_with_indicators is None:
            return False
        
        # Make predictions
        predictions = self.make_predictions(df_with_indicators)
        if predictions is None:
            return False
        
        # Save to database
        success = self.save_predictions_to_db(predictions)
        
        if success:
            safe_print("🎉 NSE prediction workflow completed successfully!")
        else:
            safe_print("❌ NSE prediction workflow failed")
        
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
            safe_print("🔍 Checking NSE data availability...")
            df = predictor.get_nse_data()
            if df is not None:
                safe_print(f"✅ Found data for {df['ticker'].nunique()} NSE stocks")
                safe_print(f"📅 Date range: {df['trading_date'].min()} to {df['trading_date'].max()}")
            return
        
        if args.all_nse:
            safe_print("🇮🇳 Processing all NSE 500 stocks...")
            success = predictor.run_prediction()
        elif args.ticker:
            safe_print(f"🇮🇳 Processing ticker: {args.ticker}")
            success = predictor.run_prediction(args.ticker)
        else:
            safe_print("🇮🇳 Processing all NSE 500 stocks (default)...")
            success = predictor.run_prediction()
        
        if success:
            safe_print("\n✅ NSE prediction completed successfully!")
            safe_print("💾 Check the following tables for results:")
            safe_print("  • ml_nse_trading_predictions")
            safe_print("  • ml_nse_technical_indicators")
            safe_print("  • ml_nse_predict_summary")
        else:
            safe_print("❌ NSE prediction failed")
    
    except Exception as e:
        safe_print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()