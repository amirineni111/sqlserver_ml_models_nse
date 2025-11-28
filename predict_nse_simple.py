"""
Simple NSE Trading Signal Prediction Script
Focuses on populating database without encoding issues
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection

warnings.filterwarnings('ignore')

class SimpleNSEPredictor:
    """Simple NSE prediction system focused on database population"""
    
    def __init__(self):
        """Initialize predictor"""
        print("Loading model artifacts...")
        
        # Load model artifacts
        self.model = joblib.load('data/best_model_extra_trees.joblib')
        self.scaler = joblib.load('data/scaler.joblib')
        self.target_encoder = joblib.load('data/target_encoder.joblib')
        
        print(f"Model loaded. Target classes: {list(self.target_encoder.classes_)}")
        
        # Database connection
        self.db = SQLServerConnection()
        
        # Feature columns (exact match to training)
        self.feature_columns = [
            'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'RSI',
            'daily_volatility', 'daily_return', 'volume_millions', 'price_range',
            'price_position', 'gap', 'volume_price_trend', 'rsi_oversold',
            'rsi_overbought', 'rsi_momentum', 'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'macd', 'macd_signal', 'macd_histogram',
            'price_vs_sma20', 'price_vs_sma50', 'price_vs_ema20', 'sma20_vs_sma50',
            'ema20_vs_ema50', 'sma5_vs_sma20', 'volume_sma_20', 'volume_sma_ratio',
            'price_momentum_5', 'price_momentum_10', 'price_volatility_10',
            'price_volatility_20', 'trend_strength_10', 'day_of_week', 'month'
        ]
    
    def get_nse_data(self, days_back=60):
        """Get NSE data"""
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
        ORDER BY h.ticker, h.trading_date
        """
        
        df = self.db.execute_query(query)
        print(f"Retrieved {len(df)} NSE records for {df['ticker'].nunique()} tickers")
        return df
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        df = df.sort_values(['ticker', 'trading_date']).reset_index(drop=True)
        
        print(f"Calculating technical indicators for {df['ticker'].nunique()} tickers...")
        
        result_dfs = []
        processed = 0
        
        for ticker, group in df.groupby('ticker'):
            group = group.sort_values('trading_date').reset_index(drop=True)
            
            if len(group) < 30:  # Need minimum data
                continue
            
            # Basic features
            group['daily_return'] = group['close_price'].pct_change()
            group['daily_volatility'] = group['daily_return'].rolling(10).std()
            group['volume_millions'] = group['volume'] / 1_000_000
            group['price_range'] = group['high_price'] - group['low_price']
            group['price_position'] = (group['close_price'] - group['low_price']) / group['price_range']
            group['gap'] = group['open_price'] - group['close_price'].shift(1)
            
            # RSI
            delta = group['close_price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            group['RSI'] = 100 - (100 / (1 + rs))
            
            # Moving averages
            group['sma_5'] = group['close_price'].rolling(5).mean()
            group['sma_10'] = group['close_price'].rolling(10).mean()
            group['sma_20'] = group['close_price'].rolling(20).mean()
            group['sma_50'] = group['close_price'].rolling(50).mean()
            
            group['ema_5'] = group['close_price'].ewm(span=5).mean()
            group['ema_10'] = group['close_price'].ewm(span=10).mean()
            group['ema_20'] = group['close_price'].ewm(span=20).mean()
            group['ema_50'] = group['close_price'].ewm(span=50).mean()
            
            # MACD
            exp1 = group['close_price'].ewm(span=12).mean()
            exp2 = group['close_price'].ewm(span=26).mean()
            group['macd'] = exp1 - exp2
            group['macd_signal'] = group['macd'].ewm(span=9).mean()
            group['macd_histogram'] = group['macd'] - group['macd_signal']
            
            # Additional features
            group['rsi_oversold'] = (group['RSI'] < 30).astype(int)
            group['rsi_overbought'] = (group['RSI'] > 70).astype(int)
            group['rsi_momentum'] = group['RSI'].diff()
            
            group['price_vs_sma20'] = group['close_price'] / group['sma_20']
            group['price_vs_sma50'] = group['close_price'] / group['sma_50']
            group['price_vs_ema20'] = group['close_price'] / group['ema_20']
            
            group['sma20_vs_sma50'] = group['sma_20'] / group['sma_50']
            group['ema20_vs_ema50'] = group['ema_20'] / group['ema_50']
            group['sma5_vs_sma20'] = group['sma_5'] / group['sma_20']
            
            group['volume_sma_20'] = group['volume'].rolling(20).mean()
            group['volume_sma_ratio'] = group['volume'] / group['volume_sma_20']
            group['volume_price_trend'] = group['volume_sma_ratio'] * group['daily_return']
            
            group['price_momentum_5'] = group['close_price'] / group['close_price'].shift(5)
            group['price_momentum_10'] = group['close_price'] / group['close_price'].shift(10)
            
            group['price_volatility_10'] = group['close_price'].rolling(10).std()
            group['price_volatility_20'] = group['close_price'].rolling(20).std()
            
            group['trend_strength_10'] = abs(group['close_price'] - group['close_price'].shift(10)) / group['close_price']
            
            # Date features
            group['trading_date'] = pd.to_datetime(group['trading_date'])
            group['day_of_week'] = group['trading_date'].dt.dayofweek
            group['month'] = group['trading_date'].dt.month
            
            # Fill NaN values
            group = group.fillna(method='ffill').fillna(0)
            
            result_dfs.append(group)
            processed += 1
            
            if processed % 50 == 0:
                print(f"Processed {processed} tickers...")
        
        if result_dfs:
            result = pd.concat(result_dfs, ignore_index=True)
            print(f"Technical indicators calculated for {len(result)} records")
            return result
        else:
            return None
    
    def predict_signals(self, df):
        """Make predictions"""
        print("Making predictions...")
        
        # Get latest data per ticker
        latest_df = df.loc[df.groupby('ticker')['trading_date'].idxmax()].copy()
        
        # Prepare features
        features_df = latest_df[self.feature_columns].copy()
        
        # Handle missing columns
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Make predictions
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        # Create results
        results = latest_df[['trading_date', 'ticker', 'company', 'close_price', 'volume', 'RSI']].copy()
        
        # Add predictions (truncate long signal names)
        signal_mapping = {
            'Overbought (Sell)': 'Sell',
            'Oversold (Buy)': 'Buy'
        }
        results['predicted_signal'] = [signal_mapping.get(self.target_encoder.classes_[pred], self.target_encoder.classes_[pred]) for pred in predictions]
        
        # Add probabilities
        for i, class_name in enumerate(self.target_encoder.classes_):
            results[f'{class_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}_probability'] = probabilities[:, i]
        
        # Calculate confidence
        results['confidence'] = np.max(probabilities, axis=1)
        results['confidence_percentage'] = results['confidence'] * 100
        
        # Signal strength
        results['signal_strength'] = pd.cut(results['confidence'], 
                                          bins=[0, 0.6, 0.8, 1.0], 
                                          labels=['Low', 'Medium', 'High'])
        
        # Confidence flags
        results['high_confidence'] = (results['confidence'] >= 0.8).astype(int)
        results['medium_confidence'] = ((results['confidence'] >= 0.6) & (results['confidence'] < 0.8)).astype(int)
        results['low_confidence'] = (results['confidence'] < 0.6).astype(int)
        
        # RSI category
        results['rsi_category'] = pd.cut(results['RSI'], 
                                       bins=[0, 30, 70, 100], 
                                       labels=['Oversold', 'Neutral', 'Overbought'])
        
        print(f"Predictions generated for {len(results)} stocks")
        return results, latest_df
    
    def save_to_database(self, predictions_df, indicators_df):
        """Save predictions and indicators to database"""
        try:
            engine = self.db.get_sqlalchemy_engine()
            
            # Save predictions
            print(f"Saving {len(predictions_df)} predictions to database...")
            
            # Prepare prediction data
            pred_data = []
            for _, row in predictions_df.iterrows():
                pred_data.append({
                    'trading_date': row['trading_date'],
                    'ticker': row['ticker'],
                    'company': row.get('company'),
                    'predicted_signal': row['predicted_signal'],
                    'confidence': row['confidence'],
                    'confidence_percentage': row['confidence_percentage'],
                    'signal_strength': str(row['signal_strength']),
                    'close_price': row['close_price'],
                    'volume': row.get('volume'),
                    'rsi': row.get('RSI'),
                    'rsi_category': str(row.get('rsi_category')),
                    'high_confidence': row['high_confidence'],
                    'medium_confidence': row['medium_confidence'],
                    'low_confidence': row['low_confidence'],
                    'sell_probability': row.get('overbought_sell_probability'),
                    'buy_probability': row.get('oversold_buy_probability'),
                    'hold_probability': None
                })
            
            pred_df = pd.DataFrame(pred_data)
            pred_df.to_sql('ml_nse_trading_predictions', engine, if_exists='append', index=False)
            print(f"Saved {len(pred_df)} prediction records")
            
            # Save technical indicators (latest per ticker)
            print(f"Saving technical indicators to database...")
            
            indicator_data = []
            for _, row in indicators_df.iterrows():
                indicator_data.append({
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
                    'rsi': row.get('RSI'),
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
            
            ind_df = pd.DataFrame(indicator_data)
            ind_df.to_sql('ml_nse_technical_indicators', engine, if_exists='append', index=False)
            print(f"Saved {len(ind_df)} indicator records")
            
            # Save summary
            analysis_date = predictions_df['trading_date'].max()
            summary_data = {
                'analysis_date': analysis_date,
                'total_predictions': len(predictions_df),
                'total_buy_signals': len(predictions_df[predictions_df['predicted_signal'].str.contains('Buy', na=False)]),
                'total_sell_signals': len(predictions_df[predictions_df['predicted_signal'].str.contains('Sell', na=False)]),
                'total_hold_signals': 0,
                'high_confidence_count': len(predictions_df[predictions_df['high_confidence'] == 1]),
                'medium_confidence_count': len(predictions_df[predictions_df['medium_confidence'] == 1]),
                'low_confidence_count': len(predictions_df[predictions_df['low_confidence'] == 1]),
                'avg_confidence': predictions_df['confidence'].mean(),
                'avg_rsi': predictions_df['RSI'].mean(),
                'avg_buy_probability': predictions_df.get('oversold_buy_probability', pd.Series([None])).mean(),
                'avg_sell_probability': predictions_df.get('overbought_sell_probability', pd.Series([None])).mean(),
                'processing_time_seconds': 0,
                'total_stocks_processed': len(predictions_df['ticker'].unique()),
                'failed_predictions': 0,
                'notes': f'NSE 500 simple prediction for {analysis_date}'
            }
            
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_sql('ml_nse_predict_summary', engine, if_exists='append', index=False)
            print(f"Saved summary record")
            
            return True
            
        except Exception as e:
            print(f"Error saving to database: {e}")
            return False
    
    def get_trend_direction(self, row):
        """Determine trend direction"""
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
    
    def run_prediction(self):
        """Run complete prediction workflow"""
        print("Starting NSE prediction workflow...")
        
        start_time = datetime.now()
        
        # Get data
        data = self.get_nse_data(days_back=90)  # Use more data for better indicators
        if data is None or data.empty:
            print("No data retrieved")
            return False
        
        # Calculate indicators
        data_with_indicators = self.calculate_technical_indicators(data)
        if data_with_indicators is None:
            print("Failed to calculate indicators")
            return False
        
        # Make predictions
        predictions, latest_indicators = self.predict_signals(data_with_indicators)
        if predictions is None:
            print("Failed to make predictions")
            return False
        
        # Save to database
        success = self.save_to_database(predictions, latest_indicators)
        
        if success:
            processing_time = (datetime.now() - start_time).total_seconds()
            print(f"\nNSE Prediction Summary:")
            print(f"- Total Predictions: {len(predictions)}")
            print(f"- Buy Signals: {len(predictions[predictions['predicted_signal'].str.contains('Buy', na=False)])}")
            print(f"- Sell Signals: {len(predictions[predictions['predicted_signal'].str.contains('Sell', na=False)])}")
            print(f"- High Confidence: {len(predictions[predictions['high_confidence'] == 1])}")
            print(f"- Processing Time: {processing_time:.1f} seconds")
            print(f"- Database Save: SUCCESS")
        
        return success

def main():
    """Main function"""
    predictor = SimpleNSEPredictor()
    success = predictor.run_prediction()
    
    if success:
        print("\nNSE prediction completed successfully!")
    else:
        print("\nNSE prediction failed!")

if __name__ == "__main__":
    main()