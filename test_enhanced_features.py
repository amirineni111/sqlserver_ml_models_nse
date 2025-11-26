#!/usr/bin/env python3
"""
Enhanced Feature Testing Script

This script tests the impact of adding MACD and SMA/EMA features
to the existing RSI-based trading signal model.

Usage:
    python test_enhanced_features.py --test-features
    python test_enhanced_features.py --compare-models
"""

import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection

class EnhancedFeatureTestor:
    """Test enhanced features for trading signal prediction"""
    
    def __init__(self):
        self.db = SQLServerConnection()
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        
    def load_data(self, limit=None):
        """Load historical data from SQL Server"""
        print("📊 Loading data from SQL Server...")
        
        query = """
        SELECT TOP {}
            h.trading_date,
            h.ticker,
            h.company,
            CAST(h.open_price AS FLOAT) as open_price,
            CAST(h.high_price AS FLOAT) as high_price,
            CAST(h.low_price AS FLOAT) as low_price,
            CAST(h.close_price AS FLOAT) as close_price,
            CAST(h.volume AS BIGINT) as volume,
            r.RSI,
            r.rsi_trade_signal
        FROM dbo.nasdaq_100_hist_data h
        INNER JOIN dbo.nasdaq_100_rsi_signals r 
            ON h.ticker = r.ticker AND h.trading_date = r.trading_date
        WHERE r.rsi_trade_signal IS NOT NULL
        ORDER BY h.trading_date DESC, h.ticker
        """.format(limit or 50000)
        
        df = self.db.execute_query(query)
        print(f"✅ Loaded {len(df):,} records")
        return df
    
    def engineer_current_features(self, df):
        """Apply current feature engineering (baseline)"""
        df_features = df.copy()
        
        # Sort by ticker and date for proper calculations
        df_features = df_features.sort_values(['ticker', 'trading_date'])
        
        # Current features (from your existing model)
        df_features['daily_volatility'] = ((df_features['high_price'] - df_features['low_price']) / df_features['close_price']) * 100
        df_features['daily_return'] = ((df_features['close_price'] - df_features['open_price']) / df_features['open_price']) * 100
        df_features['volume_millions'] = df_features['volume'] / 1000000.0
        df_features['price_range'] = df_features['high_price'] - df_features['low_price']
        df_features['price_position'] = (df_features['close_price'] - df_features['low_price']) / df_features['price_range']
        df_features['gap'] = df_features.groupby('ticker')['open_price'].diff() - df_features.groupby('ticker')['close_price'].shift(1)
        df_features['volume_price_trend'] = df_features['volume'] * df_features['daily_return']
        df_features['rsi_oversold'] = (df_features['RSI'] < 30).astype(int)
        df_features['rsi_overbought'] = (df_features['RSI'] > 70).astype(int)
        df_features['rsi_momentum'] = df_features.groupby('ticker')['RSI'].diff()
        
        # Time features
        df_features['trading_date'] = pd.to_datetime(df_features['trading_date'])
        df_features['day_of_week'] = df_features['trading_date'].dt.dayofweek
        df_features['month'] = df_features['trading_date'].dt.month
        
        return df_features
    
    def engineer_enhanced_features(self, df):
        """Add MACD and SMA/EMA features"""
        df_enhanced = self.engineer_current_features(df).copy()
        
        print("🔧 Adding MACD and Moving Average features...")
        
        # Calculate for each ticker separately
        enhanced_data = []
        
        for ticker in df_enhanced['ticker'].unique():
            ticker_data = df_enhanced[df_enhanced['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('trading_date')
            
            # Moving Averages
            ticker_data['sma_5'] = ticker_data['close_price'].rolling(window=5, min_periods=5).mean()
            ticker_data['sma_10'] = ticker_data['close_price'].rolling(window=10, min_periods=10).mean()
            ticker_data['sma_20'] = ticker_data['close_price'].rolling(window=20, min_periods=20).mean()
            ticker_data['ema_12'] = ticker_data['close_price'].ewm(span=12, adjust=False).mean()
            ticker_data['ema_26'] = ticker_data['close_price'].ewm(span=26, adjust=False).mean()
            
            # MACD
            ticker_data['macd_line'] = ticker_data['ema_12'] - ticker_data['ema_26']
            ticker_data['macd_signal'] = ticker_data['macd_line'].ewm(span=9, adjust=False).mean()
            ticker_data['macd_histogram'] = ticker_data['macd_line'] - ticker_data['macd_signal']
            
            # Moving Average Cross Signals
            ticker_data['sma_cross_5_10'] = (ticker_data['sma_5'] > ticker_data['sma_10']).astype(int)
            ticker_data['sma_cross_10_20'] = (ticker_data['sma_10'] > ticker_data['sma_20']).astype(int)
            ticker_data['price_above_sma20'] = (ticker_data['close_price'] > ticker_data['sma_20']).astype(int)
            
            # MACD Signals
            ticker_data['macd_bullish'] = (ticker_data['macd_line'] > ticker_data['macd_signal']).astype(int)
            ticker_data['macd_momentum'] = ticker_data['macd_histogram'].diff()
            
            # Trend Strength
            ticker_data['trend_strength'] = (ticker_data['close_price'] - ticker_data['sma_20']) / ticker_data['sma_20'] * 100
            
            enhanced_data.append(ticker_data)
        
        result = pd.concat(enhanced_data, ignore_index=True)
        
        # Handle NaN values
        result = result.fillna(method='bfill').fillna(0)
        
        print("✅ Enhanced features added:")
        new_features = ['sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26', 'macd_line', 
                       'macd_signal', 'macd_histogram', 'sma_cross_5_10', 'sma_cross_10_20',
                       'price_above_sma20', 'macd_bullish', 'macd_momentum', 'trend_strength']
        for feature in new_features:
            print(f"   • {feature}")
        
        return result
    
    def prepare_datasets(self, df_current, df_enhanced):
        """Prepare current and enhanced datasets for comparison"""
        
        target_column = 'rsi_trade_signal'
        exclude_cols = ['trading_date', 'ticker', 'company', target_column]
        
        # Current features (baseline)
        current_features = [
            'open_price', 'high_price', 'low_price', 'close_price', 'volume', 
            'RSI', 'daily_volatility', 'daily_return', 'volume_millions',
            'price_range', 'price_position', 'gap', 'volume_price_trend',
            'rsi_oversold', 'rsi_overbought', 'rsi_momentum',
            'day_of_week', 'month'
        ]
        
        # Enhanced features (includes current + new)
        enhanced_features = current_features + [
            'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26', 'macd_line', 
            'macd_signal', 'macd_histogram', 'sma_cross_5_10', 'sma_cross_10_20',
            'price_above_sma20', 'macd_bullish', 'macd_momentum', 'trend_strength'
        ]
        
        # Remove rows with missing targets
        df_current = df_current.dropna(subset=[target_column])
        df_enhanced = df_enhanced.dropna(subset=[target_column])
        
        # Ensure same samples for fair comparison
        common_indices = df_current.index.intersection(df_enhanced.index)
        df_current = df_current.loc[common_indices]
        df_enhanced = df_enhanced.loc[common_indices]
        
        X_current = df_current[current_features].fillna(0)
        X_enhanced = df_enhanced[enhanced_features].fillna(0)
        y = df_current[target_column]
        
        print(f"📊 Dataset prepared:")
        print(f"   • Samples: {len(X_current):,}")
        print(f"   • Current features: {len(current_features)}")
        print(f"   • Enhanced features: {len(enhanced_features)}")
        print(f"   • Target distribution: {y.value_counts().to_dict()}")
        
        return X_current, X_enhanced, y
    
    def compare_models(self, X_current, X_enhanced, y, cv_folds=5):
        """Compare current vs enhanced feature performance"""
        
        print("\n🔬 Comparing Model Performance...")
        print("=" * 60)
        
        # Use TimeSeriesSplit for proper validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Models to test
        model_current = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model_enhanced = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        # Scale features
        scaler_current = StandardScaler()
        scaler_enhanced = StandardScaler()
        
        X_current_scaled = scaler_current.fit_transform(X_current)
        X_enhanced_scaled = scaler_enhanced.fit_transform(X_enhanced)
        
        # Cross-validation scores
        scores_current = cross_val_score(model_current, X_current_scaled, y, cv=tscv, scoring='f1_macro')
        scores_enhanced = cross_val_score(model_enhanced, X_enhanced_scaled, y, cv=tscv, scoring='f1_macro')
        
        # Results
        results = {
            'current_model': {
                'mean_f1': scores_current.mean(),
                'std_f1': scores_current.std(),
                'features': len(X_current.columns)
            },
            'enhanced_model': {
                'mean_f1': scores_enhanced.mean(),
                'std_f1': scores_enhanced.std(),
                'features': len(X_enhanced.columns)
            }
        }
        
        # Calculate improvement
        improvement = ((results['enhanced_model']['mean_f1'] - results['current_model']['mean_f1']) / 
                      results['current_model']['mean_f1']) * 100
        
        print("📊 CROSS-VALIDATION RESULTS")
        print("-" * 30)
        print(f"Current Model (RSI-based):")
        print(f"   • F1-Score: {results['current_model']['mean_f1']:.4f} (±{results['current_model']['std_f1']:.4f})")
        print(f"   • Features: {results['current_model']['features']}")
        print()
        print(f"Enhanced Model (RSI + MACD + SMA/EMA):")
        print(f"   • F1-Score: {results['enhanced_model']['mean_f1']:.4f} (±{results['enhanced_model']['std_f1']:.4f})")
        print(f"   • Features: {results['enhanced_model']['features']}")
        print()
        
        if improvement > 0:
            print(f"🎉 IMPROVEMENT: +{improvement:.2f}% F1-Score")
        else:
            print(f"📉 DEGRADATION: {improvement:.2f}% F1-Score")
        
        return results
    
    def feature_importance_analysis(self, X_enhanced, y):
        """Analyze feature importance in enhanced model"""
        
        print("\n🔍 Feature Importance Analysis...")
        print("=" * 40)
        
        # Train model with enhanced features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_enhanced)
        
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': X_enhanced.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 15 Most Important Features:")
        print("-" * 30)
        for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
            feature_type = "🔴 NEW" if row['feature'] in ['sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26', 
                                                         'macd_line', 'macd_signal', 'macd_histogram', 
                                                         'sma_cross_5_10', 'sma_cross_10_20', 'price_above_sma20',
                                                         'macd_bullish', 'macd_momentum', 'trend_strength'] else "📊 CURRENT"
            print(f"{i:2d}. {row['feature']:20s} {row['importance']:.4f} {feature_type}")
        
        return importance_df

def main():
    """Main execution function"""
    
    print("🚀 Enhanced Feature Testing for SQL Server ML Trading Signals")
    print("=" * 70)
    
    tester = EnhancedFeatureTestor()
    
    try:
        # Step 1: Load data
        df_raw = tester.load_data(limit=20000)  # Limit for testing
        
        # Step 2: Create feature sets
        print("\n📊 Engineering feature sets...")
        df_current = tester.engineer_current_features(df_raw)
        df_enhanced = tester.engineer_enhanced_features(df_raw)
        
        # Step 3: Prepare datasets
        X_current, X_enhanced, y = tester.prepare_datasets(df_current, df_enhanced)
        
        # Step 4: Compare models
        results = tester.compare_models(X_current, X_enhanced, y)
        
        # Step 5: Feature importance
        importance_df = tester.feature_importance_analysis(X_enhanced, y)
        
        # Step 6: Recommendations
        print("\n💡 RECOMMENDATIONS")
        print("=" * 30)
        
        improvement = ((results['enhanced_model']['mean_f1'] - results['current_model']['mean_f1']) / 
                      results['current_model']['mean_f1']) * 100
        
        if improvement > 5:
            print("✅ IMPLEMENT ENHANCED FEATURES")
            print("   • Significant improvement detected")
            print("   • Benefits outweigh complexity cost")
        elif improvement > 1:
            print("🤔 CONSIDER ENHANCED FEATURES")
            print("   • Marginal improvement detected")
            print("   • Test with more data before deciding")
        else:
            print("❌ STICK WITH CURRENT FEATURES")
            print("   • No significant improvement")
            print("   • Additional complexity not justified")
        
        # Check for new features in top 10
        top_10_features = importance_df.head(10)['feature'].tolist()
        new_features_in_top10 = [f for f in top_10_features if f in ['sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26', 
                                                                     'macd_line', 'macd_signal', 'macd_histogram', 
                                                                     'sma_cross_5_10', 'sma_cross_10_20', 'price_above_sma20',
                                                                     'macd_bullish', 'macd_momentum', 'trend_strength']]
        
        if new_features_in_top10:
            print(f"\n🔥 High-Impact New Features ({len(new_features_in_top10)}):")
            for feature in new_features_in_top10:
                print(f"   • {feature}")
        
        return results, importance_df
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return None, None

if __name__ == "__main__":
    results, importance = main()
