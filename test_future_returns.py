#!/usr/bin/env python3
"""
Enhanced Feature Testing with Future Returns Target

This script tests whether adding MACD and SMA/EMA features improves
prediction of actual future price movements (profitable trades) instead
of just predicting RSI-derived signals.

Target Variables:
- next_day_return: Next day's return (%)
- next_3day_return: 3-day forward return (%)
- next_5day_return: 5-day forward return (%)
- profitable_trade: Binary (1 if return > threshold, 0 otherwise)

Usage:
    python test_future_returns.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection

class FutureReturnsTestor:
    """Test features against actual future returns"""
    
    def __init__(self):
        self.db = SQLServerConnection()
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load historical data from SQL Server"""
        print("📊 Loading data from SQL Server...")
        
        query = """
        SELECT 
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
        ORDER BY h.ticker, h.trading_date
        """
        
        df = self.db.execute_query(query)
        df['trading_date'] = pd.to_datetime(df['trading_date'])
        print(f"✅ Loaded {len(df):,} records for {df['ticker'].nunique()} tickers")
        return df
    
    def calculate_technical_indicators(self, df):
        """Calculate all technical indicators for each ticker"""
        print("🔧 Calculating technical indicators...")
        
        all_data = []
        
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('trading_date').reset_index(drop=True)
            
            if len(ticker_data) < 50:  # Need sufficient data for indicators
                continue
            
            # Current features (RSI-based)
            ticker_data['daily_volatility'] = ((ticker_data['high_price'] - ticker_data['low_price']) / ticker_data['close_price']) * 100
            ticker_data['daily_return'] = ((ticker_data['close_price'] - ticker_data['open_price']) / ticker_data['open_price']) * 100
            ticker_data['volume_millions'] = ticker_data['volume'] / 1_000_000
            ticker_data['price_range'] = ticker_data['high_price'] - ticker_data['low_price']
            ticker_data['price_position'] = (ticker_data['close_price'] - ticker_data['low_price']) / ticker_data['price_range']
            ticker_data['gap'] = ticker_data['open_price'] - ticker_data['close_price'].shift(1)
            ticker_data['volume_price_trend'] = ticker_data['volume'] * ticker_data['daily_return']
            ticker_data['rsi_oversold'] = (ticker_data['RSI'] < 30).astype(int)
            ticker_data['rsi_overbought'] = (ticker_data['RSI'] > 70).astype(int)
            ticker_data['rsi_momentum'] = ticker_data['RSI'].diff()
            
            # Time features
            ticker_data['day_of_week'] = ticker_data['trading_date'].dt.dayofweek
            ticker_data['month'] = ticker_data['trading_date'].dt.month
            
            # Enhanced features: Moving Averages
            ticker_data['sma_5'] = ticker_data['close_price'].rolling(window=5, min_periods=5).mean()
            ticker_data['sma_10'] = ticker_data['close_price'].rolling(window=10, min_periods=10).mean()
            ticker_data['sma_20'] = ticker_data['close_price'].rolling(window=20, min_periods=20).mean()
            ticker_data['sma_50'] = ticker_data['close_price'].rolling(window=50, min_periods=50).mean()
            
            ticker_data['ema_12'] = ticker_data['close_price'].ewm(span=12, adjust=False).mean()
            ticker_data['ema_26'] = ticker_data['close_price'].ewm(span=26, adjust=False).mean()
            
            # Enhanced features: MACD
            ticker_data['macd_line'] = ticker_data['ema_12'] - ticker_data['ema_26']
            ticker_data['macd_signal'] = ticker_data['macd_line'].ewm(span=9, adjust=False).mean()
            ticker_data['macd_histogram'] = ticker_data['macd_line'] - ticker_data['macd_signal']
            
            # Enhanced features: Price vs MA relationships
            ticker_data['price_vs_sma20'] = (ticker_data['close_price'] / ticker_data['sma_20'] - 1) * 100
            ticker_data['price_vs_sma50'] = (ticker_data['close_price'] / ticker_data['sma_50'] - 1) * 100
            ticker_data['sma20_vs_sma50'] = (ticker_data['sma_20'] / ticker_data['sma_50'] - 1) * 100
            
            # Enhanced features: Crossover signals
            ticker_data['price_above_sma20'] = (ticker_data['close_price'] > ticker_data['sma_20']).astype(int)
            ticker_data['price_above_sma50'] = (ticker_data['close_price'] > ticker_data['sma_50']).astype(int)
            ticker_data['sma20_above_sma50'] = (ticker_data['sma_20'] > ticker_data['sma_50']).astype(int)
            ticker_data['macd_bullish'] = (ticker_data['macd_line'] > ticker_data['macd_signal']).astype(int)
            
            # Enhanced features: Momentum
            ticker_data['macd_momentum'] = ticker_data['macd_histogram'].diff()
            ticker_data['price_momentum_5'] = ticker_data['close_price'].pct_change(5) * 100
            ticker_data['volume_sma_ratio'] = ticker_data['volume'] / ticker_data['volume'].rolling(20).mean()
            
            # **NEW: Future Returns (Target Variables)**
            ticker_data['next_1day_return'] = ticker_data['close_price'].shift(-1) / ticker_data['close_price'] - 1
            ticker_data['next_3day_return'] = ticker_data['close_price'].shift(-3) / ticker_data['close_price'] - 1
            ticker_data['next_5day_return'] = ticker_data['close_price'].shift(-5) / ticker_data['close_price'] - 1
            
            # Convert to percentage
            ticker_data['next_1day_return'] *= 100
            ticker_data['next_3day_return'] *= 100
            ticker_data['next_5day_return'] *= 100
            
            # Binary profitable trade targets (different thresholds)
            ticker_data['profitable_1day_1pct'] = (ticker_data['next_1day_return'] > 1.0).astype(int)
            ticker_data['profitable_3day_2pct'] = (ticker_data['next_3day_return'] > 2.0).astype(int)
            ticker_data['profitable_5day_3pct'] = (ticker_data['next_5day_return'] > 3.0).astype(int)
            
            # Outperform market binary (assuming 0.1% daily market return)
            ticker_data['beat_market_1day'] = (ticker_data['next_1day_return'] > 0.1).astype(int)
            ticker_data['beat_market_3day'] = (ticker_data['next_3day_return'] > 0.3).astype(int)
            ticker_data['beat_market_5day'] = (ticker_data['next_5day_return'] > 0.5).astype(int)
            
            all_data.append(ticker_data)
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values(['ticker', 'trading_date'])
        
        # Remove rows with insufficient data for moving averages
        result = result.dropna(subset=['sma_50', 'macd_signal'])
        
        print(f"✅ Technical indicators calculated. Final dataset: {len(result):,} records")
        return result
    
    def prepare_feature_sets(self, df):
        """Prepare current and enhanced feature sets"""
        
        # Current features (RSI-based system)
        current_features = [
            'open_price', 'high_price', 'low_price', 'close_price', 'volume',
            'RSI', 'daily_volatility', 'daily_return', 'volume_millions',
            'price_range', 'price_position', 'gap', 'volume_price_trend',
            'rsi_oversold', 'rsi_overbought', 'rsi_momentum',
            'day_of_week', 'month'
        ]
        
        # Enhanced features (adds MACD + SMA/EMA)
        enhanced_features = current_features + [
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_12', 'ema_26',
            'macd_line', 'macd_signal', 'macd_histogram',
            'price_vs_sma20', 'price_vs_sma50', 'sma20_vs_sma50',
            'price_above_sma20', 'price_above_sma50', 'sma20_above_sma50',
            'macd_bullish', 'macd_momentum',
            'price_momentum_5', 'volume_sma_ratio'
        ]
        
        # Target variables
        regression_targets = ['next_1day_return', 'next_3day_return', 'next_5day_return']
        classification_targets = [
            'profitable_1day_1pct', 'profitable_3day_2pct', 'profitable_5day_3pct',
            'beat_market_1day', 'beat_market_3day', 'beat_market_5day'
        ]
        
        return current_features, enhanced_features, regression_targets, classification_targets
    
    def test_regression_performance(self, X_current, X_enhanced, y_target, target_name):
        """Test regression performance (predicting actual returns)"""
        
        print(f"\n📈 REGRESSION TEST: {target_name}")
        print("=" * 50)
        
        # Remove NaN targets
        valid_idx = ~pd.isna(y_target)
        X_current_clean = X_current[valid_idx]
        X_enhanced_clean = X_enhanced[valid_idx]
        y_clean = y_target[valid_idx]
        
        if len(y_clean) < 100:
            print(f"⚠️  Insufficient data for {target_name}")
            return None
        
        # Split data
        X_curr_train, X_curr_test, y_train, y_test = train_test_split(
            X_current_clean, y_clean, test_size=0.2, random_state=42
        )
        X_enh_train, X_enh_test, _, _ = train_test_split(
            X_enhanced_clean, y_clean, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler_curr = StandardScaler()
        scaler_enh = StandardScaler()
        
        X_curr_train_scaled = scaler_curr.fit_transform(X_curr_train)
        X_curr_test_scaled = scaler_curr.transform(X_curr_test)
        X_enh_train_scaled = scaler_enh.fit_transform(X_enh_train)
        X_enh_test_scaled = scaler_enh.transform(X_enh_test)
        
        # Train models
        model_current = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model_enhanced = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        model_current.fit(X_curr_train_scaled, y_train)
        model_enhanced.fit(X_enh_train_scaled, y_train)
        
        # Predictions
        pred_current = model_current.predict(X_curr_test_scaled)
        pred_enhanced = model_enhanced.predict(X_enh_test_scaled)
        
        # Metrics
        mse_current = mean_squared_error(y_test, pred_current)
        mse_enhanced = mean_squared_error(y_test, pred_enhanced)
        r2_current = r2_score(y_test, pred_current)
        r2_enhanced = r2_score(y_test, pred_enhanced)
        
        # Improvement
        mse_improvement = ((mse_current - mse_enhanced) / mse_current) * 100
        r2_improvement = ((r2_enhanced - r2_current) / abs(r2_current)) * 100 if r2_current != 0 else 0
        
        print(f"📊 Current Model (RSI-based):")
        print(f"   • MSE: {mse_current:.6f}")
        print(f"   • R²:  {r2_current:.4f}")
        print(f"📊 Enhanced Model (MACD+SMA/EMA):")
        print(f"   • MSE: {mse_enhanced:.6f}")
        print(f"   • R²:  {r2_enhanced:.4f}")
        print(f"📊 Improvement:")
        print(f"   • MSE: {mse_improvement:+.2f}% (lower is better)")
        print(f"   • R²:  {r2_improvement:+.2f}% (higher is better)")
        
        return {
            'mse_current': mse_current,
            'mse_enhanced': mse_enhanced,
            'r2_current': r2_current,
            'r2_enhanced': r2_enhanced,
            'mse_improvement': mse_improvement,
            'r2_improvement': r2_improvement,
            'sample_size': len(y_test)
        }
    
    def test_classification_performance(self, X_current, X_enhanced, y_target, target_name):
        """Test classification performance (predicting profitable trades)"""
        
        print(f"\n🎯 CLASSIFICATION TEST: {target_name}")
        print("=" * 50)
        
        # Remove NaN targets
        valid_idx = ~pd.isna(y_target)
        X_current_clean = X_current[valid_idx]
        X_enhanced_clean = X_enhanced[valid_idx]
        y_clean = y_target[valid_idx]
        
        if len(y_clean) < 100:
            print(f"⚠️  Insufficient data for {target_name}")
            return None
        
        # Check class distribution
        class_counts = pd.Series(y_clean).value_counts()
        print(f"📊 Class Distribution:")
        print(f"   • Success (1): {class_counts.get(1, 0):,} ({class_counts.get(1, 0)/len(y_clean)*100:.1f}%)")
        print(f"   • Failure (0): {class_counts.get(0, 0):,} ({class_counts.get(0, 0)/len(y_clean)*100:.1f}%)")
        
        if len(class_counts) < 2:
            print(f"⚠️  Insufficient class diversity for {target_name}")
            return None
        
        # Split data
        X_curr_train, X_curr_test, y_train, y_test = train_test_split(
            X_current_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
        )
        X_enh_train, X_enh_test, _, _ = train_test_split(
            X_enhanced_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
        )
        
        # Scale features
        scaler_curr = StandardScaler()
        scaler_enh = StandardScaler()
        
        X_curr_train_scaled = scaler_curr.fit_transform(X_curr_train)
        X_curr_test_scaled = scaler_curr.transform(X_curr_test)
        X_enh_train_scaled = scaler_enh.fit_transform(X_enh_train)
        X_enh_test_scaled = scaler_enh.transform(X_enh_test)
        
        # Train models
        model_current = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model_enhanced = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        model_current.fit(X_curr_train_scaled, y_train)
        model_enhanced.fit(X_enh_train_scaled, y_train)
        
        # Predictions
        pred_current = model_current.predict(X_curr_test_scaled)
        pred_enhanced = model_enhanced.predict(X_enh_test_scaled)
        
        # Metrics
        acc_current = accuracy_score(y_test, pred_current)
        acc_enhanced = accuracy_score(y_test, pred_enhanced)
        
        improvement = ((acc_enhanced - acc_current) / acc_current) * 100
        
        print(f"📊 Current Model (RSI-based):")
        print(f"   • Accuracy: {acc_current:.4f} ({acc_current*100:.2f}%)")
        print(f"📊 Enhanced Model (MACD+SMA/EMA):")
        print(f"   • Accuracy: {acc_enhanced:.4f} ({acc_enhanced*100:.2f}%)")
        print(f"📊 Improvement: {improvement:+.2f}%")
        
        # Feature importance for enhanced model
        feature_importance = pd.DataFrame({
            'feature': X_enhanced.columns,
            'importance': model_enhanced.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Top 10 Important Features (Enhanced Model):")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            feature_type = "🔴 NEW" if row['feature'] not in X_current.columns else "📊 CURRENT"
            print(f"{i:2d}. {row['feature']:20s} {row['importance']:.4f} {feature_type}")
        
        return {
            'accuracy_current': acc_current,
            'accuracy_enhanced': acc_enhanced,
            'improvement': improvement,
            'sample_size': len(y_test),
            'feature_importance': feature_importance
        }
    
    def run_comprehensive_test(self):
        """Run comprehensive test comparing features on future returns"""
        
        print("🚀 COMPREHENSIVE FUTURE RETURNS TEST")
        print("=" * 60)
        
        # Load and prepare data
        df = self.load_data()
        df = self.calculate_technical_indicators(df)
        
        current_features, enhanced_features, regression_targets, classification_targets = self.prepare_feature_sets(df)
        
        print(f"\n📊 Feature Summary:")
        print(f"   • Current features: {len(current_features)}")
        print(f"   • Enhanced features: {len(enhanced_features)}")
        print(f"   • New features added: {len(enhanced_features) - len(current_features)}")
        
        # Prepare feature matrices
        X_current = df[current_features].fillna(0)
        X_enhanced = df[enhanced_features].fillna(0)
        
        # Test results storage
        regression_results = {}
        classification_results = {}
        
        # Test regression targets (predicting actual returns)
        print(f"\n{'='*60}")
        print("REGRESSION TESTS (Predicting Actual Returns)")
        print(f"{'='*60}")
        
        for target in regression_targets:
            if target in df.columns:
                result = self.test_regression_performance(
                    X_current, X_enhanced, df[target], target
                )
                if result:
                    regression_results[target] = result
        
        # Test classification targets (predicting profitable trades)
        print(f"\n{'='*60}")
        print("CLASSIFICATION TESTS (Predicting Profitable Trades)")
        print(f"{'='*60}")
        
        for target in classification_targets:
            if target in df.columns:
                result = self.test_classification_performance(
                    X_current, X_enhanced, df[target], target
                )
                if result:
                    classification_results[target] = result
        
        # Summary and recommendations
        self.generate_final_recommendations(regression_results, classification_results)
        
        return regression_results, classification_results
    
    def generate_final_recommendations(self, regression_results, classification_results):
        """Generate final recommendations based on all test results"""
        
        print(f"\n{'='*60}")
        print("FINAL ANALYSIS & RECOMMENDATIONS")
        print(f"{'='*60}")
        
        # Count improvements
        reg_improvements = sum(1 for r in regression_results.values() if r['r2_improvement'] > 0)
        cls_improvements = sum(1 for r in classification_results.values() if r['improvement'] > 0)
        
        total_reg_tests = len(regression_results)
        total_cls_tests = len(classification_results)
        
        print(f"\n📊 IMPROVEMENT SUMMARY:")
        print(f"   • Regression Tests: {reg_improvements}/{total_reg_tests} showed improvement")
        print(f"   • Classification Tests: {cls_improvements}/{total_cls_tests} showed improvement")
        
        # Best performing targets
        if regression_results:
            best_reg = max(regression_results.items(), key=lambda x: x[1]['r2_improvement'])
            print(f"\n🏆 Best Regression Improvement: {best_reg[0]}")
            print(f"   • R² improvement: {best_reg[1]['r2_improvement']:+.2f}%")
        
        if classification_results:
            best_cls = max(classification_results.items(), key=lambda x: x[1]['improvement'])
            print(f"\n🏆 Best Classification Improvement: {best_cls[0]}")
            print(f"   • Accuracy improvement: {best_cls[1]['improvement']:+.2f}%")
        
        # Overall recommendation
        total_improvements = reg_improvements + cls_improvements
        total_tests = total_reg_tests + total_cls_tests
        improvement_rate = total_improvements / total_tests if total_tests > 0 else 0
        
        print(f"\n💡 OVERALL RECOMMENDATION:")
        if improvement_rate > 0.7:
            print("✅ STRONGLY IMPLEMENT ENHANCED FEATURES")
            print("   • Consistent improvements across multiple targets")
            print("   • MACD and SMA/EMA features add significant predictive value")
        elif improvement_rate > 0.5:
            print("✅ IMPLEMENT ENHANCED FEATURES")
            print("   • Majority of targets show improvement")
            print("   • Benefits outweigh complexity costs")
        elif improvement_rate > 0.3:
            print("🤔 CONSIDER IMPLEMENTING SELECTIVELY")
            print("   • Mixed results - implement for specific use cases")
            print("   • Focus on best-performing targets")
        else:
            print("❌ STICK WITH CURRENT FEATURES")
            print("   • Enhanced features don't consistently improve performance")
            print("   • Additional complexity not justified")
        
        print(f"\n📈 KEY INSIGHTS:")
        print(f"   • Traditional RSI signals vs actual returns show different patterns")
        print(f"   • MACD/SMA/EMA may be more useful for trend-following strategies")
        print(f"   • Consider timeframe-specific implementations")

def main():
    """Main execution function"""
    tester = FutureReturnsTestor()
    return tester.run_comprehensive_test()

if __name__ == "__main__":
    regression_results, classification_results = main()