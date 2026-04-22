"""Investigate the April 21 prediction results"""
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from database.connection import SQLServerConnection

conn = SQLServerConnection()

# Check April 21 predictions
query = """
SELECT 
    trading_date,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN predicted_signal = 'Buy' THEN 1 ELSE 0 END) as buy_count,
    SUM(CASE WHEN predicted_signal = 'Sell' THEN 1 ELSE 0 END) as sell_count,
    AVG(buy_probability) as avg_buy_prob,
    AVG(sell_probability) as avg_sell_prob,
    AVG(confidence_percentage) as avg_confidence,
    MIN(buy_probability) as min_buy_prob,
    MAX(buy_probability) as max_buy_prob,
    model_name
FROM ml_nse_trading_predictions
WHERE trading_date = '2026-04-21'
GROUP BY trading_date, model_name
ORDER BY trading_date DESC
"""

result = conn.execute_query(query)

print("\n" + "="*80)
print("APRIL 21, 2026 PREDICTION ANALYSIS")
print("="*80)

if result is not None and not result.empty:
    for _, row in result.iterrows():
        print(f"\nTrading Date: {row['trading_date']}")
        print(f"Model: {row['model_name']}")
        print(f"Total Predictions: {row['total_predictions']:,}")
        print(f"Buy Signals: {row['buy_count']:,} ({row['buy_count']/row['total_predictions']*100:.1f}%)")
        print(f"Sell Signals: {row['sell_count']:,} ({row['sell_count']/row['total_predictions']*100:.1f}%)")
        print(f"\nProbability Analysis:")
        print(f"  Avg Buy Probability: {row['avg_buy_prob']:.4f}")
        print(f"  Avg Sell Probability: {row['avg_sell_prob']:.4f}")
        print(f"  Min Buy Probability: {row['min_buy_prob']:.4f}")
        print(f"  Max Buy Probability: {row['max_buy_prob']:.4f}")
        print(f"  Avg Confidence: {row['avg_confidence']:.2f}%")
        
    # Check sample of buy predictions
    buy_sample_query = """
    SELECT TOP 20
        ticker,
        predicted_signal,
        buy_probability,
        sell_probability,
        confidence_percentage,
        RSI,
        close_price
    FROM ml_nse_trading_predictions
    WHERE trading_date = '2026-04-21' 
        AND predicted_signal = 'Buy'
    ORDER BY confidence_percentage DESC
    """
    
    buy_samples = conn.execute_query(buy_sample_query)
    
    if buy_samples is not None and not buy_samples.empty:
        print(f"\n{'='*80}")
        print(f"SAMPLE BUY PREDICTIONS (Top 20 by confidence)")
        print(f"{'='*80}")
        print(f"{'Ticker':<12} {'Signal':<6} {'Buy Prob':<10} {'Sell Prob':<10} {'Conf%':<8} {'RSI':<8} {'Close':<10}")
        print("-"*80)
        for _, p in buy_samples.iterrows():
            print(f"{p['ticker']:<12} {p['predicted_signal']:<6} {p['buy_probability']:<10.4f} {p['sell_probability']:<10.4f} {p['confidence_percentage']:<8.2f} {p['RSI']:<8.2f} {p['close_price']:<10.2f}")
    
    # Check sample of sell predictions to compare
    sell_sample_query = """
    SELECT TOP 10
        ticker,
        predicted_signal,
        buy_probability,
        sell_probability,
        confidence_percentage,
        RSI,
        close_price
    FROM ml_nse_trading_predictions
    WHERE trading_date = '2026-04-21' 
        AND predicted_signal = 'Sell'
    ORDER BY confidence_percentage DESC
    """
    
    sell_samples = conn.execute_query(sell_sample_query)
    
    if sell_samples is not None and not sell_samples.empty:
        print(f"\n{'='*80}")
        print(f"SAMPLE SELL PREDICTIONS (Top 10 by confidence for comparison)")
        print(f"{'='*80}")
        print(f"{'Ticker':<12} {'Signal':<6} {'Buy Prob':<10} {'Sell Prob':<10} {'Conf%':<8} {'RSI':<8} {'Close':<10}")
        print("-"*80)
        for _, p in sell_samples.iterrows():
            print(f"{p['ticker']:<12} {p['predicted_signal']:<6} {p['buy_probability']:<10.4f} {p['sell_probability']:<10.4f} {p['confidence_percentage']:<8.2f} {p['RSI']:<8.2f} {p['close_price']:<10.2f}")
            
else:
    print("\n[ERROR] No predictions found for April 21, 2026")

print("\n" + "="*80)
