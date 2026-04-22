"""Check today's (April 21, 2026) NSE predictions"""
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from database.connection import SQLServerConnection

def check_today_predictions():
    """Check if predictions exist for today"""
    print("=" * 70)
    print("CHECKING TODAY'S NSE PREDICTIONS (April 21, 2026)")
    print("=" * 70)
    
    conn = SQLServerConnection()
    
    # Check today's predictions
    query = """
    SELECT 
        COUNT(*) as total_predictions,
        COUNT(DISTINCT ticker) as unique_tickers,
        SUM(CASE WHEN predicted_signal = 'Buy' THEN 1 ELSE 0 END) as buy_signals,
        SUM(CASE WHEN predicted_signal = 'Sell' THEN 1 ELSE 0 END) as sell_signals,
        SUM(CASE WHEN high_confidence = 1 THEN 1 ELSE 0 END) as high_confidence_count,
        AVG(confidence_percentage) as avg_confidence,
        MAX(created_at) as last_update
    FROM ml_nse_trading_predictions
    WHERE trading_date = '2026-04-21'
    """
    
    result = conn.execute_query(query)
    
    if result is not None and not result.empty:
        row = result.iloc[0]
        print(f"\n[SUCCESS] Today's Predictions Found!")
        print(f"  Total Predictions: {row['total_predictions']:,}")
        print(f"  Unique Tickers: {row['unique_tickers']:,}")
        print(f"  Buy Signals: {row['buy_signals']:,}")
        print(f"  Sell Signals: {row['sell_signals']:,}")
        print(f"  High Confidence: {row['high_confidence_count']:,}")
        print(f"  Avg Confidence: {row['avg_confidence']:.2f}%")
        print(f"  Last Update: {row['last_update']}")
        
        # Check summary table
        summary_query = """
        SELECT 
            trading_date,
            total_predictions,
            buy_signals,
            sell_signals,
            high_confidence_count,
            avg_confidence,
            model_name,
            model_accuracy,
            run_timestamp
        FROM ml_nse_predict_summary
        WHERE trading_date = '2026-04-21'
        ORDER BY created_at DESC
        """
        
        summary = conn.execute_query(summary_query)
        
        if summary is not None and not summary.empty:
            print(f"\n[SUMMARY] Today's Summary Record:")
            s = summary.iloc[0]
            print(f"  Trading Date: {s['trading_date']}")
            print(f"  Total Predictions: {s['total_predictions']:,}")
            print(f"  Buy/Sell: {s['buy_signals']:,}/{s['sell_signals']:,}")
            print(f"  High Confidence: {s['high_confidence_count']:,}")
            print(f"  Avg Confidence: {s['avg_confidence']:.2f}%")
            print(f"  Model: {s['model_name']}")
            print(f"  Model Accuracy: {s['model_accuracy']:.2f}%")
            print(f"  Run Timestamp: {s['run_timestamp']}")
        else:
            print("\n[WARNING] No summary record found for today")
            
        # Show top 10 high confidence predictions
        top_query = """
        SELECT TOP 10
            ticker,
            predicted_signal,
            confidence_percentage,
            signal_strength,
            RSI,
            model_name,
            sector
        FROM ml_nse_trading_predictions
        WHERE trading_date = '2026-04-21'
            AND high_confidence = 1
        ORDER BY confidence_percentage DESC
        """
        
        top_preds = conn.execute_query(top_query)
        
        if top_preds is not None and not top_preds.empty:
            print(f"\n[TOP 10] Highest Confidence Predictions for Today:")
            print(f"{'Ticker':<12} {'Signal':<6} {'Conf%':<8} {'Strength':<10} {'RSI':<8} {'Model':<20} {'Sector':<20}")
            print("-" * 110)
            for _, p in top_preds.iterrows():
                print(f"{p['ticker']:<12} {p['predicted_signal']:<6} {p['confidence_percentage']:<8.2f} {p['signal_strength']:<10} {p['RSI']:<8.2f} {p['model_name']:<20} {p['sector']:<20}")
                
        print("\n" + "=" * 70)
        print("[SUCCESS] Daily NSE job completed successfully for April 21, 2026!")
        print("=" * 70)
    else:
        print("\n[ERROR] No predictions found for today!")
        
    conn.close()

if __name__ == "__main__":
    check_today_predictions()
