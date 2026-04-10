"""
Analyze prediction accuracy by confidence tier for NSE vs NASDAQ.

This script compares:
- NSE high-confidence (≥70%) accuracy
- NSE medium-confidence (55-70%) accuracy  
- NSE low-confidence (<55%) accuracy
- NASDAQ high-confidence (>70%) accuracy

Purpose: Determine if NSE medium-confidence predictions are more accurate
than NASDAQ high-confidence predictions.
"""

import os
import sys
from datetime import datetime, timedelta
import pyodbc
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_db_connection():
    """Create database connection"""
    server = os.getenv('SQL_SERVER')
    database = os.getenv('SQL_DATABASE')
    username = os.getenv('SQL_USERNAME')
    password = os.getenv('SQL_PASSWORD')
    driver = os.getenv('SQL_DRIVER', 'ODBC Driver 17 for SQL Server')
    trusted_connection = os.getenv('SQL_TRUSTED_CONNECTION', 'no')
    
    if trusted_connection.lower() == 'yes':
        conn_str = f'DRIVER={{{driver}}};SERVER={server};DATABASE={database};Trusted_Connection=yes'
    else:
        conn_str = f'DRIVER={{{driver}}};SERVER={server};DATABASE={database};UID={username};PWD={password};TrustServerCertificate=yes'
    
    return pyodbc.connect(conn_str)


def analyze_nse_accuracy(conn, start_date, end_date):
    """Analyze NSE prediction accuracy by confidence tier"""
    
    query = """
    SELECT 
        CASE 
            WHEN p.high_confidence = 1 THEN 'High (≥70%)'
            WHEN p.medium_confidence = 1 THEN 'Medium (55-70%)' 
            ELSE 'Low (<55%)'
        END as confidence_tier,
        COUNT(*) as total_predictions,
        SUM(CASE WHEN h.direction_correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
        CAST(100.0 * SUM(CASE WHEN h.direction_correct = 1 THEN 1 ELSE 0 END) 
             / NULLIF(COUNT(*), 0) AS DECIMAL(5,2)) as accuracy_pct,
        AVG(p.confidence_percentage) as avg_confidence,
        SUM(CASE WHEN p.predicted_signal = 'Buy' THEN 1 ELSE 0 END) as buy_signals,
        SUM(CASE WHEN p.predicted_signal = 'Sell' THEN 1 ELSE 0 END) as sell_signals
    FROM dbo.ml_nse_trading_predictions p
    LEFT JOIN dbo.ai_prediction_history h 
        ON p.ticker = h.ticker 
        AND p.trading_date = h.target_date  
        AND h.market = 'NSE 500'
    WHERE p.trading_date >= ?
      AND p.trading_date <= ?
      AND h.direction_correct IS NOT NULL
    GROUP BY 
        CASE 
            WHEN p.high_confidence = 1 THEN 'High (≥70%)'
            WHEN p.medium_confidence = 1 THEN 'Medium (55-70%)' 
            ELSE 'Low (<55%)'
        END
    ORDER BY accuracy_pct DESC
    """
    
    return pd.read_sql(query, conn, params=[start_date, end_date])


def analyze_nasdaq_accuracy(conn, start_date, end_date):
    """Analyze NASDAQ high-confidence prediction accuracy"""
    
    query = """
    SELECT 
        'NASDAQ High (>70%)' as confidence_tier,
        COUNT(*) as total_predictions,
        SUM(CASE WHEN h.direction_correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
        CAST(100.0 * SUM(CASE WHEN h.direction_correct = 1 THEN 1 ELSE 0 END) 
             / NULLIF(COUNT(*), 0) AS DECIMAL(5,2)) as accuracy_pct,
        AVG(p.confidence_percentage) as avg_confidence,
        SUM(CASE WHEN p.predicted_signal LIKE '%Buy%' THEN 1 ELSE 0 END) as buy_signals,
        SUM(CASE WHEN p.predicted_signal LIKE '%Sell%' THEN 1 ELSE 0 END) as sell_signals
    FROM dbo.ml_trading_predictions p
    LEFT JOIN dbo.ai_prediction_history h 
        ON p.ticker = h.ticker 
        AND p.trading_date = h.target_date  
        AND h.market = 'NASDAQ 100'
    WHERE p.trading_date >= ?
      AND p.trading_date <= ?
      AND p.high_confidence = 1
      AND h.direction_correct IS NOT NULL
    """
    
    return pd.read_sql(query, conn, params=[start_date, end_date])


def get_signal_count_by_date(conn, start_date, end_date):
    """Get NSE signal counts by date to show the missing opportunities"""
    
    query = """
    SELECT 
        p.trading_date,
        COUNT(*) as total_predictions,
        SUM(CASE WHEN p.high_confidence = 1 THEN 1 ELSE 0 END) as high_conf_count,
        SUM(CASE WHEN p.medium_confidence = 1 THEN 1 ELSE 0 END) as medium_conf_count,
        SUM(CASE WHEN p.low_confidence = 1 THEN 1 ELSE 0 END) as low_conf_count,
        AVG(p.confidence_percentage) as avg_confidence
    FROM dbo.ml_nse_trading_predictions p
    WHERE p.trading_date >= ?
      AND p.trading_date <= ?
    GROUP BY p.trading_date
    ORDER BY p.trading_date
    """
    
    return pd.read_sql(query, conn, params=[start_date, end_date])


def print_results(nse_results, nasdaq_results, daily_counts):
    """Print formatted analysis results"""
    
    print("=" * 80)
    print("CONFIDENCE TIER ACCURACY ANALYSIS")
    print("=" * 80)
    print(f"Analysis Period: {daily_counts['trading_date'].min()} to {daily_counts['trading_date'].max()}")
    print()
    
    # NSE Results
    print("-" * 80)
    print("NSE 500 PREDICTIONS BY CONFIDENCE TIER")
    print("-" * 80)
    print(nse_results.to_string(index=False))
    print()
    
    # NASDAQ Results
    print("-" * 80)
    print("NASDAQ 100 HIGH-CONFIDENCE PREDICTIONS")
    print("-" * 80)
    print(nasdaq_results.to_string(index=False))
    print()
    
    # Daily breakdown
    print("-" * 80)
    print("NSE DAILY SIGNAL COUNTS (Showing missed opportunities)")
    print("-" * 80)
    print(daily_counts.to_string(index=False))
    print()
    
    # Key findings
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    # Extract key metrics
    nse_medium = nse_results[nse_results['confidence_tier'] == 'Medium (55-70%)']
    nse_high = nse_results[nse_results['confidence_tier'] == 'High (≥70%)']
    
    if not nasdaq_results.empty:
        nasdaq_acc = nasdaq_results['accuracy_pct'].iloc[0]
        nasdaq_total = nasdaq_results['total_predictions'].iloc[0]
        
        print(f"NASDAQ High-Confidence: {nasdaq_acc:.2f}% accuracy ({nasdaq_total:,} predictions)")
        print()
    
    if not nse_high.empty:
        nse_high_acc = nse_high['accuracy_pct'].iloc[0]
        nse_high_total = nse_high['total_predictions'].iloc[0]
        print(f"NSE High-Confidence: {nse_high_acc:.2f}% accuracy ({nse_high_total:,} predictions)")
    else:
        print(f"NSE High-Confidence: NO PREDICTIONS (0 signals)")
    
    if not nse_medium.empty:
        nse_medium_acc = nse_medium['accuracy_pct'].iloc[0]
        nse_medium_total = nse_medium['total_predictions'].iloc[0]
        print(f"NSE Medium-Confidence: {nse_medium_acc:.2f}% accuracy ({nse_medium_total:,} predictions)")
        print()
        
        # Calculate opportunity cost
        total_medium_signals = daily_counts['medium_conf_count'].sum()
        print(f"MISSED OPPORTUNITY: {total_medium_signals:,} medium-confidence signals NOT USED")
        
        if not nasdaq_results.empty:
            if nse_medium_acc > nasdaq_acc:
                diff = nse_medium_acc - nasdaq_acc
                print(f"⚠️  NSE MEDIUM is {diff:.2f}% MORE ACCURATE than NASDAQ HIGH!")
                print(f"⚠️  Recommendation: LOWER NSE threshold to 55-60% immediately")
            elif abs(nse_medium_acc - nasdaq_acc) < 5:
                print(f"✓ NSE MEDIUM and NASDAQ HIGH have similar accuracy")
                print(f"  Recommendation: Use NSE medium-confidence during volatile markets")
            else:
                diff = nasdaq_acc - nse_medium_acc
                print(f"  NASDAQ HIGH is {diff:.2f}% more accurate than NSE MEDIUM")
                print(f"  Current threshold (70%) appears appropriate")
    else:
        print("NSE Medium-Confidence: NO DATA AVAILABLE for comparison")
    
    print()
    print("=" * 80)


def main():
    """Main execution"""
    
    # Default to last 5 trading days
    end_date = datetime(2026, 4, 9)
    start_date = datetime(2026, 4, 6)
    
    print(f"\nConnecting to database...")
    
    try:
        conn = get_db_connection()
        print(f"Connected successfully!")
        print(f"\nAnalyzing predictions from {start_date.date()} to {end_date.date()}...\n")
        
        # Get analysis results
        nse_results = analyze_nse_accuracy(conn, start_date, end_date)
        nasdaq_results = analyze_nasdaq_accuracy(conn, start_date, end_date)
        daily_counts = get_signal_count_by_date(conn, start_date, end_date)
        
        # Print results
        print_results(nse_results, nasdaq_results, daily_counts)
        
        conn.close()
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
