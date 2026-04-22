"""
Diagnose why we still have only 42 Buy signals after calibration fix
"""

import pyodbc
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_connection():
    """Create database connection"""
    sql_server = os.getenv('SQL_SERVER')
    sql_database = os.getenv('SQL_DATABASE')
    sql_username = os.getenv('SQL_USERNAME')
    sql_password = os.getenv('SQL_PASSWORD')
    sql_driver = os.getenv('SQL_DRIVER', 'ODBC Driver 17 for SQL Server')
    
    conn_str = (
        f"DRIVER={{{sql_driver}}};"
        f"SERVER={sql_server};"
        f"DATABASE={sql_database};"
        f"UID={sql_username};"
        f"PWD={sql_password};"
        f"TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str)

def analyze_predictions():
    """Analyze the latest predictions"""
    conn = get_connection()
    
    query = """
    SELECT 
        ticker,
        predicted_signal,
        confidence_percentage,
        buy_probability,
        sell_probability,
        rsi,
        close_price,
        sector,
        market_cap_category
    FROM ml_nse_trading_predictions
    WHERE trading_date = '2026-04-21'
    AND model_name LIKE '%V2%'
    ORDER BY buy_probability DESC
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print("="*80)
    print("PREDICTION DISTRIBUTION ANALYSIS - April 21, 2026")
    print("="*80)
    
    print(f"\nTotal predictions: {len(df)}")
    print(f"Buy signals: {(df['predicted_signal'] == 'Buy').sum()} ({(df['predicted_signal'] == 'Buy').sum() / len(df) * 100:.1f}%)")
    print(f"Sell signals: {(df['predicted_signal'] == 'Sell').sum()} ({(df['predicted_signal'] == 'Sell').sum() / len(df) * 100:.1f}%)")
    
    print("\n" + "="*80)
    print("PROBABILITY DISTRIBUTION")
    print("="*80)
    
    print(f"\nBuy Probability Stats:")
    print(f"  Mean: {df['buy_probability'].mean():.4f}")
    print(f"  Median: {df['buy_probability'].median():.4f}")
    print(f"  Std: {df['buy_probability'].std():.4f}")
    print(f"  Min: {df['buy_probability'].min():.4f}")
    print(f"  Max: {df['buy_probability'].max():.4f}")
    print(f"  25th percentile: {df['buy_probability'].quantile(0.25):.4f}")
    print(f"  75th percentile: {df['buy_probability'].quantile(0.75):.4f}")
    
    print(f"\nSell Probability Stats:")
    print(f"  Mean: {df['sell_probability'].mean():.4f}")
    print(f"  Median: {df['sell_probability'].median():.4f}")
    print(f"  Std: {df['sell_probability'].std():.4f}")
    print(f"  Min: {df['sell_probability'].min():.4f}")
    print(f"  Max: {df['sell_probability'].max():.4f}")
    
    print("\n" + "="*80)
    print("PROBABILITY DISTRIBUTION BINS")
    print("="*80)
    
    bins = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    df['buy_prob_bin'] = pd.cut(df['buy_probability'], bins=bins)
    
    print("\nBuy Probability Distribution:")
    prob_dist = df['buy_prob_bin'].value_counts().sort_index()
    for bin_range, count in prob_dist.items():
        pct = count / len(df) * 100
        print(f"  {bin_range}: {count:4d} ({pct:5.1f}%)")
    
    print("\n" + "="*80)
    print("TOP 20 BUY PREDICTIONS (Highest buy_probability)")
    print("="*80)
    
    top_buy = df.nlargest(20, 'buy_probability')[['ticker', 'predicted_signal', 'buy_probability', 'sell_probability', 'confidence_percentage', 'rsi']]
    print(top_buy.to_string(index=False))
    
    print("\n" + "="*80)
    print("BOTTOM 20 PREDICTIONS (Lowest buy_probability)")
    print("="*80)
    
    bottom = df.nsmallest(20, 'buy_probability')[['ticker', 'predicted_signal', 'buy_probability', 'sell_probability', 'confidence_percentage', 'rsi']]
    print(bottom.to_string(index=False))
    
    print("\n" + "="*80)
    print("DISTRIBUTION BY SECTOR")
    print("="*80)
    
    sector_dist = df.groupby('sector').agg({
        'predicted_signal': lambda x: (x == 'Buy').sum(),
        'ticker': 'count',
        'buy_probability': 'mean'
    }).round(4)
    sector_dist.columns = ['Buy_Count', 'Total', 'Avg_Buy_Prob']
    sector_dist['Buy_Pct'] = (sector_dist['Buy_Count'] / sector_dist['Total'] * 100).round(1)
    sector_dist = sector_dist.sort_values('Buy_Count', ascending=False)
    print(sector_dist)
    
    print("\n" + "="*80)
    print("CONFIDENCE DISTRIBUTION")
    print("="*80)
    
    print(f"\nConfidence Stats:")
    print(f"  Mean: {df['confidence_percentage'].mean():.1f}%")
    print(f"  Median: {df['confidence_percentage'].median():.1f}%")
    print(f"  High confidence (≥60%): {(df['confidence_percentage'] >= 60).sum()} ({(df['confidence_percentage'] >= 60).sum() / len(df) * 100:.1f}%)")
    
    # Check if probabilities sum to 1
    df['prob_sum'] = df['buy_probability'] + df['sell_probability']
    print(f"\nProbability Sum Check:")
    print(f"  Mean sum: {df['prob_sum'].mean():.6f}")
    print(f"  Should be ~1.0 for each prediction")
    if not np.allclose(df['prob_sum'], 1.0, atol=0.01):
        print("  ⚠️  WARNING: Probabilities don't sum to 1.0!")
    
    # Check decision threshold
    print("\n" + "="*80)
    print("DECISION THRESHOLD ANALYSIS")
    print("="*80)
    
    print(f"\nPredictions where buy_probability > 0.5: {(df['buy_probability'] > 0.5).sum()}")
    print(f"Actual Buy signals: {(df['predicted_signal'] == 'Buy').sum()}")
    
    if (df['buy_probability'] > 0.5).sum() != (df['predicted_signal'] == 'Buy').sum():
        print("⚠️  MISMATCH: Decision threshold may not be 0.5!")
        print("\nChecking Buy signals with buy_probability <= 0.5:")
        anomalies = df[(df['predicted_signal'] == 'Buy') & (df['buy_probability'] <= 0.5)]
        if len(anomalies) > 0:
            print(f"Found {len(anomalies)} Buy signals with buy_prob <= 0.5")
            print(anomalies[['ticker', 'buy_probability', 'sell_probability']].head())

if __name__ == "__main__":
    analyze_predictions()
