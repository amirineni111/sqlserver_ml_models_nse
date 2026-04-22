"""Check predictions created today"""
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from database.connection import SQLServerConnection

conn = SQLServerConnection()

# Check predictions created today
query = """
SELECT 
    trading_date,
    COUNT(*) as prediction_count,
    MAX(created_at) as last_update,
    MIN(created_at) as first_update
FROM ml_nse_trading_predictions
WHERE CAST(created_at AS DATE) = '2026-04-21'
GROUP BY trading_date
ORDER BY trading_date DESC
"""

result = conn.execute_query(query)

if result is not None and not result.empty:
    print("\n" + "="*70)
    print("PREDICTIONS CREATED TODAY (April 21, 2026)")
    print("="*70)
    for _, row in result.iterrows():
        print(f"\nTrading Date: {row['trading_date']}")
        print(f"Count: {row['prediction_count']:,}")
        print(f"First Update: {row['first_update']}")
        print(f"Last Update: {row['last_update']}")
    print("="*70)
else:
    print("\n[INFO] No predictions were created today (April 21, 2026)")
    
    # Check most recent predictions
    query2 = """
    SELECT TOP 1
        trading_date,
        COUNT(*) as prediction_count,
        MAX(created_at) as last_update
    FROM ml_nse_trading_predictions
    GROUP BY trading_date
    ORDER BY trading_date DESC
    """
    
    result2 = conn.execute_query(query2)
    if result2 is not None and not result2.empty:
        row = result2.iloc[0]
        print(f"\n[INFO] Most recent predictions:")
        print(f"  Trading Date: {row['trading_date']}")
        print(f"  Count: {row['prediction_count']:,}")
        print(f"  Last Update: {row['last_update']}")

conn.close()
