"""Check what predictions exist for April 20 and 21"""
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from database.connection import SQLServerConnection

conn = SQLServerConnection()

# Check predictions for both dates
query = """
SELECT 
    trading_date,
    COUNT(*) as prediction_count,
    MIN(created_at) as first_created,
    MAX(created_at) as last_created
FROM ml_nse_trading_predictions
WHERE trading_date IN ('2026-04-20', '2026-04-21')
GROUP BY trading_date
ORDER BY trading_date DESC
"""

result = conn.execute_query(query)

print("\n" + "="*70)
print("PREDICTIONS FOR APRIL 20 & 21, 2026")
print("="*70)

if result is not None and not result.empty:
    for _, row in result.iterrows():
        print(f"\nTrading Date: {row['trading_date']}")
        print(f"  Count: {row['prediction_count']:,}")
        print(f"  First Created: {row['first_created']}")
        print(f"  Last Created: {row['last_created']}")
else:
    print("\n[INFO] No predictions found for April 20 or 21")

print("="*70)
