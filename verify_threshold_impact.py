"""
Verify the impact of the new 60% threshold on existing predictions.
Shows how many signals would now be high-confidence vs. medium-confidence.
"""

import os
import pyodbc
import pandas as pd
from dotenv import load_dotenv
from nse_config import HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD

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


print("=" * 80)
print("COMPARING OLD VS NEW THRESHOLD IMPACT")
print("=" * 80)
print(f"OLD Threshold: High ≥ 70%, Medium 55-70%, Low < 55%")
print(f"NEW Threshold: High ≥ {HIGH_CONFIDENCE_THRESHOLD*100:.0f}%, Medium {MEDIUM_CONFIDENCE_THRESHOLD*100:.0f}-{HIGH_CONFIDENCE_THRESHOLD*100:.0f}%, Low < {MEDIUM_CONFIDENCE_THRESHOLD*100:.0f}%")
print()

conn = get_db_connection()

query = """
SELECT 
    trading_date,
    -- OLD threshold (70%)
    SUM(CASE WHEN confidence_percentage >= 70 THEN 1 ELSE 0 END) as old_high_count,
    SUM(CASE WHEN confidence_percentage >= 55 AND confidence_percentage < 70 THEN 1 ELSE 0 END) as old_medium_count,
    SUM(CASE WHEN confidence_percentage < 55 THEN 1 ELSE 0 END) as old_low_count,
    -- NEW threshold (60%)
    SUM(CASE WHEN confidence_percentage >= 60 THEN 1 ELSE 0 END) as new_high_count,
    SUM(CASE WHEN confidence_percentage >= 55 AND confidence_percentage < 60 THEN 1 ELSE 0 END) as new_medium_count,
    SUM(CASE WHEN confidence_percentage < 55 THEN 1 ELSE 0 END) as new_low_count,
    COUNT(*) as total,
    AVG(confidence_percentage) as avg_confidence
FROM dbo.ml_nse_trading_predictions
WHERE trading_date >= '2026-04-06' AND trading_date <= '2026-04-09'
GROUP BY trading_date
ORDER BY trading_date
"""

df = pd.read_sql(query, conn)

print("-" * 80)
print("DAILY COMPARISON (April 6-9):")
print("-" * 80)
print(df.to_string(index=False))
print()

# Summary
old_total_high = df['old_high_count'].sum()
new_total_high = df['new_high_count'].sum()
total_predictions = df['total'].sum()

print("=" * 80)
print("SUMMARY:")
print("=" * 80)
print(f"Total Predictions: {total_predictions:,}")
print()
print(f"OLD Threshold (70%):")
print(f"  High Confidence: {old_total_high:,} ({old_total_high/total_predictions*100:.1f}%)")
print()
print(f"NEW Threshold (60%):")
print(f"  High Confidence: {new_total_high:,} ({new_total_high/total_predictions*100:.1f}%)")
print()
print(f"📊 IMPACT: +{new_total_high - old_total_high:,} high-confidence signals unlocked!")
print(f"   ({(new_total_high - old_total_high)/4:.0f} signals per day average)")
print()

# What percentage of predictions fall in each bucket with NEW threshold
new_total_medium = df['new_medium_count'].sum()
new_total_low = df['new_low_count'].sum()

print("-" * 80)
print("NEW THRESHOLD DISTRIBUTION:")
print("-" * 80)
print(f"High (≥60%):    {new_total_high:5,} ({new_total_high/total_predictions*100:5.1f}%)")
print(f"Medium (55-60%): {new_total_medium:5,} ({new_total_medium/total_predictions*100:5.1f}%)")
print(f"Low (<55%):      {new_total_low:5,} ({new_total_low/total_predictions*100:5.1f}%)")
print("=" * 80)

conn.close()
