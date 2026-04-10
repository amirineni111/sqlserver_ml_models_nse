"""
Check NSE prediction data and history matching.
"""

import os
import pyodbc
import pandas as pd
from dotenv import load_dotenv

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


conn = get_db_connection()

# Check NSE predictions exist
print("=" * 80)
print("Checking NSE Predictions...")
print("=" * 80)

query1 = """
SELECT TOP 5
    ticker, trading_date, predicted_signal, 
    confidence_percentage, high_confidence, medium_confidence, low_confidence
FROM dbo.ml_nse_trading_predictions
WHERE trading_date >= '2026-04-06'
ORDER BY trading_date DESC, confidence_percentage DESC
"""

df1 = pd.read_sql(query1, conn)
print(df1)
print()

# Check if ai_prediction_history has NSE data
print("=" * 80)
print("Checking ai_prediction_history for NSE 500...")
print("=" * 80)

query2 = """
SELECT TOP 5
    ticker, target_date, market, direction_correct, actual_price
FROM dbo.ai_prediction_history
WHERE market = 'NSE 500'
  AND target_date >= '2026-04-06'
ORDER BY target_date DESC
"""

df2 = pd.read_sql(query2, conn)
print(df2)
print()

print(f"Total NSE history records for April 6-9: {len(df2)}")
print()

# Check what markets exist in history
query3 = """
SELECT market, COUNT(*) as count, 
       MIN(target_date) as first_date, 
       MAX(target_date) as last_date
FROM dbo.ai_prediction_history
WHERE target_date >= '2026-04-01'
GROUP BY market
ORDER BY market
"""

df3 = pd.read_sql(query3, conn)
print("=" * 80)
print("Markets in ai_prediction_history (April 2026):")
print("=" * 80)
print(df3)
print()

# Get confidence distribution for NSE regardless of history
query4 = """
SELECT 
    CASE 
        WHEN high_confidence = 1 THEN 'High (≥70%)'
        WHEN medium_confidence = 1 THEN 'Medium (55-70%)'
        ELSE 'Low (<55%)'
    END as tier,
    COUNT(*) as count,
    AVG(confidence_percentage) as avg_conf,
    MIN(confidence_percentage) as min_conf,
    MAX(confidence_percentage) as max_conf
FROM dbo.ml_nse_trading_predictions
WHERE trading_date >= '2026-04-06' AND trading_date <= '2026-04-09'
GROUP BY 
    CASE 
        WHEN high_confidence = 1 THEN 'High (≥70%)'
        WHEN medium_confidence = 1 THEN 'Medium (55-70%)'
        ELSE 'Low (<55%)'
    END
ORDER BY avg_conf DESC
"""

df4 = pd.read_sql(query4, conn)
print("=" * 80)
print("NSE Prediction Distribution (April 6-9, no history match required):")
print("=" * 80) 
print(df4)
print()

conn.close()
