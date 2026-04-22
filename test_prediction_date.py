"""Test the new prediction date logic"""
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from database.connection import SQLServerConnection

# Test getting the latest trading date
conn = SQLServerConnection()

query = "SELECT MAX(trading_date) as latest_date FROM nse_500_hist_data"
result = conn.execute_query(query)

if result is not None and not result.empty:
    latest_date = result.iloc[0]['latest_date']
    print(f"Latest Trading Date in Database: {latest_date}")
    print(f"\nThis is the date that will be used for predictions.")
    
    # Check if we have data for this date
    check_query = f"""
    SELECT COUNT(*) as count 
    FROM nse_500_hist_data 
    WHERE trading_date = '{latest_date}'
    """
    count_result = conn.execute_query(check_query)
    if count_result is not None and not count_result.empty:
        record_count = count_result.iloc[0]['count']
        print(f"Records available for {latest_date}: {record_count:,}")
else:
    print("ERROR: Could not determine latest trading date")
