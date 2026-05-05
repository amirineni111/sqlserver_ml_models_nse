"""
Fix missing NIFTY data in market_context_daily for May 1 and May 4
Calculate from nse_500_hist_data
"""
import pyodbc
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

conn_str = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={os.getenv('SQL_SERVER')};"
    f"DATABASE={os.getenv('SQL_DATABASE')};"
    f"UID={os.getenv('SQL_USERNAME')};"
    f"PWD={os.getenv('SQL_PASSWORD')}"
)

conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

print("="*80)
print("FIXING MISSING NIFTY DATA")
print("="*80)

# Check if NIFTY 50 symbol exists in nse_500_hist_data
query = """
SELECT TOP 5
    trading_date,
    CAST(close_price AS FLOAT) as close_price,
    CAST(open_price AS FLOAT) as open_price
FROM nse_500_hist_data
WHERE ticker = '^NSEI'  -- NIFTY 50 index symbol
ORDER BY trading_date DESC
"""

try:
    df_nifty = pd.read_sql(query, conn)
    
    if len(df_nifty) > 0:
        print("\n✓ NIFTY data found in nse_500_hist_data:")
        print(df_nifty.to_string(index=False))
        
        # Calculate missing data
        query2 = """
        WITH nifty_data AS (
            SELECT 
                trading_date,
                CAST(close_price AS FLOAT) as close_price,
                LAG(CAST(close_price AS FLOAT)) OVER (ORDER BY trading_date) as prev_close
            FROM nse_500_hist_data
            WHERE ticker = '^NSEI'
              AND trading_date >= '2026-04-30'
        )
        SELECT 
            trading_date,
            close_price,
            prev_close,
            CASE 
                WHEN prev_close > 0 
                THEN ((close_price - prev_close) / prev_close) * 100
                ELSE 0 
            END as return_1d
        FROM nifty_data
        WHERE trading_date IN ('2026-05-01', '2026-05-04')
        ORDER BY trading_date
        """
        
        df_calc = pd.read_sql(query2, conn)
        
        if len(df_calc) > 0:
            print("\n[INFO] Calculated NIFTY returns:")
            print(df_calc.to_string(index=False))
            
            # Update market_context_daily
            for _, row in df_calc.iterrows():
                trading_date = row['trading_date']
                nifty_close = row['close_price']
                nifty_return = row['return_1d']
                
                update_query = """
                UPDATE market_context_daily
                SET nifty50_close = ?,
                    nifty50_return_1d = ?
                WHERE trading_date = ?
                """
                
                cursor.execute(update_query, nifty_close, nifty_return, trading_date)
                print(f"\n✓ Updated {trading_date}: close={nifty_close:.2f}, return={nifty_return:.2f}%")
            
            conn.commit()
            print("\n" + "="*80)
            print("SUCCESS: NIFTY data updated")
            print("="*80)
            print("\nNEXT STEPS:")
            print("  1. Re-run predictions: python predict_nse_signals_v2.py")
            print("  2. Hybrid features will now work correctly")
            print("  3. Expect 40-50% Buy signals (balanced distribution)")
            
        else:
            print("\n⚠ No NIFTY data for May 1 or May 4")
            print("  May be a weekend or market holiday")
            
    else:
        print("\n✗ NIFTY data NOT found in nse_500_hist_data")
        print("  Ticker might be different (try: 'NIFTY', 'NIFTY50', etc.)")
        
        # Try alternative ticker names
        alt_query = """
        SELECT DISTINCT ticker
        FROM nse_500_hist_data
        WHERE ticker LIKE '%NIFTY%' OR ticker LIKE '%NSEI%'
        """
        df_alt = pd.read_sql(alt_query, conn)
        if len(df_alt) > 0:
            print("\n  Found these NIFTY-related tickers:")
            print(df_alt.to_string(index=False))
        
except Exception as e:
    print(f"\n✗ Error: {e}")
    conn.rollback()

finally:
    conn.close()

print("\n" + "="*80)
