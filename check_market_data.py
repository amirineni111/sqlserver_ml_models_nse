"""
Check if market-wide features exist for April 17, 2026
"""
import pyodbc

conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=192.168.86.28\\MSSQLSERVER01;'
    'DATABASE=stockdata_db;'
    'UID=remote_user;'
    'PWD=YourStrongPassword123!;'
    'TrustServerCertificate=yes'
)

cursor = conn.cursor()

print("=" * 80)
print("CHECKING MARKET-WIDE FEATURES FOR APRIL 17, 2026")
print("=" * 80)

# Check if view exists
cursor.execute("""
    SELECT COUNT(*) 
    FROM INFORMATION_SCHEMA.VIEWS 
    WHERE TABLE_NAME = 'vw_market_calendar_features'
""")
view_exists = cursor.fetchone()[0] > 0
print(f"\nvw_market_calendar_features exists: {view_exists}")

if view_exists:
    # Check data for April 17
    cursor.execute("""
        SELECT TOP 1 *
        FROM vw_market_calendar_features
        WHERE trading_date = '2026-04-17'
    """)
    
    row = cursor.fetchone()
    if row:
        print("\n✓ Market data EXISTS for April 17, 2026")
        print(f"\nColumns: {[col[0] for col in cursor.description]}")
        print(f"\nValues:")
        for i, col in enumerate(cursor.description):
            print(f"  {col[0]:30} = {row[i]}")
    else:
        print("\n❌ NO MARKET DATA for April 17, 2026!")
        print("\nChecking available dates...")
        cursor.execute("""
            SELECT TOP 5 trading_date, india_vix_close, nifty50_close
            FROM vw_market_calendar_features
            ORDER BY trading_date DESC
        """)
        print("\nLatest 5 dates:")
        for row in cursor:
            print(f"  {row[0]} | VIX={row[1]} | NIFTY={row[2]}")
else:
    print("\n❌ VIEW DOES NOT EXIST!")
    print("\nSearching for alternative tables...")
    cursor.execute("""
        SELECT TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_NAME LIKE '%market%' OR TABLE_NAME LIKE '%nifty%' OR TABLE_NAME LIKE '%vix%'
    """)
    print("\nRelated tables:")
    for row in cursor:
        print(f"  - {row[0]}")

cursor.close()
conn.close()
