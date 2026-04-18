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
print("SEARCHING FOR MARKET INDEX TABLES")
print("=" * 80)

cursor.execute("""
    SELECT TABLE_NAME 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_NAME LIKE '%nifty%' 
       OR TABLE_NAME LIKE '%vix%' 
       OR TABLE_NAME LIKE '%market%' 
       OR TABLE_NAME LIKE '%sp500%'
       OR TABLE_NAME LIKE '%dxy%'
       OR TABLE_NAME LIKE '%index%'
    ORDER BY TABLE_NAME
""")

print("\nTables found:")
tables = []
for row in cursor:
    table = row[0]
    tables.append(table)
    print(f"  - {table}")

# Check each table for nifty50/vix columns
print("\n" + "=" * 80)
print("CHECKING TABLE SCHEMAS")
print("=" * 80)

for table in tables:
    try:
        cursor.execute(f"SELECT TOP 1 * FROM {table}")
        columns = [col[0] for col in cursor.description]
        
        # Check if has market data columns
        has_nifty = any('nifty' in col.lower() for col in columns)
        has_vix = any('vix' in col.lower() for col in columns)
        has_sp500 = any('sp500' in col.lower() for col in columns)
        
        if has_nifty or has_vix or has_sp500:
            print(f"\n✓ {table}:")
            print(f"  Columns ({len(columns)}): {', '.join(columns[:10])}")
            if len(columns) > 10:
                print(f"  ... and {len(columns)-10} more")
            
            # Try to get sample data
            cursor.execute(f"SELECT TOP 1 * FROM {table} ORDER BY 1 DESC")
            row = cursor.fetchone()
            if row:
                print(f"  Latest date column: {row[0]}")
    except Exception as e:
        print(f"\n⚠️  {table}: {e}")

cursor.close()
conn.close()
