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

# Get all check constraints on the table
cursor.execute("""
    SELECT 
        cc.CONSTRAINT_NAME,
        cc.CHECK_CLAUSE
    FROM INFORMATION_SCHEMA.CHECK_CONSTRAINTS cc
    JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE cu
        ON cc.CONSTRAINT_NAME = cu.CONSTRAINT_NAME
    WHERE cu.TABLE_NAME = 'ml_nse_trading_predictions'
""")

print("All check constraints on ml_nse_trading_predictions:")
for row in cursor:
    print(f"  Constraint: {row[0]}")
    print(f"  Check: {row[1]}")
    print()

conn.close()
