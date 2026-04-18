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

# Get required columns
cursor.execute("""
    SELECT COLUMN_NAME, IS_NULLABLE, DATA_TYPE 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'ml_nse_trading_predictions' 
    AND IS_NULLABLE = 'NO' 
    ORDER BY ORDINAL_POSITION
""")

print("Required columns (NOT NULL):")
for row in cursor:
    print(f"  {row[0]} ({row[2]})")

conn.close()
