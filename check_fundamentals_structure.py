import pyodbc
import pandas as pd

conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=192.168.86.28\\MSSQLSERVER01;'
    'DATABASE=stockdata_db;'
    'UID=remote_user;'
    'PWD=YourStrongPassword123!;'
    'TrustServerCertificate=yes'
)

print("=" * 80)
print("nse_500_fundamentals TABLE STRUCTURE")
print("=" * 80)

# Get column info
cursor = conn.cursor()
cursor.execute("""
    SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'nse_500_fundamentals'
    ORDER BY ORDINAL_POSITION
""")

print("\nColumns:")
for row in cursor:
    col_name, data_type, max_len = row
    if max_len:
        print(f"  {col_name:<30} {data_type}({max_len})")
    else:
        print(f"  {col_name:<30} {data_type}")

# Get sample data for one ticker with duplicates
print("\n" + "=" * 80)
print("SAMPLE DATA - RELIANCE.NS (showing all 15 duplicate rows)")
print("=" * 80)

query = """
SELECT TOP 20 *
FROM nse_500_fundamentals
WHERE ticker = 'RELIANCE.NS'
"""

df = pd.read_sql(query, conn)
print(f"\nTotal rows for RELIANCE.NS: {len(df)}")
print("\nFirst few columns of each row:")
print(df.iloc[:, :10])  # Show first 10 columns

# Check if all rows are identical
if len(df) > 1:
    print("\n" + "=" * 80)
    print("CHECKING IF ROWS ARE IDENTICAL")
    print("=" * 80)
    
    # Compare first row with all others
    first_row = df.iloc[0]
    all_identical = True
    
    for idx in range(1, len(df)):
        if not df.iloc[idx].equals(first_row):
            all_identical = False
            print(f"\nRow {idx} differs from row 0:")
            diff_cols = [col for col in df.columns if df.iloc[idx][col] != first_row[col]]
            for col in diff_cols:
                print(f"  {col}: {first_row[col]} -> {df.iloc[idx][col]}")
    
    if all_identical:
        print("\n✅ All 15 rows are IDENTICAL - these are true duplicates!")
        print("   We can safely pick ANY row (ROW_NUMBER() is fine)")
    else:
        print("\n⚠️ Rows have differences - need to investigate which to use")

conn.close()
