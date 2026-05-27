"""Quick check: is nifty50_return_1d in percentage or decimal?"""
from dotenv import load_dotenv
import os, pyodbc, pandas as pd
load_dotenv()
conn_str = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={os.getenv('SQL_SERVER')};"
    f"DATABASE={os.getenv('SQL_DATABASE')};"
    f"UID={os.getenv('SQL_USERNAME')};"
    f"PWD={os.getenv('SQL_PASSWORD')};"
    "TrustServerCertificate=yes"
)
conn = pyodbc.connect(conn_str)

df = pd.read_sql("""
SELECT TOP 10 trading_date, nifty50_return_1d, nifty50_close
FROM market_context_daily
ORDER BY trading_date DESC
""", conn)
print("nifty50_return_1d values (check if pct or decimal):")
print(df.to_string(index=False))
print("\nNote: if close goes from 24000 to 24031 (+0.13%), the value should be:")
print("  Decimal: 0.0013")
print("  Percentage: 0.13")
print()

# Also check nifty50 return_1d computed vs stored
df = df.sort_values('trading_date')
df['computed_return'] = df['nifty50_close'].pct_change(1)
df['stored_return'] = df['nifty50_return_1d']
print("Computed pct_change (decimal) vs stored nifty50_return_1d:")
print(df[['trading_date', 'nifty50_close', 'computed_return', 'stored_return']].to_string(index=False))
conn.close()
