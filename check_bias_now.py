import os
import pyodbc
from dotenv import load_dotenv
load_dotenv()
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=192.168.86.28\\MSSQLSERVER01;'
    'DATABASE=stockdata_db;'
    f'UID=remote_user;PWD={os.environ["SQL_PASSWORD"]};TrustServerCertificate=yes'
)
cursor = conn.cursor()
sql = (
    "SELECT TOP 5 trading_date, COUNT(*) total, "
    "SUM(CASE WHEN predicted_signal='Buy' THEN 1 ELSE 0 END) buys, "
    "SUM(CASE WHEN predicted_signal='Sell' THEN 1 ELSE 0 END) sells, "
    "CAST(SUM(CASE WHEN predicted_signal='Buy' THEN 1 ELSE 0 END)*100.0/COUNT(*) AS DECIMAL(5,1)) buy_pct "
    "FROM ml_nse_trading_predictions WHERE model_name LIKE '%V2%' "
    "GROUP BY trading_date ORDER BY trading_date DESC"
)
cursor.execute(sql)
print("trading_date | total | buys | sells | buy_pct")
for row in cursor.fetchall():
    print(row)
conn.close()
