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

# NASDAQ predictions
print("=" * 80)
print("NASDAQ PREDICTIONS (Latest Date)")
print("=" * 80)
cursor.execute('''
    SELECT TOP 1 trading_date 
    FROM ml_trading_predictions 
    ORDER BY trading_date DESC
''')
nasdaq_date = cursor.fetchone()[0]
print(f"Date: {nasdaq_date}\n")

cursor.execute(f'''
    SELECT 
        predicted_signal, 
        COUNT(*) as count,
        AVG(buy_probability) as avg_buy,
        AVG(sell_probability) as avg_sell
    FROM ml_trading_predictions 
    WHERE trading_date = '{nasdaq_date}'
    GROUP BY predicted_signal
    ORDER BY predicted_signal
''')

nasdaq_total = 0
for row in cursor:
    signal, count, avg_buy, avg_sell = row
    nasdaq_total += count
    print(f"{signal:6} = {count:4} | AvgBuy={avg_buy:.3f} AvgSell={avg_sell:.3f}")

print(f"TOTAL  = {nasdaq_total}")

# NSE predictions
print("\n" + "=" * 80)
print("NSE PREDICTIONS (Latest Date)")
print("=" * 80)
cursor.execute('''
    SELECT TOP 1 trading_date 
    FROM ml_nse_trading_predictions 
    ORDER BY trading_date DESC
''')
nse_date = cursor.fetchone()[0]
print(f"Date: {nse_date}\n")

cursor.execute(f'''
    SELECT 
        predicted_signal, 
        COUNT(*) as count,
        AVG(buy_probability) as avg_buy,
        AVG(sell_probability) as avg_sell
    FROM ml_nse_trading_predictions 
    WHERE trading_date = '{nse_date}'
    GROUP BY predicted_signal
    ORDER BY predicted_signal
''')

nse_total = 0
for row in cursor:
    signal, count, avg_buy, avg_sell = row
    nse_total += count
    print(f"{signal:6} = {count:4} | AvgBuy={avg_buy:.3f} AvgSell={avg_sell:.3f}")

print(f"TOTAL  = {nse_total}")

cursor.close()
conn.close()
