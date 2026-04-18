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

# Check top 20 predictions by Buy probability
print("=" * 80)
print("TOP 20 PREDICTIONS BY BUY PROBABILITY")
print("=" * 80)
cursor.execute('''
    SELECT TOP 20 
        ticker, predicted_signal, 
        buy_probability, sell_probability, confidence
    FROM ml_nse_trading_predictions 
    WHERE trading_date = '2026-04-17' 
    ORDER BY buy_probability DESC
''')

for row in cursor:
    ticker, signal, buy_prob, sell_prob, conf = row
    print(f"{ticker:15} {signal:6} Buy={buy_prob:.3f} Sell={sell_prob:.3f} Conf={conf:.3f}")

# Overall distribution
print("\n" + "=" * 80)
print("OVERALL SIGNAL DISTRIBUTION")
print("=" * 80)
cursor.execute('''
    SELECT 
        predicted_signal, 
        COUNT(*) as count,
        AVG(buy_probability) as avg_buy_prob,
        AVG(sell_probability) as avg_sell_prob,
        AVG(confidence) as avg_conf
    FROM ml_nse_trading_predictions 
    WHERE trading_date = '2026-04-17' 
    GROUP BY predicted_signal
''')

for row in cursor:
    signal, count, avg_buy, avg_sell, avg_conf = row
    print(f"{signal:6} Count={count:4} AvgBuy={avg_buy:.3f} AvgSell={avg_sell:.3f} AvgConf={avg_conf:.3f}")

cursor.close()
conn.close()
