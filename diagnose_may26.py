"""Quick diagnostic: check market context feature values for today."""
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
SELECT TOP 220 trading_date, nifty50_close, india_vix_close, sp500_close, vix_close, dxy_close
FROM market_context_daily
ORDER BY trading_date DESC
""", conn)
df = df.sort_values("trading_date")

print("=== MARKET CONTEXT DIAGNOSIS FOR MAY 26, 2026 ===\n")

# NIFTY50
nifty = df[df["nifty50_close"].notna()]
nifty_200d = nifty["nifty50_close"].tail(200).mean()
nifty_last = nifty["nifty50_close"].iloc[-1]
nifty_last_date = nifty["trading_date"].iloc[-1]
print(f"NIFTY50 last available: {nifty_last_date} = {nifty_last:.2f}")
print(f"NIFTY50 200d MA: {nifty_200d:.2f}")
print(f"nifty50_vs_200d (using last close): {(nifty_last/nifty_200d - 1)*100:.2f}%")
print(f"NOTE: May 26 NIFTY50 is NULL in DB - feature uses forward-fill or 0")
print()

# India VIX
ivix = df[df["india_vix_close"].notna()]
ivix_last = ivix["india_vix_close"].iloc[-1]
ivix_60d = ivix["india_vix_close"].tail(60).mean()
ivix_last_date = ivix["trading_date"].iloc[-1]
print(f"India VIX last available: {ivix_last_date} = {ivix_last:.2f}")
print(f"India VIX 60d avg: {ivix_60d:.2f}")
print(f"india_vix_vs_60d: {(ivix_last/ivix_60d - 1)*100:.2f}%")
print()

# SP500
sp = df[df["sp500_close"].notna()]
sp_last = sp["sp500_close"].iloc[-1]
sp_200d = sp["sp500_close"].tail(200).mean()
sp_last_date = sp["trading_date"].iloc[-1]
print(f"SP500 last available: {sp_last_date} = {sp_last:.2f}")
print(f"SP500 200d MA: {sp_200d:.2f}")
print(f"sp500_vs_200d: {(sp_last/sp_200d - 1)*100:.2f}%")
print()

# VIX
vix = df[df["vix_close"].notna()]
vix_last = vix["vix_close"].iloc[-1]
vix_60d = vix["vix_close"].tail(60).mean()
print(f"VIX last ({vix['trading_date'].iloc[-1]}): {vix_last:.2f}, 60d avg: {vix_60d:.2f}")
print(f"vix_vs_60d: {(vix_last/vix_60d - 1)*100:.2f}%")
print()

# DXY
dxy = df[df["dxy_close"].notna()]
dxy_last = dxy["dxy_close"].iloc[-1]
dxy_60d = dxy["dxy_close"].tail(60).mean()
print(f"DXY last ({dxy['trading_date'].iloc[-1]}): {dxy_last:.2f}, 60d avg: {dxy_60d:.2f}")
print(f"dxy_vs_60d: {(dxy_last/dxy_60d - 1)*100:.2f}%")
print()

print("=== HOW THE PREDICTION SCRIPT HANDLES NULLS ===")
print("If predict_nse_signals_v2.py fills NULL market context with 0 (fillna(0)),")
print("then nifty50_vs_200d=0 and india_vix_vs_60d=0 for all stocks today.")
print("But the model may interpret 0 as out-of-distribution -> unusual predictions.")

conn.close()
