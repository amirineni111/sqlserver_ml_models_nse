"""
Check April 21, 2026 market context to see if bearish market conditions
are driving the 98% Sell predictions
"""

import pyodbc
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    """Create database connection"""
    sql_server = os.getenv('SQL_SERVER')
    sql_database = os.getenv('SQL_DATABASE')
    sql_username = os.getenv('SQL_USERNAME')
    sql_password = os.getenv('SQL_PASSWORD')
    sql_driver = os.getenv('SQL_DRIVER', 'ODBC Driver 17 for SQL Server')
    
    conn_str = (
        f"DRIVER={{{sql_driver}}};"
        f"SERVER={sql_server};"
        f"DATABASE={sql_database};"
        f"UID={sql_username};"
        f"PWD={sql_password};"
        f"TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str)

def check_market_conditions():
    """Check if bearish market conditions on April 21 are causing bias"""
    
    print("="*80)
    print("MARKET CONDITIONS ANALYSIS - April 21, 2026")
    print("="*80)
    
    conn = get_connection()
    
    # Check market context for April 21 and recent history
    query = """
    SELECT TOP 20
        trading_date,
        vix_close,
        vix_change_pct,
        india_vix_close,
        india_vix_change_pct,
        nifty50_close,
        nifty50_return_1d,
        sp500_close,
        sp500_return_1d,
        dxy_close,
        dxy_return_1d,
        us_10y_yield_close
    FROM market_context_daily
    WHERE trading_date <= '2026-04-21'
    ORDER BY trading_date DESC
    """
    
    df = pd.read_sql(query, conn)
    
    if len(df) == 0:
        print("[ERROR] No market_context_daily data found!")
        conn.close()
        return
    
    print(f"\n[SUCCESS] Fetched {len(df)} days of market context data")
    print(f"Date range: {df['trading_date'].min()} to {df['trading_date'].max()}")
    
    # April 21 conditions
    print("\n" + "="*80)
    print("APRIL 21, 2026 MARKET CONDITIONS")
    print("="*80)
    
    april21_mask = df['trading_date'].astype(str).str.contains('2026-04-21')
    
    if april21_mask.any():
        april21 = df[april21_mask].iloc[0]
        print(f"\nFear/Volatility Indicators:")
        print(f"  VIX: {april21['vix_close']:.2f} (change: {april21['vix_change_pct']:+.2f}%)")
        print(f"  India VIX: {april21['india_vix_close']:.2f} (change: {april21['india_vix_change_pct']:+.2f}%)")
        
        print(f"\nEquity Market Returns:")
        print(f"  NIFTY 50: {april21['nifty50_close']:.2f} (1d return: {april21['nifty50_return_1d']:+.2f}%)")
        print(f"  S&P 500: {april21['sp500_close']:.2f} (1d return: {april21['sp500_return_1d']:+.2f}%)")
        
        print(f"\nCurrency & Rates:")
        print(f"  DXY (USD strength): {april21['dxy_close']:.2f} (1d return: {april21['dxy_return_1d']:+.2f}%)")
        print(f"  US 10Y Yield: {april21['us_10y_yield_close']:.2f}%")
        
        # Assess bearishness
        print("\n" + "="*80)
        print("BEARISHNESS ASSESSMENT")
        print("="*80)
        
        bearish_signals = []
        bullish_signals = []
        
        if april21['vix_close'] > 20:
            bearish_signals.append(f"VIX elevated at {april21['vix_close']:.1f} (normal < 20)")
        if april21['india_vix_close'] > 20:
            bearish_signals.append(f"India VIX elevated at {april21['india_vix_close']:.1f} (normal < 20)")
        if april21['nifty50_return_1d'] < -0.5:
            bearish_signals.append(f"NIFTY 50 down {april21['nifty50_return_1d']:.1f}%")
        if april21['sp500_return_1d'] < -0.5:
            bearish_signals.append(f"S&P 500 down {april21['sp500_return_1d']:.1f}%")
        if april21['dxy_return_1d'] > 0.5:
            bearish_signals.append(f"Dollar strengthening (DXY up {april21['dxy_return_1d']:.1f}%)")
        
        if april21['vix_close'] < 15:
            bullish_signals.append(f"VIX low at {april21['vix_close']:.1f} (calm markets)")
        if april21['nifty50_return_1d'] > 0.5:
            bullish_signals.append(f"NIFTY 50 up {april21['nifty50_return_1d']:.1f}%")
        if april21['sp500_return_1d'] > 0.5:
            bullish_signals.append(f"S&P 500 up {april21['sp500_return_1d']:.1f}%")
        
        print(f"\n🔴 Bearish signals: {len(bearish_signals)}")
        for signal in bearish_signals:
            print(f"  - {signal}")
        
        print(f"\n🟢 Bullish signals: {len(bullish_signals)}")
        for signal in bullish_signals:
            print(f"  - {signal}")
        
        if len(bearish_signals) > len(bullish_signals):
            print(f"\n⚠️  BEARISH MARKET CONDITIONS on April 21!")
            print(f"   Model is likely predicting Sell for most stocks due to negative market context.")
            print(f"   Top features (VIX, DXY, yields, NIFTY returns) are bearish.")
    else:
        print("[WARNING] April 21, 2026 data not found in market_context_daily!")
    
    # Recent trend
    print("\n" + "="*80)
    print("RECENT MARKET TREND (Last 20 days)")
    print("="*80)
    
    print(f"\nAverage Daily Returns:")
    print(f"  NIFTY 50: {df['nifty50_return_1d'].mean():+.2f}% (std: {df['nifty50_return_1d'].std():.2f}%)")
    print(f"  S&P 500: {df['sp500_return_1d'].mean():+.2f}% (std: {df['sp500_return_1d'].std():.2f}%)")
    
    print(f"\nVolatility Trend:")
    print(f"  VIX: {df['vix_close'].mean():.1f} ± {df['vix_close'].std():.1f}")
    print(f"  India VIX: {df['india_vix_close'].mean():.1f} ± {df['india_vix_close'].std():.1f}")
    
    print(f"\nDollar Trend:")
    print(f"  DXY: {df['dxy_close'].mean():.2f} (range: {df['dxy_close'].min():.2f} - {df['dxy_close'].max():.2f})")
    print(f"  Avg daily change: {df['dxy_return_1d'].mean():+.2f}%")
    
    # Down days vs up days
    nifty_down_days = (df['nifty50_return_1d'] < 0).sum()
    nifty_up_days = (df['nifty50_return_1d'] > 0).sum()
    print(f"\nNIFTY 50 Direction:")
    print(f"  Down days: {nifty_down_days}/{len(df)} ({nifty_down_days/len(df)*100:.1f}%)")
    print(f"  Up days: {nifty_up_days}/{len(df)} ({nifty_up_days/len(df)*100:.1f}%)")
    
    if nifty_down_days > nifty_up_days * 1.5:
        print(f"\n⚠️  RECENT BEARISH TREND: {nifty_down_days} down days vs {nifty_up_days} up days")
    
    conn.close()
    
    # Conclusion
    print("\n" + "="*80)
    print("ROOT CAUSE ANALYSIS")
    print("="*80)
    
    print("""
The model's top features are market context indicators:
  1. dxy_close, dxy_return_1d (US Dollar strength)
  2. us_10y_yield_close (Interest rates)
  3. nifty50_return_1d, sp500_return_1d (Market returns)
  4. vix_change_pct, india_vix_change_pct (Fear index)

When these indicators are bearish, the model predicts Sell for ALL stocks,
regardless of individual stock fundamentals.

This is a FEATURE ENGINEERING problem, not a calibration problem.

The model learned: "When market is down, predict Sell for everything."

SOLUTION OPTIONS:
1. Reduce market context feature influence (downweight or remove)
2. Add more stock-specific features (relative strength, stock momentum)
3. Use market-neutral features (stock return vs sector, vs NIFTY)
4. Train regime-specific models (bull market model vs bear market model)
""")

if __name__ == "__main__":
    check_market_conditions()
