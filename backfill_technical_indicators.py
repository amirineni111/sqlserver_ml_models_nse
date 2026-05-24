"""
Backfill ml_nse_technical_indicators for dates where V2 predictions exist
but technical indicators are missing (April 17 – May 22, 2026 gap).

The V2 pipeline omitted technical indicator writes during the Apr 18 refactor.
This script backfills the table using the same logic now added to
predict_nse_signals_v2.py, touching ONLY ml_nse_technical_indicators.

Usage:
    python backfill_technical_indicators.py                   # auto-detects gap
    python backfill_technical_indicators.py --start 2026-04-17 --end 2026-05-22
    python backfill_technical_indicators.py --date 2026-05-01 # single date
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import pyodbc
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

# ============================================================================
# Configuration (matches predict_nse_signals_v2.py)
# ============================================================================

SQL_SERVER = os.getenv('SQL_SERVER', '192.168.86.28\\MSSQLSERVER01')
SQL_DATABASE = os.getenv('SQL_DATABASE', 'stockdata_db')
SQL_USERNAME = os.getenv('SQL_USERNAME', 'remote_user')
SQL_PASSWORD = os.getenv('SQL_PASSWORD', 'YourStrongPassword123!')
SQL_DRIVER = os.getenv('SQL_DRIVER', 'ODBC Driver 17 for SQL Server')


def get_db_connection():
    conn_str = (
        f"DRIVER={{{SQL_DRIVER}}};"
        f"SERVER={SQL_SERVER};"
        f"DATABASE={SQL_DATABASE};"
        f"UID={SQL_USERNAME};"
        f"PWD={SQL_PASSWORD};"
        f"TrustServerCertificate=yes"
    )
    return pyodbc.connect(conn_str)


# ============================================================================
# Feature Engineering (copied from predict_nse_signals_v2.py)
# ============================================================================

def calculate_technical_indicators(df):
    """Calculate technical indicators — same logic as predict_nse_signals_v2.py."""
    results = []

    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('trading_date')

        ticker_df['return_1d'] = ticker_df['close_price'].pct_change(1)
        ticker_df['gap'] = (ticker_df['open_price'] - ticker_df['close_price'].shift(1)) / ticker_df['close_price'].shift(1)

        ticker_df['sma_5'] = ticker_df['close_price'].rolling(5).mean()
        ticker_df['sma_10'] = ticker_df['close_price'].rolling(10).mean()
        ticker_df['sma_20'] = ticker_df['close_price'].rolling(20).mean()
        ticker_df['sma_50'] = ticker_df['close_price'].rolling(50).mean()

        ticker_df['ema_5'] = ticker_df['close_price'].ewm(span=5).mean()
        ticker_df['ema_10'] = ticker_df['close_price'].ewm(span=10).mean()
        ticker_df['ema_12'] = ticker_df['close_price'].ewm(span=12).mean()
        ticker_df['ema_20'] = ticker_df['close_price'].ewm(span=20).mean()
        ticker_df['ema_26'] = ticker_df['close_price'].ewm(span=26).mean()
        ticker_df['ema_50'] = ticker_df['close_price'].ewm(span=50).mean()

        ticker_df['price_to_sma20'] = ticker_df['close_price'] / ticker_df['sma_20']
        ticker_df['price_to_sma50'] = ticker_df['close_price'] / ticker_df['sma_50']
        ticker_df['price_vs_ema20'] = ticker_df['close_price'] / ticker_df['ema_20']
        ticker_df['sma20_vs_sma50'] = ticker_df['sma_20'] / ticker_df['sma_50']
        ticker_df['ema20_vs_ema50'] = ticker_df['ema_20'] / ticker_df['ema_50']
        ticker_df['sma5_vs_sma20'] = ticker_df['sma_5'] / ticker_df['sma_20']

        # RSI
        delta = ticker_df['close_price'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        ticker_df['rsi'] = 100 - (100 / (1 + rs))
        ticker_df['rsi_oversold'] = (ticker_df['rsi'] < 30).astype(int)
        ticker_df['rsi_overbought'] = (ticker_df['rsi'] > 70).astype(int)
        ticker_df['rsi_momentum'] = ticker_df['rsi'].diff()

        # MACD
        ticker_df['macd'] = ticker_df['ema_12'] - ticker_df['ema_26']
        ticker_df['macd_signal'] = ticker_df['macd'].ewm(span=9).mean()
        ticker_df['macd_histogram'] = ticker_df['macd'] - ticker_df['macd_signal']

        # Bollinger Bands
        bb_std = ticker_df['close_price'].rolling(20).std()
        ticker_df['bb_middle'] = ticker_df['sma_20']
        ticker_df['bb_upper'] = ticker_df['bb_middle'] + (2 * bb_std)
        ticker_df['bb_lower'] = ticker_df['bb_middle'] - (2 * bb_std)

        # ATR
        high_low = ticker_df['high_price'] - ticker_df['low_price']
        high_close = np.abs(ticker_df['high_price'] - ticker_df['close_price'].shift(1))
        low_close = np.abs(ticker_df['low_price'] - ticker_df['close_price'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        ticker_df['atr'] = true_range.rolling(14).mean()
        ticker_df['atr_pct'] = ticker_df['atr'] / ticker_df['close_price']

        # Volume
        ticker_df['volume_sma_20'] = ticker_df['volume'].rolling(20).mean()
        vol_sma_nonzero = ticker_df['volume_sma_20'].replace(0, np.nan)
        ticker_df['volume_ratio'] = ticker_df['volume'] / vol_sma_nonzero

        # Price momentum (ratio-based, matching V1 definition)
        ticker_df['price_momentum_5'] = ticker_df['close_price'] / ticker_df['close_price'].shift(5)
        ticker_df['price_momentum_10'] = ticker_df['close_price'] / ticker_df['close_price'].shift(10)

        # Volatility
        ticker_df['daily_volatility'] = ticker_df['return_1d'].rolling(10).std()
        ticker_df['price_volatility_10'] = ticker_df['close_price'].pct_change().rolling(10).std()
        ticker_df['price_volatility_20'] = ticker_df['close_price'].pct_change().rolling(20).std()

        # Trend strength
        ticker_df['trend_strength_10'] = ticker_df['close_price'].rolling(10).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.std() if x.std() != 0 else 0, raw=False
        )

        # Volume-price trend
        ticker_df['volume_price_trend'] = (ticker_df['return_1d'] * ticker_df['volume']).rolling(10).mean()

        results.append(ticker_df)

    return pd.concat(results, ignore_index=True)


# ============================================================================
# Database helpers
# ============================================================================

def _safe_float(value, default=None):
    if value is None:
        return default
    try:
        f = float(value)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default


def get_dates_with_missing_indicators(conn):
    """Find dates in ml_nse_trading_predictions (V2) that have no indicators."""
    query = """
    SELECT DISTINCT p.trading_date
    FROM ml_nse_trading_predictions p
    WHERE p.model_name LIKE '%V2%'
      AND NOT EXISTS (
          SELECT 1 FROM ml_nse_technical_indicators t
          WHERE t.trading_date = p.trading_date
      )
    ORDER BY p.trading_date
    """
    df = pd.read_sql(query, conn)
    return sorted(df['trading_date'].tolist())


def load_data_for_date(conn, target_date):
    """Load 300 days of history ending on target_date."""
    query = f"""
    SELECT
        h.ticker,
        h.trading_date,
        CAST(h.open_price AS FLOAT)  AS open_price,
        CAST(h.high_price AS FLOAT)  AS high_price,
        CAST(h.low_price AS FLOAT)   AS low_price,
        CAST(h.close_price AS FLOAT) AS close_price,
        CAST(h.volume AS FLOAT)      AS volume
    FROM nse_500_hist_data h
    WHERE h.trading_date <= '{target_date}'
      AND h.trading_date >= DATEADD(day, -300, '{target_date}')
      AND h.close_price IS NOT NULL
      AND h.close_price <> '0'
      AND CAST(h.volume AS FLOAT) > 0
    ORDER BY h.ticker, h.trading_date
    """
    return pd.read_sql(query, conn)


def save_indicators_for_date(conn, df_date, target_date):
    """Write technical indicator rows for a single trading date."""
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM ml_nse_technical_indicators WHERE trading_date = ?",
        target_date
    )

    run_timestamp = datetime.now()

    insert_query = """
    INSERT INTO ml_nse_technical_indicators (
        run_timestamp, trading_date, ticker,
        rsi, rsi_oversold, rsi_overbought, rsi_momentum,
        sma_5, sma_10, sma_20, sma_50,
        ema_5, ema_10, ema_20, ema_50,
        macd, macd_signal, macd_histogram,
        price_vs_sma20, price_vs_sma50, price_vs_ema20,
        sma20_vs_sma50, ema20_vs_ema50, sma5_vs_sma20,
        volume_sma_20, volume_sma_ratio,
        price_momentum_5, price_momentum_10,
        daily_volatility, price_volatility_10, price_volatility_20,
        trend_strength_10, volume_price_trend,
        gap, bb_upper, bb_lower, bb_middle, bb_width,
        atr, atr_percentage
    ) VALUES (
        ?, ?, ?,
        ?, ?, ?, ?,
        ?, ?, ?, ?,
        ?, ?, ?, ?,
        ?, ?, ?,
        ?, ?, ?,
        ?, ?, ?,
        ?, ?,
        ?, ?,
        ?, ?, ?,
        ?, ?,
        ?, ?, ?, ?, ?,
        ?, ?
    )
    """

    rows = []
    sf = _safe_float

    for _, row in df_date.iterrows():
        bb_upper = sf(row.get('bb_upper'))
        bb_lower = sf(row.get('bb_lower'))
        bb_width = (bb_upper - bb_lower) if (bb_upper is not None and bb_lower is not None) else None

        rows.append((
            run_timestamp,
            row['trading_date'],
            row['ticker'],
            sf(row.get('rsi')),
            int(row.get('rsi_oversold', 0)),
            int(row.get('rsi_overbought', 0)),
            sf(row.get('rsi_momentum')),
            sf(row.get('sma_5')),
            sf(row.get('sma_10')),
            sf(row.get('sma_20')),
            sf(row.get('sma_50')),
            sf(row.get('ema_5')),
            sf(row.get('ema_10')),
            sf(row.get('ema_20')),
            sf(row.get('ema_50')),
            sf(row.get('macd')),
            sf(row.get('macd_signal')),
            sf(row.get('macd_histogram')),
            sf(row.get('price_to_sma20', row.get('price_vs_sma20'))),
            sf(row.get('price_to_sma50', row.get('price_vs_sma50'))),
            sf(row.get('price_vs_ema20')),
            sf(row.get('sma20_vs_sma50')),
            sf(row.get('ema20_vs_ema50')),
            sf(row.get('sma5_vs_sma20')),
            sf(row.get('volume_sma_20')),
            sf(row.get('volume_ratio')),
            sf(row.get('price_momentum_5')),
            sf(row.get('price_momentum_10')),
            sf(row.get('daily_volatility')),
            sf(row.get('price_volatility_10')),
            sf(row.get('price_volatility_20')),
            sf(row.get('trend_strength_10')),
            sf(row.get('volume_price_trend')),
            sf(row.get('gap')),
            bb_upper,
            bb_lower,
            sf(row.get('bb_middle')),
            bb_width,
            sf(row.get('atr')),
            sf(row.get('atr_pct')),
        ))

    cursor.executemany(insert_query, rows)
    conn.commit()
    cursor.close()
    return len(rows)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Backfill ml_nse_technical_indicators for missing V2 dates'
    )
    parser.add_argument('--start', help='Start date YYYY-MM-DD (inclusive)')
    parser.add_argument('--end', help='End date YYYY-MM-DD (inclusive)')
    parser.add_argument('--date', help='Single date YYYY-MM-DD')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show which dates would be processed without writing')
    args = parser.parse_args()

    print("="*80)
    print("NSE TECHNICAL INDICATORS BACKFILL")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    conn = get_db_connection()
    print("[SUCCESS] Connected to database")

    # Determine dates to process
    if args.date:
        dates_to_process = [args.date]
        print(f"[INFO] Single date mode: {args.date}")
    elif args.start or args.end:
        start = args.start or '2026-04-17'
        end = args.end or date.today().strftime('%Y-%m-%d')
        query = f"""
        SELECT DISTINCT trading_date FROM nse_500_hist_data
        WHERE trading_date >= '{start}' AND trading_date <= '{end}'
        ORDER BY trading_date
        """
        df_dates = pd.read_sql(query, conn)
        dates_to_process = [str(d)[:10] for d in df_dates['trading_date'].tolist()]
        print(f"[INFO] Date range mode: {start} → {end} ({len(dates_to_process)} trading days)")
    else:
        # Auto-detect: find prediction dates with no indicator data
        print("[INFO] Auto-detecting dates with missing indicators...")
        dates_to_process = [str(d)[:10] for d in get_dates_with_missing_indicators(conn)]
        print(f"[INFO] Found {len(dates_to_process)} dates with missing indicators")

    if not dates_to_process:
        print("[INFO] Nothing to backfill — all prediction dates already have indicator data")
        conn.close()
        return

    print(f"\nDates to process ({len(dates_to_process)}):")
    for d in dates_to_process:
        print(f"  {d}")

    if args.dry_run:
        print("\n[DRY RUN] No data written. Remove --dry-run to execute.")
        conn.close()
        return

    # Process each date
    total_rows = 0
    failed_dates = []

    for i, target_date in enumerate(dates_to_process, 1):
        print(f"\n[{i}/{len(dates_to_process)}] Processing {target_date}...")
        try:
            # Load history
            raw_df = load_data_for_date(conn, target_date)
            if raw_df.empty:
                print(f"  [WARNING] No price data found for {target_date}, skipping")
                continue

            # Compute indicators on full history
            df_indicators = calculate_technical_indicators(raw_df)

            # Filter to target date only
            df_date = df_indicators[
                pd.to_datetime(df_indicators['trading_date']).dt.strftime('%Y-%m-%d') == target_date
            ].copy()

            if df_date.empty:
                print(f"  [WARNING] No rows for {target_date} after indicator calculation, skipping")
                continue

            # Write to DB
            n = save_indicators_for_date(conn, df_date, target_date)
            total_rows += n
            print(f"  [SUCCESS] Saved {n:,} indicator rows")

        except Exception as e:
            print(f"  [ERROR] Failed for {target_date}: {e}")
            failed_dates.append(target_date)

    conn.close()

    print("\n" + "="*80)
    print("BACKFILL COMPLETE")
    print("="*80)
    print(f"Dates processed:  {len(dates_to_process) - len(failed_dates)}/{len(dates_to_process)}")
    print(f"Total rows saved: {total_rows:,}")
    if failed_dates:
        print(f"Failed dates ({len(failed_dates)}): {', '.join(failed_dates)}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
