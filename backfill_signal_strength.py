"""
One-off backfill: recompute signal_strength bands with absolute confidence thresholds.

Context (Jul 2026): signal_strength was assigned by percentile rank (top 5% = High,
next 20% = Medium), which pinned ml_nse_predict_summary's confidence bucket counts
at 96/382/1431 for every session and split confidence ties across bands by row order.
predict_nse_signals_v2.py now uses absolute thresholds (High >= 70, Medium 60-70,
Low < 60). This script rewrites the affected historical rows the same way, from the
stored per-ticker confidence_percentage values (which were always written correctly).

Usage:
    python backfill_signal_strength.py --dry-run    # preview new counts, no writes
    python backfill_signal_strength.py              # apply (default: 2026-06-23..2026-07-03)
    python backfill_signal_strength.py --start 2026-06-23 --end 2026-07-03
"""

import argparse

from predict_nse_signals_v2 import Config, get_db_connection

HIGH_PCT = Config.CONFIDENCE_STRONG_THRESHOLD * 100   # 70.0
MEDIUM_PCT = Config.CONFIDENCE_HIGH_THRESHOLD * 100   # 60.0

V2_FILTER = "model_name LIKE '%V2%'"

BAND_CASE = f"""
    CASE
        WHEN confidence_percentage >= {HIGH_PCT} THEN 'High'
        WHEN confidence_percentage >= {MEDIUM_PCT} THEN 'Medium'
        ELSE 'Low'
    END
"""


def fetch_dates(cursor, start, end):
    cursor.execute(f"""
        SELECT DISTINCT trading_date
        FROM ml_nse_trading_predictions
        WHERE trading_date BETWEEN ? AND ? AND {V2_FILTER}
        ORDER BY trading_date
    """, (start, end))
    return [row[0] for row in cursor.fetchall()]


def fetch_new_counts(cursor, trading_date):
    """Compute what the band counts WILL be under the new thresholds."""
    cursor.execute(f"""
        SELECT
            band,
            SUM(CASE WHEN predicted_signal = 'Buy' THEN 1 ELSE 0 END) AS buys,
            SUM(CASE WHEN predicted_signal = 'Sell' THEN 1 ELSE 0 END) AS sells
        FROM (
            SELECT predicted_signal, {BAND_CASE} AS band
            FROM ml_nse_trading_predictions
            WHERE trading_date = ? AND {V2_FILTER}
        ) banded
        GROUP BY band
    """, trading_date)
    counts = {band: {'buys': 0, 'sells': 0} for band in ('High', 'Medium', 'Low')}
    for band, buys, sells in cursor.fetchall():
        counts[band] = {'buys': int(buys), 'sells': int(sells)}
    return counts


def fetch_old_summary(cursor, trading_date):
    cursor.execute("""
        SELECT high_confidence_count, medium_confidence_count, low_confidence_count
        FROM ml_nse_predict_summary
        WHERE analysis_date = ?
    """, trading_date)
    row = cursor.fetchone()
    return tuple(row) if row else None


def apply_updates(cursor, trading_date, counts):
    cursor.execute(f"""
        UPDATE ml_nse_trading_predictions
        SET signal_strength = {BAND_CASE},
            high_confidence   = CASE WHEN confidence_percentage >= {HIGH_PCT} THEN 1 ELSE 0 END,
            medium_confidence = CASE WHEN confidence_percentage >= {MEDIUM_PCT}
                                      AND confidence_percentage < {HIGH_PCT} THEN 1 ELSE 0 END,
            low_confidence    = CASE WHEN confidence_percentage < {MEDIUM_PCT} THEN 1 ELSE 0 END
        WHERE trading_date = ? AND {V2_FILTER}
    """, trading_date)
    updated_rows = cursor.rowcount

    high, med, low = counts['High'], counts['Medium'], counts['Low']
    cursor.execute("""
        UPDATE ml_nse_predict_summary
        SET high_confidence_count = ?, medium_confidence_count = ?, low_confidence_count = ?,
            high_conf_buys = ?, medium_conf_buys = ?, low_conf_buys = ?,
            high_conf_sells = ?, medium_conf_sells = ?, low_conf_sells = ?
        WHERE analysis_date = ?
    """, (
        high['buys'] + high['sells'], med['buys'] + med['sells'], low['buys'] + low['sells'],
        high['buys'], med['buys'], low['buys'],
        high['sells'], med['sells'], low['sells'],
        trading_date
    ))
    return updated_rows, cursor.rowcount


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2026-06-23')
    parser.add_argument('--end', default='2026-07-03')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the recomputed counts without writing')
    args = parser.parse_args()

    conn = get_db_connection()
    cursor = conn.cursor()

    dates = fetch_dates(cursor, args.start, args.end)
    if not dates:
        print(f"[WARNING] No V2 predictions found between {args.start} and {args.end}")
        return

    mode = "DRY RUN" if args.dry_run else "APPLYING"
    print(f"[{mode}] Recomputing signal_strength bands "
          f"(High >= {HIGH_PCT:.0f}, Medium >= {MEDIUM_PCT:.0f}) for {len(dates)} date(s)\n")
    print(f"{'date':<12} {'old H/M/L':>18} {'new H/M/L':>18} {'new buys H/M/L':>18} {'new sells H/M/L':>18}")

    for trading_date in dates:
        old = fetch_old_summary(cursor, trading_date)
        counts = fetch_new_counts(cursor, trading_date)
        high, med, low = counts['High'], counts['Medium'], counts['Low']
        old_str = f"{old[0]}/{old[1]}/{old[2]}" if old else "no summary row"
        new_str = (f"{high['buys'] + high['sells']}/{med['buys'] + med['sells']}"
                   f"/{low['buys'] + low['sells']}")
        buys_str = f"{high['buys']}/{med['buys']}/{low['buys']}"
        sells_str = f"{high['sells']}/{med['sells']}/{low['sells']}"
        print(f"{str(trading_date):<12} {old_str:>18} {new_str:>18} {buys_str:>18} {sells_str:>18}")

        if not args.dry_run:
            pred_rows, summary_rows = apply_updates(cursor, trading_date, counts)
            conn.commit()
            if summary_rows == 0:
                print(f"  [WARNING] {trading_date}: no ml_nse_predict_summary row to update")
            print(f"  [SUCCESS] {trading_date}: updated {pred_rows:,} prediction rows, "
                  f"{summary_rows} summary row(s)")

    cursor.close()
    conn.close()
    print(f"\n[{'SUCCESS' if not args.dry_run else 'INFO'}] "
          f"{'Backfill complete' if not args.dry_run else 'Dry run complete -- no changes written'}")


if __name__ == '__main__':
    main()
