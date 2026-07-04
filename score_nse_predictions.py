"""
In-repo outcome scoring for NSE V2 predictions.

Settles past rows of ml_nse_trading_predictions against nse_500_hist_data at
1/5/10 trading-day horizons and writes per-date aggregates (success_rate_1d/5d/10d,
model_accuracy, ranking metrics) to ml_nse_predict_summary.

This replaces any reliance on dbo.ai_prediction_history (an external process with
a different method/horizon). The model's native label is 5-day forward direction,
so model_accuracy = success_rate_5d. Ranking quality is tracked because the
product is the top-ranked picks: per-date cross-sectional AUC and top-decile
precision of buy_probability vs realized 5d direction (stored in notes).

Usage:
    python score_nse_predictions.py                 # settle everything settleable
    python score_nse_predictions.py --start 2026-04-17 --end 2026-07-03
"""

import argparse

import numpy as np
import pandas as pd

from predict_nse_signals_v2 import get_db_connection

V2_FILTER = "model_name LIKE '%V2%'"


def settle_prediction_rows(cursor, start, end):
    """Fill actual returns and direction-correct flags from price history.

    Set-based: LEAD over nse_500_hist_data gives the close N trading days after
    each prediction's trading_date. Horizons without data yet stay NULL (pending).
    """
    cursor.execute(f"""
        WITH px AS (
            SELECT ticker, trading_date,
                   CAST(close_price AS FLOAT) AS close_px,
                   LEAD(CAST(close_price AS FLOAT), 1)  OVER (PARTITION BY ticker ORDER BY trading_date) AS px_1d,
                   LEAD(CAST(close_price AS FLOAT), 5)  OVER (PARTITION BY ticker ORDER BY trading_date) AS px_5d,
                   LEAD(CAST(close_price AS FLOAT), 10) OVER (PARTITION BY ticker ORDER BY trading_date) AS px_10d
            FROM nse_500_hist_data
            WHERE trading_date >= ?
        )
        UPDATE p SET
            actual_return_1d  = CASE WHEN px.px_1d  IS NOT NULL AND px.close_px > 0
                                     THEN (px.px_1d  - px.close_px) / px.close_px END,
            actual_return_5d  = CASE WHEN px.px_5d  IS NOT NULL AND px.close_px > 0
                                     THEN (px.px_5d  - px.close_px) / px.close_px END,
            actual_return_10d = CASE WHEN px.px_10d IS NOT NULL AND px.close_px > 0
                                     THEN (px.px_10d - px.close_px) / px.close_px END,
            direction_correct_1d = CASE WHEN px.px_1d IS NULL THEN NULL
                WHEN (p.predicted_signal = 'Buy'  AND px.px_1d > px.close_px)
                  OR (p.predicted_signal = 'Sell' AND px.px_1d < px.close_px) THEN 1 ELSE 0 END,
            direction_correct_5d = CASE WHEN px.px_5d IS NULL THEN NULL
                WHEN (p.predicted_signal = 'Buy'  AND px.px_5d > px.close_px)
                  OR (p.predicted_signal = 'Sell' AND px.px_5d < px.close_px) THEN 1 ELSE 0 END,
            prediction_accuracy = CASE WHEN px.px_5d IS NULL THEN 'Pending'
                WHEN (p.predicted_signal = 'Buy'  AND px.px_5d > px.close_px)
                  OR (p.predicted_signal = 'Sell' AND px.px_5d < px.close_px)
                THEN 'Correct' ELSE 'Incorrect' END,
            updated_at = GETDATE()
        FROM ml_nse_trading_predictions p
        JOIN px ON px.ticker = p.ticker AND px.trading_date = p.trading_date
        WHERE p.{V2_FILTER} AND p.trading_date BETWEEN ? AND ?
    """, (start, start, end))
    return cursor.rowcount


def ranking_metrics(day_df):
    """Cross-sectional ranking quality for one date: AUC + top-decile precision.

    Both use buy_probability against realized 5d direction (return > 0),
    independent of the Buy/Sell threshold, so they measure the ranking itself.
    """
    settled = day_df.dropna(subset=['actual_return_5d'])
    if len(settled) < 50:
        return None, None
    y = (settled['actual_return_5d'] > 0).astype(int)
    score = settled['buy_probability']
    auc = None
    if y.nunique() == 2:
        # Rank-based (Mann-Whitney) AUC -- no sklearn dependency needed here
        ranks = score.rank(method='average')
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    n_top = max(int(len(settled) * 0.10), 1)
    top = settled.nlargest(n_top, 'buy_probability')
    top_decile_precision = (top['actual_return_5d'] > 0).mean()
    return auc, top_decile_precision


def update_summaries(conn, start, end):
    """Aggregate settled rows per date and write them to ml_nse_predict_summary."""
    df = pd.read_sql(f"""
        SELECT trading_date, predicted_signal, buy_probability,
               actual_return_1d, actual_return_5d, actual_return_10d,
               CAST(direction_correct_1d AS INT) AS direction_correct_1d,
               CAST(direction_correct_5d AS INT) AS direction_correct_5d
        FROM ml_nse_trading_predictions
        WHERE {V2_FILTER} AND trading_date BETWEEN ? AND ?
    """, conn, params=(start, end))
    if df.empty:
        print(f"[WARNING] No V2 predictions between {start} and {end}")
        return

    cursor = conn.cursor()
    print(f"\n{'date':<12} {'settled':>8} {'sr_1d':>7} {'sr_5d':>7} {'sr_10d':>7} {'auc_5d':>7} {'top10%':>7}")

    for trading_date, day in df.groupby('trading_date'):
        sr_1d = day['direction_correct_1d'].mean()
        sr_5d = day['direction_correct_5d'].mean()
        settled_10d = day.dropna(subset=['actual_return_10d'])
        sr_10d = None
        if len(settled_10d):
            correct_10d = np.where(
                settled_10d['predicted_signal'] == 'Buy',
                settled_10d['actual_return_10d'] > 0,
                settled_10d['actual_return_10d'] < 0
            )
            sr_10d = correct_10d.mean()
        auc, top_decile = ranking_metrics(day)

        n_settled = day['direction_correct_5d'].notna().sum()
        if n_settled == 0 and pd.isna(sr_1d):
            print(f"{str(trading_date):<12} {'pending':>8}")
            continue

        pct = lambda v: None if v is None or pd.isna(v) else round(float(v) * 100, 2)
        metrics_note = (f"metrics: auc_5d={'' if auc is None else round(float(auc), 4)}, "
                        f"top_decile_precision_5d={'' if top_decile is None or pd.isna(top_decile) else round(float(top_decile), 4)}, "
                        f"settled_5d={int(n_settled)}")
        # Idempotent notes update: keep original text, replace any previous metrics suffix
        cursor.execute("""
            UPDATE ml_nse_predict_summary
            SET success_rate_1d = ?,
                success_rate_5d = ?,
                success_rate_10d = ?,
                model_accuracy = ?,
                notes = CASE
                    WHEN notes IS NULL THEN ?
                    WHEN CHARINDEX(' | metrics:', CAST(notes AS VARCHAR(MAX))) > 0
                        THEN LEFT(CAST(notes AS VARCHAR(MAX)),
                                  CHARINDEX(' | metrics:', CAST(notes AS VARCHAR(MAX))) - 1) + ' | ' + ?
                    ELSE CAST(notes AS VARCHAR(MAX)) + ' | ' + ?
                END
            WHERE analysis_date = ?
        """, (pct(sr_1d), pct(sr_5d), pct(sr_10d), pct(sr_5d),
              metrics_note, metrics_note, metrics_note, trading_date))
        if cursor.rowcount == 0:
            print(f"  [WARNING] {trading_date}: no summary row to update")

        fmt = lambda v: '  --  ' if v is None or pd.isna(v) else f"{v * 100:5.1f}%"
        fmt_auc = '  --  ' if auc is None else f"{auc:6.3f}"
        print(f"{str(trading_date):<12} {n_settled:>8} {fmt(sr_1d):>7} {fmt(sr_5d):>7} "
              f"{fmt(sr_10d):>7} {fmt_auc:>7} {fmt(top_decile):>7}")

    conn.commit()
    cursor.close()


def rolling_success_rate_5d(conn, sessions=10):
    """Rolling success_rate_5d over the last N settled sessions (retrain-trigger input)."""
    df = pd.read_sql(f"""
        SELECT TOP {int(sessions)} analysis_date, success_rate_5d, total_predictions
        FROM ml_nse_predict_summary
        WHERE success_rate_5d IS NOT NULL
        ORDER BY analysis_date DESC
    """, conn)
    if df.empty:
        return None, 0
    return float(df['success_rate_5d'].mean()), len(df)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2026-04-17',
                        help='First prediction date to settle (default: first V2 date)')
    parser.add_argument('--end', default='2099-12-31',
                        help='Last prediction date to settle')
    args = parser.parse_args()

    conn = get_db_connection()
    cursor = conn.cursor()

    print("=" * 80)
    print("SCORING NSE V2 PREDICTIONS (in-repo settlement)")
    print("=" * 80)

    updated = settle_prediction_rows(cursor, args.start, args.end)
    conn.commit()
    print(f"[SUCCESS] Settled/refreshed {updated:,} prediction rows")

    update_summaries(conn, args.start, args.end)

    rolling, n = rolling_success_rate_5d(conn)
    if rolling is not None:
        print(f"\n[INFO] Rolling success_rate_5d over last {n} settled sessions: {rolling:.1f}%")

    cursor.close()
    conn.close()
    print("[SUCCESS] Scoring complete")


if __name__ == '__main__':
    main()
