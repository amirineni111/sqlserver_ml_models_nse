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


def confidence_reliability(conn, start, end):
    """Realized 5d accuracy per confidence bucket, split by predicted signal.

    Two defects this exists to catch (Sep 2026 review):

    1. Non-monotonic reliability -- measured over 176,985 settled V2 rows,
       realized accuracy peaks at 57.3% in the 65-70 band and then DECAYS to
       ~50% above 70. Confidence above the peak carries no information, so
       anything downstream that ranks on it is ranking on noise.
    2. Buy/Sell asymmetry -- confidence_percentage is max(buy_prob, sell_prob),
       which is the probability of the WINNING class, not of the PREDICTED one.
       Whenever the relative (top-30%) threshold is active, a Buy is emitted with
       buy_prob < 0.50, so its reported confidence is literally P(Sell). That caps
       Buy confidence below Sell confidence by construction.

    Reports both so the fix can be verified rather than assumed.
    """
    df = pd.read_sql(f"""
        SELECT predicted_signal, confidence_percentage, buy_probability, sell_probability,
               signal_strength,
               COALESCE(conviction_score, confidence_percentage) AS conviction_score,
               CAST(direction_correct_5d AS INT) AS direction_correct_5d
        FROM ml_nse_trading_predictions
        WHERE {V2_FILTER} AND trading_date BETWEEN ? AND ?
          AND direction_correct_5d IS NOT NULL
    """, conn, params=(start, end))
    if df.empty:
        print()
        print("[WARNING] No settled rows -- cannot assess confidence reliability")
        return

    print()
    print("=" * 80)
    print("CONFIDENCE RELIABILITY (settled 5d outcomes)")
    print("=" * 80)

    bins = [0, 55, 60, 65, 70, 75, 85, 100]
    df['bucket'] = pd.cut(df['confidence_percentage'], bins=bins, right=False)

    print()
    print(f"{'confidence':>14} {'n':>7} {'realized':>9} {'buys':>7} {'sells':>7}")
    table = []
    for bucket, grp in df.groupby('bucket', observed=True):
        acc = grp['direction_correct_5d'].mean()
        n_buy = (grp['predicted_signal'] == 'Buy').sum()
        table.append((bucket, len(grp), acc))
        print(f"{str(bucket):>14} {len(grp):>7,} {acc * 100:>8.1f}% "
              f"{n_buy:>7,} {len(grp) - n_buy:>7,}")

    # Monotonicity: does realized accuracy rise with reported confidence?
    populated = [(b, n, a) for b, n, a in table if n >= 100]
    if len(populated) >= 2:
        top_bucket, top_n, top_acc = populated[-1]
        rest_acc = df[df['bucket'] != top_bucket]['direction_correct_5d'].mean()
        if top_acc < rest_acc:
            print()
            print(f"[WARNING] CONFIDENCE IS INVERTED at the top end: highest bucket "
                  f"{top_bucket} realized {top_acc * 100:.1f}% (n={top_n:,}) vs "
                  f"{rest_acc * 100:.1f}% for everything below it.")
            print("[WARNING] Reported confidence is anti-predictive there -- do not rank on it.")

    # Buy/Sell asymmetry, and how often confidence is the losing class's probability
    print()
    print(f"{'signal':>8} {'n':>8} {'avg_conf':>9} {'realized':>9}")
    for signal, grp in df.groupby('predicted_signal'):
        print(f"{signal:>8} {len(grp):>8,} {grp['confidence_percentage'].mean():>8.1f}% "
              f"{grp['direction_correct_5d'].mean() * 100:>8.1f}%")

    # Conviction is what signal_strength bands and what downstream selection
    # ranks on, so its reliability is the one that decides product quality.
    print()
    print(f"{'conviction':>14} {'n':>7} {'realized':>9} {'buys':>7} {'sells':>7}")
    df['conv_bucket'] = pd.cut(df['conviction_score'], bins=bins, right=False)
    for bucket, grp in df.groupby('conv_bucket', observed=True):
        n_buy = (grp['predicted_signal'] == 'Buy').sum()
        print(f"{str(bucket):>14} {len(grp):>7,} "
              f"{grp['direction_correct_5d'].mean() * 100:>8.1f}% "
              f"{n_buy:>7,} {len(grp) - n_buy:>7,}")

    mislabelled = (
        ((df['predicted_signal'] == 'Buy') & (df['buy_probability'] < df['sell_probability'])) |
        ((df['predicted_signal'] == 'Sell') & (df['sell_probability'] < df['buy_probability']))
    ).sum()
    if mislabelled:
        print()
        print(f"[WARNING] {mislabelled:,} of {len(df):,} settled rows "
              f"({mislabelled / len(df) * 100:.1f}%) report the probability of the class "
              f"that was NOT predicted -- confidence_percentage uses max(buy, sell) while "
              f"the signal came from the relative top-30% rule. Expected for rows "
              f"written before Sep 2026; should be 0 for rows after.")

    # Realized rates per signal_strength band -- the numbers any downstream
    # baseline should be recomputed from.
    print()
    print(f"{'strength':>10} {'n':>8} {'realized':>9}   <- recompute downstream baselines from these")
    for strength, grp in df.groupby('signal_strength'):
        print(f"{strength:>10} {len(grp):>8,} {grp['direction_correct_5d'].mean() * 100:>8.1f}%")


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

    confidence_reliability(conn, args.start, args.end)

    rolling, n = rolling_success_rate_5d(conn)
    if rolling is not None:
        print(f"\n[INFO] Rolling success_rate_5d over last {n} settled sessions: {rolling:.1f}%")

    cursor.close()
    conn.close()
    print("[SUCCESS] Scoring complete")


if __name__ == '__main__':
    main()
