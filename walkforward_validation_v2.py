"""
Walk-Forward Validation for the NSE V2 model.

Standalone harness (NOT part of the weekly retrain). Answers one question:
"What accuracy would this pipeline have achieved trading forward in time?"

Why this exists (June 2026): the original 60/20/20 split was effectively a
TICKER split (data arrived ordered by ticker), so test metrics were inflated
by cross-sectional leakage -- train and test covered the same dates and
stocks co-move. This harness evaluates honestly:

  - Expanding-window folds: train on everything up to T, test on the next
    ~2 months of dates, never the same dates.
  - 5-day embargo between train/cal/test segments (labels look 5 days ahead,
    so adjacent segments would otherwise share label windows).
  - Feature selection, scaling and calibration are re-fit INSIDE each fold
    using only that fold's training data.

Compare the fold accuracies against the live success_rate_5d tracking in
ml_nse_predict_summary -- they should be in the same neighbourhood. A fold
hitting 90%+ would itself be a red flag for residual leakage.

Usage:  python walkforward_validation_v2.py
Output: console summary + data/nse_models/walkforward_results.json
"""

import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

import retrain_nse_model_v2 as rt

# ============================================================================
# Configuration
# ============================================================================

N_FOLDS = 4
TEST_WINDOW_DAYS = 42   # trading days per test fold (~2 months)
EMBARGO_DAYS = 5        # matches the 5-day label horizon
CAL_RATIO = 0.20        # last 20% of each fold's pre-test dates -> calibration

RESULTS_FILE = Path('data/nse_models/walkforward_results.json')


def evaluate_fold(fold_num, X, y_encoded, dates, unique_dates, test_start_idx, encoder):
    """Train and evaluate one expanding-window fold. Returns a metrics dict."""
    test_dates = unique_dates[test_start_idx:test_start_idx + TEST_WINDOW_DAYS]
    pre_dates = unique_dates[:test_start_idx - EMBARGO_DAYS]

    train_end = int((1 - CAL_RATIO) * len(pre_dates))
    train_dates = pre_dates[:train_end]
    cal_dates = pre_dates[train_end + EMBARGO_DAYS:]

    print("\n" + "=" * 80)
    print(f"FOLD {fold_num}/{N_FOLDS}")
    print("=" * 80)
    print(f"  Train: {pd.Timestamp(train_dates[0]).date()} to {pd.Timestamp(train_dates[-1]).date()} ({len(train_dates)} days)")
    print(f"  Cal:   {pd.Timestamp(cal_dates[0]).date()} to {pd.Timestamp(cal_dates[-1]).date()} ({len(cal_dates)} days)")
    print(f"  Test:  {pd.Timestamp(test_dates[0]).date()} to {pd.Timestamp(test_dates[-1]).date()} ({len(test_dates)} days)")

    train_mask = dates.isin(train_dates).to_numpy()
    cal_mask = dates.isin(cal_dates).to_numpy()
    test_mask = dates.isin(test_dates).to_numpy()

    y_train = y_encoded[train_mask]
    y_cal = y_encoded[cal_mask]
    y_test = y_encoded[test_mask]

    # Feature selection on this fold's training data only
    selected_features, _ = rt.select_features(X[train_mask], y_train)

    X_train = X.loc[train_mask, selected_features]
    X_cal = X.loc[cal_mask, selected_features]
    X_test = X.loc[test_mask, selected_features]

    # Balance the calibration block if needed (mirrors retrain main())
    unique_cls, counts = np.unique(y_cal, return_counts=True)
    cal_imbalance = abs(counts[0] - counts[1]) / len(y_cal) if len(counts) == 2 else 1.0
    if cal_imbalance > 0.15 and len(counts) == 2:
        rng = np.random.RandomState(42)
        idx_by_class = [np.where(y_cal == cls)[0] for cls in unique_cls]
        n_keep = min(len(idx) for idx in idx_by_class)
        keep = np.sort(np.concatenate([
            rng.choice(idx, n_keep, replace=False) for idx in idx_by_class
        ]))
        X_cal = X_cal.iloc[keep]
        y_cal = y_cal[keep]
        print(f"  [INFO] Calibration imbalance {cal_imbalance:.1%} -- downsampled to {len(y_cal):,} balanced samples")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_cal_s = scaler.transform(X_cal)
    X_test_s = scaler.transform(X_test)

    train_dates_arr = dates.to_numpy()[train_mask]
    calibrated_model, _, results = rt.train_model(
        X_train_s, y_train, X_cal_s, y_cal, X_test_s, y_test,
        train_dates=train_dates_arr
    )

    y_pred = results['Test']['y_pred']
    y_proba = results['Test']['y_proba']

    up_code = list(encoder.classes_).index('Up')
    pred_up_pct = float((y_pred == up_code).mean() * 100)
    actual_up_pct = float((y_test == up_code).mean() * 100)

    # Accuracy among the top-decile most confident test predictions
    confidence = y_proba.max(axis=1)
    decile_cut = np.quantile(confidence, 0.90)
    top_mask = confidence >= decile_cut
    top_decile_acc = float(accuracy_score(y_test[top_mask], y_pred[top_mask])) if top_mask.any() else None

    return {
        'fold': fold_num,
        'train_range': f"{pd.Timestamp(train_dates[0]).date()} to {pd.Timestamp(train_dates[-1]).date()}",
        'test_range': f"{pd.Timestamp(test_dates[0]).date()} to {pd.Timestamp(test_dates[-1]).date()}",
        'n_train': int(train_mask.sum()),
        'n_test': int(test_mask.sum()),
        'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
        'f1': round(float(f1_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
        'pred_up_pct': round(pred_up_pct, 1),
        'actual_up_pct': round(actual_up_pct, 1),
        'top_decile_accuracy': round(top_decile_acc, 4) if top_decile_acc is not None else None,
        'selected_features': selected_features,
    }


def main():
    print("\n" + "=" * 80)
    print("NSE V2 WALK-FORWARD VALIDATION")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Folds: {N_FOLDS} x {TEST_WINDOW_DAYS} trading days, embargo {EMBARGO_DAYS} days")

    # Silence per-tree GB output across folds
    rt.Config.GB_PARAMS['verbose'] = 0

    conn = rt.get_db_connection()
    try:
        df = rt.load_training_data(conn)
    finally:
        conn.close()

    df = df.sort_values(['trading_date', 'ticker']).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in rt.EXCLUDE_COLS]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df['direction_5d']
    dates = df['trading_date']

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    unique_dates = np.sort(dates.unique())
    n_dates = len(unique_dates)

    needed = N_FOLDS * TEST_WINDOW_DAYS + EMBARGO_DAYS + 120  # 120 = minimum training history
    if n_dates < needed:
        print(f"[ERROR] Only {n_dates} trading days available; need >= {needed} for {N_FOLDS} folds")
        sys.exit(1)

    # Test windows tile the END of the available history
    fold_results = []
    for k in range(N_FOLDS):
        test_start_idx = n_dates - (N_FOLDS - k) * TEST_WINDOW_DAYS
        fold_results.append(
            evaluate_fold(k + 1, X, y_encoded, dates, unique_dates, test_start_idx, encoder)
        )

    # ------------------------------------------------------------------ summary
    print("\n" + "=" * 80)
    print("WALK-FORWARD SUMMARY (honest forward-in-time estimates)")
    print("=" * 80)
    header = f"{'Fold':4s} {'Test Range':26s} {'Acc':>7s} {'F1':>7s} {'PredUp%':>8s} {'ActUp%':>7s} {'Top10%Acc':>10s}"
    print(header)
    print("-" * len(header))
    for r in fold_results:
        top = f"{r['top_decile_accuracy']:.4f}" if r['top_decile_accuracy'] is not None else "n/a"
        print(f"{r['fold']:<4d} {r['test_range']:26s} {r['accuracy']:7.4f} {r['f1']:7.4f} "
              f"{r['pred_up_pct']:8.1f} {r['actual_up_pct']:7.1f} {top:>10s}")

    accs = [r['accuracy'] for r in fold_results]
    print("-" * len(header))
    print(f"Mean accuracy: {np.mean(accs):.4f}  (std {np.std(accs):.4f})")
    print("\n[NOTE] ~50% is coin-flip; 53-58% is realistic for 5-day direction.")
    print("[NOTE] A fold at 90%+ would indicate residual leakage, not skill.")

    summary = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'n_folds': N_FOLDS,
        'test_window_days': TEST_WINDOW_DAYS,
        'embargo_days': EMBARGO_DAYS,
        'mean_accuracy': round(float(np.mean(accs)), 4),
        'std_accuracy': round(float(np.std(accs)), 4),
        'folds': fold_results,
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SUCCESS] Results written to {RESULTS_FILE}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
