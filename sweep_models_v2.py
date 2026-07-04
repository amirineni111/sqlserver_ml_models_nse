"""
Model/hyperparameter sweep for the NSE V2 pipeline (Phase 4, Jul 2026).

Screens candidate models on forward-in-time folds (same protocol as
walkforward_validation_v2.py, fewer folds for speed) and ranks them by the
metrics that matter for the served product (top-ranked picks):
top-decile precision first, AUC second, accuracy third.

Deliberately skips probability calibration: isotonic and Platt both collapse
the daily cross-section's variance on this weak base signal (see Jul 2026
incident notes), so production serves RAW model probabilities -- the sweep
scores exactly what production would serve.

Respects NSE_LABEL_MODE (absolute | market_relative).
Optionally sweeps an extended history window (--extended-history) since
nse_500_hist_data reaches back to Jan 2022 while training starts 2024-06-01.

Usage:
    python sweep_models_v2.py                       # standard window
    python sweep_models_v2.py --extended-history    # also test 2022-06-01 start
Output: console table + data/nse_models/sweep_results.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score

import retrain_nse_model_v2 as rt

N_FOLDS = 2             # screening protocol; confirm winners with --folds 4
TEST_WINDOW_DAYS = 42
EMBARGO_DAYS = 5
RESULTS_FILE = Path('data/nse_models/sweep_results.json')


def candidate_models():
    """Model configs to screen. random_state pinned everywhere."""
    from lightgbm import LGBMClassifier
    return {
        'gb_baseline_200x5': GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8,
            min_samples_split=20, min_samples_leaf=10, random_state=42),
        'gb_slow_300x4': GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            min_samples_split=20, min_samples_leaf=10, random_state=42),
        'gb_shallow_400x3': GradientBoostingClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
            min_samples_split=20, min_samples_leaf=10, random_state=42),
        'lgbm_500x31': LGBMClassifier(
            n_estimators=500, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=50,
            random_state=42, verbose=-1),
        'lgbm_300x63': LGBMClassifier(
            n_estimators=300, num_leaves=63, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=100,
            random_state=42, verbose=-1),
    }


def load_data():
    conn = rt.get_db_connection()
    try:
        df = rt.load_training_data(conn)
    finally:
        conn.close()
    df = df.sort_values(['trading_date', 'ticker']).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in rt.EXCLUDE_COLS]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['direction_5d'])
    up_code = list(encoder.classes_).index('Up')
    return X, y, df['trading_date'], up_code


def fold_masks(dates):
    """Yield (train_mask, test_mask) for the last N_FOLDS test windows."""
    unique_dates = np.sort(dates.unique())
    n = len(unique_dates)
    for k in range(N_FOLDS):
        test_start = n - (N_FOLDS - k) * TEST_WINDOW_DAYS
        test_dates = unique_dates[test_start:test_start + TEST_WINDOW_DAYS]
        train_dates = unique_dates[:test_start - EMBARGO_DAYS]
        yield dates.isin(train_dates).to_numpy(), dates.isin(test_dates).to_numpy()


def score_fold(model, X_train, y_train, X_test, y_test, up_code):
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    p_up = proba[:, up_code]
    y_up = (y_test == up_code).astype(int)

    acc = accuracy_score(y_test, model.predict(X_test))
    auc = roc_auc_score(y_up, p_up) if len(np.unique(y_up)) == 2 else None
    n_top = max(int(len(p_up) * 0.10), 1)
    top_idx = np.argsort(p_up)[-n_top:]
    tdp = float(y_up[top_idx].mean())
    # Frozen-output sanity: a model whose test probabilities have no spread is useless
    p_std = float(p_up.std())
    return acc, auc, tdp, p_std


def run_sweep(tag, X, y, dates, up_code, results):
    print(f"\n{'='*80}\nSWEEP: {tag} | label_mode={rt.Config.LABEL_MODE} | rows={len(X):,}\n{'='*80}")

    # Feature selection ONCE per fold (shared across configs -- selection is
    # model-agnostic RF importance, and refitting per config would quintuple cost)
    folds = []
    for train_mask, test_mask in fold_masks(dates):
        selected, _ = rt.select_features(X[train_mask], y[train_mask])
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X.loc[train_mask, selected])
        X_te = scaler.transform(X.loc[test_mask, selected])
        folds.append((X_tr, y[train_mask], X_te, y[test_mask]))

    for name, model_proto in candidate_models().items():
        import copy
        accs, aucs, tdps, stds = [], [], [], []
        for X_tr, y_tr, X_te, y_te in folds:
            model = copy.deepcopy(model_proto)
            acc, auc, tdp, p_std = score_fold(model, X_tr, y_tr, X_te, y_te, up_code)
            accs.append(acc)
            if auc is not None:
                aucs.append(auc)
            tdps.append(tdp)
            stds.append(p_std)
        entry = {
            'data_window': tag,
            'label_mode': rt.Config.LABEL_MODE,
            'model': name,
            'mean_accuracy': round(float(np.mean(accs)), 4),
            'mean_auc': round(float(np.mean(aucs)), 4) if aucs else None,
            'mean_top_decile_precision': round(float(np.mean(tdps)), 4),
            'mean_proba_std': round(float(np.mean(stds)), 4),
        }
        results.append(entry)
        print(f"  {name:<22} acc={entry['mean_accuracy']:.4f} auc={entry['mean_auc']} "
              f"top10%P={entry['mean_top_decile_precision']:.4f} pstd={entry['mean_proba_std']:.4f}")


def main():
    global N_FOLDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extended-history', action='store_true',
                        help='Also sweep with DATA_START_DATE=2022-06-01')
    parser.add_argument('--extended-only', action='store_true',
                        help='Sweep ONLY the extended 2022-06-01 window')
    parser.add_argument('--folds', type=int, default=N_FOLDS,
                        help='Number of forward-in-time test folds (4 = confirmation protocol)')
    parser.add_argument('--models', default=None,
                        help='Comma-separated subset of candidate model names to run')
    args = parser.parse_args()

    N_FOLDS = args.folds
    if args.models:
        wanted = {m.strip() for m in args.models.split(',')}
        global candidate_models
        all_models = candidate_models

        def candidate_models_filtered():
            return {k: v for k, v in all_models().items() if k in wanted}
        candidate_models = candidate_models_filtered

    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
          f"label_mode={rt.Config.LABEL_MODE} | folds={N_FOLDS}")
    rt.Config.GB_PARAMS['verbose'] = 0

    results = []

    if args.extended_only:
        rt.Config.DATA_START_DATE = '2022-06-01'
        X, y, dates, up_code = load_data()
        run_sweep('extended (2022-06-01)', X, y, dates, up_code, results)
    else:
        X, y, dates, up_code = load_data()
        run_sweep(f'standard ({rt.Config.DATA_START_DATE})', X, y, dates, up_code, results)

        if args.extended_history:
            rt.Config.DATA_START_DATE = '2022-06-01'
            X, y, dates, up_code = load_data()
            run_sweep('extended (2022-06-01)', X, y, dates, up_code, results)

    results.sort(key=lambda r: (r['mean_top_decile_precision'] or 0), reverse=True)
    print(f"\n{'='*80}\nSWEEP RANKING (by top-decile precision)\n{'='*80}")
    for r in results:
        print(f"  {r['model']:<22} {r['data_window']:<24} "
              f"top10%P={r['mean_top_decile_precision']:.4f} auc={r['mean_auc']} acc={r['mean_accuracy']:.4f}")

    # Confirmation runs (non-default fold count) get their own file so the
    # screening results are preserved
    results_file = RESULTS_FILE if N_FOLDS == 2 else \
        RESULTS_FILE.with_name(f'sweep_results_{N_FOLDS}fold.json')
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump({'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                   'n_folds': N_FOLDS, 'results': results}, f, indent=2)
    print(f"\n[SUCCESS] Results written to {results_file}")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
