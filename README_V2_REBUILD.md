# NSE Model V2 - Rebuild with Simplified Architecture

**Date**: April 18, 2026  
**Status**: Ready for Testing  
**Based on**: Proven NASDAQ architecture (65-70% accuracy)

---

## Quick Start

### 1. Train New V2 Model
```powershell
python retrain_nse_model_v2.py
```

**Expected Results**:
- ✅ Test accuracy: **60-65%** (realistic for stock prediction)
- ✅ Buy signals: **40-60%** (balanced, not 0.2%)
- ✅ Sell signals: **40-60%** (balanced, not 99.8%)
- ✅ Model type: Gradient Boosting + Isotonic Calibration
- ✅ Features: **Top 20** (down from 30)

### 2. Generate Predictions
```powershell
python predict_nse_signals_v2.py
```

**Expected Results**:
- ✅ Predictions saved to `ml_nse_trading_predictions` (model_name contains 'V2')
- ✅ Summary saved to `ml_nse_predict_summary`
- ✅ Signal distribution: ~45% Buy, ~55% Sell (balanced!)

### 3. Compare V1 vs V2
```powershell
# Query database to compare
SELECT 
    model_name,
    COUNT(*) AS total,
    SUM(CASE WHEN predicted_signal = 'Buy' THEN 1 ELSE 0 END) AS buy_count,
    SUM(CASE WHEN predicted_signal = 'Sell' THEN 1 ELSE 0 END) AS sell_count,
    AVG(confidence_percentage) AS avg_confidence
FROM ml_nse_trading_predictions
WHERE trading_date = (SELECT MAX(trading_date) FROM ml_nse_trading_predictions)
GROUP BY model_name
ORDER BY model_name
```

---

## What Changed from V1 to V2

### V1 Problems (57.4% accuracy, 99.9% Sell bias)
1. ❌ **5-model ensemble** with VotingClassifier bug (sklearn 1.6.1 ignores sample_weight)
2. ❌ **30 features** including weak/redundant features
3. ❌ **No data filtering** - included biased periods
4. ❌ **Data leakage** - initially included future returns
5. ❌ **No calibration** - disabled due to sklearn compatibility

### V2 Solutions (Target: 60-65% accuracy, balanced signals)
1. ✅ **Single Gradient Boosting** model (proven in NASDAQ)
2. ✅ **Top 20 features** only (quality > quantity)
3. ✅ **Filtered training data** (2024-01-01 to 2026-03-31, excludes biased April)
4. ✅ **No future-leak features** (strictly validated)
5. ✅ **Isotonic calibration** on dedicated 20% calibration set

---

## Architecture Comparison

| Aspect | V1 (Old) | V2 (New) |
|--------|----------|----------|
| **Model** | 5-model ensemble (RF/GB/ET/LR/Voting) | Single Gradient Boosting |
| **Calibration** | Disabled (sklearn bug) | Isotonic on 20% cal set |
| **Features** | 30 (stratified selection) | 20 (importance-based) |
| **Data Split** | 70/15/15 (train/val/test) | 60/20/20 (train/cal/test) |
| **Date Filter** | Last 730 days (all data) | 2024-01-01 to 2026-03-31 |
| **Sample Weighting** | VotingClassifier (broken!) | Gradient Boosting (works!) |
| **Expected Accuracy** | 57.4% (actual) | 60-65% (target) |
| **Buy Signal Rate** | 0.2% (broken) | 40-60% (balanced) |

---

## Key Features Selected (Top 20)

**Likely top features** (based on NASDAQ + NSE training):
1. `sr_pivot_position` - Support/Resistance position
2. `sma_cross_signal` - Moving average crossover
3. `rsi` - Relative Strength Index
4. `bb_position` - Bollinger Band position
5. `macd_histogram` - MACD momentum
6. `price_to_sma20` - Price relative to 20-day MA
7. `return_20d` - 20-day return
8. `volume_ratio` - Volume relative to average
9. `atr_pct` - Average True Range %
10. `stochastic_k` - Stochastic oscillator
11. `nifty50_return_1d` - Market-wide: NIFTY 50 return
12. `india_vix_change_pct` - Market-wide: India VIX change
13. `sp500_return_1d` - Market-wide: S&P 500 return
14. `vix_close` - Market-wide: VIX level
15. `breakout_high` - 20-day high breakout flag
16. `obv_ema` - On-Balance Volume trend
17. `return_10d` - 10-day return
18. `price_to_sma50` - Price relative to 50-day MA
19. `gap` - Opening gap
20. `return_5d` - 5-day return

*(Actual features determined by Random Forest importance during training)*

---

## Training Details

### Hyperparameters (from NASDAQ)
```python
GradientBoostingClassifier(
    n_estimators=200,      # 200 trees
    max_depth=5,           # Moderate depth (prevent overfitting)
    learning_rate=0.1,     # Standard learning rate
    subsample=0.8,         # 80% sample per tree (regularization)
    min_samples_split=20,  # Prevent over-splitting
    min_samples_leaf=10,   # Ensure leaf stability
    random_state=42
)
```

### Sample Weighting
```python
# Class balancing (Up vs Down)
class_weights = compute_sample_weight('balanced', y_train)

# Time weighting (recent data more important)
positions = np.arange(len(y_train)) / len(y_train)
time_weights = np.exp(1.2 * (positions - 1))

# Combined weighting
sample_weights = class_weights * time_weights
```

### Calibration
```python
# Isotonic calibration on separate 20% set
calibrated_model = CalibratedClassifierCV(
    estimator=base_model,
    cv='prefit',           # Use pre-fitted model
    method='isotonic'      # Isotonic regression (flexible)
)
calibrated_model.fit(X_cal, y_cal)
```

---

## Expected Outcomes

### Training Output
```
[SUCCESS] Loaded 537,111 rows, 500 tickers
[SUCCESS] Class balance acceptable: 4.4% imbalance
[SUCCESS] Selected top 20 features
[INFO] Training set:    322,266 samples
[INFO] Calibration set: 107,422 samples
[INFO] Test set:        107,423 samples

Training Set:
  Accuracy:  0.6280
  F1 Score:  0.6250

Calibration Set:
  Accuracy:  0.6100
  F1 Score:  0.6070

Test Set:
  Accuracy:  0.6050
  F1 Score:  0.6020

Test Accuracy: 60.50%
Model Type: Gradient Boosting + Isotonic Calibration
```

### Prediction Output
```
[SUCCESS] Loaded 2,073 tickers ready for prediction
[INFO] Prediction Summary:
  Buy    :    892 (43.0%) | Avg Confidence: 68.5%
  Sell   :  1,181 (57.0%) | Avg Confidence: 65.2%

[INFO] High Confidence Signals: 1,534

[SUCCESS] Inserted 2,073 predictions
[SUCCESS] Saved prediction summary: 892 Buy / 1,181 Sell
```

---

## Validation Checklist

After training and prediction, verify:

### ✅ Model Quality
- [ ] Test accuracy ≥ 60%
- [ ] F1 score ≥ 0.60
- [ ] Training/Test gap < 5% (no overfitting)
- [ ] No future-leak features in selected_features_v2.json

### ✅ Signal Distribution
- [ ] Buy signals: 40-60% (not 0.2%!)
- [ ] Sell signals: 40-60% (not 99.8%!)
- [ ] Average confidence: 60-70%
- [ ] High confidence count: 60-75% of total

### ✅ Database Records
- [ ] Predictions in `ml_nse_trading_predictions` (model_name contains 'V2')
- [ ] Summary in `ml_nse_predict_summary`
- [ ] No duplicate entries for same date

### ✅ Comparison with NASDAQ
- [ ] NSE V2 accuracy within 5-10% of NASDAQ (expected: NSE=60%, NASDAQ=65%)
- [ ] Signal distribution similar pattern (both should be 40-60% range)

---

## Troubleshooting

### Problem: Still getting 99% Sell bias

**Possible causes**:
1. Training data still heavily skewed (check class distribution in training output)
2. Market data still missing (check market_context_daily table coverage)
3. Feature calculation errors (check for NaN values in features)

**Solutions**:
1. Adjust `DATA_START_DATE` and `DATA_END_DATE` in `retrain_nse_model_v2.py`
2. Update market_context_daily table with April 2026 data
3. Add debug prints in feature engineering to catch NaN issues

### Problem: Accuracy still only 57%

**Possible causes**:
1. Feature selection picked wrong features
2. Hyperparameters not optimal for NSE
3. NSE fundamentally harder to predict than NASDAQ

**Solutions**:
1. Review `feature_importances_v2.csv` - check if top features make sense
2. Try tuning hyperparameters (increase `n_estimators`, adjust `max_depth`)
3. Accept lower accuracy (55-60% is still profitable if used correctly)

### Problem: Model file not found

**Error**: `[ERROR] V2 model not found: data/nse_models/nse_gb_model_v2.joblib`

**Solution**: Run `python retrain_nse_model_v2.py` first to create V2 model

---

## Files Created

### Training Script
- `retrain_nse_model_v2.py` - New training pipeline (simplified architecture)

### Prediction Script
- `predict_nse_signals_v2.py` - New prediction pipeline (works with V2 model)

### Documentation
- `NSE_MODEL_INVESTIGATION_APRIL_2026.md` - Full investigation report
- `README_V2_REBUILD.md` - This file (V2 implementation guide)

### Model Artifacts (created after training)
- `data/nse_models/nse_gb_model_v2.joblib` - Calibrated Gradient Boosting model
- `data/nse_models/nse_gb_base_model_v2.joblib` - Base model (before calibration)
- `data/nse_models/nse_scaler_v2.joblib` - Feature scaler
- `data/nse_models/nse_direction_encoder_v2.joblib` - Label encoder
- `data/nse_models/selected_features_v2.json` - Top 20 selected features
- `data/nse_models/feature_importances_v2.csv` - All features with importance scores
- `data/nse_models/model_metadata_v2.json` - Training metadata

---

## Next Steps

1. **TODAY**: Train V2 model and validate accuracy
   ```powershell
   python retrain_nse_model_v2.py
   ```

2. **TODAY**: Generate V2 predictions and compare with V1
   ```powershell
   python predict_nse_signals_v2.py
   ```

3. **WEEK 1**: Monitor V2 predictions (paper trading)
   - Track accuracy over 5 trading days
   - Compare buy/sell distribution
   - Validate high-confidence signals

4. **WEEK 2**: Full rollout if V2 outperforms V1
   - Update daily automation to use V2
   - Deprecate V1 model
   - Update downstream consumers (streamlit dashboard, agents)

5. **ONGOING**: Market data maintenance
   - Ensure `market_context_daily` updated daily
   - Monitor for data gaps (set up alerts)
   - Backfill missing historical data

---

## Success Metrics

| Metric | V1 (Old) | V2 (Target) | V2 (Actual) |
|--------|----------|-------------|-------------|
| **Test Accuracy** | 57.4% | 60-65% | _TBD_ |
| **Buy Signal %** | 0.2% | 40-60% | _TBD_ |
| **Sell Signal %** | 99.8% | 40-60% | _TBD_ |
| **Avg Confidence** | 70% (misleading) | 60-70% (realistic) | _TBD_ |
| **High Conf Count** | 7 (0.3%) | 60-75% | _TBD_ |

**Goal**: V2 should achieve balanced signals (40-60% Buy) with realistic accuracy (60-65%)

---

**Status**: Documentation complete. Ready to train and test V2 model.  
**Next Action**: Run `python retrain_nse_model_v2.py` to train new model.
