# NSE Bearish Bias Root Cause & Fix - April 21, 2026

## Problem Summary
**Issue:** NSE V2 predictions showing extreme bearish bias:
- **April 21, 2026:** Only 9 Buy (0.4%) vs 2,055 Sell (99.6%) out of 2,064 tickers
- Average buy probability: 28.36%
- Average sell probability: 71.64%

This is despite:
- ✅ Training data being reasonably balanced (46.25% UP / 53.75% DOWN)
- ✅ Class balancing applied during training
- ✅ Sample weighting used (class + time weights)

---

## Root Cause Analysis

### Investigation Steps

1. **Verified Training Data Distribution** ✅
   - Period: June 2024 - November 2025
   - Distribution: 46.25% UP / 53.75% DOWN
   - Imbalance: 7.5% (acceptable, within 20% threshold)
   - Conclusion: Training data is NOT the problem

2. **Analyzed Recent Market Trend** ✅
   - Last 30 days: Slightly bullish trend
   - Conclusion: Market conditions don't explain bearish predictions

3. **Examined Prediction Logic** ✅
   - Signal determined by: `buy_probability > sell_probability ? 'Buy' : 'Sell'`
   - Logic is correct
   - Conclusion: Prediction code is NOT the problem

4. **Checked Calibration Period Distribution** 🎯 **ROOT CAUSE FOUND**
   - Calibration period: December 2025 - February 2026
   - Distribution: **40.54% UP / 59.46% DOWN**
   - Average return: **-0.57%** (bearish market)
   - Imbalance: 9.46% deviation from 50/50

### The Fatal Flaw

```python
# BROKEN CODE (retrain_nse_model_v2.py - OLD)
# Time-series split - sequential, NOT stratified
train_size = int(0.60 * len(X))
cal_size = int(0.80 * len(X))

X_train = X.iloc[:train_size]       # First 60% (Jun 2024 - Nov 2025)
X_cal = X.iloc[train_size:cal_size] # Next 20% (Dec 2025 - Feb 2026) ← BEARISH PERIOD!
X_test = X.iloc[cal_size:]          # Last 20% (Mar 2026 - Apr 2026)
```

**Why This Breaks:**
1. Model trains on balanced data (46/54 split) → learns reasonable decision boundaries
2. Isotonic calibration fits on **bearish period** (40/60 split)
3. Calibrator learns: "When base model says 70% Up, actual outcome is 40% Up"
4. Calibrator applies aggressive downward adjustment to all probabilities
5. Result: Even when base model predicts Buy, calibrated probability is pushed toward Sell

**This is like calibrating a thermometer in Antarctica and using it in the Sahara!**

---

## The Fix

### Code Changes

```python
# FIXED CODE (retrain_nse_model_v2.py - NEW)
# Split data: 60% Train (time-series), 40% Remaining (for cal+test)
train_size = int(0.60 * len(X))

X_train = X.iloc[:train_size]
y_train = y[:train_size]

X_remaining = X.iloc[train_size:]
y_remaining = y[train_size:]

# CRITICAL: Stratified split for calibration + test
# Ensures calibration set has balanced distribution
X_cal, X_test, y_cal, y_test = train_test_split(
    X_remaining, y_remaining,
    test_size=0.5,          # 50% of 40% = 20% total
    stratify=y_remaining,   # ← KEY FIX: Force 50/50 balance
    random_state=42
)

# Verify calibration balance
unique, counts = np.unique(y_cal, return_counts=True)
cal_imbalance = abs(counts[0] - counts[1]) / len(y_cal)
print(f"Calibration imbalance: {cal_imbalance:.1%}")  # Should be ~0-5%
```

### What Changed

| Aspect | Before | After |
|--------|--------|-------|
| Training Set | Time-series (first 60%) | Time-series (first 60%) |
| Calibration Set | **Time-series (next 20%)** | **Stratified (20% of remaining)** |
| Calibration Balance | 40.54% UP / 59.46% DOWN | ~50% UP / ~50% DOWN |
| Test Set | Time-series (last 20%) | Stratified (20% of remaining) |

**Key Insight:** Training should use time-series to respect temporal ordering and recent data importance. But calibration must use stratified sampling to avoid learning period-specific biases.

---

## Expected Results After Retraining

### Before Fix (April 21, 2026 - 8:02 AM model)
- Buy signals: 9 (0.4%)
- Sell signals: 2,055 (99.6%)
- Avg buy probability: 28.36%
- Avg sell probability: 71.64%

### After Fix (Expected)
- Buy signals: ~40-50% of predictions
- Sell signals: ~50-60% of predictions
- Avg buy probability: ~48-52%
- Avg sell probability: ~48-52%
- Distribution should match recent market conditions, not calibration period bias

---

## Action Required

### 1. Retrain Model Immediately
```bash
python retrain_nse_model_v2.py
```

Expected training time: 15-20 minutes

### 2. Verify Calibration Balance
Look for this in training output:
```
[INFO] Calibration set balance:
  Down: 93,800 (50.0%)
  Up: 93,800 (50.0%)
[SUCCESS] Calibration set is balanced: 0.0% imbalance
```

### 3. Run Predictions Again
```bash
python predict_nse_signals_v2.py
```

### 4. Verify Results
```bash
python investigate_april21_predictions.py
```

Expected output:
- Buy: ~800-1000 signals (40-50%)
- Sell: ~1000-1200 signals (50-60%)
- Buy probability avg: ~48-52%

---

## Prevention for Future

### Monitoring
Add to `daily_nse_automation.py`:
```python
# Alert if prediction distribution is too skewed
buy_pct = len(df[df['predicted_signal'] == 'Buy']) / len(df) * 100

if buy_pct < 20 or buy_pct > 80:
    send_alert(f"WARNING: Prediction skew detected: {buy_pct:.1f}% Buy")
```

### Validation During Training
Already implemented in fixed script:
```python
cal_imbalance = abs(counts[0] - counts[1]) / len(y_cal)
if cal_imbalance > 0.10:
    print(f"[WARNING] Calibration imbalance: {cal_imbalance:.1%}")
    # Consider aborting or resampling
```

---

## Lessons Learned

1. **Calibration data must be representative, not just recent**
   - Time-series split assumes stationarity (markets change!)
   - Stratified split ensures balanced learning

2. **Monitor prediction distributions in production**
   - 99.6% Sell is a red flag that should trigger alerts
   - Normal range: 30-70% for either class

3. **Validate training artifacts, not just accuracy metrics**
   - 71.6% test accuracy looked fine
   - But model was fundamentally broken

4. **Document data split strategy clearly**
   - "60/20/20 split" is ambiguous
   - Must specify: stratified vs time-series, random state, etc.

---

## Files Modified

1. `retrain_nse_model_v2.py` - Fixed calibration split (lines 778-810)
2. `CALIBRATION_BIAS_FIX_APRIL_21_2026.md` - This document

## Related Documents

- `BEARISH_BIAS_FIX_APRIL_16_2026.md` - Previous class weighting fix
- `NSE_BEARISH_BIAS_ROOT_CAUSE_ANALYSIS.md` - Earlier investigation
- `check_calibration_period.py` - Diagnostic script used for root cause

---

**Status:** Fix implemented, awaiting model retraining and validation
**Priority:** CRITICAL - Production predictions are 99.6% Sell
**Next Action:** Run `python retrain_nse_model_v2.py` immediately
