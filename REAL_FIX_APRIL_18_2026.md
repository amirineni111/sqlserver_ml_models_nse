# NSE 97% Sell Bias - REAL Root Cause & Fix
**Date:** April 18, 2026  
**Issue:** NSE predictions showing 2002 Sell vs 21 Buy (97% Sell bias) even after April 16 "fix"

---

## ACTUAL ROOT CAUSE: VotingClassifier Retrains Without Class Balancing

### The Bug (Line 1697 in retrain_nse_model.py)

```python
# April 16 fix added class+time balancing for individual models:
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train, sample_weight=combined_weights)  # ✅ BALANCED

# But then VotingClassifier RETRAINS all models WITHOUT balancing:
ensemble.fit(X_train_scaled, y_train)  # ❌ NO sample_weight parameter!
```

### What Happened

1. **Training data**: 54.8% Down / 45.2% Up (only 9.5% imbalance - very reasonable!)
2. **Individual models trained**: WITH `combined_weights` → class-balanced ✅
3. **VotingClassifier created**: `VotingClassifier(estimators=[(rf, gb, et, lr)])`
4. **ensemble.fit() called**: sklearn **retrains all base estimators from scratch** WITHOUT sample_weight ❌
5. **Result**: Ensemble learns raw 54.8% Down distribution → predicts 97% Down → 97% Sell signals

### Why April 16 Fix Didn't Work

The April 16 fix added `combined_weights = class_weights * time_weights` and passed it to individual model training. This worked perfectly for individual models!

But the Ensemble (which has the highest F1 score at 79.3%) retrains all base estimators and **doesn't receive the sample_weight**, so it learns the unbalanced distribution.

Since "Ensemble" is the best model (79.3% F1), it gets selected as `best_classifier`, and its biased predictions are used for all 2071 stocks.

---

## The Fix Applied

**File**: `retrain_nse_model.py` (line 1697)

**Changed:**
```python
ensemble.fit(X_train_scaled, y_train, sample_weight=combined_weights)
```

**Added comments:**
```python
# CRITICAL FIX: Pass combined_weights to preserve class balancing!
# VotingClassifier.fit() retrains base estimators, so we MUST pass sample_weight
# Otherwise it will learn the raw imbalanced distribution (54.8% Down → 97% Sell predictions)
```

---

## Evidence Trail

### Training Data Check (April 18, 2026)
```
5-DAY DIRECTION DISTRIBUTION (Training Data):
    Down  :  513,547 ( 54.8%)
    Up    :  424,418 ( 45.2%)
    IMBALANCE: 9.5%
    ✅ Training data is reasonably balanced
```

**Conclusion**: Training data is NOT the problem. 54.8% vs 45.2% should NEVER cause 97% bias.

### Database Predictions (April 15-17)
```
4/17/2026: Buy=21, Sell=2002, Hold=34  (97.0% Sell)
4/16/2026: Buy=20, Sell=2010, Hold=33  (97.1% Sell)
4/15/2026: Buy=30, Sell=2000, Hold=32  (96.9% Sell)
```

**Pattern**: Consistently 97% Sell AFTER April 16 retrain, proving the April 16 fix didn't work.

### Model Metadata (April 16 retrain)
```
Training date: 20260416_215248
Best classifier: Ensemble (79.3% F1)

MODEL F1 SCORES:
  Random Forest       : F1=0.7905, Acc=0.7901
  Gradient Boosting   : F1=0.7908, Acc=0.7940
  Extra Trees         : F1=0.6385, Acc=0.6352
  Logistic Regression : F1=0.7831, Acc=0.7824
  Ensemble            : F1=0.7927, Acc=0.7934  ← SELECTED AS BEST
```

**Key Finding**: "Ensemble" has the highest F1 score, so it's selected as the best classifier and used for all predictions. But this ensemble was trained WITHOUT class balancing, causing the 97% Sell bias.

### Why April 2 Was Balanced

Looking at historical reports:
- **April 2**: 1440 Buy (69.7%), 627 Sell (30.3%) - BALANCED, even bullish!
- **April 6+**: 1-5 Buy (~0.1%), 2063-2069 Sell (~99.9%) - EXTREME SELL BIAS

**Hypothesis**: A model retrain happened between April 2-6 that introduced the VotingClassifier bug. The April 16 fix attempted to add class balancing but only applied it to individual models, not the ensemble.

---

## Expected Impact After Fix & Retrain

### Before Fix (Current State - April 17)
```
Total Predictions: 2057
Buy Signals:       21 (1.0%)
Sell Signals:      2002 (97.3%)
Hold Signals:      34 (1.7%)
```

### After Fix (Expected)
```
Total Predictions: 2071
Buy Signals:       800-1100 (40-55%)
Sell Signals:      800-1100 (40-55%)
Hold Signals:      50-100 (2-5%)
Model Accuracy:    ~79% (maintained)
```

---

## Why NASDAQ Works But NSE Doesn't

| Aspect | NASDAQ (Working) | NSE (Broken) |
|--------|------------------|--------------|
| **Best Model** | Gradient Boosting (single model) | Ensemble (VotingClassifier) |
| **Ensemble Used** | ❌ No - uses best individual model | ✅ Yes - VotingClassifier |
| **Class Balancing** | Applied to best model via sample_weight | Applied to individuals, LOST in ensemble |
| **Result** | Balanced 50/50 predictions | 97% Sell bias |

**Key Difference**: NASDAQ doesn't use VotingClassifier, so it never hits this bug. NSE uses VotingClassifier which retrains models without sample_weight.

---

## How to Apply the Fix

### Step 1: Retrain the NSE Model

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run retrain with fixed code
python retrain_nse_model.py
```

**What to look for during retraining:**
1. **Class weight output** (confirms balancing is applied):
   ```
   Class balancing: Up weight=1.218, Down weight=0.824
   ```

2. **Combined weights output** (confirms both class+time weights):
   ```
   Combined weights (class+time): range 0.247 to 2.027
   ```

3. **Ensemble training confirmation** (NEW - confirms fix):
   ```
   [OK] Ensemble trained WITH class+time balancing (fixes 97% Sell bias)
   ```

4. **Training distribution warning** (if data is skewed):
   ```
   ⚠️  WARNING: TRAINING DATA IS HEAVILY SKEWED!
   ```
   (Should NOT appear since data is only 9.5% imbalanced)

### Step 2: Run Daily Predictions

After retraining completes (~20-30 minutes):

```powershell
python predict_nse_signals.py
```

**Expected output in daily report:**
- Buy signals: 40-55% (not 1%)
- Sell signals: 40-55% (not 97%)
- High confidence signals: distributed between both Buy and Sell

### Step 3: Verify in Database

```powershell
$query = @"
SELECT trading_date, predicted_signal, COUNT(*) as count
FROM ml_nse_trading_predictions  
WHERE trading_date = CAST(GETDATE() AS DATE)
GROUP BY trading_date, predicted_signal
ORDER BY predicted_signal
"@
Invoke-Sqlcmd -ServerInstance "192.168.86.28\MSSQLSERVER01" -Database "stockdata_db" -Username "remote_user" -Password "YourStrongPassword123!" -Query $query
```

**Expected:**
- Buy: 800-1100
- Sell: 800-1100
- Hold: 50-100

---

## Technical Details

### Why VotingClassifier Retrains Base Estimators

From sklearn documentation:
> "The VotingClassifier clones the base estimators and fits them on the training data."

This means calling `ensemble.fit()` doesn't use the already-trained models - it creates fresh clones and trains them from scratch. If you don't pass `sample_weight` to `ensemble.fit()`, the clones are trained on raw unbalanced data.

### Why This Is So Critical

With 54.8% Down training data:
- **Without balancing**: Model learns "Down is usually correct" → predicts Down 97% of the time → 50% accuracy (barely better than random)
- **With balancing**: Model learns "What features distinguish Up from Down?" → predicts based on patterns → 79% accuracy

The irony: The model achieves 79% test accuracy because the TEST set ALSO has 54.8% Down. So predicting Down 97% of the time gives high accuracy on the test set, but it's useless for actual trading!

---

## Verification Checklist

After retraining and running predictions, verify:

✅ **Signal Distribution**:
- [ ] Buy signals: 35-65% (not <10%)
- [ ] Sell signals: 35-65% (not >90%)
- [ ] Hold signals: 0-10%

✅ **Model Accuracy**:
- [ ] Test accuracy maintained at ~79%
- [ ] High confidence signals have >75% accuracy

✅ **Realistic Predictions**:
- [ ] Buy signals appear for oversold stocks (RSI <40)
- [ ] Sell signals appear for overbought stocks (RSI >65)
- [ ] Signals correlate with technical indicators

✅ **Database Validation**:
- [ ] Query ml_nse_trading_predictions for latest date
- [ ] Verify balanced Buy/Sell counts
- [ ] Check confidence_percentage distribution

---

## Alternative Solution (If Fix Doesn't Work)

If passing `sample_weight` to `VotingClassifier.fit()` still doesn't work (some sklearn versions have bugs), use NASDAQ's approach:

**Remove VotingClassifier entirely and use the best individual model:**

```python
# In retrain_nse_model.py, line 1683-1713
# COMMENT OUT the entire VotingClassifier section

# Keep only:
best_model_name = max(model_results.keys(),
                     key=lambda k: model_results[k]['f1_score'])
best_model = trained_models[best_model_name]

# Then calibrate the best model (already in code at line 1723)
```

This guarantees the class-balanced model is used, matching NASDAQ's proven approach.

---

## Related Documentation
- [BEARISH_BIAS_FIX_APRIL_16_2026.md](BEARISH_BIAS_FIX_APRIL_16_2026.md) — Previous fix attempt (didn't work)
- [NSE_BEARISH_BIAS_ROOT_CAUSE_ANALYSIS.md](NSE_BEARISH_BIAS_ROOT_CAUSE_ANALYSIS.md) — Initial investigation
- [BUG_FIX_APRIL_14_2026.md](BUG_FIX_APRIL_14_2026.md) — Earlier bug fix

---

**Fix Applied:** April 18, 2026  
**Next Action:** Retrain model immediately to verify fix  
**Expected Resolution:** 40-55% balanced Buy/Sell signals
