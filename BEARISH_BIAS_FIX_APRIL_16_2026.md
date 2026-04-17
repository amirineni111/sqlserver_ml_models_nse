# NSE Bearish Bias Fix - April 16, 2026

## Problem Summary
**Issue:** NSE predictions showing extreme bearish bias:
- **NSE:** 23 Buy (1.1%) vs 2014 Sell (97.3%) out of 2071 tickers
- **NASDAQ (for comparison):** 1521 Buy (64.6%) vs 834 Sell (35.4%) out of 2355 tickers

Despite high test accuracy (79.4%), NSE model is producing unrealistic signal distribution while NSE market is performing well.

---

## Root Cause Analysis

### Primary Issue: Missing Class Balancing for Gradient Boosting
**File:** `retrain_nse_model.py` (lines 1565-1587, 1613)

**Problem:**
1. **Gradient Boosting** is the best-performing classifier (79.4% F1-score) and dominates the ensemble
2. sklearn's `GradientBoostingClassifier` does **NOT support** the `class_weight='balanced'` parameter
3. Only **time-weighted** sample weights were used (giving more weight to recent data)
4. **NO class balancing** was applied → model learned training data's class distribution as ground truth

**Comparison:**
- ✅ **Random Forest:** Has `class_weight='balanced'`
- ❌ **Gradient Boosting:** Missing class balancing (best model, 79.4% F1)
- ✅ **Extra Trees:** Has `class_weight='balanced'`
- ✅ **Logistic Regression:** Has `class_weight='balanced'`

### Secondary Issue: Training Data Reflects Bearish NSE Market (2024-2026)
**Training window:** 730 days (last 2 years ending April 2026)

If NSE market was predominantly bearish during 2024-2026:
- Training data would be 70-80% "Down" samples
- Model correctly learns this distribution
- Produces bearish predictions even when current market conditions change

**NASDAQ avoids this by filtering training periods** (excludes Nov 2025 which had 64.5% Buy vs 35.5% Sell imbalance).

---

## Fix Applied

### 1. Added Class Balancing (IMMEDIATE FIX)
**File:** `retrain_nse_model.py`

**Changes:**
1. **Added import:**
   ```python
   from sklearn.utils.class_weight import compute_sample_weight
   ```

2. **Compute class-balanced weights:**
   ```python
   # Compute class-balanced weights (CRITICAL FIX: prevents model learning training data's class imbalance)
   class_weights = compute_sample_weight('balanced', y_train)
   print(f"  Class balancing: Up weight={class_weights[y_train==1][0]:.3f}, Down weight={class_weights[y_train==0][0]:.3f}")
   ```

3. **Combine class weights with time weights:**
   ```python
   # Combine class weights with time weights (multiplicative)
   combined_weights = class_weights * time_weights
   combined_weights = combined_weights / combined_weights.mean()  # Normalize to mean=1
   print(f"  Combined weights (class+time): range {combined_weights.min():.3f} to {combined_weights.max():.3f}")
   ```

4. **Updated model training:**
   ```python
   # Train with combined class+time weighted sample weights
   # This ensures models don't just learn the training data's class distribution
   model.fit(X_train_scaled, y_train, sample_weight=combined_weights)
   ```

### 2. Added Training Distribution Monitoring
**File:** `retrain_nse_model.py` (in `create_target_variable` method)

**Added warning system:**
```python
# WARNING CHECK: Detect heavily skewed training data (like NASDAQ does)
up_pct = direction_dist_5d.get('Up', 0) / len(df_target)
down_pct = direction_dist_5d.get('Down', 0) / len(df_target)
imbalance = abs(up_pct - down_pct)
if imbalance > 0.20:  # More than 20% difference (e.g., 60% vs 40%)
    print(f"")
    print(f"  ⚠️  WARNING: TRAINING DATA IS HEAVILY SKEWED!")
    print(f"  ⚠️  Up: {up_pct:.1%} vs Down: {down_pct:.1%} (imbalance: {imbalance:.1%})")
    print(f"  ⚠️  This may cause model to predict predominantly {direction_dist_5d.idxmax()}")
    print(f"  ⚠️  Class balancing (compute_sample_weight) will help, but consider:")
    print(f"  ⚠️    - Filtering training period to exclude extreme bear/bull markets")
    print(f"  ⚠️    - Extending training window for more balanced data")
    print(f"  ⚠️  (See NASDAQ retrain_model.py lines 300-310 for date filtering example)")
    print(f"")
```

This will alert you during retraining if the training data is heavily skewed, allowing you to take corrective action.

---

## Expected Impact After Retraining

### Before Fix (Current State)
```
Total Predictions: 2071
Buy Signals:       23 (1.1%)
Sell Signals:      2014 (97.3%)
Hold Signals:      34 (1.6%)
Average Confidence: 70%
```

### After Fix (Expected)
```
Total Predictions: 2071
Buy Signals:       800-1200 (40-60%)
Sell Signals:      800-1200 (40-60%)
Hold Signals:      50-100 (2-5%)
Average Confidence: 65-70%
Model Accuracy:    ~79% (maintained)
```

**Key Changes:**
- ✅ **Balanced signal distribution** (40-60% range for both Buy/Sell)
- ✅ **Maintains high accuracy** (~79% test accuracy)
- ✅ **Realistic predictions** aligned with current market conditions
- ✅ **Warning system** alerts you to training data issues during retraining

---

## How to Apply the Fix

### Step 1: Retrain the NSE Model
```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Run retrain script
python retrain_nse_model.py
```

**What to look for during retraining:**
1. **Class weight output:**
   ```
   Class balancing: Up weight=X.XXX, Down weight=X.XXX
   ```
   - If Up weight > Down weight → training data had more Down samples (good, balancing is working)
   - If Down weight > Up weight → training data had more Up samples

2. **Training distribution warning:**
   ```
   ⚠️  WARNING: TRAINING DATA IS HEAVILY SKEWED!
   ⚠️  Up: 35.0% vs Down: 65.0% (imbalance: 30.0%)
   ```
   - If you see this, the model is learning from biased data
   - Class balancing will help, but consider filtering the training period

3. **Combined weights output:**
   ```
   Combined weights (class+time): range 0.234 to 2.456
   ```
   - This shows the final weights applied to each training sample

### Step 2: Run Daily Predictions
After retraining completes, run daily predictions to verify the fix:

```powershell
python predict_nse_signals.py
```

**Expected output:**
- Check `daily_reports/nse_daily_report_YYYYMMDD.txt`
- Buy/Sell signals should be balanced (40-60% range)
- High confidence signals should be evenly distributed

### Step 3: Monitor Results Over Next Week
Track prediction distribution for the next 7 days:

```powershell
# View recent daily reports
Get-Content daily_reports\nse_daily_report_*.txt | Select-String "Buy Signals|Sell Signals"
```

---

## Additional Recommendations

### Medium-Term: Filter Training Period (Like NASDAQ)
If training data continues to show >20% imbalance, consider filtering the training period:

**Option 1: Exclude specific bearish periods**
```python
# In retrain_nse_model.py, modify load_historical_data query
hist_query = f"""
WHERE h.trading_date >= '2023-01-01'  -- Wider window
  AND h.trading_date <= '2026-04-16'
  AND h.trading_date NOT IN (
      -- Exclude periods with extreme directional bias
      SELECT DISTINCT trading_date FROM nse_500_hist_data
      WHERE trading_date BETWEEN '2024-12-01' AND '2025-02-28'
  )
"""
```

**Option 2: Extend training window**
```python
# Change from 730 days (2 years) to 1095 days (3 years)
self.days_back = 1095
```

### Long-Term: Continuous Monitoring
Add daily prediction distribution logging to monitor model drift:

**Create:** `monitor_prediction_distribution.py`
```python
"""Monitor daily prediction distribution to detect model drift."""
import pandas as pd
from datetime import datetime, timedelta

# Query last 30 days of predictions
query = """
SELECT 
    trading_date,
    COUNT(*) as total,
    SUM(CASE WHEN predicted_signal = 'Buy' THEN 1 ELSE 0 END) as buy_count,
    SUM(CASE WHEN predicted_signal = 'Sell' THEN 1 ELSE 0 END) as sell_count,
    AVG(confidence_percentage) as avg_confidence
FROM ml_nse_trading_predictions
WHERE trading_date >= DATEADD(day, -30, CAST(GETDATE() AS DATE))
GROUP BY trading_date
ORDER BY trading_date DESC
"""

# Alert if any day has >75% bias
# Send email if pattern persists for >3 days
```

---

## Technical Details

### Why Class Balancing Works
When training data has 70% "Down" and 30% "Up":
- **Without balancing:** Model learns "predicting Down is usually correct" → 70% accuracy
- **With balancing:** Model is penalized equally for Up and Down errors → learns actual patterns

**Formula:**
```python
class_weight = n_samples / (n_classes * np.bincount(y))
# Example: 100 samples, 70 Down, 30 Up
# Down weight = 100 / (2 * 70) = 0.714
# Up weight   = 100 / (2 * 30) = 1.667
# Ratio: Up samples get 2.3x more weight than Down samples
```

### Why Combine with Time Weights
Time weights give more importance to recent market conditions:
```python
time_weight = exp(1.2 * (position_in_training_set - 1))
# Oldest sample: exp(1.2 * (0 - 1)) = 0.301
# Newest sample: exp(1.2 * (1 - 1)) = 1.000
# Ratio: Recent data is 3.3x more important than old data
```

**Combined effect:**
- Recent Up samples: 1.667 (class) × 1.000 (time) = 1.667 weight
- Old Down samples: 0.714 (class) × 0.301 (time) = 0.215 weight
- **Result:** Recent minority class samples are 7.7x more important than old majority class samples

---

## Comparison with NASDAQ Fix

### NASDAQ Approach (from MODEL_REBALANCING_SUMMARY.md)
**Problem:** All 95 stocks predicted as Buy (100%) due to biased November 2025 data

**Fix:**
1. Excluded November 2025 training data (64.5% Buy vs 35.5% Sell)
2. Used `class_weight='balanced'` on Extra Trees (best model)
3. Result: 86 Buy (90.5%) vs 9 Sell (9.5%) — still slightly biased but functional

### NSE Approach (This Fix)
**Problem:** 2014 Sell (97.3%) vs 23 Buy (1.1%) due to bearish 2024-2026 training data + missing GB class balance

**Fix:**
1. **Added class balancing to ALL models** (including Gradient Boosting via sample_weight)
2. **Combined class+time weights** for optimal balance
3. **Added training distribution monitoring** to detect future issues
4. **Expected result:** 40-60% balanced signal distribution

**Key Difference:**
- NASDAQ only fixed one model (Extra Trees) and excluded data
- NSE fixes ALL models and adds comprehensive monitoring
- NSE approach is more robust and maintainable

---

## Success Criteria

After retraining and running predictions, verify:

✅ **Signal Distribution:**
- Buy signals: 35-65% (not <10% or >90%)
- Sell signals: 35-65% (not <10% or >90%)
- Hold signals: 0-10%

✅ **Model Accuracy:**
- Test accuracy maintained at ~79%
- High confidence signals have >80% accuracy
- Medium confidence signals have >70% accuracy

✅ **Realistic Predictions:**
- Signals align with technical indicators (RSI, MACD, etc.)
- Buy signals appear for oversold stocks (RSI <30)
- Sell signals appear for overbought stocks (RSI >70)

✅ **No Warning During Retraining:**
- If "TRAINING DATA IS HEAVILY SKEWED" warning appears:
  - Note the imbalance percentage
  - Consider filtering training period if >30% imbalance

---

## Rollback Plan (If Needed)

If the fix causes unexpected issues:

### 1. Restore Previous Model
```powershell
# Models are backed up automatically during retrain
# Check backups/2026-04/ for previous model files

# Restore from backup
Copy-Item "backups\2026-04\nse_models_backup_20260413*\*" "data\nse_models\" -Force
```

### 2. Revert Code Changes
```powershell
git checkout retrain_nse_model.py
```

### 3. Run Predictions with Old Model
```powershell
python predict_nse_signals.py
```

---

## Related Documentation
- [NSE_BEARISH_BIAS_ROOT_CAUSE_ANALYSIS.md](NSE_BEARISH_BIAS_ROOT_CAUSE_ANALYSIS.md) — Detailed investigation
- [BUG_FIX_APRIL_14_2026.md](BUG_FIX_APRIL_14_2026.md) — Previous bug fix (model name access)
- [CONFIDENCE_THRESHOLD_ANALYSIS.md](CONFIDENCE_THRESHOLD_ANALYSIS.md) — Threshold lowering from 70% to 60%
- [MODEL_REBALANCING_SUMMARY.md](MODEL_REBALANCING_SUMMARY.md) — NASDAQ rebalancing fix (reference)

---

## Questions?
If you have questions or observe unexpected behavior after applying the fix, check:

1. **Training output logs** — Look for class weight values and warning messages
2. **Daily prediction reports** — Check signal distribution in `daily_reports/`
3. **Test accuracy** — Ensure it remains ~79% (should not drop significantly)
4. **Database predictions** — Query `ml_nse_trading_predictions` for the last 7 days:
   ```sql
   SELECT 
       trading_date,
       COUNT(*) as total,
       SUM(CASE WHEN predicted_signal = 'Buy' THEN 1 ELSE 0 END) as buy_count,
       SUM(CASE WHEN predicted_signal = 'Sell' THEN 1 ELSE 0 END) as sell_count,
       AVG(confidence_percentage) as avg_confidence
   FROM ml_nse_trading_predictions
   WHERE trading_date >= DATEADD(day, -7, CAST(GETDATE() AS DATE))
   GROUP BY trading_date
   ORDER BY trading_date DESC
   ```

---

**Fix Applied:** April 16, 2026  
**Next Retrain:** Run manually now, then resume weekly schedule (Sundays 2 AM)  
**Status:** Ready for testing
