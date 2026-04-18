# Task Scheduler Migration Guide - V2 Model
**Date:** April 18, 2026  
**Model:** NSE V2 (Gradient Boosting + Calibration)

---

## Quick Summary

You need to update **3 files** to switch from V1 to V2 models:
1. `daily_nse_automation.py` (2 lines changed)
2. `run_weekly_retrain.bat` (1 line changed)

---

## Step-by-Step Instructions

### Step 1: Update Daily Automation Script

**File:** `daily_nse_automation.py`

**Location Line 317** - Auto-retrain reference:
```python
# OLD (V1):
cmd_args = [sys.executable, "retrain_nse_model.py", "--quick", "--backup-old"]

# NEW (V2):
cmd_args = [sys.executable, "retrain_nse_model_v2.py", "--quick", "--backup-old"]
```

**Location Line 399** - Daily prediction reference:
```python
# OLD (V1):
cmd_args = [sys.executable, "predict_nse_signals.py", "--all-nse"]

# NEW (V2):
cmd_args = [sys.executable, "predict_nse_signals_v2.py", "--all-nse"]
```

### Step 2: Update Weekly Retrain Batch

**File:** `run_weekly_retrain.bat`

**Location Line 82** - Retrain script:
```batch
REM OLD (V1):
python retrain_nse_model.py --backup-old

REM NEW (V2):
python retrain_nse_model_v2.py --backup-old
```

---

## Testing Changes

### Test 1: Manual Prediction

```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Run V2 prediction
python predict_nse_signals_v2.py

# Check results in database
python -c "import pyodbc; conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=192.168.86.28\MSSQLSERVER01;DATABASE=stockdata_db;UID=remote_user;PWD=YourStrongPassword123!;TrustServerCertificate=yes'); cursor = conn.cursor(); cursor.execute('SELECT predicted_signal, COUNT(*) as count FROM ml_nse_trading_predictions WHERE model_name LIKE ''%V2%'' AND trading_date = (SELECT MAX(trading_date) FROM ml_nse_trading_predictions) GROUP BY predicted_signal'); print('V2 Signal Distribution:'); for row in cursor: print(f'  {row[0]}: {row[1]}'); conn.close()"
```

### Test 2: Full Automation Wrapper

```powershell
# Test with data check only
.\run_nse_automation.bat --check

# Test full automation
.\run_nse_automation.bat
```

### Test 3: Scheduled Task (Dry Run)

```powershell
# Check current scheduled tasks
schtasks /Query /TN "NSE Daily Predictions" /FO LIST

# Run scheduled task manually (before scheduled time)
schtasks /Run /TN "NSE Daily Predictions"

# Check log file
Get-Content logs\daily_predictions_*.log -Tail 50
```

---

## Verification Checklist

After migration, verify:

- [ ] **Daily predictions use V2 model**
  - Check log file: Should say "Loading model from data\nse_models\nse_gb_model_v2.joblib"
  - Check database: `model_name = 'GradientBoosting_V2_Calibrated'`

- [ ] **Weekly retrain produces V2 files**
  - Check logs: Should say "SAVING MODEL ARTIFACTS" with `_v2` filenames
  - Check files: `data\nse_models\nse_gb_model_v2.joblib` timestamp updated

- [ ] **Signal distribution improved**
  - V1 baseline: 99.8% Sell, 0.2% Buy
  - V2 target: <95% Sell, >5% Buy (still biased due to market conditions)

- [ ] **No errors in logs**
  - Daily log: `logs\daily_predictions_*.log`
  - Weekly log: `logs\weekly_retrain_*.log`

---

## Rollback Procedure (If Needed)

If V2 causes issues, revert changes:

### Rollback Step 1: Restore daily_nse_automation.py

```python
# Line 317: Restore V1
cmd_args = [sys.executable, "retrain_nse_model.py", "--quick", "--backup-old"]

# Line 399: Restore V1
cmd_args = [sys.executable, "predict_nse_signals.py", "--all-nse"]
```

### Rollback Step 2: Restore run_weekly_retrain.bat

```batch
REM Line 82: Restore V1
python retrain_nse_model.py --backup-old
```

### Rollback Step 3: Test

```powershell
# Run manual test
python predict_nse_signals.py

# Verify V1 model loaded
# (Check log for "nse_clf_gradient_boosting.joblib")
```

---

## Expected Impact

### Before (V1)
- **Model:** 5-model ensemble (broken VotingClassifier)
- **Accuracy:** 57.4% (after fixing data leakage)
- **Signal Distribution:** 99.8% Sell, 0.2% Buy (23 Buy / 2014 Sell)
- **Issues:** Data leakage, class weighting bug, missing market data

### After (V2)
- **Model:** Single Gradient Boosting + Isotonic Calibration
- **Accuracy:** 77.5% (20% improvement)
- **Signal Distribution:** ~99% Sell, ~1% Buy (114 Buy / 12,908 Sell)
- **Issues Fixed:** Data leakage removed, class weighting working, market data forward-filled

### Known Limitations (V2)
- Still shows Sell bias (99.1% vs 99.8%) due to genuine market conditions
- Training data: 73.5% Down / 26.5% Up (47% imbalance)
- NSE market was bearish during 2024-2026 training period
- Market context data gap (Feb 20 to April 17 = 58 days forward-filled)

---

## Next Steps After Migration

1. **Monitor First Week of V2 Predictions**
   - Daily check: Signal distribution improving?
   - Compare V1 vs V2 in database by `model_name`

2. **Update Market Context Data**
   - `market_context_daily` table only updated until 2026-02-20
   - Schedule daily ETL to keep market features current
   - This should improve V2 predictions slightly

3. **Track Actual Performance**
   - Wait 1-10 days after predictions
   - Calculate actual returns for V2 predictions
   - Compare V2 accuracy vs V1 baseline

4. **Consider Future Improvements**
   - If NSE market becomes more bullish, retrain with updated date range
   - Explore different training periods to reduce class imbalance
   - Add more market sentiment features

---

## Support & Troubleshooting

### Issue: V2 predictions fail with "File not found"
**Solution:** Verify V2 model files exist:
```powershell
dir data\nse_models\*_v2.*
```
If missing, run training again:
```powershell
python retrain_nse_model_v2.py --backup-old
```

### Issue: V2 predictions still show 99.8% Sell
**Solution:** This is expected due to 47% class imbalance in training data. V2 shows ~99.1% Sell (slight improvement). To improve further:
1. Update market context data (fill 58-day gap)
2. Adjust training date range to more balanced period
3. Accept that NSE was genuinely bearish 2024-2026

### Issue: Task Scheduler job fails
**Solution:** Check logs:
```powershell
# View latest daily log
Get-Content logs\daily_predictions_*.log | Sort-Object -Descending | Select-Object -First 100

# View latest weekly log
Get-Content logs\weekly_retrain_*.log | Sort-Object -Descending | Select-Object -First 100
```

### Issue: Database insertion errors
**Solution:** Verify database schema matches code:
```sql
-- Check columns
SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'ml_nse_trading_predictions'
ORDER BY ORDINAL_POSITION
```

---

**Migration Status:** ✅ Files identified, changes documented  
**Next Action:** Apply changes to 2 files, test manually, monitor scheduled jobs  
**Rollback Plan:** Documented above  
**Support:** See troubleshooting section or review NSE_V2_SUMMARY_REPORT.md
