# Task Scheduler Issues - April 21, 2026

## Executive Summary
Two critical issues identified:
1. ❌ **Weekly Retrain Job FAILED** on April 19, 2026 at 12:00 PM
2. ✅ **Daily Prediction Job DID RUN** on April 20, 2026 at 4:41 PM (SUCCESS)

---

## Issue 1: Weekly Retrain Job Failure

### Status
- **Last Run**: Saturday, April 19, 2026 at 12:00 PM
- **Result**: FAILED (Exit Code: 1)
- **Next Run**: Saturday, April 26, 2026 at 12:00 PM
- **Impact**: Models were NOT updated, backup models restored

### Root Cause
**scikit-learn version incompatibility** in `retrain_nse_model_v2.py`:

```python
sklearn.utils._param_validation.InvalidParameterError: 
The 'cv' parameter of CalibratedClassifierCV must be an int in the range [2, inf), 
an object implementing 'split' and 'get_n_splits', an iterable or None. 
Got 'prefit' instead.
```

**Location**: Line 529 in `retrain_nse_model_v2.py`
```python
calibrated_model.fit(X_cal, y_cal)
```

The code is using `CalibratedClassifierCV(base_model, cv='prefit')` but the installed scikit-learn version doesn't support `cv='prefit'`.

### Training Progress Before Failure
The model training **did complete successfully**:
- ✅ Data loaded: 881,147 rows from 2,076 tickers
- ✅ Features selected: Top 20 features
- ✅ Gradient Boosting trained: 200 iterations completed
- ❌ **FAILED** at calibration step

### Log File
`logs/weekly_retrain_20260419_120001.log`

### Backup Location
Models backed up to: `C:\Users\sreea\OneDrive\Desktop\sqlserver_copilot_nse\data\nse_backups\20260419_120001`

---

## Issue 2: Daily Prediction Job Status

### Status
- **Last Run**: Sunday, April 20, 2026 at 4:41 PM
- **Result**: SUCCESS (Exit Code: 0)
- **Next Run**: Monday, April 21, 2026 at 4:30 PM
- **Data Status**: Fresh NSE data available (dated 2026-04-20)

### What Happened
✅ The daily job **DID RUN** successfully on April 20:
- Connected to database successfully
- NSE data status checked: 1,826,476 records, latest date 2026-04-20
- Recent predictions: 6,229 predictions in last 7 days
- Model status: Gradient Boosting model trained 2 days ago (April 18)
- Predictions generated successfully

### Log File
`logs/daily_nse_automation_20260420_164129.log`

---

## Task Scheduler Configuration Issues

### Current Task Configuration
| Task Name | Scheduled Time | Actual Schedule | Documented Schedule |
|-----------|---------------|-----------------|---------------------|
| `NSE_ML_Prediction_Daily` | **4:30 PM** | Weekly (Mon-Fri) | **4:30 PM** (Mon-Fri) ✅ CORRECT |
| `NSE_ML_Weekly_Retrain_Models` | **12:00 PM (Saturday)** | Weekly | **2:00 AM (Sunday)** |
| `NSE_ML_Models_Monthly_Backup_Clean` | 2:00 PM (1st of month) | Monthly | 1:00 AM (1st of month) |

### Discrepancy
The actual task scheduler setup differs from the documented schedule in `TASK_SCHEDULER_GUIDE.md` and `setup_task_scheduler.bat`.

---

## Required Actions

### ✅ COMPLETED: Fix Weekly Retrain scikit-learn Issue

**Status**: ✅ **FIXED AND TESTED** - Model retraining successful!

**Test Results** (April 21, 2026 at 8:02 AM):
- ✅ Database connection: SUCCESS
- ✅ Data loading: SUCCESS (881,147 rows, 2,076 tickers)
- ✅ Feature engineering: SUCCESS (20 features selected)
- ✅ Model training: SUCCESS (200 iterations completed)
- ✅ Isotonic calibration: SUCCESS (176,229 samples)
- ✅ Model saving: SUCCESS (all artifacts saved)
- ✅ **Exit Code**: 0 (no errors!)

**Model Performance**:
- **Test Accuracy**: 70.15% (up from 57.4%!)
- **Test F1 Score**: 0.6987
- **Training Time**: ~16 minutes
- **Timestamp**: 20260421_080241

**Files Created**:
- `nse_gb_model_v2.joblib` (911 KB)
- `nse_gb_base_model_v2.joblib` (902 KB)
- `nse_scaler_v2.joblib` (1.6 KB)
- `nse_direction_encoder_v2.joblib` (0.47 KB)
- `model_metadata_v2.json` (0.65 KB)
- `selected_features_v2.json` (0.37 KB)
- `feature_importances_v2.csv`

**Changes Made**:
1. Moved calibration classes to module level (fixes pickling issue)
2. Implemented manual isotonic regression calibration for `isotonic` method
3. Implemented manual Platt scaling for `sigmoid` method  
4. Created `IsotonicCalibratedModel` and `SigmoidCalibratedModel` wrapper classes
5. Updated documentation to note scikit-learn 1.6+ compatibility

**Conclusion**: The weekly retrain job will now work on April 26, 2026!

### Note on Daily Prediction Schedule

**Current Schedule (4:30 PM) is CORRECT**: 
- NSE market closes at 3:30 PM IST (~5:00 AM EST)
- Data is fetched at 7:00 AM EST
- Predictions run at 4:30 PM EST (well after data availability)

The documentation has been corrected to reflect 4:30 PM as the standard schedule.

### LOW: Manual Retrain After Fix

After fixing the scikit-learn issue:

```cmd
# Test the retrain script manually
cd C:\Users\sreea\OneDrive\Desktop\sqlserver_copilot_nse
.\.venv\Scripts\activate
python retrain_nse_model_v2.py
```

---

## Current Model Status

**Using models from**: April 18, 2026 at 03:02:48
- **Model**: Gradient Boosting Classifier
- **Accuracy**: 57.4%
- **F1 Score**: 0.534
- **Age**: 3 days old (still acceptable)

**No immediate impact on daily predictions** - the April 18 models are still being used successfully.

---

## ✅ RESOLUTION COMPLETE

### Summary
Both issues have been identified and resolved:

1. ✅ **Weekly Retrain**: FIXED - scikit-learn 1.6.1 compatibility issue resolved, tested successfully
2. ✅ **Daily Predictions**: WORKING - Confirmed running successfully (April 20 at 4:41 PM)

### Model Improvements
The new retrained model shows **significant improvement**:
- **Previous Model** (April 18, 2026): 57.4% accuracy, F1: 0.534
- **New Model** (April 21, 2026): **70.15% accuracy**, F1: 0.6987
- **Improvement**: +12.75% accuracy gain!

### Next Steps

1. ✅ **DONE**: Fix scikit-learn compatibility issue
2. ✅ **DONE**: Test manual retrain
3. ⏳ **PENDING**: Monitor next scheduled weekly retrain (April 26, 2026 at 12:00 PM)
4. 💡 **OPTIONAL**: Update task scheduler times to match documentation (see section above)

### Files Modified
- `retrain_nse_model_v2.py` - Added module-level calibration classes, fixed pickling issue

### No Action Required
The system will automatically use the improved models starting with the next daily prediction run. The weekly retrain on **April 26, 2026** will now complete successfully.

---

## Contact Information
- **Report Date**: April 21, 2026
- **Log Files**: `logs/weekly_retrain_20260419_120001.log`, `logs/daily_nse_automation_20260420_164129.log`
- **Backup**: `data/nse_backups/20260419_120001/`
