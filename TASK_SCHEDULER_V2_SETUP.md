# NSE V2 Task Scheduler Setup Guide
**Updated: April 18, 2026**  
**Model: Balanced V2 with Natural Factor Weighting**

---

## Quick Setup (Automated)

Run the existing setup script (already updated for V2):

```cmd
setup_task_scheduler.bat
```

**✅ This creates:**
- Daily predictions (Mon-Fri 9:30 AM)
- Weekly retraining (Sundays 2:00 AM)
- Monthly backup (1st of month 1:00 AM)

---

## Manual Setup (Detailed Commands)

### 1. Daily Predictions (Monday-Friday, 9:30 AM)

```cmd
schtasks /create ^
  /tn "NSE 500 Daily Predictions V2" ^
  /tr "C:\Users\sreea\OneDrive\Desktop\sqlserver_copilot_nse\run_daily_predictions.bat" ^
  /sc weekly ^
  /d MON,TUE,WED,THU,FRI ^
  /st 09:30 ^
  /f ^
  /rl highest
```

**What it does:**
- Runs `daily_nse_automation.py` which calls `predict_nse_signals_v2.py`
- Generates predictions using balanced V2 model (70% accuracy, natural weighting)
- Saves to `ml_nse_trading_predictions` table
- Creates daily report in `daily_reports/`
- Logs to `logs/daily_predictions_*.log`

---

### 2. Weekly Model Retraining (Sundays, 2:00 AM)

```cmd
schtasks /create ^
  /tn "NSE 500 Weekly Retrain V2" ^
  /tr "C:\Users\sreea\OneDrive\Desktop\sqlserver_copilot_nse\run_weekly_retrain.bat" ^
  /sc weekly ^
  /d SUN ^
  /st 02:00 ^
  /f ^
  /rl highest
```

**What it does:**
- Runs `retrain_nse_model_v2.py` (balanced feature selection)
- Trains on data from June 2024 to latest (auto-updates)
- Implements balanced feature selection (no feature >10% importance)
- Backs up old models to `data/nse_backups/`
- Saves new models to `data/nse_models/`
- Takes ~15-20 minutes (881K samples)
- Logs to `logs/weekly_retrain_*.log`

---

### 3. Monthly Backup (1st of month, 1:00 AM)

```cmd
schtasks /create ^
  /tn "NSE 500 Monthly Backup V2" ^
  /tr "C:\Users\sreea\OneDrive\Desktop\sqlserver_copilot_nse\run_monthly_backup.bat" ^
  /sc monthly ^
  /d 1 ^
  /st 01:00 ^
  /f ^
  /rl highest
```

**What it does:**
- Archives models, predictions, and logs
- Saves to `backups/YYYY-MM/`
- Includes model files, feature lists, metadata

---

## Task Management Commands

### View All NSE Tasks
```cmd
schtasks /query /fo list /v | findstr /i "NSE 500"
```

### Run Task Immediately (Manual Test)
```cmd
:: Test daily predictions
schtasks /run /tn "NSE 500 Daily Predictions V2"

:: Test weekly retraining
schtasks /run /tn "NSE 500 Weekly Retrain V2"
```

### Check Task Status
```cmd
schtasks /query /tn "NSE 500 Daily Predictions V2"
schtasks /query /tn "NSE 500 Weekly Retrain V2"
```

### View Last Run Result
```cmd
schtasks /query /tn "NSE 500 Daily Predictions V2" /fo list /v | findstr /i "Last Result"
```

### Delete Task (If Needed)
```cmd
schtasks /delete /tn "NSE 500 Daily Predictions V2" /f
schtasks /delete /tn "NSE 500 Weekly Retrain V2" /f
```

### Modify Schedule Time
```cmd
:: Change daily predictions to 10:00 AM
schtasks /change /tn "NSE 500 Daily Predictions V2" /st 10:00

:: Change weekly retrain to Saturdays
schtasks /change /tn "NSE 500 Weekly Retrain V2" /d SAT
```

---

## File Locations

| Component | File Path |
|-----------|-----------|
| **V2 Training Script** | `retrain_nse_model_v2.py` |
| **V2 Prediction Script** | `predict_nse_signals_v2.py` |
| **Daily Automation** | `daily_nse_automation.py` (updated to use V2) |
| **Daily Batch** | `run_daily_predictions.bat` (uses V2 via automation) |
| **Weekly Batch** | `run_weekly_retrain.bat` (updated to use V2) |
| **V2 Model Files** | `data/nse_models/nse_gb_model_v2.joblib` |
| **Feature List** | `data/nse_models/selected_features_v2.json` |
| **Daily Logs** | `logs/daily_predictions_*.log` |
| **Weekly Logs** | `logs/weekly_retrain_*.log` |
| **Database Table** | `ml_nse_trading_predictions` |

---

## V2 Model Characteristics

### Balanced Feature Selection
- **Max feature importance: 10%** (no single feature dominates)
- **Before:** india_vix_close 15.3% → caused 100% Sell predictions
- **After:** vix_close 10.0% → realistic 75% Sell / 25% Buy

### Feature Distribution (20 features)
| Category | Importance | Features |
|----------|-----------|----------|
| Market Context | ~30% | VIX, Dollar, Yields, NIFTY50 return |
| Technical Indicators | ~25% | Bollinger Bands, Stochastic, MACD |
| Price Momentum | ~15% | 5-day, 10-day returns |
| Volume & Volatility | ~20% | Volume ratio, price ratios |
| Moving Averages | ~10% | SMA 20, SMA 200 |

### Model Performance
- **Test Accuracy:** 70.14%
- **Training Period:** June 2024 - April 2026 (881K samples)
- **Class Balance:** 55% Down / 45% Up (10.6% imbalance)
- **Prediction Confidence:** 55-56% average (appropriately uncertain)
- **High Confidence Signals:** 1.3% (27 out of 2,064 stocks)

---

## Validation Checks

### After Daily Predictions
```sql
-- Check latest predictions
SELECT 
    trading_date,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN predicted_signal = 'Buy' THEN 1 ELSE 0 END) as buy_count,
    SUM(CASE WHEN predicted_signal = 'Sell' THEN 1 ELSE 0 END) as sell_count,
    AVG(confidence_percentage) as avg_confidence
FROM ml_nse_trading_predictions
WHERE model_name = 'GradientBoosting_V2_Calibrated'
GROUP BY trading_date
ORDER BY trading_date DESC;
```

**Expected:** 
- 2,000-2,100 predictions per day
- 60-75% Sell, 25-40% Buy (varies with market conditions)
- 50-60% average confidence
- 1-5% high confidence (>60%)

### After Weekly Retraining
```cmd
:: Check model files exist
dir data\nse_models\*_v2.joblib

:: Check feature count
python -c "import json; f=json.load(open('data/nse_models/selected_features_v2.json')); print(f'Features: {len(f)}'); print('\\n'.join(f))"

:: Check feature balance
python -c "import pandas as pd; df=pd.read_csv('data/nse_models/feature_importances_v2.csv'); print('Top feature importance:', df.iloc[0]['importance']); print('✅ Balanced' if df.iloc[0]['importance'] <= 0.10 else '❌ Over-dominant')"
```

**Expected:**
- 20 features selected
- Top feature ≤ 10% importance
- Model size ~50-100 MB

---

## Troubleshooting

### Issue: Task doesn't run

**Check:**
```cmd
:: Verify task exists
schtasks /query /tn "NSE 500 Daily Predictions V2"

:: Check last run status
schtasks /query /tn "NSE 500 Daily Predictions V2" /fo list /v
```

**Fix:**
```cmd
:: Re-create task
schtasks /delete /tn "NSE 500 Daily Predictions V2" /f
schtasks /create /tn "NSE 500 Daily Predictions V2" /tr "C:\Users\sreea\OneDrive\Desktop\sqlserver_copilot_nse\run_daily_predictions.bat" /sc weekly /d MON,TUE,WED,THU,FRI /st 09:30 /f /rl highest
```

### Issue: Predictions are still 100% one direction

**Check:**
```cmd
:: Verify using V2 model
python -c "import joblib; m=joblib.load('data/nse_models/nse_gb_model_v2.joblib'); print('Model type:', type(m)); import json; f=json.load(open('data/nse_models/selected_features_v2.json')); print('Features:', len(f))"

:: Check feature balance
python -c "import pandas as pd; df=pd.read_csv('data/nse_models/feature_importances_v2.csv'); print(df.head(5))"
```

**Expected:** Top feature ≤ 10% importance

### Issue: Retraining takes too long

**Normal:** 15-20 minutes for 881K samples  
**If >30 min:** Check CPU usage, available RAM

**Quick retrain (skip if recently trained):**
```cmd
python predict_nse_signals_v2.py
```

### Issue: Database connection failed

**Check:**
```cmd
python -c "import pyodbc; conn=pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=192.168.86.28\\MSSQLSERVER01;DATABASE=stockdata_db;UID=remote_user;PWD=YourStrongPassword123!;TrustServerCertificate=yes'); print('✅ Connected')"
```

---

## Migration from V1 to V2

### What Changed

| Component | V1 (Old) | V2 (New - Balanced) |
|-----------|----------|---------------------|
| Training Script | `retrain_nse_model.py` | `retrain_nse_model_v2.py` |
| Prediction Script | `predict_nse_signals.py` | `predict_nse_signals_v2.py` |
| Model File | Various ensemble files | `nse_gb_model_v2.joblib` |
| Feature Selection | Top 20 by importance | Top 20, capped at 10% each |
| Training Period | 2024-06-01 to 2026-02-28 | 2024-06-01 to 2026-04-15 |
| Model Type | VotingClassifier ensemble | Single GradientBoosting + Calibration |
| Prediction Quality | 99% one direction (broken) | 75/25 split (realistic) |

### Files Updated

✅ `daily_nse_automation.py` - Lines 317, 399 (now call V2 scripts)  
✅ `run_weekly_retrain.bat` - Line 82 (now calls V2 retrain)  
✅ `setup_task_scheduler.bat` - No changes needed (batch files handle routing)

---

## Production Readiness Checklist

- [x] V2 model trained with balanced features
- [x] Feature importance capped at 10%
- [x] Predictions tested (75/25 split achieved)
- [x] Daily automation updated to use V2
- [x] Weekly retraining updated to use V2
- [x] Task Scheduler commands documented
- [x] Database tables configured correctly
- [x] Logging enabled for all tasks
- [ ] Task Scheduler tasks created/updated
- [ ] First manual test run completed
- [ ] Email alerts configured (optional)

---

## Next Steps

1. **Create/Update Task Scheduler Tasks:**
   ```cmd
   cd C:\Users\sreea\OneDrive\Desktop\sqlserver_copilot_nse
   setup_task_scheduler.bat
   ```

2. **Test Immediately:**
   ```cmd
   schtasks /run /tn "NSE 500 Daily Predictions V2"
   ```

3. **Verify Output:**
   - Check logs: `logs/daily_predictions_*.log`
   - Check database: `SELECT * FROM ml_nse_trading_predictions WHERE trading_date = CAST(GETDATE() AS DATE)`
   - Check signal distribution (should be 60-75% Sell, 25-40% Buy)

4. **Monitor for 1 Week:**
   - Daily predictions should complete in 2-3 minutes
   - Signals should vary with market conditions (not stuck at 99%)
   - Confidence levels should be 50-60% average

5. **Optional Enhancements:**
   - Add email notifications on task failure
   - Set up dashboard to visualize predictions
   - Create alerts for unusual prediction patterns

---

## Support

**Model Issues:** Check `FEATURE_WEIGHTING_ANALYSIS.md`  
**Data Issues:** Check `check_data_availability.py`  
**Prediction Issues:** Check `diagnose_single_prediction.py`  
**Database Issues:** Verify connection string in `.env`

**Logs Location:** `logs/` directory  
**Backups Location:** `backups/` directory  
**Model Files:** `data/nse_models/`
