# NSE 500 Windows Task Scheduler Guide

## Overview
This guide explains how to set up automated NSE 500 trading signal generation using Windows Task Scheduler. The system includes daily predictions, weekly model retraining, and monthly backups.

## Available BAT Files

### 1. `run_daily_predictions.bat`
**Purpose**: Daily NSE 500 trading signal generation  
**Recommended Schedule**: Monday-Friday at 9:30 AM (after market open)  
**What it does**:
- Runs `predict_nse_simple.py` to generate trading signals
- Runs `daily_nse_automation.py` to create reports
- Logs all activities with timestamps
- Populates database tables with fresh predictions

### 2. `run_weekly_retrain.bat`
**Purpose**: Weekly ML model retraining with fresh data  
**Recommended Schedule**: Sundays at 2:00 AM  
**What it does**:
- Backs up current model files
- Runs `retrain_model.py` to retrain models with latest data
- Tests new models for validity
- Restores backup if retraining fails
- Maintains model performance over time

### 3. `run_monthly_backup.bat`
**Purpose**: Monthly data backup and cleanup  
**Recommended Schedule**: 1st of every month at 1:00 AM  
**What it does**:
- Backs up models, results, and reports
- Cleans old log files (>90 days)
- Cleans old result files (>180 days)
- Cleans old backups (>1 year)
- Maintains system storage

### 4. `setup_task_scheduler.bat`
**Purpose**: One-time setup script for Task Scheduler  
**Requirements**: Administrator privileges  
**What it does**:
- Creates all scheduled tasks
- Configures proper timing and frequency
- Sets up test task for verification

### 5. `run_nse_automation.bat` (Interactive)
**Purpose**: Manual execution with options  
**Usage**: For testing and manual runs  
**Features**:
- Interactive mode with pause
- Command-line options (--check, --export)
- Suitable for development and debugging

## Quick Setup Instructions

### Step 1: One-Time Setup
1. **Right-click** on `setup_task_scheduler.bat`
2. Select **"Run as Administrator"**
3. Follow the prompts to create scheduled tasks

### Step 2: Verify Tasks
1. Open Task Scheduler: `Win+R` → `taskschd.msc`
2. Navigate to **Task Scheduler Library**
3. Look for NSE 500 tasks:
   - NSE 500 Daily Predictions
   - NSE 500 Weekly Retrain
   - NSE 500 Monthly Backup
   - NSE 500 Test Run

### Step 3: Test Execution
```cmd
# Test daily predictions immediately
schtasks /run /tn "NSE 500 Test Run"

# Or run manually
run_daily_predictions.bat
```

## Schedule Details

| Task | Frequency | Time | Days | Purpose |
|------|-----------|------|------|---------|
| **Daily Predictions** | Daily | 9:30 AM | Mon-Fri | Generate trading signals |
| **Weekly Retrain** | Weekly | 2:00 AM | Sunday | Update ML models |
| **Monthly Backup** | Monthly | 1:00 AM | 1st of month | Backup & cleanup |

## Directory Structure

```
📁 Project Root
├── 📄 run_daily_predictions.bat     # Daily automation
├── 📄 run_weekly_retrain.bat       # Weekly retraining
├── 📄 run_monthly_backup.bat       # Monthly maintenance
├── 📄 setup_task_scheduler.bat     # Setup script
├── 📁 logs/                        # Execution logs
│   ├── daily_predictions_*.log
│   ├── weekly_retrain_*.log
│   └── monthly_backup_*.log
├── 📁 data/                        # Model files
│   └── 📁 backups/                 # Model backups
├── 📁 results/                     # CSV outputs
├── 📁 daily_reports/              # Daily summaries
└── 📁 backups/                    # Monthly archives
```

## Log Files

All BAT files generate detailed log files with timestamps:
- **Format**: `task_type_YYYYMMDD_HHMMSS.log`
- **Location**: `logs/` directory
- **Content**: Complete execution details, errors, results

### Log Examples:
```
logs/daily_predictions_20251130_093015.log
logs/weekly_retrain_20251201_020005.log
logs/monthly_backup_20251201_010003.log
```

## Manual Execution

### Run Individual Tasks:
```cmd
# Daily predictions
run_daily_predictions.bat

# Weekly retraining  
run_weekly_retrain.bat

# Monthly backup
run_monthly_backup.bat

# Interactive automation (with pause)
run_nse_automation.bat
```

### Command Line Task Management:
```cmd
# List NSE tasks
schtasks /query /fo table | findstr "NSE 500"

# Run specific task
schtasks /run /tn "NSE 500 Daily Predictions"

# Disable task
schtasks /change /tn "NSE 500 Daily Predictions" /disable

# Enable task
schtasks /change /tn "NSE 500 Daily Predictions" /enable

# Delete task
schtasks /delete /tn "NSE 500 Daily Predictions" /f
```

## Troubleshooting

### Common Issues:

1. **Python not found**
   - Ensure Python is installed and in PATH
   - Test: `python --version`

2. **Permission errors**
   - Run setup script as Administrator
   - Ensure write permissions to project directory

3. **Database connection issues**
   - Check SQL Server connectivity
   - Verify credentials in `.env` file

4. **Task not running**
   - Check Task Scheduler for error messages
   - Verify file paths are correct
   - Check log files for detailed errors

### Verification Steps:
```cmd
# Check if tasks were created
schtasks /query /fo table | findstr "NSE"

# Test Python environment
python -c "from predict_nse_simple import SimpleNSEPredictor; print('OK')"

# Test database connection
python -c "from src.database.connection import SQLServerConnection; SQLServerConnection().test_connection()"
```

## Monitoring & Maintenance

### Daily Monitoring:
- Check log files in `logs/` directory
- Verify database tables are populated:
  - `ml_nse_trading_predictions` (daily)
  - `ml_nse_technical_indicators` (daily)
  - `ml_nse_predict_summary` (daily)

### Weekly Monitoring:
- Review retraining logs
- Check model backup creation
- Monitor prediction accuracy

### Monthly Monitoring:
- Review backup completion
- Check disk space usage
- Clean up if necessary

## Security Considerations

1. **Administrator Rights**: Setup requires admin privileges
2. **File Permissions**: Ensure proper read/write access
3. **Database Security**: Use secure connection strings
4. **Log Protection**: Logs may contain sensitive information

## Performance Optimization

1. **Timing**: 
   - Daily tasks after market open (9:30 AM)
   - Weekly tasks during low activity (Sunday 2:00 AM)
   - Monthly tasks during maintenance windows

2. **Resources**:
   - Monitor CPU/memory usage during execution
   - Adjust schedules if conflicts occur
   - Use SSD for better performance

## Backup Strategy

| Component | Frequency | Retention | Location |
|-----------|-----------|-----------|----------|
| **Model Files** | Weekly | 1 year | `data/backups/` |
| **Results** | Monthly | 6 months | `backups/YYYY-MM/` |
| **Reports** | Monthly | 1 year | `backups/YYYY-MM/` |
| **Logs** | Continuous | 90 days | `logs/` (auto-cleanup) |

---

## Summary

✅ **Ready for Production**:
- 5 BAT files for complete automation
- Proper scheduling for different frequencies  
- Comprehensive logging and error handling
- Automatic backup and cleanup
- Easy setup with one-click installation

🎯 **Next Steps**:
1. Run `setup_task_scheduler.bat` as Administrator
2. Test with `schtasks /run /tn "NSE 500 Test Run"`
3. Monitor logs for first few executions
4. Adjust schedules if needed

The system is now ready for 24/7 automated NSE 500 trading signal generation! 🚀