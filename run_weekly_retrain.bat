@echo off
:: NSE 500 Weekly Model Retraining - Windows Task Scheduler
:: Run this script weekly to retrain the ML models with fresh data
:: Recommended schedule: Sundays at 2:00 AM

setlocal enabledelayedexpansion

:: Set log file with timestamp
set TIMESTAMP=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set LOGFILE=%~dp0logs\weekly_retrain_%TIMESTAMP%.log

:: Create logs directory if it doesn't exist
if not exist "%~dp0logs" mkdir "%~dp0logs"

:: Redirect all output to log file
call :MAIN > "%LOGFILE%" 2>&1
exit /b %errorlevel%

:MAIN
echo ============================================================
echo NSE 500 Weekly Model Retraining - %date% %time%
echo ============================================================

:: Set working directory
cd /d "%~dp0"

:: Check if Python is available
python --version
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    exit /b 1
)

:: Backup existing models
echo Creating backup of current models...
if not exist "%~dp0data\backups" mkdir "%~dp0data\backups"
set BACKUP_DIR=%~dp0data\backups\%TIMESTAMP%
mkdir "%BACKUP_DIR%"
copy "%~dp0data\*.joblib" "%BACKUP_DIR%\" >nul 2>&1
echo Models backed up to: %BACKUP_DIR%

:: Run model retraining
echo Starting model retraining...
python retrain_model.py
set RETRAIN_RESULT=%errorlevel%

:: Test new models with a quick prediction
if %RETRAIN_RESULT%==0 (
    echo Testing retrained models...
    python -c "from predict_nse_simple import SimpleNSEPredictor; p = SimpleNSEPredictor(); print('✅ Model test successful')"
    set TEST_RESULT=%errorlevel%
) else (
    set TEST_RESULT=1
)

:: Log results
echo.
echo ============================================================
echo WEEKLY RETRAINING SUMMARY - %date% %time%
echo ============================================================
if %RETRAIN_RESULT%==0 (
    echo ✅ Model Retraining: SUCCESS
) else (
    echo ❌ Model Retraining: FAILED (Exit Code: %RETRAIN_RESULT%)
    echo 🔄 Restoring backup models...
    copy "%BACKUP_DIR%\*.joblib" "%~dp0data\" >nul 2>&1
    echo 📋 Backup restored
)

if %TEST_RESULT%==0 (
    echo ✅ Model Testing: SUCCESS
) else (
    echo ❌ Model Testing: FAILED (Exit Code: %TEST_RESULT%)
)

echo 📁 Backup location: %BACKUP_DIR%
echo 📝 Log saved to: %LOGFILE%
echo ============================================================

:: Return overall result
if %RETRAIN_RESULT%==0 if %TEST_RESULT%==0 exit /b 0
exit /b 1