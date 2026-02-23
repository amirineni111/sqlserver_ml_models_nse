@echo off
:: NSE 500 Weekly Model Retraining - Windows Task Scheduler
:: Run this script weekly to retrain the ML models with fresh NSE data
:: Recommended schedule: Sundays at 2:00 AM EST
::
:: IMPORTANT: This now calls retrain_nse_model.py which trains on NSE 500 data
:: (not the old retrain_model.py which trained on NASDAQ data)

setlocal enabledelayedexpansion

:: Set UTF-8 code page for proper character encoding
chcp 65001 >nul 2>&1

:: Set Python UTF-8 mode (Python 3.7+)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

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
echo NSE 500 Weekly Model Retraining - %date% %time% (EST)
echo ============================================================
echo NOTE: Training on NSE 500 data (not NASDAQ)

:: Set working directory
cd /d "%~dp0"

:: Activate virtual environment
if exist "%~dp0venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call "%~dp0venv\Scripts\activate.bat"
) else (
    echo WARNING: Virtual environment not found at %~dp0venv
    echo Running with system Python - packages may be missing!
)

:: Check if Python is available
python --version
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    exit /b 1
)

:: Backup existing NSE models
echo.
echo Creating backup of current NSE models...
if not exist "%~dp0data\nse_backups" mkdir "%~dp0data\nse_backups"
set BACKUP_DIR=%~dp0data\nse_backups\%TIMESTAMP%
mkdir "%BACKUP_DIR%"

:: Backup NSE-specific models (new location)
if exist "%~dp0data\nse_models" (
    copy "%~dp0data\nse_models\*.joblib" "%BACKUP_DIR%\" >nul 2>&1
    copy "%~dp0data\nse_models\*.pkl" "%BACKUP_DIR%\" >nul 2>&1
    echo NSE models backed up to: %BACKUP_DIR%
) else (
    echo No existing NSE models found - first time training
)

:: Also backup legacy models if they exist (one-time safety net)
if exist "%~dp0data\best_model_extra_trees.joblib" (
    copy "%~dp0data\*.joblib" "%BACKUP_DIR%\" >nul 2>&1
    echo Legacy models also backed up
)

:: Run NSE-specific model retraining
echo.
echo ============================================================
echo Starting NSE model retraining (retrain_nse_model.py)...
echo ============================================================
python retrain_nse_model.py --backup-old
set RETRAIN_RESULT=%errorlevel%

:: Test new models with a quick prediction check
echo.
if %RETRAIN_RESULT%==0 (
    echo Testing retrained NSE models...
    python -c "from predict_nse_signals import NSETradingSignalPredictor; p = NSETradingSignalPredictor(); print('[SUCCESS] NSE model loaded:', 'NSE-specific' if p.use_nse_models else 'Legacy'); print('[SUCCESS] Classifiers:', len(p.clf_models)); print('[SUCCESS] Regressors:', len(p.reg_models))"
    set TEST_RESULT=%errorlevel%
) else (
    set TEST_RESULT=1
)

:: Log results
echo.
echo ============================================================
echo WEEKLY RETRAINING SUMMARY - %date% %time% (EST)
echo ============================================================
if %RETRAIN_RESULT%==0 (
    echo [SUCCESS] NSE Model Retraining: SUCCESS
) else (
    echo [ERROR] NSE Model Retraining: FAILED (Exit Code: %RETRAIN_RESULT%)
    echo [RESTORE] Restoring backup models...
    if exist "%BACKUP_DIR%\nse_best_classifier.joblib" (
        if not exist "%~dp0data\nse_models" mkdir "%~dp0data\nse_models"
        copy "%BACKUP_DIR%\*.joblib" "%~dp0data\nse_models\" >nul 2>&1
        copy "%BACKUP_DIR%\*.pkl" "%~dp0data\nse_models\" >nul 2>&1
        echo [INFO] NSE model backup restored
    ) else (
        echo [WARN] No NSE backup models to restore
    )
)

if %TEST_RESULT%==0 (
    echo [SUCCESS] Model Testing: SUCCESS
) else (
    echo [ERROR] Model Testing: FAILED (Exit Code: %TEST_RESULT%)
)

echo [FILE] Backup location: %BACKUP_DIR%
echo [LOG] Log saved to: %LOGFILE%
echo ============================================================

:: Return overall result
if %RETRAIN_RESULT%==0 if %TEST_RESULT%==0 exit /b 0
exit /b 1
