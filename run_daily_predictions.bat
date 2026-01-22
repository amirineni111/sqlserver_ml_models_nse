@echo off
:: NSE 500 Daily Predictions - Windows Task Scheduler
:: Run this script daily to generate NSE trading signals
:: Recommended schedule: Monday-Friday at 9:30 AM (after market open)

setlocal enabledelayedexpansion

:: Set UTF-8 code page for proper character encoding
chcp 65001 >nul 2>&1

:: Set Python UTF-8 mode (Python 3.7+)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

:: Set log file with timestamp
set TIMESTAMP=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set LOGFILE=%~dp0logs\daily_predictions_%TIMESTAMP%.log

:: Create logs directory if it doesn't exist
if not exist "%~dp0logs" mkdir "%~dp0logs"

:: Redirect all output to log file
call :MAIN > "%LOGFILE%" 2>&1
exit /b %errorlevel%

:MAIN
echo ============================================================
echo NSE 500 Daily Predictions - %date% %time%
echo ============================================================

:: Set working directory
cd /d "%~dp0"

:: Check if Python is available
python --version
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    exit /b 1
)

:: Run the simple prediction script (most reliable)
echo Running NSE 500 predictions...
python predict_nse_simple.py
set PRED_RESULT=%errorlevel%

:: Also run the full automation for reports
echo Running full automation for reports...
python daily_nse_automation.py
set AUTO_RESULT=%errorlevel%

:: Log results
echo.
echo ============================================================
echo DAILY PREDICTIONS SUMMARY - %date% %time%
echo ============================================================
if %PRED_RESULT%==0 (
    echo ✅ Predictions: SUCCESS
) else (
    echo ❌ Predictions: FAILED (Exit Code: %PRED_RESULT%)
)

if %AUTO_RESULT%==0 (
    echo ✅ Reports: SUCCESS
) else (
    echo ❌ Reports: FAILED (Exit Code: %AUTO_RESULT%)
)

echo Log saved to: %LOGFILE%
echo ============================================================

:: Return overall result (0 if both succeeded, 1 if either failed)
if %PRED_RESULT%==0 if %AUTO_RESULT%==0 exit /b 0
exit /b 1