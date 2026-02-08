@echo off
:: NSE 500 Daily Predictions - Windows Task Scheduler
:: Run this script daily to generate NSE trading signals
:: Schedule: Monday-Friday at 7:30 AM EST
::
:: DATA PIPELINE (EST):
::   3:30 PM IST (~5:00 AM EST) - NSE market closes
::   7:00 AM EST - yfinance data fetch loads today's NSE data into DB
::   7:30 AM EST - THIS SCRIPT runs ML predictions on fresh data
::
:: This calls daily_nse_automation.py which handles:
::   1. Data status check (verifies today's NSE data is loaded)
::   2. Model status check (auto-retrain if needed)
::   3. NSE predictions using NSE-trained ensemble models
::   4. Model performance monitoring
::   5. Daily report generation

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
echo NSE 500 Daily Predictions - %date% %time% (EST)
echo ============================================================
echo Data pipeline: yfinance fetch at 7 AM EST, ML predictions at 7:30 AM EST

:: Set working directory
cd /d "%~dp0"

:: Check if Python is available
python --version
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    exit /b 1
)

:: Run the full daily automation (handles model check + predictions + reports)
echo.
echo Running NSE daily automation...
echo (includes: data check, model check, predictions, performance monitoring, reports)
echo.
python daily_nse_automation.py
set AUTO_RESULT=%errorlevel%

:: Log results
echo.
echo ============================================================
echo DAILY PREDICTIONS SUMMARY - %date% %time% (EST)
echo ============================================================
if %AUTO_RESULT%==0 (
    echo [SUCCESS] NSE Daily Automation: SUCCESS
    echo [INFO] Predictions saved to: ml_nse_trading_predictions
    echo [INFO] Technical indicators saved to: ml_nse_technical_indicators
    echo [INFO] Summary saved to: ml_nse_predict_summary
    echo [INFO] Report saved to: daily_reports\
) else (
    echo [ERROR] NSE Daily Automation: FAILED (Exit Code: %AUTO_RESULT%)
    echo [INFO] Check log for details: %LOGFILE%
)

echo [LOG] Log saved to: %LOGFILE%
echo ============================================================

exit /b %AUTO_RESULT%
