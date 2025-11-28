@echo off
:: NSE 500 Daily Automation Batch Script
:: This script runs the daily NSE 500 trading signal automation

echo.
echo ===========================================
echo 🇮🇳 NSE 500 Daily Trading Automation
echo ===========================================
echo.

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python and add it to PATH.
    pause
    exit /b 1
)

:: Set the working directory to the script location
cd /d "%~dp0"

:: Show current date and time
echo 📅 Current Date/Time: %date% %time%
echo 📁 Working Directory: %cd%
echo.

:: Check command line arguments
if "%1"=="--help" (
    echo Usage:
    echo   run_nse_automation.bat           ^(Run full automation^)
    echo   run_nse_automation.bat --check   ^(Check data status only^)
    echo   run_nse_automation.bat --export  ^(Export results only^)
    echo.
    pause
    exit /b 0
)

:: Run the automation based on arguments
if "%1"=="--check" (
    echo 🔍 Running NSE data status check only...
    python daily_nse_automation.py --check-only
) else if "%1"=="--export" (
    echo 📁 Running NSE export only...
    python daily_nse_automation.py --export-only
) else (
    echo 🚀 Running full NSE automation...
    python daily_nse_automation.py
)

:: Check the exit code
if errorlevel 1 (
    echo.
    echo ❌ NSE automation encountered errors. Check the logs for details.
    echo 📝 Log files are saved in the 'logs' directory.
    echo.
    pause
    exit /b 1
) else (
    echo.
    echo ✅ NSE automation completed successfully!
    echo 📊 Check the 'results' directory for CSV exports.
    echo 📝 Check the 'daily_reports' directory for daily reports.
    echo 📄 Log files are in the 'logs' directory.
    echo.
)

pause