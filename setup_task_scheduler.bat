@echo off
:: Setup Windows Task Scheduler for NSE 500 Automation
:: Run this script once to create scheduled tasks
:: Requires Administrator privileges

echo ============================================================
echo NSE 500 Task Scheduler Setup
echo ============================================================
echo.

:: Check for admin privileges
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ This script requires Administrator privileges.
    echo Please right-click and "Run as Administrator"
    pause
    exit /b 1
)

echo ✅ Administrator privileges confirmed
echo.

:: Get current directory
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

echo 📁 Script directory: %SCRIPT_DIR%
echo.

:: Create Daily Predictions Task (Monday to Friday, 9:30 AM)
echo Creating daily predictions task...
schtasks /create /tn "NSE 500 Daily Predictions" /tr "\"%SCRIPT_DIR%\run_daily_predictions.bat\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 09:30 /f /rl highest
if errorlevel 1 (
    echo ❌ Failed to create daily predictions task
) else (
    echo ✅ Daily predictions task created (9:30 AM, Mon-Fri)
)
echo.

:: Create Weekly Retraining Task (Sundays, 2:00 AM)
echo Creating weekly retraining task...
schtasks /create /tn "NSE 500 Weekly Retrain" /tr "\"%SCRIPT_DIR%\run_weekly_retrain.bat\"" /sc weekly /d SUN /st 02:00 /f /rl highest
if errorlevel 1 (
    echo ❌ Failed to create weekly retraining task
) else (
    echo ✅ Weekly retraining task created (2:00 AM Sundays)
)
echo.

:: Create Monthly Backup Task (1st of month, 1:00 AM)
echo Creating monthly backup task...
schtasks /create /tn "NSE 500 Monthly Backup" /tr "\"%SCRIPT_DIR%\run_monthly_backup.bat\"" /sc monthly /d 1 /st 01:00 /f /rl highest
if errorlevel 1 (
    echo ❌ Failed to create monthly backup task
) else (
    echo ✅ Monthly backup task created (1:00 AM 1st of month)
)
echo.

:: Create test task to run immediately for verification
echo Creating test task...
schtasks /create /tn "NSE 500 Test Run" /tr "\"%SCRIPT_DIR%\run_daily_predictions.bat\"" /sc once /st 23:59 /sd 01/01/2030 /f /rl highest
if errorlevel 1 (
    echo ❌ Failed to create test task
) else (
    echo ✅ Test task created (can be run manually)
)
echo.

echo ============================================================
echo TASK SCHEDULER SETUP COMPLETE
echo ============================================================
echo.
echo 📋 Created Tasks:
echo   • NSE 500 Daily Predictions  (9:30 AM Monday-Friday)
echo   • NSE 500 Weekly Retrain     (2:00 AM Sundays)
echo   • NSE 500 Monthly Backup     (1:00 AM 1st of month)
echo   • NSE 500 Test Run           (Manual execution)
echo.
echo 🔧 To manage tasks:
echo   1. Open Task Scheduler (Win+R, type: taskschd.msc)
echo   2. Navigate to Task Scheduler Library
echo   3. Find and manage the NSE 500 tasks
echo.
echo 🧪 To test immediately:
echo   schtasks /run /tn "NSE 500 Test Run"
echo.
echo 📝 All logs will be saved in the 'logs' directory
echo 📁 Backups will be saved in the 'backups' directory
echo.
echo ⚡ Tasks are now ready for automated execution!
echo.
pause