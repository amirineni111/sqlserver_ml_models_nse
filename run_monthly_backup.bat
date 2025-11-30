@echo off
:: NSE 500 Monthly Backup - Windows Task Scheduler
:: Run this script monthly to backup data and clean old logs
:: Recommended schedule: 1st of every month at 1:00 AM

setlocal enabledelayedexpansion

:: Set log file with timestamp
set TIMESTAMP=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set LOGFILE=%~dp0logs\monthly_backup_%TIMESTAMP%.log

:: Create logs directory if it doesn't exist
if not exist "%~dp0logs" mkdir "%~dp0logs"

:: Redirect all output to log file
call :MAIN > "%LOGFILE%" 2>&1
exit /b %errorlevel%

:MAIN
echo ============================================================
echo NSE 500 Monthly Backup - %date% %time%
echo ============================================================

:: Set working directory
cd /d "%~dp0"

:: Create backup directory with year-month format
set BACKUP_ROOT=%~dp0backups
set BACKUP_DIR=%BACKUP_ROOT%\%date:~-4,4%-%date:~-10,2%
if not exist "%BACKUP_ROOT%" mkdir "%BACKUP_ROOT%"
if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"

echo Creating monthly backup...

:: Backup models
echo Backing up models...
xcopy "%~dp0data" "%BACKUP_DIR%\data" /E /I /Q
echo Models backed up

:: Backup results (last 30 days)
echo Backing up recent results...
xcopy "%~dp0results" "%BACKUP_DIR%\results" /E /I /Q /D:%date%
echo Results backed up

:: Backup daily reports (last 30 days)
echo Backing up recent reports...
xcopy "%~dp0daily_reports" "%BACKUP_DIR%\daily_reports" /E /I /Q
echo Reports backed up

:: Clean old log files (older than 90 days)
echo Cleaning old log files...
forfiles /p "%~dp0logs" /s /m *.log /d -90 /c "cmd /c del @path" 2>nul
echo Old logs cleaned

:: Clean old result files (older than 180 days)
echo Cleaning old result files...
forfiles /p "%~dp0results" /s /m *.csv /d -180 /c "cmd /c del @path" 2>nul
echo Old results cleaned

:: Clean old backup directories (older than 1 year)
echo Cleaning old backups...
forfiles /p "%BACKUP_ROOT%" /s /m *.* /d -365 /c "cmd /c rd /s /q @path" 2>nul
echo Old backups cleaned

:: Generate backup summary
echo.
echo ============================================================
echo MONTHLY BACKUP SUMMARY - %date% %time%
echo ============================================================
echo ✅ Monthly backup completed successfully
echo 📁 Backup location: %BACKUP_DIR%
echo 🧹 Cleaned logs older than 90 days
echo 🧹 Cleaned results older than 180 days  
echo 🧹 Cleaned backups older than 1 year
echo 📝 Log saved to: %LOGFILE%
echo ============================================================

exit /b 0