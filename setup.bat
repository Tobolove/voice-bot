@echo off
echo ========================================
echo   IndexTTS-2 Voice Bot Setup
echo ========================================
echo.

REM Run the PowerShell setup script
powershell -ExecutionPolicy Bypass -File "%~dp0setup.ps1"

pause
