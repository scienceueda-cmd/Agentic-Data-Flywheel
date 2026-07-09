@echo off
color 0B
title Agentic Data Flywheel - Control Panel

:menu
cls
echo ========================================================
echo        Agentic Data Flywheel - Control Panel
echo ========================================================
echo.
echo  [1] Setup Auto-Start for Monitor ^& Start Now
echo  [2] Disable Auto-Start ^& Stop Monitor
echo  [3] Start CLI Agent
echo  [4] Start FastAPI Server for Open WebUI
echo  [5] Start LoRA Fine-tuning
echo  [6] Install Requirements
echo  [7] Exit
echo.
echo ========================================================
set /p choice="Please enter your choice (1-7): "

if "%choice%"=="1" goto startup
if "%choice%"=="2" goto disable
if "%choice%"=="3" goto agent
if "%choice%"=="4" goto api
if "%choice%"=="5" goto train
if "%choice%"=="6" goto install
if "%choice%"=="7" goto exit

echo Invalid input. Please try again.
pause
goto menu


:train
echo.
echo --------------------------------------------------------
echo Starting LoRA Fine-tuning process...
echo --------------------------------------------------------
py -3.12 src\train_lora.py
pause
goto menu

:agent
echo.
echo --------------------------------------------------------
echo Starting CLI Agent...
echo --------------------------------------------------------
py -3.12 src\run_agent.py
pause
goto menu

:install
echo.
echo --------------------------------------------------------
echo Installing requirements...
echo --------------------------------------------------------
py -3.12 -m pip install -r requirements.txt
pause
goto menu

:api
echo.
echo --------------------------------------------------------
echo Starting FastAPI Server for WebUI integration...
echo --------------------------------------------------------
py -3.12 src\run_api.py
pause
goto menu

:startup
echo.
echo --------------------------------------------------------
echo Registering Background Monitor to Windows Startup...
echo --------------------------------------------------------
set "startupFolder=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "shortcutFile=%startupFolder%\Agent_Monitor.vbs"
echo Set WshShell = CreateObject("WScript.Shell") > "%shortcutFile%"
echo WshShell.Run "py -3.12 ""%~dp0src\log_monitor.py""", 0, False >> "%shortcutFile%"

echo Starting the monitor now...
cscript //nologo "%shortcutFile%"

echo.
echo Registration successful and Monitor started!
echo The monitor is now running in the background, AND it will start automatically whenever you turn on your PC.
pause
goto menu

:disable
echo.
echo --------------------------------------------------------
echo Disabling Auto-Start and Stopping Monitor...
echo --------------------------------------------------------
set "startupFolder=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "shortcutFile=%startupFolder%\Agent_Monitor.vbs"
if exist "%shortcutFile%" (
    del "%shortcutFile%"
    echo Removed Auto-Start script from Windows Startup folder.
) else (
    echo Auto-Start is not currently enabled.
)
echo.
echo Stopping any running background monitor processes...
wmic process where "name='python.exe' and commandline like '%%log_monitor.py%%'" call terminate >nul 2>&1
wmic process where "name='py.exe' and commandline like '%%log_monitor.py%%'" call terminate >nul 2>&1
echo Monitor stopped!
pause
goto menu

:exit
exit
