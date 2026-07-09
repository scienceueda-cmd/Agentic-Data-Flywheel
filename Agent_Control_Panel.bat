@echo off
chcp 65001 >nul
color 0B
title Agentic Data Flywheel - Control Panel

:menu
cls
echo ========================================================
echo        Agentic Data Flywheel - Control Panel
echo ========================================================
echo.
echo  [1] 監視エージェントの起動 (Background Monitor)
echo  [2] 脳みその学習を開始 (LoRA Fine-tuning)
echo  [3] エージェントを起動 (CLI Agent)
echo  [4] 必須ライブラリのインストール (Install Requirements)
echo  [5] WebUI連携用 APIサーバー起動 (FastAPI Server)
echo  [6] 終了 (Exit)
echo.
echo ========================================================
set /p choice="実行したいメニューの番号を入力してください: "

if "%choice%"=="1" goto monitor
if "%choice%"=="2" goto train
if "%choice%"=="3" goto agent
if "%choice%"=="4" goto install
if "%choice%"=="5" goto api
if "%choice%"=="6" goto exit

echo 不正な入力です。もう一度入力してください。
pause
goto menu

:monitor
echo.
echo 監視エージェントをバックグラウンドでサイレント起動します...
echo Set WshShell = CreateObject("WScript.Shell") > "%temp%\run_monitor.vbs"
echo WshShell.Run "py -3.12 """%~dp0src\log_monitor.py"""", 0, False >> "%temp%\run_monitor.vbs"
cscript //nologo "%temp%\run_monitor.vbs"
echo 起動しました！裏でログの監視と学習データの生成を行います。
pause
goto menu

:train
echo.
echo --------------------------------------------------------
echo 脳みその学習（LoRAファインチューニング）を開始します...
echo --------------------------------------------------------
py -3.12 src\train_lora.py
pause
goto menu

:agent
echo.
echo --------------------------------------------------------
echo エージェント（CLI）を起動します...
echo --------------------------------------------------------
py -3.12 src\run_agent.py
pause
goto menu

:install
echo.
echo --------------------------------------------------------
echo 必須ライブラリ (requirements.txt) をインストールします...
echo --------------------------------------------------------
py -3.12 -m pip install -r requirements.txt
pause
goto menu

:api
echo.
echo --------------------------------------------------------
echo WebUI連携用 APIサーバー (FastAPI) を起動します...
echo --------------------------------------------------------
py -3.12 src\run_api.py
pause
goto menu

:exit
exit
