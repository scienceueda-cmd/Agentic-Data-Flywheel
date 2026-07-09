@echo off
color 0A
title Agentic Data Flywheel - Initial Setup

echo ========================================================
echo        Agentic Data Flywheel - Initial Setup
echo ========================================================
echo.
echo このスクリプトは以下の初期セットアップを全自動で行います。
echo 1. 必要なPythonライブラリのインストール
echo 2. Windows起動時のログ監視エージェントの自動起動登録
echo 3. 個人用設定ファイル（config.json）の自動生成
echo.
pause

echo.
echo --------------------------------------------------------
echo 1. 必須ライブラリをインストールしています...
echo --------------------------------------------------------
py -3.12 -m pip install -r requirements.txt

echo.
echo --------------------------------------------------------
echo 2. Windowsスタートアップへの登録を行っています...
echo --------------------------------------------------------
set "STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "VBS_FILE=%STARTUP_DIR%\WebUI_Agent_Monitor.vbs"

:: スタートアップフォルダに監視スクリプトをサイレント起動するVBSを作成
echo Set WshShell = CreateObject("WScript.Shell") > "%VBS_FILE%"
echo WshShell.Run "py -3.12 """%~dp0src\log_monitor.py"""", 0, False >> "%VBS_FILE%"

echo.
echo スタートアップへの登録が完了しました！
echo 今後、PCを起動するだけで裏側で自動的に監視フライホイールが回ります。
echo ※解除したい場合は、Windowsの「スタートアップ」フォルダからVBSファイルを削除してください。

echo.
echo --------------------------------------------------------
echo 3. 設定ファイル（config.json）を生成しています...
echo --------------------------------------------------------
if not exist "%~dp0config.json" (
    copy "%~dp0config.example.json" "%~dp0config.json" > nul
    echo config.json を作成しました。必要に応じて中身（モデルパス等）を書き換えてください。
) else (
    echo 既に config.json が存在するため、作成をスキップしました。
)
echo.
echo ========================================================
echo 全てのセットアップが完了しました！
echo 今後は「Agent_Control_Panel.bat」からエージェントを起動してください。
echo ========================================================
pause
