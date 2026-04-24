@echo off
REM 在当前目录下直接启动 GUI（已预置 PyQt5 插件路径）
REM Launches the PyQt5 GUI using the local .venv interpreter
cd /d "%~dp0"
set QT_QPA_PLATFORM_PLUGIN_PATH=%~dp0.venv\Lib\site-packages\PyQt5\Qt5\plugins\platforms
.venv\Scripts\python.exe main.py
pause
