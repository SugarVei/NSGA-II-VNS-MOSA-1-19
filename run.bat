@echo off
REM 快速启动程序脚本
cd /d "%~dp0"
call .venv\Scripts\activate.bat
cd NSGA-II-VNS-MOSA-26-1-11-main
python main.py
pause
