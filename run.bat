@echo off
REM 快速启动程序脚本（一键运行 NSGA-II-VNS-MOSA 混合算法 GUI）
REM Quick-launch script for the NSGA-II-VNS-MOSA hybrid algorithm GUI
cd /d "%~dp0"
call .venv\Scripts\activate.bat
cd "NSGA-II-VNS-MOSA Algorithm"
python main.py
pause
