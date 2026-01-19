@echo off
REM ============================================================
REM 多目标调度优化系统 - 一键环境配置脚本 (Windows)
REM NSGA-II-VNS-MOSA 算法对比实验平台
REM ============================================================

echo.
echo ========================================
echo    多目标调度优化系统 - 环境配置
echo ========================================
echo.

REM 检查 Python 是否已安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.9+ 
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/5] Python 已检测到
python --version

REM 进入项目目录
cd /d "%~dp0"
cd NSGA-II-VNS-MOSA-26-1-11-main

echo.
echo [2/5] 正在创建虚拟环境...
if exist "..\\.venv" (
    echo      虚拟环境已存在，跳过创建
) else (
    python -m venv ..\\.venv
    echo      虚拟环境创建完成
)

echo.
echo [3/5] 正在激活虚拟环境...
call ..\\.venv\\Scripts\\activate.bat

echo.
echo [4/5] 正在安装依赖包...
echo      这可能需要几分钟，请耐心等待...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo [5/5] 正在验证安装...
python -c "import PyQt5; import numpy; import pandas; import matplotlib; print('所有依赖包安装成功!')"

if errorlevel 1 (
    echo.
    echo [错误] 部分依赖安装失败，请检查网络连接后重试
    pause
    exit /b 1
)

echo.
echo ========================================
echo    环境配置完成!
echo ========================================
echo.
echo 启动程序请运行:
echo    cd NSGA-II-VNS-MOSA-26-1-11-main
echo    python main.py
echo.
echo 或者直接双击 run.bat 启动
echo.
pause
