@echo off
echo ========================================
echo 启动羽毛球视频对比分析系统
echo ========================================
echo.

REM 激活虚拟环境
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo 错误: 找不到虚拟环境
    echo 请先运行安装脚本
    pause
    exit /b 1
)

REM 检查Python和依赖
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: Python不可用
    pause
    exit /b 1
)

echo 启动GUI...
python pose_detection_compare_gui.py

@REM pause
