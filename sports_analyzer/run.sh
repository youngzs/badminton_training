#!/bin/bash

# AI运动训练分析系统启动脚本

echo "==================================="
echo "  AI运动训练分析系统 v1.0.0"
echo "==================================="

# 检查Python版本
python_version=$(python3 --version 2>&1)
echo "Python版本: $python_version"

# 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "检查并安装依赖..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# 创建必要的目录
mkdir -p uploads
mkdir -p recordings
mkdir -p logs

# 启动API服务
echo "启动API服务..."
echo "API地址: http://localhost:8000"
echo "API文档: http://localhost:8000/docs"
echo "前端界面: 请在浏览器打开 frontend/index.html"
echo ""
echo "按 Ctrl+C 停止服务"
echo "==================================="

# 启动FastAPI应用
cd api
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload