#!/bin/bash

echo "==================================="
echo "  无需sudo权限的系统设置"
echo "==================================="

cd /home/yangzs/projects/badminton_training/sports_analyzer

# 检查是否有pip
if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
    echo "正在下载并安装pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py --user
    rm get-pip.py
    
    # 添加用户pip路径到环境变量
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

# 使用pip安装依赖到用户目录
echo "安装Python依赖包..."
python3 -m pip install --user opencv-python==4.9.0.80
python3 -m pip install --user mediapipe==0.10.9
python3 -m pip install --user numpy==1.26.3
python3 -m pip install --user fastapi==0.109.0
python3 -m pip install --user uvicorn==0.27.0
python3 -m pip install --user websockets==12.0
python3 -m pip install --user python-multipart==0.0.6

# 安装其他依赖
echo "安装其他依赖..."
python3 -m pip install --user scipy==1.12.0
python3 -m pip install --user scikit-learn==1.4.0
python3 -m pip install --user pillow==10.2.0
python3 -m pip install --user pydantic==2.5.3
python3 -m pip install --user python-dotenv==1.0.0
python3 -m pip install --user aiofiles==23.2.1

echo "==================================="
echo "安装完成！"
echo "现在可以运行："
echo "python3 api/main.py"
echo "==================================="