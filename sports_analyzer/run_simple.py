#!/usr/bin/env python3
"""
简化版运行脚本 - 不需要虚拟环境
"""

import sys
import subprocess
import os
from pathlib import Path

def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        'cv2', 'mediapipe', 'numpy', 'fastapi', 'uvicorn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'mediapipe':
                import mediapipe
            elif package == 'numpy':
                import numpy
            elif package == 'fastapi':
                import fastapi
            elif package == 'uvicorn':
                import uvicorn
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """安装依赖"""
    print("正在安装依赖包...")
    
    packages = [
        'opencv-python==4.9.0.80',
        'mediapipe==0.10.9', 
        'numpy==1.26.3',
        'fastapi==0.109.0',
        'uvicorn==0.27.0',
        'websockets==12.0',
        'python-multipart==0.0.6',
        'scipy==1.12.0',
        'pydantic==2.5.3'
    ]
    
    for package in packages:
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '--user', package
            ], check=True, capture_output=True)
            print(f"✓ 安装成功: {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ 安装失败: {package}")
            print(f"  错误: {e}")
            return False
    
    return True

def main():
    print("=================================")
    print("  AI运动训练分析系统")
    print("=================================")
    
    # 检查依赖
    missing = check_dependencies()
    
    if missing:
        print(f"缺少依赖: {', '.join(missing)}")
        print("正在尝试自动安装...")
        
        if not install_dependencies():
            print("依赖安装失败，请手动安装：")
            print("python3 -m pip install --user opencv-python mediapipe numpy fastapi uvicorn")
            return
    else:
        print("所有依赖已安装 ✓")
    
    # 创建必要目录
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('recordings', exist_ok=True)
    
    print("\n启动API服务...")
    print("API地址: http://localhost:8000")
    print("API文档: http://localhost:8000/docs")
    print("前端界面: frontend/index.html")
    print("\n按 Ctrl+C 停止服务")
    print("=================================")
    
    # 切换到api目录并启动服务
    api_dir = Path(__file__).parent / 'api'
    os.chdir(api_dir)
    
    try:
        import uvicorn
        from main import app
        
        # 直接运行FastAPI应用
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
        
    except ImportError:
        print("uvicorn未安装，尝试直接运行...")
        subprocess.run([sys.executable, 'main.py'])
    except KeyboardInterrupt:
        print("\n服务已停止")
    except Exception as e:
        print(f"启动失败: {e}")
        print("请检查依赖是否正确安装")

if __name__ == "__main__":
    main()