#!/bin/bash

echo "==================================="
echo "  安装系统Python包"
echo "==================================="

echo "请在终端中手动运行以下命令来安装系统包："
echo ""
echo "sudo apt update"
echo "sudo apt install python3-pip python3-venv python3-opencv python3-numpy python3-scipy"
echo "sudo apt install python3-pil python3-requests"
echo ""
echo "或者如果您想在用户目录安装（可能破坏系统），可以运行："
echo "python3 -m pip install --break-system-packages --user fastapi uvicorn opencv-python mediapipe numpy"
echo ""
echo "==================================="