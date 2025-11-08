# PowerShell脚本 - 设置UTF-8编码并运行程序
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

# 使用虚拟环境中的Python
& "venv\Scripts\python.exe" pose_detection.py $args

