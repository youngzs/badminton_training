# 设置编码为UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# 切换到脚本目录
Set-Location $PSScriptRoot

# 激活虚拟环境并运行
if (Test-Path "venv\Scripts\Activate.ps1") {
    & .\venv\Scripts\Activate.ps1
    python pose_detection_gui.py
} else {
    Write-Host "虚拟环境不存在，请先创建虚拟环境"
    Read-Host "按回车键退出"
}

