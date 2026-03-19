@echo off
REM 设置 Python 路径变量
set PYTHON_EXE=C:\PX4PSP\Python38\python.exe

REM 使用国内源安装所有库，包括 CUDA 12.4 对应的 PyTorch
%PYTHON_EXE% -m pip install ^
    smolagents==1.20.0 ^
    torch==2.5.0+cu124 torchvision==0.20.0+cu124 torchaudio==2.5.0+cu124 ^
    supervision ^
    transformers ^
    addict ^
    yapf ^
    pycocotools ^
    timm ^
    ultralytics ^
    --index-url https://mirrors.aliyun.com/pypi/simple/ ^
    --extra-index-url https://download.pytorch.org/whl/cu124 ^
    --trusted-host mirrors.aliyun.com

echo.
echo ✅ 所有库安装完成
pause
