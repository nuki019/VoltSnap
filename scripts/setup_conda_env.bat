@echo off
echo === VoltSnap Conda Environment Setup ===

call conda create -n voltsnap python=3.11 -y
if errorlevel 1 (
    echo [ERROR] Failed to create conda environment
    exit /b 1
)

call conda activate voltsnap
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment
    exit /b 1
)

pip install schemdraw "opencv-contrib-python>=4.8" numpy matplotlib networkx pyyaml pytest
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    exit /b 1
)

echo.
echo === Running environment verification ===
python tools\verify_env.py

echo.
echo === Setup complete ===
echo Run 'conda activate voltsnap' to use the environment.
