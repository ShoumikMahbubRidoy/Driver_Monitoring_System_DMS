@echo off
REM Driver Monitoring System - Quick Start Script (Windows)
REM This script automates the entire setup process

echo ==================================================================
echo   Driver Monitoring System - Automated Setup (Windows)
echo ==================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] Python is not installed
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [+] Python found
python --version

REM Create virtual environment
echo.
echo Step 1: Creating virtual environment...
if not exist "dms_env\" (
    python -m venv dms_env
    echo [+] Virtual environment created
) else (
    echo [!] Virtual environment already exists
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call dms_env\Scripts\activate.bat

REM Install requirements
echo.
echo Step 2: Installing Python packages...
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q
echo [+] All packages installed

REM Download datasets
echo.
echo Step 3: Downloading datasets...
python 1_download_datasets.py

REM Manual download instructions
echo.
echo ==================================================================
echo MANUAL ACTION REQUIRED:
echo ==================================================================
echo Please download the following datasets manually:
echo.
echo 1. Drowsiness Dataset:
echo    URL: https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset
echo    - Download 'archive.zip'
echo    - Place it in: .\datasets\drowsiness\
echo.
echo 2. (Optional) State Farm Distraction Dataset:
echo    URL: https://www.kaggle.com/c/state-farm-distracted-driver-detection
echo.
pause

REM Prepare datasets
echo.
echo Step 4: Preparing datasets...
python 2_prepare_datasets.py

REM Ask if user wants to train now
echo.
echo ==================================================================
echo READY TO TRAIN!
echo ==================================================================
echo Training will take approximately:
echo   - With GPU: 2-3 hours
echo   - With CPU: 12-18 hours
echo.
set /p train_now="Do you want to start training now? (y/n): "

if /i "%train_now%"=="y" (
    echo.
    echo Step 5: Training models...
    echo This will train all 3 models (eye_state, yawn, drowsiness)
    python 3_train_models.py --model all --epochs 25 --batch-size 64
    
    REM Export to ONNX
    echo.
    echo Step 6: Exporting to ONNX...
    python 4_export_onnx.py --model all
    
    echo.
    echo ==================================================================
    echo [+] SETUP COMPLETE!
    echo ==================================================================
    echo.
    echo Next steps:
    echo   1. Test on webcam: python 5_test_webcam.py
    echo   2. Deploy to OAK-D: python 6_deploy_oakd.py
    echo.
    echo All trained models are in: .\runs\
    echo ONNX exports are in: .\exports\
) else (
    echo.
    echo ==================================================================
    echo Setup complete! Training skipped.
    echo ==================================================================
    echo.
    echo To train models later, run:
    echo   python 3_train_models.py --model all --epochs 25
    echo.
    echo Or train individually:
    echo   python 3_train_models.py --model eye_state --epochs 25
    echo   python 3_train_models.py --model yawn --epochs 25
    echo   python 3_train_models.py --model drowsiness --epochs 25
)

echo.
echo For detailed instructions, see README.md
echo.
pause