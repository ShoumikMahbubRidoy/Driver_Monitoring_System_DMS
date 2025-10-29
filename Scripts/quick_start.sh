#!/bin/bash

# Driver Monitoring System - Quick Start Script
# This script automates the entire setup process

echo "=================================================================="
echo "  Driver Monitoring System - Automated Setup"
echo "=================================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo -e "${GREEN}✓ Python found: $(python3 --version)${NC}"

# Create virtual environment
echo ""
echo "Step 1: Creating virtual environment..."
if [ ! -d "dms_env" ]; then
    python3 -m venv dms_env
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source dms_env/bin/activate 2>/dev/null || source dms_env/Scripts/activate 2>/dev/null

# Install requirements
echo ""
echo "Step 2: Installing Python packages..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "${GREEN}✓ All packages installed${NC}"

# Download datasets
echo ""
echo "Step 3: Downloading datasets..."
python 1_download_datasets.py

# Check if manual downloads are needed
echo ""
echo -e "${YELLOW}=================================================================="
echo "MANUAL ACTION REQUIRED:"
echo "=================================================================="
echo "Please download the following datasets manually:"
echo ""
echo "1. Drowsiness Dataset:"
echo "   URL: https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset"
echo "   - Download 'archive.zip'"
echo "   - Place it in: ./datasets/drowsiness/"
echo ""
echo "2. (Optional) State Farm Distraction Dataset:"
echo "   URL: https://www.kaggle.com/c/state-farm-distracted-driver-detection"
echo ""
read -p "Press ENTER after you've downloaded the datasets..." dummy

# Prepare datasets
echo ""
echo "Step 4: Preparing datasets..."
python 2_prepare_datasets.py

# Ask if user wants to train now
echo ""
echo -e "${YELLOW}=================================================================="
echo "READY TO TRAIN!"
echo "=================================================================="
echo "Training will take approximately:"
echo "  - With GPU: 2-3 hours"
echo "  - With CPU: 12-18 hours"
echo ""
read -p "Do you want to start training now? (y/n): " train_now

if [ "$train_now" = "y" ] || [ "$train_now" = "Y" ]; then
    echo ""
    echo "Step 5: Training models..."
    echo "This will train all 3 models (eye_state, yawn, drowsiness)"
    python 3_train_models.py --model all --epochs 25 --batch-size 64
    
    # Export to ONNX
    echo ""
    echo "Step 6: Exporting to ONNX..."
    python 4_export_onnx.py --model all
    
    echo ""
    echo -e "${GREEN}=================================================================="
    echo "✓ SETUP COMPLETE!"
    echo "=================================================================="
    echo ""
    echo "Next steps:"
    echo "  1. Test on webcam: python 5_test_webcam.py"
    echo "  2. Deploy to OAK-D: python 6_deploy_oakd.py"
    echo ""
    echo "All trained models are in: ./runs/"
    echo "ONNX exports are in: ./exports/"
    echo -e "${NC}"
else
    echo ""
    echo -e "${YELLOW}=================================================================="
    echo "Setup complete! Training skipped."
    echo "=================================================================="
    echo ""
    echo "To train models later, run:"
    echo "  python 3_train_models.py --model all --epochs 25"
    echo ""
    echo "Or train individually:"
    echo "  python 3_train_models.py --model eye_state --epochs 25"
    echo "  python 3_train_models.py --model yawn --epochs 25"
    echo "  python 3_train_models.py --model drowsiness --epochs 25"
    echo -e "${NC}"
fi

echo ""
echo "For detailed instructions, see README.md"
echo ""