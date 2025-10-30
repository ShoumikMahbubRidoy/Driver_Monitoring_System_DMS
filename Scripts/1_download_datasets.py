"""
Step 1: Download and Prepare Datasets for Driver Monitoring System
-------------------------------------------------------------------
This script downloads and organizes datasets for:
- Drowsiness detection
- Eye state classification
- Yawn detection
- Driver distraction detection

Run: python 1_download_datasets.py
"""

import os
import urllib.request
import zipfile
import gdown
from pathlib import Path

# Create project structure
DATASET_ROOT = "../datasets"
MODELS_DIR = "../models"
RUNS_DIR = "../runs"

def create_project_structure():
    """Create all necessary folders"""
    folders = [
        f"{DATASET_ROOT}/drowsiness",
        f"{DATASET_ROOT}/eye_state",
        f"{DATASET_ROOT}/yawn",
        f"{DATASET_ROOT}/distraction",
        f"{MODELS_DIR}/pretrained",
        f"{RUNS_DIR}/drowsiness",
        f"{RUNS_DIR}/eye_state",
        f"{RUNS_DIR}/yawn",
        "./logs",
        "./exports",
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("✓ Project structure created")

def download_mrl_eye_dataset():
    """
    MRL Eye Dataset - 84,898 images
    Classes: Open, Closed
    Perfect for eye state detection
    """
    print("\n[1/4] Downloading MRL Eye Dataset...")
    url = "http://mrl.cs.vsb.cz/eyedataset/mrlEyes_2018_01.zip"
    output = f"{DATASET_ROOT}/eye_state/mrlEyes.zip"
    
    if not os.path.exists(output):
        print("Downloading from MRL (this may take 10-15 minutes)...")
        urllib.request.urlretrieve(url, output)
        print("✓ Downloaded")
    
    # Extract
    extract_dir = f"{DATASET_ROOT}/eye_state/mrlEyes"
    if not os.path.exists(extract_dir):
        print("Extracting...")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("✓ MRL Eye Dataset ready")
    else:
        print("✓ MRL Eye Dataset already extracted")

def download_yawdd_dataset():
    """
    YawDD - Yawn Detection Dataset
    Google Drive: https://drive.google.com/file/d/1QZICv4wqpqKdZPvZQ0F_FnLaDLgxzJqT
    """
    print("\n[2/4] Downloading YawDD (Yawn Detection Dataset)...")
    gdrive_id = "1QZICv4wqpqKdZPvZQ0F_FnLaDLgxzJqT"
    output = f"{DATASET_ROOT}/yawn/yawdd.zip"
    
    if not os.path.exists(output):
        print("Downloading from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={gdrive_id}", output, quiet=False)
        print("✓ Downloaded")
    
    extract_dir = f"{DATASET_ROOT}/yawn/YawDD"
    if not os.path.exists(extract_dir):
        print("Extracting...")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(f"{DATASET_ROOT}/yawn/")
        print("✓ YawDD ready")
    else:
        print("✓ YawDD already extracted")

def download_drowsiness_dataset():
    """
    Drowsiness Detection Dataset from Kaggle
    Manual download required: https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset
    """
    print("\n[3/4] Drowsiness Detection Dataset")
    print("=" * 60)
    print("MANUAL DOWNLOAD REQUIRED:")
    print("1. Go to: https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset")
    print("2. Download 'archive.zip'")
    print(f"3. Place it in: {DATASET_ROOT}/drowsiness/")
    print("4. The script will auto-extract it")
    print("=" * 60)
    
    zip_path = f"{DATASET_ROOT}/drowsiness/archive.zip"
    if os.path.exists(zip_path):
        extract_dir = f"{DATASET_ROOT}/drowsiness/data"
        if not os.path.exists(extract_dir):
            print("Found archive.zip, extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(f"{DATASET_ROOT}/drowsiness/")
            print("✓ Drowsiness dataset extracted")
        else:
            print("✓ Drowsiness dataset already extracted")
    else:
        print("⚠ archive.zip not found - please download manually")

def download_distraction_dataset():
    """
    State Farm Distracted Driver Detection
    Kaggle: https://www.kaggle.com/c/state-farm-distracted-driver-detection
    """
    print("\n[4/4] Driver Distraction Dataset")
    print("=" * 60)
    print("MANUAL DOWNLOAD (OPTIONAL):")
    print("1. Go to: https://www.kaggle.com/c/state-farm-distracted-driver-detection/data")
    print("2. Download 'train.zip' and 'test.zip'")
    print(f"3. Place in: {DATASET_ROOT}/distraction/")
    print("Note: This is optional - we'll focus on drowsiness first")
    print("=" * 60)
    
    zip_path = f"{DATASET_ROOT}/distraction/train.zip"
    if os.path.exists(zip_path):
        extract_dir = f"{DATASET_ROOT}/distraction/train"
        if not os.path.exists(extract_dir):
            print("Extracting distraction dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(f"{DATASET_ROOT}/distraction/")
            print("✓ Distraction dataset extracted")
        else:
            print("✓ Distraction dataset already extracted")
    else:
        print("⚠ Distraction dataset not found (optional)")

def download_pretrained_models():
    """Download pretrained models for facial landmarks"""
    print("\n[5/5] Downloading pretrained models...")
    
    # Dlib shape predictor
    shape_predictor_url = "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"
    shape_predictor_path = f"{MODELS_DIR}/pretrained/shape_predictor_68_face_landmarks.dat"
    
    if not os.path.exists(shape_predictor_path):
        print("Downloading dlib shape predictor (99.7 MB)...")
        urllib.request.urlretrieve(shape_predictor_url, shape_predictor_path)
        print("✓ Shape predictor downloaded")
    else:
        print("✓ Shape predictor already exists")

def create_readme():
    """Create README with dataset info"""
    readme = """
# Driver Monitoring System - Dataset Guide

## Downloaded Datasets

### 1. MRL Eye Dataset (Auto-downloaded)
- **Size**: 84,898 images
- **Classes**: Open (42,449), Closed (42,449)
- **Location**: `datasets/eye_state/mrlEyes/`
- **Usage**: Eye state classification (open/closed detection)

### 2. YawDD - Yawn Detection (Auto-downloaded)
- **Size**: ~3,000 videos
- **Classes**: Normal driving, Talking, Yawning
- **Location**: `datasets/yawn/YawDD/`
- **Usage**: Yawn detection for fatigue monitoring

### 3. Drowsiness Dataset (Manual Download)
- **Link**: https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset
- **Classes**: Yawn, No_yawn, Open, Closed
- **Location**: `datasets/drowsiness/`
- **Usage**: Combined drowsiness detection

### 4. State Farm Distraction (Optional)
- **Link**: https://www.kaggle.com/c/state-farm-distracted-driver-detection
- **Classes**: 10 types of driver distraction
- **Location**: `datasets/distraction/`
- **Usage**: Distraction detection (phone use, drinking, etc.)

## Next Steps

Run the following scripts in order:
1. `python 2_prepare_datasets.py` - Organize and split data
2. `python 3_train_eye_state.py` - Train eye classifier
3. `python 4_train_yawn_detector.py` - Train yawn detector
4. `python 5_train_drowsiness.py` - Train drowsiness model
5. `python 6_export_onnx.py` - Export for OAK-D
6. `python 7_run_inference.py` - Test on webcam
7. `python 8_deploy_oakd.py` - Deploy to OAK-D camera

## Dataset Statistics (After Download)

Total disk space required: ~15-20 GB
- MRL Eyes: ~5 GB
- YawDD: ~8 GB
- Drowsiness: ~2 GB
- Distraction: ~5 GB (optional)

Training time (GPU): 2-4 hours total
Training time (CPU): 12-24 hours total
"""
    
    with open("DATASET_README.md", "w") as f:
        f.write(readme)
    print("\n✓ Created DATASET_README.md")

if __name__ == "__main__":
    print("=" * 70)
    print("Driver Monitoring System - Dataset Downloader")
    print("=" * 70)
    
    try:
        # Install required packages
        print("\nInstalling required packages...")
        os.system("pip install gdown pillow opencv-python -q")
        
        create_project_structure()
        download_mrl_eye_dataset()
        download_yawdd_dataset()
        download_drowsiness_dataset()
        download_distraction_dataset()
        download_pretrained_models()
        create_readme()
        
        print("\n" + "=" * 70)
        print("SETUP COMPLETE!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. If you haven't already, manually download the Kaggle datasets")
        print("2. Run: python 2_prepare_datasets.py")
        print("\nFor detailed instructions, see DATASET_README.md")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nIf download fails, please download datasets manually:")
        print("See DATASET_README.md for direct links")