"""
Step 2: Prepare and Organize Datasets
--------------------------------------
This script organizes all datasets into train/val/test splits
with proper folder structure for training.

Run: python 2_prepare_datasets.py
"""

import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

random.seed(42)
np.random.seed(42)

DATASET_ROOT = "./datasets"
PROCESSED_ROOT = "./datasets_processed"

def prepare_eye_state_dataset():
    """
    Organize MRL Eye Dataset into train/val/test
    Structure: datasets_processed/eye_state/{train,val,test}/{open,closed}/
    """
    print("\n[1/3] Preparing Eye State Dataset...")
    
    source = f"{DATASET_ROOT}/eye_state/mrlEyes/mrlEyes_2018_01"
    output = f"{PROCESSED_ROOT}/eye_state"
    
    if not os.path.exists(source):
        print("⚠ MRL Eye dataset not found. Run 1_download_datasets.py first")
        return
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        for cls in ['open', 'closed']:
            os.makedirs(f"{output}/{split}/{cls}", exist_ok=True)
    
    # Collect all images
    open_imgs = []
    closed_imgs = []
    
    for root, dirs, files in os.walk(source):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                # MRL dataset naming: *_0_* = closed, *_1_* = open
                if '_0_' in file or 'close' in root.lower():
                    closed_imgs.append(path)
                elif '_1_' in file or 'open' in root.lower():
                    open_imgs.append(path)
    
    print(f"Found {len(open_imgs)} open eyes, {len(closed_imgs)} closed eyes")
    
    # Split each class
    def split_and_copy(images, class_name):
        random.shuffle(images)
        train_imgs, temp = train_test_split(images, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(temp, test_size=0.5, random_state=42)
        
        for img_path in train_imgs:
            dst = f"{output}/train/{class_name}/{Path(img_path).name}"
            if not os.path.exists(dst):
                shutil.copy2(img_path, dst)
        
        for img_path in val_imgs:
            dst = f"{output}/val/{class_name}/{Path(img_path).name}"
            if not os.path.exists(dst):
                shutil.copy2(img_path, dst)
        
        for img_path in test_imgs:
            dst = f"{output}/test/{class_name}/{Path(img_path).name}"
            if not os.path.exists(dst):
                shutil.copy2(img_path, dst)
        
        print(f"  {class_name}: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")
    
    split_and_copy(open_imgs, 'open')
    split_and_copy(closed_imgs, 'closed')
    
    print("✓ Eye state dataset prepared")

def prepare_drowsiness_dataset():
    """
    Organize Drowsiness Dataset
    Structure: datasets_processed/drowsiness/{train,val,test}/{alert,drowsy}/
    """
    print("\n[2/3] Preparing Drowsiness Dataset...")
    
    # Check multiple possible locations
    possible_sources = [
        f"{DATASET_ROOT}/drowsiness/Drowsiness Dataset",
        f"{DATASET_ROOT}/drowsiness/data",
        f"{DATASET_ROOT}/drowsiness",
    ]
    
    source = None
    for src in possible_sources:
        if os.path.exists(src):
            source = src
            break
    
    if not source:
        print("⚠ Drowsiness dataset not found. Download from Kaggle:")
        print("  https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset")
        return
    
    output = f"{PROCESSED_ROOT}/drowsiness"
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        for cls in ['alert', 'drowsy']:
            os.makedirs(f"{output}/{split}/{cls}", exist_ok=True)
    
    # Collect images
    alert_imgs = []
    drowsy_imgs = []
    
    for root, dirs, files in os.walk(source):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                lower_path = path.lower()
                
                # Classification logic
                if 'yawn' in lower_path or 'closed' in lower_path or 'drowsy' in lower_path:
                    drowsy_imgs.append(path)
                elif 'open' in lower_path or 'alert' in lower_path or 'no_yawn' in lower_path:
                    alert_imgs.append(path)
    
    print(f"Found {len(alert_imgs)} alert images, {len(drowsy_imgs)} drowsy images")
    
    if len(alert_imgs) == 0 and len(drowsy_imgs) == 0:
        print("⚠ No images found. Check dataset structure.")
        return
    
    # Split and copy
    def split_and_copy(images, class_name):
        if len(images) == 0:
            return
        random.shuffle(images)
        train_imgs, temp = train_test_split(images, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(temp, test_size=0.5, random_state=42)
        
        for img_path in train_imgs:
            dst = f"{output}/train/{class_name}/{Path(img_path).name}"
            if not os.path.exists(dst):
                try:
                    shutil.copy2(img_path, dst)
                except:
                    pass
        
        for img_path in val_imgs:
            dst = f"{output}/val/{class_name}/{Path(img_path).name}"
            if not os.path.exists(dst):
                try:
                    shutil.copy2(img_path, dst)
                except:
                    pass
        
        for img_path in test_imgs:
            dst = f"{output}/test/{class_name}/{Path(img_path).name}"
            if not os.path.exists(dst):
                try:
                    shutil.copy2(img_path, dst)
                except:
                    pass
        
        print(f"  {class_name}: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")
    
    split_and_copy(alert_imgs, 'alert')
    split_and_copy(drowsy_imgs, 'drowsy')
    
    print("✓ Drowsiness dataset prepared")

def prepare_yawn_dataset():
    """
    Organize YawDD dataset (video frames)
    Structure: datasets_processed/yawn/{train,val,test}/{yawn,no_yawn}/
    """
    print("\n[3/3] Preparing Yawn Dataset...")
    
    source = f"{DATASET_ROOT}/yawn/YawDD"
    output = f"{PROCESSED_ROOT}/yawn"
    
    if not os.path.exists(source):
        print("⚠ YawDD dataset not found")
        return
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        for cls in ['yawn', 'no_yawn']:
            os.makedirs(f"{output}/{split}/{cls}", exist_ok=True)
    
    yawn_imgs = []
    no_yawn_imgs = []
    
    # YawDD has video folders with frames
    for root, dirs, files in os.walk(source):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                lower_path = path.lower()
                
                if 'yawn' in lower_path:
                    yawn_imgs.append(path)
                else:
                    no_yawn_imgs.append(path)
    
    print(f"Found {len(yawn_imgs)} yawn images, {len(no_yawn_imgs)} no-yawn images")
    
    # Split and copy
    def split_and_copy(images, class_name):
        if len(images) == 0:
            return
        random.shuffle(images)
        train_imgs, temp = train_test_split(images, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(temp, test_size=0.5, random_state=42)
        
        for img_path in train_imgs[:min(len(train_imgs), 5000)]:  # Limit to 5k per class
            dst = f"{output}/train/{class_name}/{Path(img_path).name}"
            if not os.path.exists(dst):
                try:
                    shutil.copy2(img_path, dst)
                except:
                    pass
        
        for img_path in val_imgs[:min(len(val_imgs), 1000)]:
            dst = f"{output}/val/{class_name}/{Path(img_path).name}"
            if not os.path.exists(dst):
                try:
                    shutil.copy2(img_path, dst)
                except:
                    pass
        
        for img_path in test_imgs[:min(len(test_imgs), 1000)]:
            dst = f"{output}/test/{class_name}/{Path(img_path).name}"
            if not os.path.exists(dst):
                try:
                    shutil.copy2(img_path, dst)
                except:
                    pass
        
        print(f"  {class_name}: copied to train/val/test")
    
    split_and_copy(yawn_imgs, 'yawn')
    split_and_copy(no_yawn_imgs, 'no_yawn')
    
    print("✓ Yawn dataset prepared")

def create_summary():
    """Create a summary of prepared datasets"""
    print("\n" + "=" * 70)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 70)
    
    datasets = ['eye_state', 'drowsiness', 'yawn']
    
    for ds in datasets:
        print(f"\n{ds.upper()}:")
        for split in ['train', 'val', 'test']:
            path = f"{PROCESSED_ROOT}/{ds}/{split}"
            if os.path.exists(path):
                total = sum([len(files) for _, _, files in os.walk(path)])
                print(f"  {split}: {total} images")
    
    print("\n" + "=" * 70)
    print("✓ All datasets prepared!")
    print("\nNext step: Run python 3_train_models.py")
    print("=" * 70)

if __name__ == "__main__":
    print("=" * 70)
    print("Driver Monitoring System - Dataset Preparation")
    print("=" * 70)
    
    try:
        prepare_eye_state_dataset()
        prepare_drowsiness_dataset()
        prepare_yawn_dataset()
        create_summary()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()