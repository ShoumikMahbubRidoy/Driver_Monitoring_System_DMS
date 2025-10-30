"""
Step 2: Prepare and Organize Datasets
--------------------------------------
Adapted for actual Kaggle dataset structures
"""

import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

random.seed(42)
np.random.seed(42)

DATASET_ROOT = "./datasets"
PROCESSED_ROOT = "./datasets_processed"

def prepare_drowsiness_dataset():
    """Prepare Drowsiness Dataset from train folder"""
    print("\n[1/3] Preparing Drowsiness Dataset...")
    
    source = f"{DATASET_ROOT}/drowsiness/train"
    output = f"{PROCESSED_ROOT}/drowsiness"
    
    if not os.path.exists(source):
        print(f"⚠ Drowsiness dataset not found at {source}")
        return
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        for cls in ['alert', 'drowsy']:
            os.makedirs(f"{output}/{split}/{cls}", exist_ok=True)
    
    alert_imgs = []
    drowsy_imgs = []
    
    # Collect Open eyes (alert)
    open_path = os.path.join(source, "Open")
    if os.path.exists(open_path):
        for file in os.listdir(open_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                alert_imgs.append(os.path.join(open_path, file))
    
    # Collect no_yawn (alert)
    no_yawn_path = os.path.join(source, "no_yawn")
    if os.path.exists(no_yawn_path):
        for file in os.listdir(no_yawn_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                alert_imgs.append(os.path.join(no_yawn_path, file))
    
    # Collect Closed eyes (drowsy)
    closed_path = os.path.join(source, "Closed")
    if os.path.exists(closed_path):
        for file in os.listdir(closed_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                drowsy_imgs.append(os.path.join(closed_path, file))
    
    # Collect yawn (drowsy)
    yawn_path = os.path.join(source, "yawn")
    if os.path.exists(yawn_path):
        for file in os.listdir(yawn_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                drowsy_imgs.append(os.path.join(yawn_path, file))
    
    print(f"Found {len(alert_imgs)} alert, {len(drowsy_imgs)} drowsy images")
    
    def split_and_copy(images, class_name):
        if len(images) == 0:
            print(f"  Warning: No images for {class_name}")
            return
        
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
    
    split_and_copy(alert_imgs, 'alert')
    split_and_copy(drowsy_imgs, 'drowsy')
    print("✓ Drowsiness dataset prepared")

def prepare_eye_state_dataset():
    """Prepare Eye State from Driver Drowsiness Dataset (DDD)"""
    print("\n[2/3] Preparing Eye State Dataset...")
    
    source = f"{DATASET_ROOT}/eye_state/Driver Drowsiness Dataset (DDD)"
    output = f"{PROCESSED_ROOT}/eye_state"
    
    if not os.path.exists(source):
        print(f"⚠ Eye state dataset not found at {source}")
        return
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        for cls in ['open', 'closed']:
            os.makedirs(f"{output}/{split}/{cls}", exist_ok=True)
    
    open_imgs = []
    closed_imgs = []
    
    # Map: Non Drowsy → open, Drowsy → closed
    non_drowsy_path = os.path.join(source, "Non Drowsy")
    drowsy_path = os.path.join(source, "Drowsy")
    
    if os.path.exists(non_drowsy_path):
        for root, dirs, files in os.walk(non_drowsy_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    open_imgs.append(os.path.join(root, file))
    
    if os.path.exists(drowsy_path):
        for root, dirs, files in os.walk(drowsy_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    closed_imgs.append(os.path.join(root, file))
    
    print(f"Found {len(open_imgs)} open, {len(closed_imgs)} closed eyes")
    
    def split_and_copy(images, class_name):
        if len(images) == 0:
            print(f"  Warning: No images for {class_name}")
            return
        
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

def prepare_yawn_dataset():
    """Prepare Yawn Dataset"""
    print("\n[3/3] Preparing Yawn Dataset...")
    
    source = f"{DATASET_ROOT}/yawn/dataset_new"
    output = f"{PROCESSED_ROOT}/yawn"
    
    if not os.path.exists(source):
        print(f"⚠ Yawn dataset not found at {source}")
        return
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        for cls in ['yawn', 'no_yawn']:
            os.makedirs(f"{output}/{split}/{cls}", exist_ok=True)
    
    yawn_imgs = []
    no_yawn_imgs = []
    
    # Process both train and test folders
    for subfolder in ['train', 'test']:
        subfolder_path = os.path.join(source, subfolder)
        
        # Collect yawn images
        yawn_path = os.path.join(subfolder_path, "yawn")
        if os.path.exists(yawn_path):
            for file in os.listdir(yawn_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    yawn_imgs.append(os.path.join(yawn_path, file))
        
        # Collect no_yawn images
        no_yawn_path = os.path.join(subfolder_path, "no_yawn")
        if os.path.exists(no_yawn_path):
            for file in os.listdir(no_yawn_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    no_yawn_imgs.append(os.path.join(no_yawn_path, file))
    
    print(f"Found {len(yawn_imgs)} yawn, {len(no_yawn_imgs)} no-yawn images")
    
    def split_and_copy(images, class_name):
        if len(images) == 0:
            print(f"  Warning: No images for {class_name}")
            return
        
        random.shuffle(images)
        train_imgs, temp = train_test_split(images, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(temp, test_size=0.5, random_state=42)
        
        # Limit size if needed
        train_imgs = train_imgs[:min(len(train_imgs), 5000)]
        val_imgs = val_imgs[:min(len(val_imgs), 1000)]
        test_imgs = test_imgs[:min(len(test_imgs), 1000)]
        
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
    
    split_and_copy(yawn_imgs, 'yawn')
    split_and_copy(no_yawn_imgs, 'no_yawn')
    print("✓ Yawn dataset prepared")

def create_summary():
    """Create summary"""
    print("\n" + "=" * 70)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 70)
    
    for ds in ['eye_state', 'drowsiness', 'yawn']:
        print(f"\n{ds.upper()}:")
        for split in ['train', 'val', 'test']:
            path = f"{PROCESSED_ROOT}/{ds}/{split}"
            if os.path.exists(path):
                total = sum([len(files) for _, _, files in os.walk(path)])
                print(f"  {split}: {total} images")
    
    print("\n" + "=" * 70)
    print("✓ All datasets prepared!")
    print("\nNext step: python Scripts/3_train_models.py --model all --epochs 25")
    print("=" * 70)

if __name__ == "__main__":
    print("=" * 70)
    print("Driver Monitoring System - Dataset Preparation")
    print("=" * 70)
    
    try:
        prepare_drowsiness_dataset()
        prepare_eye_state_dataset()
        prepare_yawn_dataset()
        create_summary()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()