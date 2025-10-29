"""
Step 3: Train Driver Monitoring Models
---------------------------------------
Trains 3 specialized models:
1. Eye State Classifier (Open/Closed)
2. Yawn Detector (Yawn/No-Yawn)
3. Drowsiness Detector (Alert/Drowsy)

Each uses EfficientNet-B0 backbone with transfer learning.

Run: python 3_train_models.py --model eye_state
     python 3_train_models.py --model yawn
     python 3_train_models.py --model drowsiness
     python 3_train_models.py --model all  (trains all 3)
"""

import os
import time
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# -------------------------
# Dataset Loader
# -------------------------
class DMSDataset(Dataset):
    """Generic dataset loader for DMS tasks"""
    def __init__(self, root, classes, img_size=112, augment=True):
        self.paths = []
        self.labels = []
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        
        for cls in classes:
            cls_path = os.path.join(root, cls)
            if not os.path.exists(cls_path):
                print(f"Warning: {cls_path} does not exist")
                continue
            
            for fname in os.listdir(cls_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.paths.append(os.path.join(cls_path, fname))
                    self.labels.append(self.class_to_idx[cls])
        
        print(f"  Loaded {len(self.paths)} images from {root}")
        
        # Normalization for pretrained models
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
        
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                norm,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                norm,
            ])
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        img = self.transform(img)
        label = self.labels[idx]
        return img, label

# -------------------------
# Model Architecture
# -------------------------
class DMSModel(nn.Module):
    """EfficientNet-B0 based classifier for DMS tasks"""
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        # Use EfficientNet-B0 (lighter than MobileNet for edge devices)
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# -------------------------
# Training Functions
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device, class_names=None):
    """Evaluate model and return metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    accuracy = correct / total
    
    # Print detailed metrics
    if class_names:
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, 
                                    target_names=class_names, 
                                    digits=4))
    
    return accuracy, all_labels, all_preds

def train_model(model_name, epochs=25, batch_size=64, lr=1e-3, img_size=112):
    """Train a specific DMS model"""
    
    # Define paths and classes for each model type
    configs = {
        'eye_state': {
            'classes': ['open', 'closed'],
            'data_root': './datasets_processed/eye_state',
            'save_dir': './runs/eye_state'
        },
        'yawn': {
            'classes': ['no_yawn', 'yawn'],
            'data_root': './datasets_processed/yawn',
            'save_dir': './runs/yawn'
        },
        'drowsiness': {
            'classes': ['alert', 'drowsy'],
            'data_root': './datasets_processed/drowsiness',
            'save_dir': './runs/drowsiness'
        }
    }
    
    if model_name not in configs:
        raise ValueError(f"Model {model_name} not recognized")
    
    config = configs[model_name]
    classes = config['classes']
    data_root = config['data_root']
    save_dir = config['save_dir']
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Training {model_name.upper()} Model")
    print(f"{'='*70}")
    
    # Check if data exists
    if not os.path.exists(data_root):
        print(f"❌ Data not found at {data_root}")
        print(f"   Run: python 2_prepare_datasets.py")
        return
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = DMSDataset(f"{data_root}/train", classes, img_size, augment=True)
    val_dataset = DMSDataset(f"{data_root}/val", classes, img_size, augment=False)
    
    if len(train_dataset) == 0:
        print(f"❌ No training data found!")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = DMSModel(num_classes=len(classes), dropout=0.3)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Training loop
    best_acc = 0.0
    best_model_path = f"{save_dir}/best_model.pth"
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Progress bar
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}", end='\r')
        
        # Epoch statistics
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation
        val_acc, _, _ = evaluate(model, val_loader, device)
        
        # Learning rate step
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch [{epoch}/{epochs}] "
              f"Loss: {train_loss:.4f} "
              f"Train Acc: {train_acc*100:.2f}% "
              f"Val Acc: {val_acc*100:.2f}% "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'classes': classes,
                'img_size': img_size
            }, best_model_path)
            print(f"  ✓ Best model saved! (Val Acc: {val_acc*100:.2f}%)")
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_acc*100:.2f}%")
    print(f"Model saved to: {best_model_path}")
    print(f"{'='*70}")
    
    # Plot training history
    plot_history(history, save_dir, model_name)
    
    # Final evaluation with detailed metrics
    print(f"\nFinal Evaluation on Validation Set:")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    val_acc, labels, preds = evaluate(model, val_loader, device, classes)
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, classes, save_dir, model_name)
    
    return best_acc

def plot_history(history, save_dir, model_name):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Training Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_history.png")
    plt.close()
    print(f"  Training history saved to {save_dir}/training_history.png")

def plot_confusion_matrix(cm, classes, save_dir, model_name):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png")
    plt.close()
    print(f"  Confusion matrix saved to {save_dir}/confusion_matrix.png")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DMS Models')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['eye_state', 'yawn', 'drowsiness', 'all'],
                       help='Which model to train')
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--img-size', type=int, default=112,
                       help='Image size')
    
    args = parser.parse_args()
    
    # Install required packages
    try:
        import matplotlib
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError:
        print("Installing required packages...")
        os.system("pip install matplotlib scikit-learn -q")
    
    print("=" * 70)
    print("Driver Monitoring System - Model Training")
    print("=" * 70)
    
    if args.model == 'all':
        models_to_train = ['eye_state', 'yawn', 'drowsiness']
        results = {}
        
        for model_name in models_to_train:
            try:
                acc = train_model(model_name, args.epochs, args.batch_size, 
                                args.lr, args.img_size)
                results[model_name] = acc
            except Exception as e:
                print(f"\n❌ Error training {model_name}: {e}")
                results[model_name] = 0.0
        
        # Summary
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        for model_name, acc in results.items():
            print(f"{model_name:15s}: {acc*100:.2f}% validation accuracy")
        print("=" * 70)
        print("\nNext step: python 4_export_onnx.py")
        
    else:
        train_model(args.model, args.epochs, args.batch_size, 
                   args.lr, args.img_size)
        print("\nNext step: python 4_export_onnx.py")