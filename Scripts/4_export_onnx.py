"""
Step 4: Export Models to ONNX for OAK-D Deployment
---------------------------------------------------
Exports trained PyTorch models to ONNX format for edge deployment.

Run: python Scripts/4_export_onnx.py --model all
"""

import os
import argparse
import torch
import torch.nn as nn
from torchvision import models

class DMSModel(nn.Module):
    """EfficientNet-B0 based classifier"""
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def export_to_onnx(model_name, img_size=112):
    """Export a trained model to ONNX format"""
    
    model_paths = {
        'eye_state': './runs/eye_state/best_model.pth',
        'yawn': './runs/yawn/best_model.pth',
        'drowsiness': './runs/drowsiness/best_model.pth'
    }
    
    output_paths = {
        'eye_state': './exports/eye_state_model.onnx',
        'yawn': './exports/yawn_model.onnx',
        'drowsiness': './exports/drowsiness_model.onnx'
    }
    
    if model_name not in model_paths:
        print(f"❌ Unknown model: {model_name}")
        return False
    
    checkpoint_path = model_paths[model_name]
    onnx_path = output_paths[model_name]
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"   Train the model first: python Scripts/3_train_models.py --model {model_name}")
        return False
    
    print(f"\n[Exporting {model_name}]")
    print(f"  Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    num_classes = len(checkpoint['classes'])
    
    # Create model
    model = DMSModel(num_classes=num_classes, dropout=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    # Create export directory
    os.makedirs('./exports', exist_ok=True)
    
    # Export to ONNX
    print(f"  Exporting to: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"  ✓ Exported successfully!")
    print(f"  Classes: {checkpoint['classes']}")
    print(f"  Val Accuracy: {checkpoint['val_acc']*100:.2f}%")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"  ✓ ONNX model verified")
    except ImportError:
        print("  ⚠ onnx package not installed, skipping verification")
    except Exception as e:
        print(f"  ⚠ ONNX verification warning: {e}")
    
    return True

def create_blob_for_oakd(onnx_path, output_path):
    """
    Convert ONNX to .blob format for OAK-D
    Requires: blobconverter package
    """
    try:
        import blobconverter
        
        print(f"\n  Converting to .blob for OAK-D...")
        blob_path = blobconverter.from_onnx(
            model=onnx_path,
            output_dir=os.path.dirname(output_path),
            data_type="FP16",
            shaves=6,
            version="2021.4"
        )
        print(f"  ✓ Blob created: {blob_path}")
        return blob_path
    except ImportError:
        print("\n  ⚠ blobconverter not installed")
        print("  Install with: pip install blobconverter")
        print("  Or convert online: https://blobconverter.luxonis.com/")
        return None
    except Exception as e:
        print(f"  ⚠ Blob conversion failed: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export DMS Models to ONNX')
    parser.add_argument('--model', type=str, default='all',
                       choices=['eye_state', 'yawn', 'drowsiness', 'all'],
                       help='Which model to export')
    parser.add_argument('--img-size', type=int, default=112,
                       help='Image size')
    parser.add_argument('--create-blob', action='store_true',
                       help='Also create .blob files for OAK-D')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Driver Monitoring System - ONNX Export")
    print("=" * 70)
    
    models_to_export = ['eye_state', 'yawn', 'drowsiness'] if args.model == 'all' else [args.model]
    
    success_count = 0
    for model_name in models_to_export:
        if export_to_onnx(model_name, args.img_size):
            success_count += 1
            
            # Optionally create .blob
            if args.create_blob:
                onnx_path = f'./exports/{model_name}_model.onnx'
                blob_path = f'./exports/{model_name}_model.blob'
                create_blob_for_oakd(onnx_path, blob_path)
    
    print("\n" + "=" * 70)
    print(f"Export Summary: {success_count}/{len(models_to_export)} models exported")
    print("=" * 70)
    
    if success_count > 0:
        print("\nExported models:")
        for model_name in models_to_export:
            onnx_file = f'./exports/{model_name}_model.onnx'
            if os.path.exists(onnx_file):
                size_mb = os.path.getsize(onnx_file) / (1024 * 1024)
                print(f"  {model_name:15s}: {onnx_file} ({size_mb:.2f} MB)")
        
        print("\nNext steps:")
        print("  1. Test on webcam: python Scripts/5_test_webcam.py")
        print("  2. Deploy to OAK-D: python Scripts/6_deploy_oakd.py")
        
        if not args.create_blob:
            print("\nTo create .blob files for OAK-D:")
            print("  python Scripts/4_export_onnx.py --model all --create-blob")
    else:
        print("\n❌ No models exported. Train models first:")
        print("  python Scripts/3_train_models.py --model all")