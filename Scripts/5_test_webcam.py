"""
Step 5: Test Driver Monitoring System on Webcam
------------------------------------------------
Real-time testing using webcam before deploying to OAK-D.

Run: python Scripts/5_test_webcam.py
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from collections import deque
import time

# -------------------------
# Model Definition
# -------------------------
class DMSModel(nn.Module):
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

# -------------------------
# DMS System
# -------------------------
class DriverMonitoringSystem:
    def __init__(self, img_size=112, device='cpu'):
        self.device = device
        self.img_size = img_size
        
        # Load models
        print("Loading models...")
        self.eye_model = self.load_model('./runs/eye_state/best_model.pth')
        self.yawn_model = self.load_model('./runs/yawn/best_model.pth')
        self.drowsy_model = self.load_model('./runs/drowsiness/best_model.pth')
        
        # Face detector (Haar Cascade - lightweight)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Eye detector
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Temporal smoothing (reduce false alarms)
        self.eye_history = deque(maxlen=10)
        self.yawn_history = deque(maxlen=20)
        self.drowsy_history = deque(maxlen=15)
        
        # Alert thresholds
        self.eye_closed_threshold = 0.7
        self.yawn_threshold = 0.4
        self.drowsy_threshold = 0.6
    
    def load_model(self, checkpoint_path):
        """Load trained model from checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            num_classes = len(checkpoint['classes'])
            
            model = DMSModel(num_classes=num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            print(f"  ✓ Loaded {checkpoint_path}")
            print(f"    Classes: {checkpoint['classes']}, Acc: {checkpoint['val_acc']*100:.1f}%")
            return model
        except Exception as e:
            print(f"  ⚠ Failed to load {checkpoint_path}: {e}")
            return None
    
    @torch.no_grad()
    def predict(self, model, image):
        """Run inference on a single image"""
        if model is None:
            return 0, 0.0
        
        try:
            # Preprocess
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Inference
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)
            
            return pred.item(), confidence.item()
        except Exception as e:
            return 0, 0.0
    
    def detect_drowsiness(self, frame):
        """Main detection pipeline"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
        
        if len(faces) == 0:
            return frame, "No Face Detected", (0, 0, 255)
        
        # Process largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        
        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract face ROI
        face_roi = rgb[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]
        
        # Detect eyes within face
        eyes = self.eye_cascade.detectMultiScale(face_gray, 1.1, 5, minSize=(20, 20))
        
        # Analyze face for drowsiness
        drowsy_pred, drowsy_conf = self.predict(self.drowsy_model, face_roi)
        self.drowsy_history.append(drowsy_pred)
        
        # Analyze for yawning
        yawn_pred, yawn_conf = self.predict(self.yawn_model, face_roi)
        self.yawn_history.append(yawn_pred)
        
        # Eye state detection
        eye_closed_count = 0
        if len(eyes) > 0:
            for (ex, ey, ew, eh) in eyes[:2]:
                # Draw eye boxes
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 1)
                
                # Extract eye ROI
                eye_roi = rgb[y+ey:y+ey+eh, x+ex:x+ex+ew]
                if eye_roi.size > 0:
                    eye_pred, eye_conf = self.predict(self.eye_model, eye_roi)
                    
                    # Debug: Print prediction
                    print(f"Eye prediction: {eye_pred}, confidence: {eye_conf:.2f}")
                    
                    # Reverse the logic if needed
                    # If class 0 = closed, class 1 = open, then reverse:
                    if eye_pred == 0:  # Changed from 1 to 0
                        eye_closed_count += 1
                    
                    # Display eye state
                    state_text = "Closed" if eye_pred == 0 else "Open"  # Changed
                    cv2.putText(frame, state_text, (x+ex, y+ey-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Record eye state (True if eyes closed)
        eyes_closed = eye_closed_count >= len(eyes) if len(eyes) > 0 else False
        self.eye_history.append(1 if eyes_closed else 0)
        
        # Calculate temporal averages
        eye_closed_ratio = sum(self.eye_history) / len(self.eye_history) if self.eye_history else 0
        yawn_ratio = sum(self.yawn_history) / len(self.yawn_history) if self.yawn_history else 0
        drowsy_ratio = sum(self.drowsy_history) / len(self.drowsy_history) if self.drowsy_history else 0
        
        # Determine alert status
        if eye_closed_ratio > self.eye_closed_threshold:
            alert_text = "⚠ EYES CLOSED - WAKE UP!"
            alert_color = (0, 0, 255)  # Red
        elif yawn_ratio > self.yawn_threshold:
            alert_text = "⚠ YAWNING - FATIGUE!"
            alert_color = (0, 165, 255)  # Orange
        elif drowsy_ratio > self.drowsy_threshold:
            alert_text = "⚠ DROWSINESS DETECTED!"
            alert_color = (0, 165, 255)  # Orange
        else:
            alert_text = "ALERT - SAFE DRIVING"
            alert_color = (0, 255, 0)  # Green
        
        # Display stats
        stats_y = 30
        cv2.putText(frame, f"Eyes Closed: {eye_closed_ratio*100:.0f}%", (10, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Yawn: {yawn_ratio*100:.0f}%", (10, stats_y+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Drowsy: {drowsy_ratio*100:.0f}%", (10, stats_y+60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame, alert_text, alert_color
    
    def run(self, camera_id=0):
        """Run real-time monitoring"""
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        print("\n" + "="*70)
        print("Driver Monitoring System - RUNNING")
        print("="*70)
        print("Press 'q' to quit")
        print("="*70 + "\n")
        
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            # Process frame
            processed_frame, alert_text, alert_color = self.detect_drowsiness(frame)
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            # Display FPS
            cv2.putText(processed_frame, f"FPS: {fps}", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display alert banner
            cv2.rectangle(processed_frame, (0, frame.shape[0]-60), 
                         (frame.shape[1], frame.shape[0]), alert_color, -1)
            cv2.putText(processed_frame, alert_text, (10, frame.shape[0]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Driver Monitoring System', processed_frame)
            
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Monitoring stopped")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test DMS on Webcam')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--img-size', type=int, default=112,
                       help='Image size for models')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    try:
        dms = DriverMonitoringSystem(img_size=args.img_size, device=args.device)
        dms.run(camera_id=args.camera)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have trained models:")
        print("  python Scripts/3_train_models.py --model all")
    except KeyboardInterrupt:
        print("\n\n✓ Stopped by user")