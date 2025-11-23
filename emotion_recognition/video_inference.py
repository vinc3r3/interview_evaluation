#!/usr/bin/env python3
"""
Video Emotion Inference - Standalone Script for VSCode
Run this script to analyze emotions in videos using your trained model.

Usage:
    python video_inference.py --model best_model_fixed.pth --video test_video.mp4
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import pandas as pd
from tqdm import tqdm
import argparse
import os
from datetime import datetime

# ============================================================================
# MODEL ARCHITECTURE (Must match training!)
# ============================================================================

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.feature_dim = 2048
        
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        return self.features(x)


class SpatialTransformerEncoder(nn.Module):
    def __init__(self, feature_dim=2048, embed_dim=512, num_heads=8, 
                 num_layers=4, dropout=0.2):
        super().__init__()
        
        self.feature_projection = nn.Linear(feature_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 49, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            activation='gelu', batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.feature_projection(x)
        x = x + self.pos_embedding[:, :H*W, :]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.dropout(x)
        x = self.transformer(x)
        
        return x[:, 0]


class ResNetViTHybrid(nn.Module):
    def __init__(self, num_classes=7, pretrained_resnet=True, freeze_resnet=True,
                 embed_dim=512, num_heads=8, num_layers=4):
        super().__init__()
        
        self.resnet_extractor = ResNetFeatureExtractor(
            pretrained=pretrained_resnet, freeze_backbone=freeze_resnet
        )
        
        self.transformer_encoder = SpatialTransformerEncoder(
            feature_dim=2048, embed_dim=embed_dim, num_heads=num_heads,
            num_layers=num_layers, dropout=0.2
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.resnet_extractor(x)
        encoded = self.transformer_encoder(features)
        logits = self.classifier(encoded)
        return logits


# ============================================================================
# VIDEO EMOTION DETECTOR
# ============================================================================

class VideoEmotionDetector:
    EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def __init__(self, model_path, device='cuda', image_size=224):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = ResNetViTHybrid(
            num_classes=7, freeze_resnet=True,
            embed_dim=512, num_heads=8, num_layers=4
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("✓ Model loaded successfully!\n")
        
        # Face detector
        print("Initializing face detector...")
        self.face_detector = MTCNN(
            image_size=image_size, margin=20, keep_all=False,
            device=self.device, post_process=False
        )
        print("✓ Face detector ready!\n")
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def detect_face(self, frame):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # ⭐ ADD THIS: Convert to grayscale then back to 3-channel (match training!)
            frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            frame_rgb = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)
            # ⭐ END ADD
            boxes, probs = self.face_detector.detect(frame_rgb)
            
            if boxes is not None and len(boxes) > 0:
                box = boxes[0].astype(int)
                h, w = frame_rgb.shape[:2]
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
                face = frame_rgb[y1:y2, x1:x2]
                
                if face.size > 0:
                    return Image.fromarray(face)
            return None
        except:
            return None
    
    def predict_emotion(self, face_image):
        if face_image is None:
            return None
        
        try:
            img_tensor = self.transform(face_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()[0]
        except Exception as e:
            return None
    
    def process_video(self, video_path, sample_rate=10, output_csv=None):
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print("=" * 70)
        print("VIDEO INFORMATION")
        print("=" * 70)
        print(f"FPS: {fps:.2f}")
        print(f"Total Frames: {total_frames:,}")
        print(f"Duration: {duration:.2f}s ({duration/60:.2f} min)")
        print(f"Sample Rate: Every {sample_rate} frame(s)")
        print(f"Frames to Process: {total_frames//sample_rate:,}")
        print("=" * 70 + "\n")
        
        results = []
        frame_idx = 0
        faces_detected = 0
        
        pbar = tqdm(total=total_frames//sample_rate, desc="Processing", unit="frame")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                timestamp = frame_idx / fps
                face = self.detect_face(frame)
                
                if face is not None:
                    probs = self.predict_emotion(face)
                    
                    if probs is not None:
                        result = {'timestamp': timestamp, 'frame_number': frame_idx}
                        for i, label in enumerate(self.EMOTION_LABELS):
                            result[label] = float(probs[i])
                        results.append(result)
                        faces_detected += 1
                
                pbar.update(1)
            frame_idx += 1
        
        pbar.close()
        cap.release()
        
        print("\n" + "=" * 70)
        print("PROCESSING COMPLETE")
        print("=" * 70)
        print(f"Frames processed: {total_frames//sample_rate:,}")
        print(f"Faces detected: {faces_detected:,}")
        print(f"Detection rate: {(faces_detected/(total_frames//sample_rate)*100):.1f}%")
        print("=" * 70 + "\n")
        
        if len(results) == 0:
            print("⚠️  No faces detected in the video!")
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df['dominant_emotion'] = df[self.EMOTION_LABELS].idxmax(axis=1)
        df['confidence'] = df[self.EMOTION_LABELS].max(axis=1)
        
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"✓ Results saved to: {output_csv}\n")
        
        return df
    
    def print_summary(self, df):
        if len(df) == 0:
            return
        
        emotion_pct = df['dominant_emotion'].value_counts(normalize=True) * 100
        avg_conf = df['confidence'].mean()
        most_common = df['dominant_emotion'].mode()[0]
        
        print("=" * 70)
        print("EMOTION SUMMARY")
        print("=" * 70)
        print(f"\nTotal detections: {len(df)}")
        print(f"Average confidence: {avg_conf:.1%}")
        print(f"Most common emotion: {most_common}")
        
        print("\nEmotion Distribution:")
        for emotion, percentage in sorted(emotion_pct.items(), key=lambda x: x[1], reverse=True):
            bar = '█' * int(percentage / 2)
            print(f"  {emotion:10s}: {percentage:5.1f}%  {bar}")
        
        print("=" * 70 + "\n")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze emotions in video')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--sample-rate', type=int, default=10, help='Process every Nth frame (default: 10)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file (optional)')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu (default: cuda)')
    
    args = parser.parse_args()
    
    # Verify files exist
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file not found: {args.video}")
        return
    
    # Auto-generate output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        args.output = f'emotions_{video_name}_{timestamp}.csv'
    
    print("\n" + "=" * 70)
    print("VIDEO EMOTION ANALYZER")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Video: {args.video}")
    print(f"Sample rate: Every {args.sample_rate} frame(s)")
    print(f"Output: {args.output}")
    print("=" * 70 + "\n")
    
    # Initialize detector
    detector = VideoEmotionDetector(args.model, device=args.device)
    
    # Process video
    results = detector.process_video(args.video, sample_rate=args.sample_rate, output_csv=args.output)
    
    # Print summary
    detector.print_summary(results)
    
    if len(results) > 0:
        print(f"✓ Analysis complete!")
        print(f"✓ Results saved to: {args.output}")
        print(f"\nFirst 5 predictions:")
        print(results[['timestamp', 'dominant_emotion', 'confidence']].head())


if __name__ == '__main__':
    main()
