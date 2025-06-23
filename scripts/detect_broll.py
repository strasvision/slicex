#!/usr/bin/env python3
"""
Detect B-Roll content in videos by identifying scenes without humans.
A scene is considered B-Roll if no humans are detected in any of its frames.
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Constants
MODEL_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence score to consider a detection valid
FRAME_SAMPLE_RATE = 1  # Sample 1 frame per second
HUMAN_CLASS_ID = 1  # COCO class ID for 'person'
OUTPUT_CSV = 'broll_detection.csv'

def load_model():
    """Load the pre-trained Faster R-CNN model."""
    print("Loading pre-trained Faster R-CNN model...")
    model = fasterrcnn_resnet50_fpn(pretrained=True, progress=True)
    model.eval()
    return model

def process_video(video_path, model, sample_rate=FRAME_SAMPLE_RATE):
    """Process a video file to detect if it contains humans."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to sample (1 frame per second by default)
    frame_indices = range(0, total_frames, max(1, int(fps * sample_rate)))
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        # Convert frame to tensor and add batch dimension
        img_tensor = F.to_tensor(frame).unsqueeze(0)
        
        # Run detection
        with torch.no_grad():
            predictions = model(img_tensor)
        
        # Check for human detections above confidence threshold
        for _, scores, labels in zip(
            predictions[0]['boxes'],
            predictions[0]['scores'],
            predictions[0]['labels']
        ):
            if labels.item() == HUMAN_CLASS_ID and scores.item() > MODEL_CONFIDENCE_THRESHOLD:
                cap.release()
                return True  # Human detected
    
    cap.release()
    return False  # No humans detected

def process_directory(input_dir, output_csv):
    """Process all video files in the input directory."""
    input_dir = Path(input_dir)
    video_files = list(input_dir.glob('*.mp4')) + list(input_dir.glob('*.MP4')) + \
                  list(input_dir.glob('*.mov')) + list(input_dir.glob('*.MOV'))
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} video files to process.")
    
    # Load model
    model = load_model()
    
    results = []
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        try:
            has_human = process_video(str(video_file), model)
            is_broll = not has_human
            results.append({
                'filename': video_file.name,
                'has_human': has_human,
                'is_broll': is_broll
            })
            print(f"Processed {video_file.name}: Human detected: {has_human}, B-Roll: {is_broll}")
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")
        print(f"Total videos: {len(df)}")
        print(f"Videos with humans: {df['has_human'].sum()}")
        print(f"B-Roll videos: {df['is_broll'].sum()}")
    else:
        print("No results to save.")

def main():
    parser = argparse.ArgumentParser(description='Detect B-Roll content in videos')
    parser.add_argument('--input_dir', default='../data/raw_videos',
                       help='Directory containing video files')
    parser.add_argument('--output_csv', default='../data/broll_detection.csv',
                       help='Output CSV file path')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    
    process_directory(args.input_dir, args.output_csv)

if __name__ == "__main__":
    main()
