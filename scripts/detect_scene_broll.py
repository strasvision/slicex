#!/usr/bin/env python3
"""
Detect B-Roll content in video scenes by identifying scenes without humans/avatars.
This script analyzes each scene separately using pre-computed scene timestamps.
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import json
import argparse
from datetime import datetime

# Constants
MODEL_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence score to consider a detection valid
FRAMES_PER_SCENE = 3  # Number of frames to sample per scene for detection
OUTPUT_CSV = 'scene_broll_detection.csv'

def load_model():
    """Load the pre-trained Faster R-CNN model."""
    print("Loading pre-trained Faster R-CNN model...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()
    return model

def detect_humans_in_frame(model, frame):
    """Detect humans in a single frame."""
    # Convert frame to RGB (YOLOv5 expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run detection
    results = model(rgb_frame)
    
    # Check for person class (class 0 in YOLOv5)
    detections = results.xyxy[0]  # x1, y1, x2, y2, confidence, class
    person_detections = [d for d in detections if int(d[5]) == 0 and d[4] >= MODEL_CONFIDENCE_THRESHOLD]
    
    return len(person_detections) > 0

def process_scene(video_path, start_time, end_time, model):
    """Process a single scene to detect if it contains humans."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Convert times to frame numbers
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), total_frames - 1)
    
    # Sample frames from the scene
    frame_indices = np.linspace(start_frame, end_frame, min(FRAMES_PER_SCENE, end_frame - start_frame + 1), dtype=int)
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        # Check for humans in this frame
        if detect_humans_in_frame(model, frame):
            cap.release()
            return True  # Human detected in this scene
    
    cap.release()
    return False  # No humans detected in any sampled frame

def process_metadata(metadata_path, output_dir):
    """Process all videos in the metadata file for scene-based B-roll detection."""
    # Load the metadata
    metadata_df = pd.read_csv(metadata_path)
    
    # Initialize YOLOv5 model
    model = load_model()
    
    results = []
    
    # Process each video
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing videos"):
        try:
            video_path = row['file_path']
            if not os.path.exists(video_path):
                print(f"Video not found: {video_path}")
                continue
                
            # Extract scene information
            scene_cols = [col for col in row.index if 'scene_list_' in col and '_scene_number' in col]
            num_scenes = len(scene_cols)
            
            for i in range(num_scenes):
                scene_num = i + 1
                start_time_col = f'processing_scene_detection_scene_list_{i}_start_time'
                end_time_col = f'processing_scene_detection_scene_list_{i}_end_time'
                
                if start_time_col not in row or pd.isna(row[start_time_col]) or \
                   end_time_col not in row or pd.isna(row[end_time_col]):
                    continue
                
                start_time = float(row[start_time_col])
                end_time = float(row[end_time_col])
                
                # Skip very short scenes
                if end_time - start_time < 0.5:  # Less than 0.5 seconds
                    continue
                
                # Check for humans in this scene
                has_human = process_scene(video_path, start_time, end_time, model)
                
                results.append({
                    'filename': os.path.basename(video_path),
                    'scene_number': scene_num,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'has_human': has_human,
                    'is_broll': not has_human
                })
                
                print(f"Processed {os.path.basename(video_path)} - Scene {scene_num}: "
                      f"{start_time:.2f}s to {end_time:.2f}s - "
                      f"{'Human' if has_human else 'B-Roll'}")
                
        except Exception as e:
            print(f"Error processing {row.get('filename', 'unknown')}: {str(e)}")
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        output_path = os.path.join(output_dir, f"scene_{OUTPUT_CSV}")
        df.to_csv(output_path, index=False)
        print(f"\nScene-based B-roll detection complete!")
        print(f"Results saved to: {output_path}")
        print(f"Total scenes processed: {len(df)}")
        print(f"B-Roll scenes: {df['is_broll'].sum()} ({(df['is_broll'].mean()*100):.1f}%)")
        
        return output_path
    else:
        print("No results to save.")
        return None

def main():
    parser = argparse.ArgumentParser(description='Detect B-Roll content in video scenes')
    parser.add_argument('--metadata', default='../data/processed/video_metadata_20250621_185934.csv',
                       help='Path to the video metadata CSV file')
    parser.add_argument('--output_dir', default='../data/processed',
                       help='Directory to save the output CSV')
    
    args = parser.parse_args()
    
    # Ensure paths are absolute
    metadata_path = os.path.abspath(args.metadata)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting scene-based B-roll detection...")
    print(f"Metadata file: {metadata_path}")
    print(f"Output directory: {output_dir}")
    
    process_metadata(metadata_path, output_dir)

if __name__ == "__main__":
    main()
