#!/usr/bin/env python3
"""
Analyze motion in videos using OpenCV's optical flow.

This script calculates motion vectors and statistics for video files,
which can be used to understand camera movement and motion patterns.
"""

import os
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Configure argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Analyze motion in videos using optical flow")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="../data/raw_videos",
        help="Directory containing input video files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="../data/motion_analysis",
        help="Directory to save motion analysis results"
    )
    parser.add_argument(
        "--every_n_frames",
        type=int,
        default=5,
        help="Process every N frames to reduce computation (default: 5)"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=1000,
        help="Maximum number of frames to process per video (default: 1000)"
    )
    parser.add_argument(
        "--resize_factor",
        type=float,
        default=0.5,
        help="Factor to resize frames for faster processing (default: 0.5)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode to visualize optical flow"
    )
    return parser.parse_args()


def calculate_optical_flow(prev_gray: np.ndarray, frame_gray: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Calculate dense optical flow using Farneback's algorithm.
    
    Args:
        prev_gray: Previous frame in grayscale
        frame_gray: Current frame in grayscale
        
    Returns:
        Tuple of (flow, stats) where flow is the optical flow field and stats contains motion statistics
    """
    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, frame_gray, 
        None,  # flow
        0.5,  # pyr_scale
        3,  # levels
        15,  # winsize
        3,  # iterations
        5,  # poly_n
        1.2,  # poly_sigma
        0  # flags
    )
    
    # Calculate magnitude and angle of flow vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Calculate motion statistics
    stats = {
        'mean_magnitude': float(np.mean(magnitude)),
        'std_magnitude': float(np.std(magnitude)),
        'max_magnitude': float(np.max(magnitude)),
        'mean_angle': float(np.mean(angle)),
        'motion_energy': float(np.sum(magnitude ** 2)),
        'motion_hist': np.histogram(
            magnitude, 
            bins=10, 
            range=(0, np.percentile(magnitude, 95))  # Exclude outliers
        )[0].tolist()
    }
    
    return flow, stats


def analyze_video_motion(
    video_path: str, 
    every_n_frames: int = 5, 
    max_frames: int = 1000,
    resize_factor: float = 0.5,
    debug: bool = False
) -> Dict:
    """
    Analyze motion in a video file using optical flow.
    
    Args:
        video_path: Path to the video file
        every_n_frames: Process every N frames to reduce computation
        max_frames: Maximum number of frames to process
        resize_factor: Factor to resize frames for faster processing
        debug: Whether to visualize the optical flow
        
    Returns:
        Dictionary containing motion analysis results
    """
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return {}
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_factor)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_factor)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize variables
    prev_gray = None
    frame_count = 0
    processed_count = 0
    frame_stats = []
    
    # For visualization
    if debug:
        cv2.namedWindow('Optical Flow', cv2.WINDOW_NORMAL)
    
    # Skip first frame
    ret, frame = cap.read()
    if not ret:
        return {}
    
    # Resize and convert to grayscale
    frame = cv2.resize(frame, (width, height))
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize progress bar
    pbar = tqdm(total=min(total_frames, max_frames), desc=f"Analyzing {os.path.basename(video_path)}")
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        pbar.update(1)
        
        # Skip frames to reduce computation
        if frame_count % every_n_frames != 0:
            continue
            
        # Resize and convert to grayscale
        frame = cv2.resize(frame, (width, height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Skip if not enough frames have been processed yet
        if prev_gray is None:
            prev_gray = gray
            continue
        
        # Calculate optical flow
        flow, stats = calculate_optical_flow(prev_gray, gray)
        
        # Add frame-level stats
        frame_stats.append({
            'frame': frame_count,
            'timestamp': frame_count / fps,
            **stats
        })
        
        # For visualization
        if debug:
            # Create a visualization of the optical flow
            hsv = np.zeros_like(frame)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = angle * 180 / np.pi / 2  # Convert to degrees
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Show the result
            cv2.imshow('Optical Flow', flow_rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Update previous frame
        prev_gray = gray
        processed_count += 1
    
    # Clean up
    pbar.close()
    cap.release()
    if debug:
        cv2.destroyAllWindows()
    
    # Calculate video-level statistics
    if not frame_stats:
        return {}
    
    # Aggregate statistics across all frames
    magnitudes = [s['mean_magnitude'] for s in frame_stats]
    angles = [s['mean_angle'] for s in frame_stats]
    energies = [s['motion_energy'] for s in frame_stats]
    
    # Calculate motion statistics
    motion_stats = {
        'video_duration': total_frames / fps,
        'frames_analyzed': len(frame_stats),
        'mean_motion': float(np.mean(magnitudes)),
        'std_motion': float(np.std(magnitudes)),
        'max_motion': float(np.max(magnitudes)),
        'mean_energy': float(np.mean(energies)),
        'motion_variation': float(np.std(energies) / (np.mean(energies) + 1e-6)),
        'motion_hist': np.histogram(
            magnitudes, 
            bins=10, 
            range=(0, np.percentile(magnitudes, 95) if magnitudes else 0)
        )[0].tolist(),
        'frame_stats': frame_stats
    }
    
    return motion_stats


def process_videos(
    input_dir: str, 
    output_dir: str, 
    every_n_frames: int = 5,
    max_frames: int = 1000,
    resize_factor: float = 0.5,
    debug: bool = False
) -> None:
    """
    Process all videos in the input directory and save motion analysis results.
    
    Args:
        input_dir: Directory containing input video files
        output_dir: Directory to save motion analysis results
        every_n_frames: Process every N frames to reduce computation
        max_frames: Maximum number of frames to process per video
        resize_factor: Factor to resize frames for faster processing
        debug: Whether to visualize the optical flow
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all video files in the input directory
    video_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} video files to process\n")
    
    all_results = {}
    
    # Process each video
    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        print(f"\nProcessing: {video_file}")
        
        # Analyze motion in the video
        motion_stats = analyze_video_motion(
            video_path, 
            every_n_frames=every_n_frames,
            max_frames=max_frames,
            resize_factor=resize_factor,
            debug=debug
        )
        
        if not motion_stats:
            print(f"Skipped {video_file} - no motion data")
            continue
        
        # Save individual results
        output_file = os.path.splitext(video_file)[0] + "_motion.json"
        output_path = os.path.join(output_dir, output_file)
        
        with open(output_path, 'w') as f:
            json.dump(motion_stats, f, indent=2)
        
        print(f"✓ Saved motion analysis to {output_path}")
        
        # Store for combined results
        all_results[video_file] = motion_stats
    
    # Save combined results
    if all_results:
        combined_path = os.path.join(output_dir, "combined_motion_analysis.json")
        with open(combined_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nSaved combined motion analysis to {combined_path}")
    
    print("\nMotion analysis complete!")


if __name__ == "__main__":
    args = parse_args()
    
    print(f"Starting motion analysis with settings:")
    print(f"- Input directory: {args.input_dir}")
    print(f"- Output directory: {args.output_dir}")
    print(f"- Processing every {args.every_n_frames} frames")
    print(f"- Max frames per video: {args.max_frames}")
    print(f"- Resize factor: {args.resize_factor}")
    print(f"- Debug mode: {'ON' if args.debug else 'OFF'}\n")
    
    process_videos(
        args.input_dir,
        args.output_dir,
        every_n_frames=args.every_n_frames,
        max_frames=args.max_frames,
        resize_factor=args.resize_factor,
        debug=args.debug
    )
