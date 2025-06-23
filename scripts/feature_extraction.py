"""
Feature extraction for video style prediction.

This module provides functions to extract features from videos that are used
for training and prediction in the video style classification pipeline.
"""

import os
import cv2
import numpy as np
import pandas as pd
import librosa
from scenedetect import VideoManager, SceneManager, ContentDetector
from scenedetect.scene_manager import save_images
from scenedetect.video_splitter import split_video_ffmpeg
from scenedetect.frame_timecode import FrameTimecode
import subprocess
import tempfile
import json
from pathlib import Path
from tqdm import tqdm

# Constants
FRAME_RATE = 30  # Default frame rate if not detected
MIN_SCENE_LEN = 15  # Minimum scene length in frames
AUDIO_SAMPLE_RATE = 22050  # Target sample rate for audio analysis
HOP_LENGTH = 512  # Hop length for audio analysis
N_FFT = 2048  # FFT window size for audio analysis

def extract_features(video_path):
    """
    Extract all features from a video file.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        dict: Dictionary containing all extracted features
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Initialize result dictionary with all required features
    features = {
        'video_path': video_path,
        'duration': 0,
        'num_scenes': 1,  # Default to 1 scene
        'avg_scene_length': 0,
        'scene_length_std': 0,
        'min_scene_length': 0,
        'max_scene_length': 0,
        'b_roll_usage': 0.5,  # Default value
        'num_broll_scenes': 0,
        'num_human_scenes': 1,  # Default to 1 human scene
        'transition_pct_cut': 100.0,  # Default to 100% cut
        'transition_pct_fade': 0.0,
        'transition_pct_dissolve': 0.0,
        'transition_pct_quick_cut': 0.0,
        'beat_density': 0.5,  # Default value
        'energy_entropy': 0.0,
        'zero_crossing_rate': 0.0,
        'spectral_centroid': 0.0,
        'spectral_bandwidth': 0.0,
        'spectral_contrast': 0.0,
        'spectral_rolloff': 0.0,
        'mfcc_1': 0.0,  # Add individual MFCCs
        'mfcc_2': 0.0,
        'mfcc_3': 0.0,
        'mfcc_4': 0.0,
        'mfcc_5': 0.0,
        'rms_energy': 0.0,
        'motion_intensity': 0.0,
        'motion_entropy': 0.0,
        'motion_std': 0.0,
        'avg_cut_interval': 0.0,  # Added missing features
        'max_cut_interval': 0.0,
        'min_cut_interval': 0.0,
        'total_duration': 0.0
    }
    
    try:
        # Get basic video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        features['duration'] = duration
        cap.release()
        
        # Detect scenes
        scenes = detect_scenes(video_path)
        features.update(analyze_scenes(scenes, duration))
        
        # Extract audio features
        audio_features = extract_audio_features(video_path)
        features.update(audio_features)
        
        # Analyze motion (simplified for this example)
        motion_features = analyze_motion(video_path, max_frames=100)  # Limit frames for performance
        features.update(motion_features)
        
        # Analyze transitions (simplified for this example)
        transition_features = analyze_transitions(video_path, scenes)
        features.update(transition_features)
        
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        # Return partial results if available
        if all(v == 0 for k, v in features.items() if k != 'video_path'):
            raise
    
    return features

def detect_scenes(video_path, threshold=30.0):
    """
    Detect scenes in a video using PySceneDetect.
    
    Args:
        video_path (str): Path to the video file
        threshold (float): Threshold for scene detection (lower = more sensitive)
        
    Returns:
        list: List of (start_time, end_time) tuples in seconds
    """
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=MIN_SCENE_LEN))
    
    try:
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        
        # Convert to list of (start_time, end_time) in seconds
        scenes = []
        for i, (start, end) in enumerate(scene_list):
            start_sec = start.get_seconds()
            end_sec = end.get_seconds()
            scenes.append((start_sec, end_sec))
            
        return scenes
        
    finally:
        video_manager.release()

def analyze_scenes(scenes, video_duration):
    """Analyze scene cut statistics."""
    if not scenes:
        return {
            'num_scenes': 1,  # At least one scene exists
            'avg_scene_length': video_duration,
            'scene_length_std': 0,
            'min_scene_length': video_duration,
            'max_scene_length': video_duration,
            'cut_frequency': 0,
            'avg_cut_interval': 0,
            'min_cut_interval': 0,
            'max_cut_interval': 0,
            'total_duration': video_duration
        }
    
    scene_lengths = [end - start for start, end in scenes]
    cut_intervals = [scenes[i+1][0] - scenes[i][1] for i in range(len(scenes)-1)]
    
    if not cut_intervals:  # Only one scene
        cut_intervals = [0]
    
    return {
        'num_scenes': len(scenes) if scenes else 1,
        'avg_scene_length': float(np.mean(scene_lengths)) if scene_lengths else video_duration,
        'scene_length_std': float(np.std(scene_lengths)) if len(scene_lengths) > 1 else 0,
        'min_scene_length': float(min(scene_lengths)) if scene_lengths else video_duration,
        'max_scene_length': float(max(scene_lengths)) if scene_lengths else video_duration,
        'cut_frequency': len(scenes) / video_duration if video_duration > 0 else 0,
        'avg_cut_interval': float(np.mean(cut_intervals)) if cut_intervals else 0,
        'min_cut_interval': float(min(cut_intervals)) if cut_intervals else 0,
        'max_cut_interval': float(max(cut_intervals)) if cut_intervals else 0,
        'total_duration': video_duration
    }

def extract_audio_features(video_path, target_sr=AUDIO_SAMPLE_RATE):
    """Extract audio features using librosa."""
    try:
        # Extract audio
        y, sr = librosa.load(video_path, sr=target_sr, mono=True)
        
        # Basic features
        features = {
            'beat_density': float(librosa.feature.tempo(y=y, sr=sr)[0] / 200.0),  # Normalized
            'energy_entropy': float(np.mean(librosa.feature.spectral_flatness(y=y))),
            'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y=y))),
            'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
            'spectral_bandwidth': float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
            'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
            'rms_energy': float(np.mean(librosa.feature.rms(y=y))),
        }
        
        # MFCCs (take mean of first 5 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(5):
            features[f'mfcc_{i+1}'] = float(np.mean(mfccs[i]))
        
        return features
        
    except Exception as e:
        print(f"Error extracting audio features: {str(e)}")
        return {k: 0.0 for k in [
            'beat_density', 'energy_entropy', 'zero_crossing_rate',
            'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
            'rms_energy', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5'
        ]}

def analyze_motion(video_path, max_frames=100):
    """Analyze motion in the video using optical flow."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            'motion_intensity': 0,
            'motion_entropy': 0,
            'motion_std': 0
        }
    
    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return {
            'motion_intensity': 0,
            'motion_entropy': 0,
            'motion_std': 0
        }
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(prev_frame)
    hsv[..., 1] = 255
    
    motion_magnitudes = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calculate magnitude and angle of the 2D vectors
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_magnitudes.extend(magnitude.flatten())
        
        prev_gray = gray
        frame_count += 1
    
    cap.release()
    
    if not motion_magnitudes:
        return {
            'motion_intensity': 0,
            'motion_entropy': 0,
            'motion_std': 0
        }
    
    motion_magnitudes = np.array(motion_magnitudes)
    
    # Calculate motion statistics
    return {
        'motion_intensity': float(np.mean(motion_magnitudes)),
        'motion_entropy': float(-np.sum(motion_magnitudes * np.log1p(motion_magnitudes + 1e-10))),
        'motion_std': float(np.std(motion_magnitudes))
    }

def analyze_transitions(video_path, scenes):
    """Analyze transition types between scenes."""
    # This is a simplified version - in a real implementation, 
    # you would analyze the frames around scene cuts to detect transition types
    if not scenes or len(scenes) < 2:
        return {
            'transition_pct_cut': 100.0,  # If only one scene, it's a "cut"
            'transition_pct_fade': 0.0,
            'transition_pct_dissolve': 0.0,
            'transition_pct_quick_cut': 0.0
        }
    
    # For demo purposes, we'll just return some default values
    # In a real implementation, you would analyze the video to detect transition types
    return {
        'transition_pct_cut': 80.0,  # Most transitions are cuts
        'transition_pct_fade': 10.0,  # Some fades
        'transition_pct_dissolve': 5.0,  # Fewer dissolves
        'transition_pct_quick_cut': 5.0  # Few quick cuts
    }

# For testing
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        features = extract_features(video_path)
        print("Extracted features:")
        for k, v in features.items():
            print(f"{k}: {v}")
    else:
        print("Please provide a video file path")
