#!/usr/bin/env python3
"""
Enhanced Video Feature Extraction Tool

This script performs comprehensive analysis of video files, including:
- Shot boundary detection using PySceneDetect
- Audio feature extraction (energy, spectral features, music/speech classification)
- Visual feature extraction (color histograms, brightness, contrast, zoom detection)

Output is saved as a CSV file with timestamps and extracted features.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import cv2
import librosa
import scenedetect
import tempfile
import subprocess
from scenedetect import VideoManager, SceneManager, ContentDetector
from scenedetect.scene_detector import SceneDetector
from typing import List, Tuple, Dict, Optional, Union, Any
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass, asdict

# Constants
DEFAULT_SAMPLE_RATE = 22050  # Hz for audio analysis
FRAME_SAMPLE_RATE = 1  # Frames per second to sample for visual features
MIN_SCENE_DURATION = 1.5  # Minimum scene duration in seconds
ZOOM_THRESHOLD = 1.1  # Threshold for zoom detection (10% change in frame size)
HISTOGRAM_BINS = 16  # Number of bins for color histograms

@dataclass
class VideoFeatures:
    """Container for extracted video features."""
    # Timestamp information
    timestamp: float
    duration: float
    
    # Shot detection
    shot_id: int
    is_shot_start: bool
    
    # Audio features
    rms_energy: float
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_rolloff: float
    zero_crossing_rate: float
    is_speech: bool
    is_music: bool
    
    # Visual features
    brightness: float
    contrast: float
    sharpness: float
    zoom_factor: float
    is_zooming: bool
    
    # Color features (histograms)
    color_histogram: np.ndarray
    
    # Motion features
    motion_intensity: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert features to a dictionary for CSV export."""
        data = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(data['color_histogram'], np.ndarray):
            data['color_histogram'] = data['color_histogram'].tolist()
        return data


def detect_scenes(video_path: Union[str, Path], 
                  threshold: float = 30.0, 
                  min_scene_len: float = MIN_SCENE_DURATION) -> List[Dict[str, float]]:
    """
    Detect scenes in a video using PySceneDetect with improved parameters.
    
    Args:
        video_path: Path to the video file
        threshold: Threshold for scene detection (higher = fewer scenes)
        min_scene_len: Minimum scene duration in seconds
        
    Returns:
        List of dictionaries containing scene information with start/end times and duration
    """
    video_path = str(video_path)
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    
    # Use ContentDetector with better parameters
    scene_manager.add_detector(ContentDetector(
        threshold=threshold,
        min_scene_len=int(min_scene_len * 1000),  # Convert to milliseconds
        luma_only=True,  # Process only luminance channel for better performance
        min_content_val=15  # Minimum content change to trigger a new scene
    ))
    
    try:
        # Set downscale factor for faster processing
        video_manager.set_downscale_factor(2)  # 2x downscaling
        video_manager.start()
        
        # Get video properties
        fps = video_manager.get_framerate()
        duration = video_manager.get_duration()
        
        # Detect scenes
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        
        # Convert scene list to list of dictionaries with detailed info
        scenes = []
        for i, (start, end) in enumerate(scene_list):
            start_sec = float(start.get_frames()) / fps
            end_sec = float(end.get_frames()) / fps
            scene_duration = end_sec - start_sec
            
            # Skip very short scenes
            if scene_duration >= min_scene_len:
                scenes.append({
                    'scene_id': i,
                    'start_time': start_sec,
                    'end_time': end_sec,
                    'duration': scene_duration,
                    'start_frame': start.get_frames(),
                    'end_frame': end.get_frames(),
                    'fps': fps
                })
        
        # Ensure the last scene goes to the end of the video
        if scenes and abs(scenes[-1]['end_time'] - duration) > 1.0:
            scenes[-1]['end_time'] = duration
            scenes[-1]['duration'] = scenes[-1]['end_time'] - scenes[-1]['start_time']
            scenes[-1]['end_frame'] = int(duration * fps)
        
        return scenes
        
    except Exception as e:
        print(f"Error detecting scenes: {e}")
        # Return the full video as a single scene if detection fails
        return [{
            'scene_id': 0,
            'start_time': 0.0,
            'end_time': duration if 'duration' in locals() else 60.0,
            'duration': duration if 'duration' in locals() else 60.0,
            'start_frame': 0,
            'end_frame': int(duration * fps) if 'fps' in locals() else 1800,
            'fps': fps if 'fps' in locals() else 30.0,
            'error': str(e)
        }]
        
    finally:
        video_manager.release()


def extract_audio_features(audio_path: Union[str, np.ndarray], 
                         start_time: float, 
                         end_time: float, 
                         sr: int = DEFAULT_SAMPLE_RATE) -> Dict[str, Any]:
    """
    Extract comprehensive audio features for a given time segment.
    
    Args:
        audio_path: Path to the audio file or numpy array of audio samples
        start_time: Start time in seconds
        end_time: End time in seconds
        sr: Sample rate for audio loading
        
    Returns:
        Dictionary containing audio features including MFCCs, spectral features,
        and music/speech classification
    """
    try:
        # Load audio segment
        duration = end_time - start_time
        if duration <= 0:
            raise ValueError(f"Invalid time range: {start_time}s to {end_time}s")
            
        if isinstance(audio_path, (str, Path)):
            y, sr = librosa.load(audio_path, sr=sr, 
                               offset=start_time, 
                               duration=duration,
                               mono=True,  # Convert to mono if needed
                               res_type='kaiser_fast')  # Faster resampling
        else:
            y = audio_path  # Assume it's already a numpy array
            
        if len(y) < 10:  # Need at least 10 samples
            raise ValueError("Audio segment too short for analysis")
        
        # Extract time-domain features
        rms_energy = float(np.mean(librosa.feature.rms(y=y)[0]))
        zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))
        
        # Short-time Fourier transform
        S = np.abs(librosa.stft(y))
        
        # Spectral features
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)[0]))
        spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]))
        spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]))
        spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)[0]))
        
        # Mel-frequency cepstral coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1).tolist()
        mfccs_std = np.std(mfccs, axis=1).tolist()
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(S=S, sr=sr)
        chroma_mean = np.mean(chroma, axis=1).tolist()
        
        # Beat and tempo analysis
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(tempo)
        except:
            tempo = 0.0
        
        # Music/speech classification using a simple rule-based approach
        # Note: This could be replaced with a trained ML model for better accuracy
        is_speech = (spectral_centroid < 2000 and 
                    zero_crossing_rate > 0.08 and 
                    spectral_flatness < 0.5)
                    
        is_music = (spectral_centroid > 1000 and 
                   rms_energy > 0.01 and 
                   spectral_bandwidth > 1000)
        
        # Ensure mutual exclusivity
        if is_speech and is_music:
            if spectral_centroid > 3000:
                is_speech = False
            else:
                is_music = False
        
        # Return all features in a structured format
        return {
            # Basic audio features
            'rms_energy': rms_energy,
            'zero_crossing_rate': zero_crossing_rate,
            
            # Spectral features
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_rolloff': spectral_rolloff,
            'spectral_flatness': spectral_flatness,
            'tempo': tempo,
            
            # MFCCs (mean and std of first 13 coefficients)
            'mfccs_mean': mfccs_mean,
            'mfccs_std': mfccs_std,
            
            # Chroma features (12-dimensional)
            'chroma_mean': chroma_mean,
            
            # Classification
            'is_speech': is_speech,
            'is_music': is_music,
            'audio_quality': 'good' if (rms_energy > 0.01 and duration > 0.5) else 'poor',
            
            # Metadata
            'sample_rate': sr,
            'duration': duration,
            'num_samples': len(y)
        }
        
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        # Return a feature dictionary with default values
        return {
            'rms_energy': 0.0,
            'zero_crossing_rate': 0.0,
            'spectral_centroid': 0.0,
            'spectral_bandwidth': 0.0,
            'spectral_rolloff': 0.0,
            'spectral_flatness': 0.0,
            'tempo': 0.0,
            'mfccs_mean': [0.0] * 13,
            'mfccs_std': [0.0] * 13,
            'chroma_mean': [0.0] * 12,
            'is_speech': False,
            'is_music': False,
            'audio_quality': 'error',
            'sample_rate': sr,
            'duration': duration if 'duration' in locals() else 0.0,
            'num_samples': 0,
            'error': str(e)
        }


def extract_visual_features(video_path: Union[str, Path], 
                          timestamp: float, 
                          prev_frame: Optional[np.ndarray] = None,
                          frame_rate: float = 30.0) -> Dict[str, Any]:
    """
    Extract comprehensive visual features from a specific frame in the video.
    
    Args:
        video_path: Path to the video file
        timestamp: Timestamp in seconds
        prev_frame: Previous frame for motion analysis (optional)
        frame_rate: Video frame rate (used for motion analysis)
        
    Returns:
        Dictionary containing visual features including:
        - Basic: brightness, contrast, sharpness
        - Color: histograms (RGB, HSV, LAB)
        - Motion: optical flow, motion intensity
        - Composition: edges, corners, focus
        - Zoom: zoom factor and direction
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    
    try:
        # Set video to the specified timestamp
        fps = cap.get(cv2.CAP_PROP_FPS) or frame_rate
        frame_num = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret or frame is None:
            raise ValueError(f"Could not read frame at {timestamp:.2f}s")
        
        # Convert to different color spaces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # --- Basic Features ---
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Sharpness using Laplacian variance (higher is sharper)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # --- Color Analysis ---
        def compute_histogram(image: np.ndarray, bins: int = 256) -> List[float]:
            """Compute normalized histogram for a single channel image."""
            hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
            return (hist / hist.sum()).flatten().tolist()
        
        # Color histograms
        b_hist = compute_histogram(frame[:,:,0])  # Blue channel
        g_hist = compute_histogram(frame[:,:,1])  # Green channel
        r_hist = compute_histogram(frame[:,:,2])  # Red channel
        h_hist = compute_histogram(hsv[:,:,0])    # Hue
        s_hist = compute_histogram(hsv[:,:,1])    # Saturation
        v_hist = compute_histogram(hsv[:,:,2])    # Value
        
        # --- Edge and Corner Detection ---
        # Canny edges
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges) / 255.0  # Normalize to [0,1]
        
        # Harris corners
        corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        corner_density = (corners > 0.01 * corners.max()).mean()
        
        # --- Motion Analysis ---
        motion_intensity = 0.0
        flow_magnitude = 0.0
        flow_angle = 0.0
        
        if prev_frame is not None and prev_frame.shape == gray.shape:
            # Calculate dense optical flow (Farneback method)
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, gray, 
                None,  # No previous flow
                pyr_scale=0.5,  # Image scale (<1 for pyramid)
                levels=3,  # Number of pyramid layers
                winsize=15,  # Window size
                iterations=3,  # Iterations per level
                poly_n=5,  # Size of pixel neighborhood
                poly_sigma=1.2,  # Standard deviation of Gaussian
                flags=0
            )
            
            # Calculate magnitude and angle of flow vectors
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_intensity = float(np.mean(magnitude))
            flow_magnitude = float(np.median(magnitude))
            flow_angle = float(np.median(angle))  # Dominant direction
        
        # --- Zoom Detection ---
        zoom_factor = 1.0
        is_zooming = False
        zoom_direction = 'none'
        
        # Simple zoom detection using center vs edge brightness difference
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        center_brightness = np.mean(center_region)
        
        # Calculate edge brightness (border regions)
        border_width = min(h, w) // 8  # Use 1/8th of smaller dimension
        top_edge = gray[:border_width, :]
        bottom_edge = gray[-border_width:, :]
        left_edge = gray[:, :border_width]
        right_edge = gray[:, -border_width:]
        
        edge_brightness = (np.mean(top_edge) + np.mean(bottom_edge) + 
                          np.mean(left_edge) + np.mean(right_edge)) / 4
        
        brightness_diff = abs(center_brightness - edge_brightness)
        
        if brightness_diff > 20:  # Threshold for zoom detection
            if center_brightness > edge_brightness:
                zoom_factor = 1.2  # Zoom in (brighter center)
                zoom_direction = 'in'
            else:
                zoom_factor = 0.8  # Zoom out (darker center)
                zoom_direction = 'out'
            is_zooming = True
        
        # --- Composition Analysis ---
        # Rule of thirds grid (simplified)
        rule_of_thirds = {
            'top_third': np.mean(gray[:h//3, :]),
            'middle_third': np.mean(gray[h//3:2*h//3, :]),
            'bottom_third': np.mean(gray[2*h//3:, :]),
            'left_third': np.mean(gray[:, :w//3]),
            'center_third': np.mean(gray[:, w//3:2*w//3]),
            'right_third': np.mean(gray[:, 2*w//3:])
        }
        
        # Return all features in a structured format
        return {
            # Basic features
            'brightness': float(brightness),
            'contrast': float(contrast),
            'sharpness': float(sharpness),
            'edge_density': float(edge_density),
            'corner_density': float(corner_density),
            
            # Color features
            'color_histogram': {
                'b': b_hist,
                'g': g_hist,
                'r': r_hist,
                'h': h_hist,
                's': s_hist,
                'v': v_hist
            },
            'dominant_color': {
                'b': float(np.mean(frame[:,:,0])),
                'g': float(np.mean(frame[:,:,1])),
                'r': float(np.mean(frame[:,:,2]))
            },
            
            # Motion features
            'motion_intensity': motion_intensity,
            'flow_magnitude': flow_magnitude,
            'flow_angle': flow_angle,
            
            # Zoom features
            'zoom': {
                'factor': zoom_factor,
                'is_zooming': is_zooming,
                'direction': zoom_direction,
                'center_brightness': float(center_brightness),
                'edge_brightness': float(edge_brightness)
            },
            
            # Composition
            'composition': rule_of_thirds,
            'aspect_ratio': frame.shape[1] / frame.shape[0],
            
            # Metadata
            'timestamp': timestamp,
            'frame_num': frame_num,
            'frame_size': f"{frame.shape[1]}x{frame.shape[0]}",
            'channels': frame.shape[2] if len(frame.shape) > 2 else 1
        }
        
    except Exception as e:
        print(f"Error extracting visual features: {e}")
        # Return a minimal feature set with error information
        return {
            'error': str(e),
            'brightness': 0.0,
            'contrast': 0.0,
            'sharpness': 0.0,
            'motion_intensity': 0.0,
            'zoom': {
                'factor': 1.0,
                'is_zooming': False,
                'direction': 'none'
            },
            'frame_num': int(timestamp * (cap.get(cv2.CAP_PROP_FPS) or frame_rate)) if 'cap' in locals() else 0
        }
        
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()


def extract_audio_from_video(video_path: Union[str, Path], 
                           audio_path: Optional[Union[str, Path]] = None) -> str:
    """
    Extract audio from video file using ffmpeg.
    
    Args:
        video_path: Path to the input video file
        audio_path: Optional path to save the extracted audio
        
    Returns:
        Path to the extracted audio file
    """
    video_path = Path(video_path)
    if audio_path is None:
        audio_path = video_path.with_suffix('.wav')
    
    if not audio_path.exists():
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', str(video_path),
            '-vn',  # Disable video
            '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian
            '-ar', str(DEFAULT_SAMPLE_RATE),  # Sample rate
            '-ac', '1',  # Mono audio
            '-loglevel', 'error',  # Only show errors
            str(audio_path)
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio: {e}")
            raise
    
    return str(audio_path)


def process_video(video_path: Union[str, Path], 
                 output_csv: Union[str, Path], 
                 threshold: float = 30.0,
                 sample_rate: float = 1.0) -> None:
    """
    Process a video file to extract comprehensive audio and visual features.
    
    Args:
        video_path: Path to the input video file
        output_csv: Path to save the output CSV
        threshold: Threshold for scene detection (higher = fewer scenes)
        sample_rate: Frames per second to sample for visual features
    """
    video_path = Path(video_path)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"Processing video: {video_path.name}")
    print(f"Output will be saved to: {output_csv}")
    print(f"Sample rate: {sample_rate} fps")
    print(f"Scene detection threshold: {threshold}")
    print(f"Minimum scene duration: {MIN_SCENE_DURATION}s")
    
    # Extract audio first for more efficient processing
    print("\n[1/4] Extracting audio track...")
    try:
        audio_path = extract_audio_from_video(video_path)
        print(f"Audio extracted to: {audio_path}")
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return
    
    # Detect scenes
    print("\n[2/4] Detecting scenes...")
    scenes = detect_scenes(video_path, threshold)
    print(f"Found {len(scenes)} scenes")
    if not scenes:
        print("No scenes detected, using full video as a single scene")
        cap = cv2.VideoCapture(str(video_path))
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        scenes = [{
            'scene_id': 0,
            'start_time': 0.0,
            'end_time': duration,
            'duration': duration
        }]
    # Initialize feature containers
    all_features = []
    prev_frame = None
    
    # Process each scene
    print("\n[3/4] Extracting features from each scene...")
    for scene in tqdm(scenes, desc="Processing scenes"):
        try:
            scene_id = scene['scene_id']
            start_time = scene['start_time']
            end_time = scene['end_time']
            duration = scene['duration']
            
            # Sample timestamps within the scene
            num_samples = max(1, int(duration * sample_rate))
            timestamps = np.linspace(start_time, end_time, num=num_samples, endpoint=False)
            
            # Process each sampled frame
            for timestamp in timestamps:
                try:
                    # Visual features (with motion analysis)
                    visual_features = extract_visual_features(
                        video_path, 
                        timestamp,
                        prev_frame=prev_frame
                    )
                    
                    # Update previous frame for motion analysis
                    if 'frame' in locals():
                        prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Audio features for this time window
                    window_duration = min(1.0, duration)  # 1-second analysis window
                    audio_start = max(0, timestamp - window_duration/2)
                    audio_end = min(end_time, timestamp + window_duration/2)
                    
                    audio_features = extract_audio_features(
                        audio_path,
                        audio_start,
                        audio_end
                    )
                    
                    # Combine all features
                    features = {
                        'video_path': str(video_path),
                        'scene_id': scene_id,
                        'timestamp': timestamp,
                        'duration': duration,
                        'scene_start': start_time,
                        'scene_end': end_time,
                        **audio_features,
                        **visual_features
                    }
                    
                    all_features.append(features)
                    
                except Exception as e:
                    print(f"Error processing frame at {timestamp:.2f}s: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error processing scene {scene_id}: {e}")
            continue
    
    if not all_features:
        print("No features were extracted. Exiting.")
        return
    
    # Convert to DataFrame and save
    print("\n[4/4] Saving results...")
    try:
        df = pd.DataFrame(all_features)
        
        # Clean up any problematic columns for CSV
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                df[col] = df[col].apply(json.dumps)
        
        df.to_csv(output_csv, index=False)
        print(f"Successfully saved {len(df)} feature vectors to {output_csv}")
        
        # Print some statistics
        print("\nExtraction Summary:")
        print(f"- Total scenes processed: {len(scenes)}")
        print(f"- Total frames analyzed: {len(df)}")
        print(f"- Average frames per scene: {len(df)/len(scenes):.1f}")
        
    except Exception as e:
        print(f"Error saving results: {e}")
        return
    finally:
        # Clean up temporary audio file
        if 'audio_path' in locals() and Path(audio_path).exists():
            try:
                Path(audio_path).unlink()
            except Exception as e:
                print(f"Error cleaning up temporary audio file: {e}")

def main():
    """Main function to handle command line arguments and process videos."""
    parser = argparse.ArgumentParser(description='Extract features from video files.')
    parser.add_argument('video_path', type=str, help='Path to the input video file or directory')
    parser.add_argument('--output', '-o', type=str, default='output/features.csv',
                      help='Output CSV file path (default: output/features.csv)')
    parser.add_argument('--threshold', '-t', type=float, default=30.0,
                      help='Threshold for scene detection (default: 30.0)')
    parser.add_argument('--sample-rate', '-s', type=float, default=1.0,
                      help='Frames per second to sample (default: 1.0)')
    parser.add_argument('--recursive', '-r', action='store_true',
                      help='Process videos in subdirectories recursively')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process single file or directory
    video_path = Path(args.video_path)
    video_files = []
    
    if video_path.is_file():
        if video_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.append(video_path)
        else:
            print(f"Error: Unsupported file format: {video_path}")
            return 1
    elif video_path.is_dir():
        # Find all video files in directory
        extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        for ext in extensions:
            if args.recursive:
                video_files.extend(video_path.rglob(ext))
            else:
                video_files.extend(video_path.glob(ext))
        
        if not video_files:
            print(f"No video files found in {video_path}")
            return 1
    else:
        print(f"Error: File or directory not found: {video_path}")
        return 1
    
    print(f"Found {len(video_files)} video(s) to process")
    
    # Process each video
    all_dfs = []
    for i, video_file in enumerate(video_files, 1):
        print(f"\nProcessing video {i}/{len(video_files)}: {video_file.name}")
        
        # Create output path for this video
        if len(video_files) > 1:
            video_output = output_path.parent / f"{video_file.stem}_features.csv"
        else:
            video_output = output_path
        
        # Process the video
        try:
            process_video(
                video_path=video_file,
                output_csv=video_output,
                threshold=args.threshold,
                sample_rate=args.sample_rate
            )
            
            # Load and append results if processing multiple videos
            if len(video_files) > 1 and video_output.exists():
                try:
                    df = pd.read_csv(video_output)
                    df['source_video'] = video_file.name
                    all_dfs.append(df)
                except Exception as e:
                    print(f"Error reading {video_output}: {e}")
                    
        except Exception as e:
            print(f"Error processing video {video_file}: {e}")
            continue
    
    # Combine results from multiple videos if needed
    if len(all_dfs) > 1:
        print("\nCombining results from all videos...")
        try:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df.to_csv(output_path, index=False)
            print(f"Combined results saved to {output_path}")
        except Exception as e:
            print(f"Error combining results: {e}")
    
    print("\nAll done!")
    return 0


if __name__ == "__main__":
    main()
