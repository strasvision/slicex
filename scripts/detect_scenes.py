#!/usr/bin/env python3
"""
Scene Cut Detection Script

This script detects scene cuts in video files using PySceneDetect and OpenCV,
and saves the results as a JSON file containing scene change timestamps and metadata.

Usage:
    python detect_scenes.py --input video.mp4 --output scenes.json
    python detect_scenes.py --input videos/ --output output_dir/
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass, asdict

import cv2
from scenedetect import VideoManager, SceneManager, StatsManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_detector import SceneDetector
from scenedetect.video_splitter import split_video_ffmpeg

@dataclass
class SceneCut:
    """Data class to store scene cut information."""
    start_frame: int
    end_frame: int
    start_time: float  # in seconds
    end_time: float    # in seconds
    duration: float    # in seconds

@dataclass
class VideoSceneData:
    """Data class to store video and scene information."""
    filename: str
    file_path: str
    file_size: int  # in bytes
    duration: float  # in seconds
    fps: float
    width: int
    height: int
    total_frames: int
    scene_count: int
    avg_scene_length: float  # in seconds
    scenes: List[Dict[str, Any]]

def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """Extract basic metadata from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration,
        'file_size': os.path.getsize(video_path)
    }

def detect_scenes(
    video_path: str,
    threshold: float = 30.0,
    min_scene_len: int = 15,
    show_progress: bool = True
) -> List[SceneCut]:
    """
    Detect scenes in a video using PySceneDetect.
    
    Args:
        video_path: Path to the video file
        threshold: Threshold value for content-aware detection (higher = fewer scenes)
        min_scene_len: Minimum number of frames in a scene
        show_progress: Whether to show progress bar
        
    Returns:
        List of SceneCut objects
    """
    # Create video manager and scene manager
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    
    # Add content detector with specified threshold
    scene_manager.add_detector(ContentDetector(
        threshold=threshold,
        min_scene_len=min_scene_len
    ))
    
    try:
        # Set downscale factor for faster processing
        video_manager.set_downscale_factor()
        
        # Start video manager
        video_manager.start()
        
        # Detect scenes
        scene_manager.detect_scenes(
            frame_source=video_manager,
            show_progress=show_progress
        )
        
        # Get list of scenes
        scene_list = scene_manager.get_scene_list()
        
        # Convert to list of SceneCut objects
        scenes = []
        for i, (start, end) in enumerate(scene_list):
            start_frame = start.get_frames()
            end_frame = end.get_frames()
            start_time = start.get_seconds()
            end_time = end.get_seconds()
            
            scenes.append(SceneCut(
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time
            ))
            
        return scenes
        
    finally:
        video_manager.release()

def process_video(
    video_path: str,
    output_path: Optional[str] = None,
    threshold: float = 30.0,
    min_scene_len: int = 15,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    Process a single video and return scene cut information.
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the JSON output (None to skip saving)
        threshold: Threshold for scene detection
        min_scene_len: Minimum scene length in frames
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary containing video and scene information
    """
    # Get video metadata
    video_meta = get_video_metadata(video_path)
    
    # Detect scenes
    scenes = detect_scenes(
        video_path,
        threshold=threshold,
        min_scene_len=min_scene_len,
        show_progress=show_progress
    )
    
    # Calculate average scene length
    avg_scene_len = video_meta['duration'] / len(scenes) if scenes else 0
    
    # Create result dictionary
    result = {
        'filename': os.path.basename(video_path),
        'file_path': os.path.abspath(video_path),
        'file_size': video_meta['file_size'],
        'duration': video_meta['duration'],
        'fps': video_meta['fps'],
        'resolution': f"{video_meta['width']}x{video_meta['height']}",
        'width': video_meta['width'],
        'height': video_meta['height'],
        'total_frames': video_meta['frame_count'],
        'scene_count': len(scenes),
        'avg_scene_length': avg_scene_len,
        'scenes': [asdict(scene) for scene in scenes],
        'detection_settings': {
            'threshold': threshold,
            'min_scene_len': min_scene_len
        }
    }
    
    # Save to file if output path is provided
    if output_path:
        # If output_path is a directory, create a filename based on input
        if os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)
            output_filename = f"{Path(video_path).stem}_scenes.json"
            output_path = os.path.join(output_path, output_filename)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Scene detection complete. Results saved to: {output_path}")
    
    return result

def process_directory(
    input_dir: str,
    output_dir: str,
    threshold: float = 30.0,
    min_scene_len: int = 15,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    Process all video files in a directory.
    
    Args:
        input_dir: Directory containing video files
        output_dir: Directory to save JSON results
        threshold: Threshold for scene detection
        min_scene_len: Minimum scene length in frames
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary containing results for all processed videos
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported video extensions
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    
    # Process each video file in the directory
    results = {}
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(video_extensions):
            video_path = os.path.join(input_dir, filename)
            try:
                print(f"\nProcessing: {filename}")
                result = process_video(
                    video_path=video_path,
                    output_path=output_dir,
                    threshold=threshold,
                    min_scene_len=min_scene_len,
                    show_progress=show_progress
                )
                results[filename] = result
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    # Save combined results
    if results:
        combined_path = os.path.join(output_dir, "combined_scene_data.json")
        with open(combined_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved combined results to: {combined_path}")
    
    return results

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Detect scene cuts in video files and save results as JSON.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input video file or directory containing video files'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output JSON file or directory to save results'
    )
    
    # Optional arguments
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=30.0,
        help='Threshold for scene detection (higher = fewer scenes)'
    )
    
    parser.add_argument(
        '-m', '--min-scene-len',
        type=int,
        default=15,
        help='Minimum number of frames in a scene'
    )
    
    parser.add_argument(
        '--no-progress',
        action='store_false',
        dest='show_progress',
        help='Disable progress bar'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' does not exist.")
        return
    
    try:
        # Process single file or directory
        if os.path.isfile(args.input):
            process_video(
                video_path=args.input,
                output_path=args.output,
                threshold=args.threshold,
                min_scene_len=args.min_scene_len,
                show_progress=args.show_progress
            )
        else:
            process_directory(
                input_dir=args.input,
                output_dir=args.output,
                threshold=args.threshold,
                min_scene_len=args.min_scene_len,
                show_progress=args.show_progress
            )
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
