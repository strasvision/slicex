#!/usr/bin/env python3
"""
Test script to verify the video processing pipeline.
"""

import cv2
import numpy as np
import os
from scenedetect import VideoManager, SceneManager, ContentDetector

def test_video_loading(video_path):
    """Test if OpenCV can load the video file."""
    print(f"\nTesting video loading: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"Video properties:")
    print(f"- Resolution: {width}x{height}")
    print(f"- FPS: {fps:.2f}")
    print(f"- Frames: {frame_count}")
    print(f"- Duration: {duration:.2f} seconds")
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        cap.release()
        return False
    
    print(f"First frame shape: {frame.shape}")
    cap.release()
    return True

def test_scene_detection(video_path):
    """Test if PySceneDetect can detect scenes."""
    print(f"\nTesting scene detection: {video_path}")
    
    # Create video manager
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))
    
    try:
        # Set downscale factor to improve processing speed
        video_manager.set_downscale_factor()
        
        # Start the video manager
        video_manager.start()
        
        # Perform scene detection
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        
        # Get frame rate for time conversion
        fps = video_manager.get_framerate()
        
        print(f"Detected {len(scene_list)} scenes:")
        for i, (start_frame, end_frame) in enumerate(scene_list):
            start_time = start_frame / fps
            end_time = end_frame / fps
            duration = (end_frame - start_frame) / fps
            print(f"Scene {i+1}: {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)")
        
        return True
        
    except Exception as e:
        print(f"Error during scene detection: {e}")
        return False
    finally:
        video_manager.release()

def main():
    # Test with the original video first
    video_path = "test_video.mp4"
    print("="*50)
    print("TESTING ORIGINAL TEST VIDEO")
    print("="*50)
    
    if os.path.exists(video_path):
        video_ok = test_video_loading(video_path)
        if video_ok:
            test_scene_detection(video_path)
    else:
        print(f"Original test video not found: {video_path}")
    
    # Test with the new scene-based video
    scene_video_path = "test_video_scenes.mp4"
    print("\n" + "="*50)
    print("TESTING SCENE-BASED TEST VIDEO")
    print("="*50)
    
    if os.path.exists(scene_video_path):
        video_ok = test_video_loading(scene_video_path)
        if video_ok:
            test_scene_detection(scene_video_path)
    else:
        print(f"Scene-based test video not found: {scene_video_path}")

if __name__ == "__main__":
    main()
