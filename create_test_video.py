#!/usr/bin/env python3
"""
Create a test video with multiple distinct scenes for testing the video analysis pipeline.
"""

import cv2
import numpy as np
import os
from tqdm import tqdm

def create_scene_1(frames, width, height):
    """Create a color fade scene."""
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        t = i / frames
        color = (int(255 * (1 - t)), int(255 * t), int(255 * t * (1 - t)))
        frame[:, :] = color
        yield frame

def create_scene_2(frames, width, height):
    """Create a moving gradient scene."""
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                t = (x + i * 2) % width / width
                frame[y, x] = [int(255 * t), int(128 * (1 - t)), int(255 * (1 - t))]
        yield frame

def create_scene_3(frames, width, height):
    """Create a bouncing ball scene."""
    ball_radius = min(width, height) // 10
    ball_pos = [width // 4, height // 2]
    ball_vel = [5, 3]
    
    for _ in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Update ball position
        ball_pos[0] += ball_vel[0]
        ball_pos[1] += ball_vel[1]
        
        # Bounce off walls
        if ball_pos[0] <= ball_radius or ball_pos[0] >= width - ball_radius:
            ball_vel[0] *= -1
        if ball_pos[1] <= ball_radius or ball_pos[1] >= height - ball_radius:
            ball_vel[1] *= -1
        
        # Draw ball
        cv2.circle(frame, (int(ball_pos[0]), int(ball_pos[1])), ball_radius, (0, 0, 255), -1)
        yield frame

def create_test_video(output_path, fps=30, duration_per_scene=3):
    """Create a test video with multiple scenes."""
    width, height = 640, 480
    frames_per_scene = int(fps * duration_per_scene)
    
    # Define scenes
    scenes = [
        create_scene_1(frames_per_scene, width, height),
        create_scene_2(frames_per_scene, width, height),
        create_scene_3(frames_per_scene, width, height)
    ]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating test video with {len(scenes)} scenes...")
    
    # Write each scene to the video
    for scene_num, scene in enumerate(scenes, 1):
        print(f"\nGenerating scene {scene_num}...")
        for frame in tqdm(scene, total=frames_per_scene, desc=f"Scene {scene_num}"):
            out.write(frame)
    
    # Release the video writer
    out.release()
    print(f"\nTest video saved to: {output_path}")

if __name__ == "__main__":
    output_path = "test_video_scenes.mp4"
    if os.path.exists(output_path):
        os.remove(output_path)
    create_test_video(output_path)
