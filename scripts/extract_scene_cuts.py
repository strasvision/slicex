#!/usr/bin/env python3
"""
Extract and list all scene cut timecodes from video metadata.
"""

import os
import pandas as pd
import json
from pathlib import Path
from datetime import timedelta

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS.mmm format."""
    return str(timedelta(seconds=seconds)).ljust(12, '0')

def extract_scene_cuts(metadata_path, output_dir):
    """Extract scene cuts from the metadata file."""
    print(f"Loading metadata from {metadata_path}...")
    metadata = pd.read_csv(metadata_path)
    
    all_cuts = []
    
    # Process each video
    for _, row in metadata.iterrows():
        video_path = row.get('file_path', '')
        if not video_path or not isinstance(video_path, str) or not os.path.exists(video_path):
            print(f"Warning: Video not found or invalid path: {video_path}")
            continue
            
        video_name = os.path.basename(video_path)
        print(f"\nProcessing: {video_name}")
        
        # Find all scene start/end time columns
        scene_cols = [col for col in row.index if 'processing_scene_detection_scene_list_' in col and ('start_time' in col or 'end_time' in col)]
        
        if not scene_cols:
            print("  No scene data found for this video")
            continue
            
        # Extract scene numbers and timestamps
        scenes = {}
        for col in scene_cols:
            if pd.isna(row[col]):
                continue
                
            # Extract scene number from column name like 'processing_scene_detection_scene_list_0_start_time'
            parts = col.split('_')
            try:
                # The scene number is the 6th part (0-based index 5)
                scene_num = int(parts[5])
                time_type = parts[-1]  # 'start_time' or 'end_time'
                
                if scene_num not in scenes:
                    scenes[scene_num] = {}
                scenes[scene_num][time_type] = float(row[col])
            except (IndexError, ValueError) as e:
                print(f"  Warning: Could not parse scene info from column: {col}")
                continue
        
        # Sort scenes by number and collect cuts
        cuts = []
        for scene_num in sorted(scenes.keys()):
            scene = scenes[scene_num]
            if 'start_time' in scene:
                start_time = scene['start_time']
                end_time = scene.get('end_time', start_time)
                duration = end_time - start_time
                
                cuts.append({
                    'scene': scene_num,
                    'start': start_time,
                    'start_timestamp': format_timestamp(start_time),
                    'end': end_time,
                    'end_timestamp': format_timestamp(end_time),
                    'duration': duration,
                    'duration_formatted': format_timestamp(duration).split('.')[0]  # Show duration without ms
                })
        
        # Add to all cuts
        for cut in cuts:
            cut['video'] = video_name
            all_cuts.append(cut)
        
        # Print summary for this video
        print(f"  Found {len(cuts)} scenes")
        print("  Scene cuts:")
        for cut in cuts:
            print(f"    Scene {cut['scene']}: {cut['start_timestamp']} - {cut['end_timestamp']} "
                  f"(duration: {cut['duration_formatted']})")
    
    # Save to CSV if we have data
    if all_cuts:
        output_csv = os.path.join(output_dir, 'scene_cuts.csv')
        df = pd.DataFrame(all_cuts)
        
        # Reorder columns for better readability
        columns = ['video', 'scene', 'start', 'start_timestamp', 'end', 'end_timestamp', 
                  'duration', 'duration_formatted']
        df = df[columns]
        
        df.to_csv(output_csv, index=False)
        print(f"\nSaved scene cuts to: {output_csv}")
        
        return df
    
    return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract scene cut timecodes from video metadata')
    parser.add_argument('--metadata', default='../data/processed/video_metadata_20250621_185934.csv',
                       help='Path to the video metadata CSV file')
    parser.add_argument('--output_dir', default='../data/processed',
                       help='Directory to save the output CSV')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract and display scene cuts
    scene_cuts = extract_scene_cuts(args.metadata, args.output_dir)
    
    if scene_cuts is not None:
        print("\nScene Cut Summary:")
        print(f"Total videos processed: {scene_cuts['video'].nunique()}")
        print(f"Total scene cuts: {len(scene_cuts)}")
        print(f"Average scenes per video: {len(scene_cuts) / scene_cuts['video'].nunique():.1f}")
        print(f"Average scene duration: {scene_cuts['duration'].mean():.2f} seconds")

if __name__ == "__main__":
    main()
