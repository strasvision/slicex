#!/usr/bin/env python3
"""
List all scene cuts from the scene detection CSV.
"""

import os
import pandas as pd
from pathlib import Path
from tabulate import tabulate

def format_duration(seconds):
    """Format seconds into HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def list_scene_cuts(scene_csv_path):
    """List all scene cuts from the scene detection CSV."""
    print(f"Loading scene data from {scene_csv_path}...")
    
    if not os.path.exists(scene_csv_path):
        print(f"Error: Scene CSV file not found: {scene_csv_path}")
        return None
    
    try:
        df = pd.read_csv(scene_csv_path)
        
        if df.empty:
            print("No scene data found in the CSV file.")
            return None
            
        # Sort by filename and start time
        df = df.sort_values(['filename', 'start_time'])
        
        # Group by video
        grouped = df.groupby('filename')
        
        all_cuts = []
        
        for video_name, group in grouped:
            print(f"\n\n=== {video_name} ===")
            print(f"Total scenes: {len(group)}")
            print(f"Duration: {format_duration(group['end_time'].max())}")
            
            # Prepare table data
            table_data = []
            for _, row in group.iterrows():
                table_data.append([
                    row['scene_number'],
                    format_duration(row['start_time']),
                    format_duration(row['end_time']),
                    format_duration(row['duration']),
                    'B-Roll' if row['is_broll'] else 'Avatar',
                    'Human' if row['has_human'] else 'No Human'
                ])
            
            # Print table
            print("\nScene cuts:")
            print(tabulate(
                table_data,
                headers=['Scene', 'Start Time', 'End Time', 'Duration', 'Type', 'Human'],
                tablefmt='grid',
                floatfmt='.3f',
                colalign=('right', 'center', 'center', 'center', 'center', 'center')
            ))
            
            # Add to all cuts
            for _, row in group.iterrows():
                all_cuts.append({
                    'video': video_name,
                    'scene': int(row['scene_number']),
                    'start_time': row['start_time'],
                    'end_time': row['end_time'],
                    'duration': row['duration'],
                    'is_broll': bool(row['is_broll']),
                    'has_human': bool(row['has_human'])
                })
        
        # Create a summary
        print("\n" + "="*60)
        print("SCENE CUT SUMMARY")
        print("="*60)
        
        summary = []
        for video_name, group in grouped:
            total_scenes = len(group)
            broll_scenes = group['is_broll'].sum()
            human_scenes = group['has_human'].sum()
            
            summary.append([
                video_name,
                total_scenes,
                broll_scenes,
                f"{broll_scenes/total_scenes:.1%}",
                human_scenes,
                f"{human_scenes/total_scenes:.1%}",
                format_duration(group['duration'].mean())
            ])
        
        # Add totals
        total_scenes = len(df)
        total_broll = df['is_broll'].sum()
        total_human = df['has_human'].sum()
        
        summary.append([
            "TOTAL",
            total_scenes,
            total_broll,
            f"{total_broll/total_scenes:.1%}",
            total_human,
            f"{total_human/total_scenes:.1%}",
            format_duration(df['duration'].mean())
        ])
        
        print("\n" + tabulate(
            summary,
            headers=['Video', 'Scenes', 'B-Roll', 'B-Roll %', 'Human', 'Human %', 'Avg Scene'],
            tablefmt='grid',
            colalign=('left', 'right', 'right', 'right', 'right', 'right', 'right')
        ))
        
        return pd.DataFrame(all_cuts)
        
    except Exception as e:
        print(f"Error processing scene data: {str(e)}")
        return None

def save_scene_cuts(scene_df, output_path):
    """Save scene cuts to a CSV file."""
    if scene_df is None or scene_df.empty:
        print("No scene data to save.")
        return
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        scene_df.to_csv(output_path, index=False)
        print(f"\nSaved detailed scene cuts to: {output_path}")
        
    except Exception as e:
        print(f"Error saving scene cuts: {str(e)}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='List all scene cuts from scene detection CSV')
    parser.add_argument('--scenes', default='data/processed/scene_scene_broll_detection.csv',
                       help='Path to the scene detection CSV file')
    parser.add_argument('--output', default='data/processed/detailed_scene_cuts.csv',
                       help='Path to save the detailed scene cuts CSV')
    
    args = parser.parse_args()
    
    # List scene cuts
    scene_cuts = list_scene_cuts(args.scenes)
    
    # Save detailed scene cuts if requested
    if args.output and scene_cuts is not None and not scene_cuts.empty:
        save_scene_cuts(scene_cuts, args.output)

if __name__ == "__main__":
    main()
