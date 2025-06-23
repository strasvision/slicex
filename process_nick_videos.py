#!/usr/bin/env python3
"""
Process all videos in the Nick Video Data folder.
"""

import os
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm

def process_videos(input_dir, output_dir):
    """Process all MP4 videos in the input directory."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all MP4 files in the input directory
    video_files = list(Path(input_dir).glob('*.mp4'))
    
    if not video_files:
        print(f"No MP4 files found in {input_dir}")
        return []
    
    print(f"Found {len(video_files)} video files to process")
    
    processed_files = []
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        try:
            # Create output CSV path
            output_csv = Path(output_dir) / f"{video_path.stem}_features.csv"
            
            print(f"\nProcessing {video_path.name}...")
            
            # Run extract_features.py on the video
            cmd = [
                'python', 'extract_features.py',
                str(video_path),
                '--output', str(output_csv)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Successfully processed {video_path.name}")
                processed_files.append(output_csv)
            else:
                print(f"Error processing {video_path.name}:")
                print(result.stderr)
                
        except Exception as e:
            print(f"Error processing {video_path.name}: {str(e)}")
    
    return processed_files

def combine_csvs(csv_files, output_file):
    """Combine multiple CSV files into one."""
    if not csv_files:
        print("No CSV files to combine")
        return
    
    import pandas as pd
    
    print(f"\nCombining {len(csv_files)} CSV files into {output_file}...")
    
    # Read and concatenate all CSV files
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df['source_video'] = Path(csv_file).name
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {str(e)}")
    
    if not dfs:
        print("No valid CSV files to combine")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save combined CSV
    combined_df.to_csv(output_file, index=False)
    print(f"Saved combined features to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Process Nick video files and extract features.')
    parser.add_argument('--input-dir', default='/Users/bingo/Desktop/Nick Video Data',
                       help='Directory containing the Nick video files')
    parser.add_argument('--output-dir', default='nick_features',
                       help='Directory to save feature CSVs')
    parser.add_argument('--combined-output', default='nick_combined_features.csv',
                       help='Output path for combined features CSV')
    
    args = parser.parse_args()
    
    # Process all videos
    csv_files = process_videos(args.input_dir, args.output_dir)
    
    # Combine all CSVs
    if csv_files:
        combine_csvs(csv_files, args.combined_output)
        print("\nProcessing complete!")
        print(f"- Processed {len(csv_files)} videos")
        print(f"- Combined features saved to: {os.path.abspath(args.combined_output)}")
    else:
        print("\nNo videos were successfully processed.")

if __name__ == "__main__":
    main()
