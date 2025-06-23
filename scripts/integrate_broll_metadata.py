#!/usr/bin/env python3
"""
Integrate B-roll detection results with the main video metadata.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path

def load_metadata(metadata_path):
    """Load the main video metadata CSV."""
    print(f"Loading metadata from {metadata_path}...")
    return pd.read_csv(metadata_path)

def load_broll_data(broll_path):
    """Load the B-roll detection results."""
    print(f"Loading B-roll data from {broll_path}...")
    return pd.read_csv(broll_path)

def calculate_broll_stats(broll_df):
    """Calculate B-roll statistics for each video."""
    print("Calculating B-roll statistics...")
    
    # Group by video and calculate statistics
    stats = broll_df.groupby('filename').agg(
        total_scenes=('scene_number', 'count'),
        broll_scenes=('is_broll', 'sum'),
        total_duration=('duration', 'sum'),
        broll_duration=('duration', lambda x: x[broll_df['is_broll']].sum())
    ).reset_index()
    
    # Calculate percentages
    stats['broll_scene_ratio'] = stats['broll_scenes'] / stats['total_scenes']
    stats['broll_duration_ratio'] = stats['broll_duration'] / stats['total_duration']
    
    # Rename columns to be more descriptive
    stats = stats.rename(columns={
        'broll_scenes': 'broll_scene_count',
        'broll_duration': 'broll_duration_seconds'
    })
    
    return stats

def integrate_metadata(metadata_df, broll_stats):
    """Integrate B-roll statistics with the main metadata."""
    print("Integrating B-roll data with metadata...")
    
    # Create a clean filename column in metadata for merging
    metadata_df['clean_filename'] = metadata_df['file_path'].apply(
        lambda x: os.path.basename(x) if pd.notnull(x) else None
    )
    
    # Merge the data
    merged_df = pd.merge(
        metadata_df,
        broll_stats,
        left_on='clean_filename',
        right_on='filename',
        how='left'
    )
    
    # Drop the temporary columns if they exist
    cols_to_drop = [col for col in ['clean_filename', 'filename'] if col in merged_df.columns]
    if cols_to_drop:
        merged_df = merged_df.drop(columns=cols_to_drop)
    
    # Fill NaN values for videos without B-roll data
    broll_cols = [
        'total_scenes', 'broll_scene_count', 'total_duration',
        'broll_duration_seconds', 'broll_scene_ratio', 'broll_duration_ratio'
    ]
    
    for col in broll_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(0)
    
    return merged_df

def save_enhanced_metadata(df, output_dir):
    """Save the enhanced metadata to a new CSV file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"enhanced_video_metadata_{timestamp}.csv")
    
    print(f"Saving enhanced metadata to {output_path}...")
    df.to_csv(output_path, index=False)
    
    return output_path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrate B-roll detection with video metadata')
    parser.add_argument('--metadata', default='../data/processed/video_metadata_20250621_185934.csv',
                       help='Path to the main video metadata CSV')
    parser.add_argument('--broll', default='../data/processed/scene_scene_broll_detection.csv',
                       help='Path to the B-roll detection CSV')
    parser.add_argument('--output_dir', default='../data/processed',
                       help='Directory to save the enhanced metadata')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the data
    metadata_df = load_metadata(args.metadata)
    broll_df = load_broll_data(args.broll)
    
    # Calculate B-roll statistics
    broll_stats = calculate_broll_stats(broll_df)
    
    # Integrate with main metadata
    enhanced_df = integrate_metadata(metadata_df, broll_stats)
    
    # Save the results
    output_path = save_enhanced_metadata(enhanced_df, args.output_dir)
    
    # Print summary
    print("\nIntegration complete!")
    print(f"Processed {len(metadata_df)} videos")
    print(f"Found {len(broll_df)} total scenes ({broll_df['is_broll'].mean()*100:.1f}% B-roll)")
    print(f"Enhanced metadata saved to: {output_path}")
    
    # Print some sample statistics
    if 'broll_scene_ratio' in enhanced_df.columns:
        print("\nB-Roll Statistics:")
        print(f"Average B-roll scene ratio: {enhanced_df['broll_scene_ratio'].mean()*100:.1f}%")
        print(f"Average B-roll duration ratio: {enhanced_df['broll_duration_ratio'].mean()*100:.1f}%")

if __name__ == "__main__":
    main()
