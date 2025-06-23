#!/usr/bin/env python3
"""
Integrate B-roll detection results with video metadata.

This script combines the B-roll detection results with the existing video metadata
and saves the enhanced metadata to a new file.
"""

import os
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime

def integrate_broll_metadata(metadata_path, broll_path, output_dir=None):
    """
    Integrate B-roll detection results with video metadata.
    
    Args:
        metadata_path (str): Path to the original metadata CSV
        broll_path (str): Path to the B-roll detection CSV
        output_dir (str, optional): Directory to save the output. If None, uses the metadata directory.
    
    Returns:
        str: Path to the saved enhanced metadata file
    """
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.path.dirname(metadata_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"enhanced_metadata_{timestamp}.csv")
    
    print(f"Loading metadata from: {metadata_path}")
    print(f"Loading B-roll data from: {broll_path}")
    
    # Load the data
    try:
        metadata_df = pd.read_csv(metadata_path, low_memory=False)
        broll_df = pd.read_csv(broll_path)
        
        print(f"Original metadata shape: {metadata_df.shape}")
        print(f"B-roll data shape: {broll_df.shape}")
        
        # Clean filenames for matching (remove paths if present)
        metadata_df['filename_clean'] = metadata_df.iloc[:, 0].apply(lambda x: os.path.basename(x) if isinstance(x, str) else x)
        broll_df['filename_clean'] = broll_df['filename'].apply(lambda x: os.path.basename(x) if isinstance(x, str) else x)
        
        print("Sample filenames in metadata:", metadata_df['filename_clean'].head().tolist())
        print("Sample filenames in broll:", broll_df['filename_clean'].head().tolist())
        
        # Merge the dataframes
        print("Merging data...")
        enhanced_df = pd.merge(
            metadata_df,
            broll_df[['filename_clean', 'has_human', 'is_broll']],
            on='filename_clean',
            how='left'
        )
        
        # Clean up the merged dataframe
        enhanced_df.drop(columns=['filename_clean'], inplace=True, errors='ignore')
        
        # Save the enhanced metadata
        enhanced_df.to_csv(output_path, index=False)
        print(f"Enhanced metadata saved to: {output_path}")
        print(f"Enhanced metadata shape: {enhanced_df.shape}")
        print(f"Number of videos with B-roll: {enhanced_df['is_broll'].sum()}")
        
        return output_path
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Integrate B-roll detection results with video metadata')
    parser.add_argument('--metadata', default='data/processed/video_metadata_20250621_185934.csv',
                       help='Path to the original metadata CSV file')
    parser.add_argument('--broll', default='data/processed/broll_detection.csv',
                       help='Path to the B-roll detection CSV file')
    parser.add_argument('--output_dir', default='data/processed',
                       help='Directory to save the enhanced metadata file')
    
    args = parser.parse_args()
    
    # Ensure paths are absolute
    metadata_path = os.path.abspath(args.metadata)
    broll_path = os.path.abspath(args.broll)
    output_dir = os.path.abspath(args.output_dir)
    
    # Run the integration
    integrate_broll_metadata(metadata_path, broll_path, output_dir)

if __name__ == "__main__":
    main()
