#!/usr/bin/env python3
"""
Generate a summary of B-roll statistics across videos.
"""

import pandas as pd
import os
from tabulate import tabulate

def load_broll_data(broll_path):
    """Load the B-roll detection results."""
    print(f"Loading B-roll data from {broll_path}...")
    return pd.read_csv(broll_path)

def generate_summary(broll_df):
    """Generate a summary of B-roll statistics."""
    # Calculate statistics per video
    video_stats = broll_df.groupby('filename').agg(
        total_scenes=('scene_number', 'count'),
        broll_scenes=('is_broll', 'sum'),
        avg_scene_duration=('duration', 'mean')
    ).reset_index()
    
    # Calculate percentages and round values
    video_stats['broll_ratio'] = (video_stats['broll_scenes'] / video_stats['total_scenes']).round(3)
    video_stats['avg_scene_duration'] = video_stats['avg_scene_duration'].round(2)
    
    # Sort by B-roll ratio (descending)
    video_stats = video_stats.sort_values('broll_ratio', ascending=False)
    
    # Calculate totals
    total_scenes = video_stats['total_scenes'].sum()
    total_broll = video_stats['broll_scenes'].sum()
    
    # Format the table
    summary = video_stats[[
        'filename', 'total_scenes', 'broll_scenes', 
        'broll_ratio', 'avg_scene_duration'
    ]].rename(columns={
        'filename': 'Video',
        'total_scenes': 'Total Scenes',
        'broll_scenes': 'B-Roll Scenes',
        'broll_ratio': 'B-Roll %',
        'avg_scene_duration': 'Avg Scene (s)'
    })
    
    # Add a total row
    totals = pd.DataFrame([{
        'Video': 'TOTAL',
        'Total Scenes': total_scenes,
        'B-Roll Scenes': total_broll,
        'B-Roll %': round(total_broll / total_scenes, 3),
        'Avg Scene (s)': round(broll_df['duration'].mean(), 2)
    }])
    
    # Combine and format percentages
    summary = pd.concat([summary, totals], ignore_index=True)
    summary['B-Roll %'] = (summary['B-Roll %'] * 100).round(1).astype(str) + '%'
    
    return summary

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate B-roll statistics summary')
    parser.add_argument('--broll', default='../data/processed/scene_scene_broll_detection.csv',
                       help='Path to the B-roll detection CSV')
    
    args = parser.parse_args()
    
    # Load the data
    broll_df = load_broll_data(args.broll)
    
    # Generate and display the summary
    print("\n=== B-Roll Statistics Summary ===\n")
    
    # Get the summary table
    summary = generate_summary(broll_df)
    
    # Print the table with borders
    print(tabulate(summary, headers='keys', tablefmt='grid', showindex=False))
    
    # Print some key insights
    print("\nKey Insights:")
    print(f"- Total videos analyzed: {len(summary)-1}")
    print(f"- Total scenes: {summary['Total Scenes'].iloc[-1]}")
    print(f"- Total B-roll scenes: {summary['B-Roll Scenes'].iloc[-1]} ({summary['B-Roll %'].iloc[-1]})")
    print(f"- Average scene duration: {summary['Avg Scene (s)'].iloc[-1]:.2f} seconds")
    
    # Find videos with most and least B-roll
    most_broll = summary.iloc[0]
    least_broll = summary.iloc[-2]  # -2 because last row is totals
    
    print(f"\nMost B-Roll: {most_broll['Video']} ({most_broll['B-Roll %']} B-roll)")
    print(f"Least B-Roll: {least_broll['Video']} ({least_broll['B-Roll %']} B-roll)")

if __name__ == "__main__":
    main()
