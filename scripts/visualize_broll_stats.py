#!/usr/bin/env python3
"""
Visualize B-roll statistics across videos.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_broll_data(broll_path):
    """Load the B-roll detection results."""
    print(f"Loading B-roll data from {broll_path}...")
    return pd.read_csv(broll_path)

def plot_scene_distribution(broll_df, output_dir):
    """Plot scene count and B-roll count per video."""
    print("Generating scene distribution plots...")
    
    # Calculate scene counts per video
    scene_counts = broll_df.groupby('filename').agg(
        total_scenes=('scene_number', 'count'),
        broll_scenes=('is_broll', 'sum')
    ).reset_index()
    
    # Sort by total scenes
    scene_counts = scene_counts.sort_values('total_scenes', ascending=False)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Total scenes per video
    sns.barplot(
        data=scene_counts, 
        x='filename', 
        y='total_scenes', 
        color='skyblue',
        label='Total Scenes',
        ax=ax1
    )
    ax1.set_title('Total Scenes per Video', fontsize=14)
    ax1.set_xlabel('')
    ax1.set_ylabel('Number of Scenes')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.legend()
    
    # Plot 2: B-roll scenes per video
    sns.barplot(
        data=scene_counts, 
        x='filename', 
        y='broll_scenes', 
        color='lightcoral',
        label='B-Roll Scenes',
        ax=ax2
    )
    ax2.set_title('B-Roll Scenes per Video', fontsize=14)
    ax2.set_xlabel('Video Filename')
    ax2.set_ylabel('Number of B-Roll Scenes')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'scene_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scene distribution plot saved to: {output_path}")
    return output_path

def plot_broll_ratio(broll_df, output_dir):
    """Plot B-roll ratio per video."""
    print("Generating B-roll ratio plot...")
    
    # Calculate B-roll ratio per video
    video_stats = broll_df.groupby('filename').agg(
        total_scenes=('scene_number', 'count'),
        broll_scenes=('is_broll', 'sum')
    ).reset_index()
    video_stats['broll_ratio'] = video_stats['broll_scenes'] / video_stats['total_scenes']
    
    # Sort by B-roll ratio
    video_stats = video_stats.sort_values('broll_ratio', ascending=False)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=video_stats, 
        x='filename', 
        y='broll_ratio',
        color='mediumseagreen'
    )
    
    # Add value labels on top of bars
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.1%}", 
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center', 
            xytext=(0, 5), 
            textcoords='offset points'
        )
    
    # Customize the plot
    plt.title('B-Roll Scene Ratio per Video', fontsize=14)
    plt.xlabel('Video Filename')
    plt.ylabel('B-Roll Scene Ratio')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)  # Add some space at the top for the labels
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'broll_ratio.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"B-roll ratio plot saved to: {output_path}")
    return output_path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize B-roll statistics')
    parser.add_argument('--broll', default='../data/processed/scene_scene_broll_detection.csv',
                       help='Path to the B-roll detection CSV')
    parser.add_argument('--output_dir', default='../reports/figures',
                       help='Directory to save the output figures')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the data
    broll_df = load_broll_data(args.broll)
    
    # Generate visualizations
    scene_plot = plot_scene_distribution(broll_df, args.output_dir)
    ratio_plot = plot_broll_ratio(broll_df, args.output_dir)
    
    print("\nVisualization complete!")
    print(f"- Scene distribution plot: {scene_plot}")
    print(f"- B-roll ratio plot: {ratio_plot}")

if __name__ == "__main__":
    main()
