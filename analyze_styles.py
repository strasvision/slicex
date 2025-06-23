#!/usr/bin/env python3
"""
Video Style Analysis Tool

This script analyzes extracted video features to discover common editing patterns and styles.
It performs clustering, statistical analysis, and generates visualizations.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
import seaborn as sns
from tqdm import tqdm

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette('viridis')

def load_data(input_pattern: str) -> pd.DataFrame:
    """
    Load data from one or more CSV files matching the input pattern.
    
    Args:
        input_pattern: Glob pattern for input CSV files
        
    Returns:
        Combined DataFrame with all data
    """
    files = glob.glob(input_pattern)
    if not files:
        raise ValueError(f"No files found matching pattern: {input_pattern}")
    
    print(f"Loading data from {len(files)} files...")
    dfs = []
    for file in tqdm(files, desc="Loading files"):
        try:
            df = pd.read_csv(file)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not dfs:
        raise ValueError("No valid data found in any input files")
    
    return pd.concat(dfs, ignore_index=True)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data for analysis.
    
    Args:
        df: Input DataFrame with video features
        
    Returns:
        Processed DataFrame with additional features
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Convert string representations of lists to actual lists
    if 'color_hist_avg' in df.columns and isinstance(df['color_hist_avg'].iloc[0], str):
        df['color_hist_avg'] = df['color_hist_avg'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    # Extract RGB components if they're in a list
    if 'color_hist_avg' in df.columns and isinstance(df['color_hist_avg'].iloc[0], list):
        df[['color_r', 'color_g', 'color_b']] = pd.DataFrame(
            df['color_hist_avg'].tolist(), 
            index=df.index
        )
    
    # Add some derived features
    df['log_duration'] = np.log1p(df['duration'])
    df['energy_ratio'] = df['rms_energy'] / (df['spectral_centroid'] + 1e-6)
    
    return df


def analyze_shot_patterns(df: pd.DataFrame, output_dir: str = 'analysis_results'):
    """
    Analyze and visualize shot patterns.
    
    Args:
        df: DataFrame with video features
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nAnalyzing shot patterns...")
    
    # 1. Shot duration analysis
    plt.figure(figsize=(12, 6))
    sns.histplot(df['duration'], bins=50, kde=True)
    plt.title('Distribution of Shot Durations')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'shot_duration_dist.png'), bbox_inches='tight')
    plt.close()
    
    # 2. Shot duration by video
    if 'video_id' in df.columns:
        plt.figure(figsize=(14, 8))
        video_stats = df.groupby('video_id')['duration'].agg(['mean', 'std', 'count']).sort_values('mean')
        
        plt.subplot(2, 1, 1)
        sns.boxplot(data=df, x='video_id', y='duration', showfliers=False)
        plt.xticks(rotation=90)
        plt.title('Shot Duration Distribution by Video')
        plt.tight_layout()
        
        plt.subplot(2, 1, 2)
        sns.scatterplot(data=video_stats, x='mean', y='count', size='std', alpha=0.7)
        plt.xlabel('Mean Shot Duration (s)')
        plt.ylabel('Number of Shots')
        plt.title('Shot Count vs. Mean Duration (bubble size = std dev)')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'shot_duration_by_video.png'), bbox_inches='tight')
        plt.close()
    
    # 3. Print summary statistics
    print("\nShot Duration Statistics:")
    print(f"- Mean: {df['duration'].mean():.2f} seconds")
    print(f"- Median: {df['duration'].median():.2f} seconds")
    print(f"- Std Dev: {df['duration'].std():.2f} seconds")
    print(f"- 90th percentile: {df['duration'].quantile(0.9):.2f} seconds")
    print(f"- Max: {df['duration'].max():.2f} seconds")


def analyze_audio_patterns(df: pd.DataFrame, output_dir: str = 'analysis_results'):
    """
    Analyze and visualize audio patterns.
    
    Args:
        df: DataFrame with audio features
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nAnalyzing audio patterns...")
    
    # 1. Audio feature distributions
    audio_cols = ['rms_energy', 'zero_crossing_rate', 'spectral_centroid']
    
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(audio_cols, 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[col], bins=50, kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'audio_features_dist.png'), bbox_inches='tight')
    plt.close()
    
    # 2. Speech vs Music analysis
    if 'is_speech' in df.columns and 'is_music' in df.columns:
        speech_music = pd.crosstab(df['is_speech'], df['is_music'])
        speech_music_percent = speech_music / speech_music.sum().sum()
        
        plt.figure(figsize=(8, 6))
        # Convert to percentage and format as string with % sign
        annot_data = (speech_music_percent * 100).round(1).astype(str) + '%'
        sns.heatmap(speech_music_percent, annot=annot_data, fmt='', cmap='viridis')
        plt.title('Speech vs Music Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'speech_music_heatmap.png'))
        plt.close()
        
        print("\nSpeech vs Music Analysis:")
        print(speech_music)
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='rms_energy', y='spectral_centroid', 
                       hue='is_music', alpha=0.6, palette='viridis')
        plt.title('Audio Feature Space')
        plt.xlabel('RMS Energy')
        plt.ylabel('Spectral Centroid')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'audio_feature_space.png'))
        plt.close()


def analyze_visual_patterns(df: pd.DataFrame, output_dir: str = 'analysis_results'):
    """
    Analyze and visualize visual patterns.
    
    Args:
        df: DataFrame with visual features
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nAnalyzing visual patterns...")
    
    # 1. Color analysis
    if all(col in df.columns for col in ['color_r', 'color_g', 'color_b']):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        sns.histplot(df['color_r'], bins=50, color='red', alpha=0.7)
        plt.title('Red Channel Distribution')
        
        plt.subplot(1, 3, 2)
        sns.histplot(df['color_g'], bins=50, color='green', alpha=0.7)
        plt.title('Green Channel Distribution')
        
        plt.subplot(1, 3, 3)
        sns.histplot(df['color_b'], bins=50, color='blue', alpha=0.7)
        plt.title('Blue Channel Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'color_distributions.png'), bbox_inches='tight')
        plt.close()
        
        # 2. Color palette visualization
        plt.figure(figsize=(10, 2))
        colors = df[['color_r', 'color_g', 'color_b']].sample(min(1000, len(df))).values
        plt.imshow([colors], aspect='auto')
        plt.axis('off')
        plt.title('Color Palette Sample')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'color_palette.png'), bbox_inches='tight', dpi=300)
        plt.close()
    
    # 3. Zoom detection
    if 'zoom_detected' in df.columns:
        zoom_stats = df['zoom_detected'].value_counts(normalize=True) * 100
        
        plt.figure(figsize=(8, 6))
        zoom_stats.plot(kind='bar')
        plt.title('Zoom Detection Frequency')
        plt.xticks(rotation=0)
        plt.ylabel('Percentage of Shots')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'zoom_detection.png'), bbox_inches='tight')
        plt.close()
        
        print("\nZoom Detection:")
        print(zoom_stats)


def perform_clustering(df: pd.DataFrame, output_dir: str = 'analysis_results', n_clusters: int = 5):
    """
    Perform clustering on the features to discover patterns.
    
    Args:
        df: DataFrame with features
        output_dir: Directory to save visualizations
        n_clusters: Number of clusters to use
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nPerforming clustering with {n_clusters} clusters...")
    
    # Select features for clustering
    features = [
        'duration', 'rms_energy', 'zero_crossing_rate', 'spectral_centroid',
        'brightness', 'contrast'
    ]
    
    if all(col in df.columns for col in ['color_r', 'color_g', 'color_b']):
        features.extend(['color_r', 'color_g', 'color_b'])
    
    # Filter out non-numeric columns and handle missing values
    features = [f for f in features if f in df.columns]
    X = df[features].copy()
    
    # Fill missing values with column means
    X = X.fillna(X.mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Visualize clusters using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['cluster'], palette='viridis', alpha=0.7)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(f'Cluster Visualization (PCA)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_visualization_pca.png'), bbox_inches='tight')
    plt.close()
    
    # Analyze cluster characteristics
    cluster_stats = df.groupby('cluster')[features].mean()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(cluster_stats, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Feature Means by Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_characteristics.png'), bbox_inches='tight')
    plt.close()
    
    print("\nCluster Sizes:")
    print(df['cluster'].value_counts().sort_index())
    
    return df


def generate_report(df: pd.DataFrame, output_dir: str = 'analysis_results'):
    """
    Generate a summary report of the analysis.
    
    Args:
        df: DataFrame with analysis results
        output_dir: Directory to save the report
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=== Video Style Analysis Report ===\n\n")
        
        # Basic statistics
        f.write(f"Total number of shots analyzed: {len(df)}\n")
        if 'video_id' in df.columns:
            f.write(f"Number of videos: {df['video_id'].nunique()}\n\n")
        
        # Shot duration summary
        f.write("Shot Duration Summary (seconds):\n")
        duration_stats = df['duration'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        f.write(duration_stats.to_string() + "\n\n")
        
        # Audio analysis summary
        if 'rms_energy' in df.columns:
            f.write("Audio Feature Summary:\n")
            audio_cols = ['rms_energy', 'zero_crossing_rate', 'spectral_centroid']
            audio_cols = [col for col in audio_cols if col in df.columns]
            f.write(df[audio_cols].describe().to_string() + "\n\n")
        
        # Visual analysis summary
        if 'brightness' in df.columns:
            f.write("Visual Feature Summary:\n")
            visual_cols = ['brightness', 'contrast']
            if all(col in df.columns for col in ['color_r', 'color_g', 'color_b']):
                visual_cols.extend(['color_r', 'color_g', 'color_b'])
            f.write(df[visual_cols].describe().to_string() + "\n\n")
        
        # Cluster summary if available
        if 'cluster' in df.columns:
            f.write("Cluster Sizes:\n")
            f.write(df['cluster'].value_counts().sort_index().to_string() + "\n\n")
    
    print(f"\nAnalysis report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze video style patterns')
    parser.add_argument('input_pattern', help='Input file pattern (e.g., "*.csv" or "features/*.csv")')
    parser.add_argument('--output', '-o', default='analysis_results',
                       help='Output directory for results (default: analysis_results)')
    parser.add_argument('--clusters', '-c', type=int, default=5,
                       help='Number of clusters to use (default: 5)')
    
    args = parser.parse_args()
    
    try:
        # Load and preprocess data
        df = load_data(args.input_pattern)
        df = preprocess_data(df)
        
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Perform analyses
        analyze_shot_patterns(df, args.output)
        analyze_audio_patterns(df, args.output)
        analyze_visual_patterns(df, args.output)
        
        # Perform clustering if we have enough data
        if len(df) >= args.clusters * 10:  # At least 10 samples per cluster
            df = perform_clustering(df, args.output, args.clusters)
        else:
            print(f"\nNot enough data for clustering (have {len(df)} samples, need at least {args.clusters * 10})")
        
        # Generate final report
        generate_report(df, args.output)
        
        # Save the clustered data to a CSV file
        clustered_csv_path = os.path.join(args.output, 'clustered_features.csv')
        df.to_csv(clustered_csv_path, index=False)
        print(f"\nClustered data saved to: {os.path.abspath(clustered_csv_path)}")
        
        print("\nAnalysis complete!")
        print(f"Results saved to: {os.path.abspath(args.output)}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
