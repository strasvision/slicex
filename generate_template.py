#!/usr/bin/env python3
"""
Style Template Generator

This script analyzes clustered video features and generates JSON style templates
that capture the key characteristics of each style cluster.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette('viridis')

@dataclass
class StyleTemplate:
    """Class to hold style template data."""
    name: str
    description: str
    avg_shot_duration: float
    shot_duration_std: float
    zoom_rate: float
    music_energy_peaks: List[float]
    color_grade: str
    brightness: float
    contrast: float
    color_palette: List[List[float]]
    audio_energy: float
    speech_ratio: float
    music_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'avg_shot_duration': round(self.avg_shot_duration, 2),
            'shot_duration_std': round(self.shot_duration_std, 2),
            'zoom_rate': round(self.zoom_rate, 2),
            'music_energy_peaks': [round(x, 2) for x in self.music_energy_peaks],
            'color_grade': self.color_grade,
            'brightness': round(self.brightness, 2),
            'contrast': round(self.contrast, 2),
            'color_palette': [[round(c, 2) for c in color] for color in self.color_palette],
            'audio_energy': round(self.audio_energy, 2),
            'speech_ratio': round(self.speech_ratio, 2),
            'music_ratio': round(self.music_ratio, 2)
        }

def detect_color_grade(colors: np.ndarray) -> str:
    """
    Detect the color grade based on RGB statistics.
    
    Args:
        colors: Nx3 array of RGB colors (0-1 range)
        
    Returns:
        String describing the color grade
    """
    if colors.size == 0:
        return "neutral"
        
    # Calculate average color
    avg_color = np.mean(colors, axis=0)
    
    # Calculate color ratios (normalized)
    r, g, b = avg_color
    max_val = max(r, g, b) + 1e-6
    r_ratio = r / max_val
    g_ratio = g / max_val
    b_ratio = b / max_val
    
    # Determine dominant color
    if r_ratio > 0.8 and g_ratio < 0.6 and b_ratio < 0.6:
        base = "warm_red"
    elif r_ratio > 0.7 and g_ratio > 0.6 and b_ratio < 0.5:
        base = "warm_orange"
    elif r_ratio > 0.6 and g_ratio > 0.7 and b_ratio < 0.6:
        base = "warm_yellow"
    elif r_ratio < 0.6 and g_ratio > 0.7 and b_ratio < 0.6:
        base = "cool_green"
    elif r_ratio < 0.6 and g_ratio < 0.7 and b_ratio > 0.7:
        base = "cool_blue"
    elif r_ratio > 0.7 and g_ratio < 0.7 and b_ratio > 0.7:
        base = "cool_purple"
    else:
        base = "neutral"
    
    # Determine contrast level based on standard deviation
    color_std = np.std(colors, axis=0).mean()
    if color_std > 0.2:
        contrast = "high_contrast"
    elif color_std > 0.1:
        contrast = "mid_contrast"
    else:
        contrast = "low_contrast"
    
    # Combine base and contrast
    if base == "neutral":
        return contrast
    else:
        return f"{base}_{contrast}"

def detect_energy_peaks(energy_series: pd.Series, n_peaks: int = 3) -> List[float]:
    """
    Detect peaks in energy time series.
    
    Args:
        energy_series: Series of energy values
        n_peaks: Number of peaks to return
        
    Returns:
        List of peak positions (0-1 normalized)
    """
    if len(energy_series) < 3:
        return [0.25, 0.5, 0.75]  # Default peaks if not enough data
    
    # Simple peak detection
    peaks = []
    for i in range(1, len(energy_series)-1):
        if energy_series.iloc[i-1] < energy_series.iloc[i] > energy_series.iloc[i+1]:
            peaks.append(i / len(energy_series))
    
    # If not enough peaks, distribute them evenly
    if len(peaks) < n_peaks:
        return [i/(n_peaks+1) for i in range(1, n_peaks+1)]
    
    # Sort peaks by energy and take top n
    peak_energies = [(i, energy_series.iloc[int(p*len(energy_series))]) for i, p in enumerate(peaks)]
    peak_energies.sort(key=lambda x: -x[1])
    top_peaks = [peaks[i] for i, _ in peak_energies[:n_peaks]]
    
    return sorted(top_peaks)

def generate_style_template(cluster_data: pd.DataFrame, cluster_id: int, cluster_size: int) -> StyleTemplate:
    """
    Generate a style template for a cluster.
    
    Args:
        cluster_data: DataFrame containing the cluster data
        cluster_id: ID of the cluster to analyze
        cluster_size: Number of samples in the cluster
        
    Returns:
        StyleTemplate object
    """
    # Filter data for this cluster
    cluster = cluster_data[cluster_data['cluster'] == cluster_id].copy()
    
    # Basic statistics
    avg_shot_duration = cluster['duration'].mean()
    shot_duration_std = cluster['duration'].std()
    
    # Zoom rate
    zoom_rate = cluster.get('zoom_detected', pd.Series(0)).mean()
    
    # Audio features
    audio_energy = cluster.get('rms_energy', pd.Series(0)).mean()
    
    # Speech/music ratios
    speech_ratio = cluster.get('is_speech', pd.Series(0)).mean()
    music_ratio = cluster.get('is_music', pd.Series(0)).mean()
    
    # Detect energy peaks (normalized 0-1)
    energy_peaks = []
    if 'rms_energy' in cluster.columns and not cluster.empty:
        energy_peaks = detect_energy_peaks(
            cluster.sort_values('start_time')['rms_energy'].reset_index(drop=True)
        )
    
    # Color analysis
    color_grade = "neutral_mid_contrast"
    color_palette = []
    
    if all(c in cluster.columns for c in ['color_r', 'color_g', 'color_b']):
        colors = cluster[['color_r', 'color_g', 'color_b']].values
        color_grade = detect_color_grade(colors)
        
        # Get representative colors (k-means could be better here)
        n_colors = min(5, len(colors))
        if n_colors > 0:
            color_palette = colors[np.linspace(0, len(colors)-1, n_colors, dtype=int)].tolist()
    
    # Visual features
    brightness = cluster.get('brightness', pd.Series(128)).mean() / 255.0
    contrast = cluster.get('contrast', pd.Series(0)).mean() / 128.0
    
    # Generate name and description
    name = f"style_cluster_{cluster_id}"
    
    # Create description based on features
    duration_desc = ""
    if avg_shot_duration < 2:
        duration_desc = "rapid_cuts"
    elif avg_shot_duration < 5:
        duration_desc = "moderate_pacing"
    else:
        duration_desc = "leisurely_pacing"
    
    energy_desc = "high_energy" if audio_energy > 0.5 else "moderate_energy" if audio_energy > 0.2 else "low_energy"
    
    description = f"{duration_desc}_{energy_desc}_{color_grade}".replace("_", " ").title()
    
    return StyleTemplate(
        name=name,
        description=description,
        avg_shot_duration=avg_shot_duration,
        shot_duration_std=shot_duration_std,
        zoom_rate=zoom_rate,
        music_energy_peaks=energy_peaks[:3],  # Top 3 peaks
        color_grade=color_grade,
        brightness=brightness,
        contrast=contrast,
        color_palette=color_palette,
        audio_energy=audio_energy,
        speech_ratio=speech_ratio,
        music_ratio=music_ratio
    )

def visualize_style_template(template: StyleTemplate, output_dir: str):
    """
    Generate visualizations for a style template.
    
    Args:
        template: StyleTemplate object
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])
    
    # Color palette
    ax1 = fig.add_subplot(gs[0, 1])
    if template.color_palette:
        colors = np.array(template.color_palette)
        ax1.imshow([colors[0]], aspect='auto')
        for i in range(1, len(colors)):
            ax1.imshow([colors[i]], aspect='auto')
    ax1.set_title('Color Palette')
    ax1.axis('off')
    
    # Audio energy peaks
    ax2 = fig.add_subplot(gs[1, 1])
    if template.music_energy_peaks:
        peaks = np.array(template.music_energy_peaks)
        ax2.vlines(peaks, 0, 1, colors='r', alpha=0.7, label='Energy Peaks')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1.1)
        ax2.set_title('Music Energy Peaks')
        ax2.legend()
    
    # Style attributes
    ax3 = fig.add_subplot(gs[:, 0])
    attributes = {
        'Shot Duration': f"{template.avg_shot_duration:.1f}s",
        'Zoom Rate': f"{template.zoom_rate*100:.0f}%",
        'Audio Energy': f"{template.audio_energy:.2f}",
        'Brightness': f"{template.brightness:.2f}",
        'Contrast': f"{template.contrast:.2f}",
        'Speech Ratio': f"{template.speech_ratio*100:.0f}%",
        'Music Ratio': f"{template.music_ratio*100:.0f}%"
    }
    
    ax3.text(0.1, 0.9, template.description, fontsize=14, fontweight='bold')
    for i, (key, value) in enumerate(attributes.items()):
        ax3.text(0.1, 0.8 - i*0.1, f"{key}: {value}", fontsize=12)
    
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{template.name}_preview.png"), dpi=150, bbox_inches='tight')
    plt.close()

def save_template(template: StyleTemplate, output_dir: str):
    """
    Save template to JSON file.
    
    Args:
        template: StyleTemplate object
        output_dir: Directory to save the template
    """
    os.makedirs(output_dir, exist_ok=True)
    template_dict = template.to_dict()
    
    # Save as JSON
    output_path = os.path.join(output_dir, f"{template.name}.json")
    with open(output_path, 'w') as f:
        json.dump(template_dict, f, indent=2)
    
    print(f"Saved template to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate style templates from clustered video features')
    parser.add_argument('input_csv', help='Input CSV file with clustered features')
    parser.add_argument('--output', '-o', default='style_templates',
                       help='Output directory for templates (default: style_templates)')
    parser.add_argument('--min-cluster-size', type=int, default=10,
                       help='Minimum cluster size to consider (default: 10)')
    
    args = parser.parse_args()
    
    try:
        # Load data
        print(f"Loading data from {args.input_csv}...")
        df = pd.read_csv(args.input_csv)
        
        # Check if data has clusters
        if 'cluster' not in df.columns:
            raise ValueError("Input CSV must contain a 'cluster' column. Run clustering first.")
        
        # Process each cluster
        cluster_ids = df['cluster'].unique()
        print(f"Found {len(cluster_ids)} clusters")
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each cluster
        for cluster_id in cluster_ids:
            cluster_data = df[df['cluster'] == cluster_id]
            cluster_size = len(cluster_data)
            
            # Skip small clusters
            if cluster_size < args.min_cluster_size:
                print(f"Skipping cluster {cluster_id} (only {cluster_size} samples)")
                continue
                
            print(f"\nGenerating template for cluster {cluster_id} ({cluster_size} samples)")
            
            # Generate template
            template = generate_style_template(df, cluster_id, cluster_size)
            
            # Save template
            cluster_dir = output_dir / f"cluster_{cluster_id}"
            save_template(template, str(cluster_dir))
            
            # Generate visualizations
            visualize_style_template(template, str(cluster_dir))
            
            # Print summary
            print(f"  - {template.description}")
            print(f"  - Avg shot duration: {template.avg_shot_duration:.2f}s")
            print(f"  - Color grade: {template.color_grade}")
            print(f"  - Audio energy: {template.audio_energy:.2f}")
        
        print("\nTemplate generation complete!")
        print(f"Templates saved to: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
