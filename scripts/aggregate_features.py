#!/usr/bin/env python3
"""
Aggregate video features into a standardized JSON format for style analysis.

Inputs:
- Scene cuts and B-roll detection from scene_scene_broll_detection.csv
- Audio features from enhanced_video_metadata_*.csv
- Manual labels (to be added)

Output: JSON files with standardized feature format.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

def load_scene_data(scene_csv: str) -> Dict[str, List[Dict]]:
    """Load scene cut and B-roll data from CSV."""
    if not os.path.exists(scene_csv):
        raise FileNotFoundError(f"Scene CSV not found: {scene_csv}")
    
    df = pd.read_csv(scene_csv)
    video_scenes = {}
    
    for video_name, group in df.groupby('filename'):
        scenes = []
        for _, row in group.sort_values('scene_number').iterrows():
            scenes.append({
                'scene_number': int(row['scene_number']),
                'start_time': float(row['start_time']),
                'end_time': float(row['end_time']),
                'duration': float(row['duration']),
                'is_broll': bool(row['is_broll']),
                'has_human': bool(row['has_human'])
            })
        video_scenes[video_name] = scenes
    
    return video_scenes

def load_audio_features(audio_csv: str) -> Dict[str, Dict[str, Any]]:
    """Load audio features from enhanced metadata CSV."""
    if not os.path.exists(audio_csv):
        print(f"Warning: Audio features CSV not found: {audio_csv}")
        return {}
    
    df = pd.read_csv(audio_csv)
    audio_features = {}
    
    for _, row in df.iterrows():
        video_name = row.get('filename_x', '')
        if not video_name or pd.isna(video_name):
            continue
            
        # Extract tempo (beat density) if available
        tempo = float(row.get('tempo', 0)) if 'tempo' in row and pd.notna(row['tempo']) else 0
        
        # Estimate if background music is present (simplified)
        has_background_music = tempo > 80  # Simple heuristic
        
        audio_features[video_name] = {
            'tempo': tempo,
            'has_background_music': has_background_music,
            'energy': float(row.get('energy', 0)) if 'energy' in row else 0,
            'loudness': float(row.get('loudness', 0)) if 'loudness' in row else 0
        }
    
    return audio_features

def detect_transition_types(scenes: List[Dict]) -> List[str]:
    """Detect transition types between scenes (simplified)."""
    transitions = []
    for i in range(len(scenes) - 1):
        # Simple heuristic: if scenes are very short, might be a quick cut
        if scenes[i]['duration'] < 1.0 and scenes[i+1]['duration'] < 1.0:
            transitions.append("quick_cut")
        # If going from B-roll to non-B-roll, might be a fade
        elif scenes[i]['is_broll'] != scenes[i+1]['is_broll']:
            transitions.append("fade")
        else:
            transitions.append("cut")
    return transitions

def aggregate_features(
    video_name: str,
    scenes: List[Dict],
    audio_features: Dict[str, Any],
    manual_labels: Optional[Dict] = None
) -> Dict[str, Any]:
    """Aggregate all features into the target JSON format."""
    if not scenes:
        return {}
    
    # Calculate scene statistics
    durations = [s['duration'] for s in scenes]
    cut_intervals = [round(scenes[i+1]['start_time'] - scenes[i]['end_time'], 2) 
                    for i in range(len(scenes)-1)]
    
    # Calculate B-roll usage
    broll_duration = sum(s['duration'] for s in scenes if s['is_broll'])
    total_duration = sum(durations)
    
    # Get audio features
    audio = audio_features.get(video_name, {})
    
    # Detect transition types
    transitions = detect_transition_types(scenes)
    
    # Create the feature dictionary
    features = {
        'video_id': os.path.splitext(video_name)[0],
        'scene_cuts': [[round(s['start_time'], 2), round(s['end_time'], 2)] for s in scenes],
        'cut_intervals': cut_intervals,
        'avg_scene_length': round(np.mean(durations), 2),
        'has_background_music': audio.get('has_background_music', False),
        'beat_density': int(round(audio.get('tempo', 0))),  # Using tempo as beat density
        'b_roll_usage': round(broll_duration / total_duration, 2) if total_duration > 0 else 0,
        'transition_types': transitions,
        'total_duration': round(total_duration, 2),
        'num_scenes': len(scenes),
        'num_broll_scenes': sum(1 for s in scenes if s['is_broll']),
        'num_human_scenes': sum(1 for s in scenes if s['has_human'])
    }
    
    # Add manual labels if provided
    if manual_labels and video_name in manual_labels:
        features.update({
            'style_label': manual_labels[video_name].get('style_label', 'unknown'),
            'energy_level': manual_labels[video_name].get('energy_level', 'medium')
        })
    
    return features

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Aggregate video features into standardized JSON format')
    parser.add_argument('--scenes', default='data/processed/scene_scene_broll_detection.csv',
                       help='Path to scene detection CSV')
    parser.add_argument('--audio', default='data/processed/enhanced_video_metadata_*.csv',
                       help='Glob pattern for audio features CSVs')
    parser.add_argument('--labels', default='data/manual_labels.json',
                       help='Path to manual labels JSON (optional)')
    parser.add_argument('--output', default='data/processed/aggregated_features',
                       help='Output directory for JSON files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load manual labels if available
    manual_labels = {}
    if os.path.exists(args.labels):
        with open(args.labels, 'r') as f:
            manual_labels = json.load(f)
    
    # Load scene data
    try:
        scene_data = load_scene_data(args.scenes)
    except Exception as e:
        print(f"Error loading scene data: {e}")
        return
    
    # Load audio features (handle glob pattern)
    import glob
    audio_csvs = glob.glob(args.audio)
    audio_features = {}
    for csv_file in audio_csvs:
        audio_features.update(load_audio_features(csv_file))
    
    # Process each video
    all_features = {}
    for video_name, scenes in scene_data.items():
        try:
            features = aggregate_features(video_name, scenes, audio_features, manual_labels)
            if features:
                all_features[video_name] = features
                
                # Save individual video features
                output_file = os.path.join(args.output, f"{Path(video_name).stem}_features.json")
                with open(output_file, 'w') as f:
                    json.dump(features, f, indent=2)
                print(f"Saved features for {video_name}")
                
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
    
    # Save combined features
    if all_features:
        combined_file = os.path.join(args.output, 'all_features.json')
        with open(combined_file, 'w') as f:
            json.dump(all_features, f, indent=2)
        print(f"\nSaved combined features to {combined_file}")
        
        # Print a sample feature set
        sample_video = next(iter(all_features.keys()))
        print("\nSample feature set:")
        print(json.dumps(all_features[sample_video], indent=2))

if __name__ == "__main__":
    main()
