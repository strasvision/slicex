#!/usr/bin/env python3
"""
Flatten nested JSON video features into a tabular format for machine learning.

This script processes video analysis results from extract_metadata.py and converts them
into a flat tabular format suitable for training machine learning models. It handles:
- Individual video feature files
- Combined feature collections
- Various output formats (CSV, Parquet, JSON)

Example:
    python flatten_json.py --input_dir data/processed --output_file data/processed/video_features
"""

import json
import os
import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from tqdm import tqdm
import argparse

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, returning a default on failure.
    
    Args:
        value: The value to convert to float
        default: Default value to return if conversion fails
        
    Returns:
        float: The converted float value or default if conversion fails
        
    Example:
        >>> safe_float("3.14")
        3.14
        >>> safe_float("invalid", 0.0)
        0.0
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to int, returning a default on failure.
    
    Args:
        value: The value to convert to int
        default: Default value to return if conversion fails
        
    Returns:
        int: The converted integer or default if conversion fails
        
    Example:
        >>> safe_int("42")
        42
        >>> safe_int("invalid", -1)
        -1
    """
    if value is None:
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

def clean_key(key: str) -> str:
    """Clean and standardize key names for better CSV compatibility.
    
    Args:
        key: The input key string to clean
        
    Returns:
        str: A cleaned version of the key with only alphanumeric and underscore characters
        
    Example:
        >>> clean_key("Video.Duration (s)")
        'video_duration_s'
    """
    if not isinstance(key, str):
        key = str(key)
    # Convert to lowercase and strip whitespace
    key = key.lower().strip()
    # Replace dots and spaces with underscores
    key = re.sub(r'[.\s]+', '_', key)
    # Remove any remaining special characters
    key = re.sub(r'[^a-z0-9_]', '', key)
    return key

def analyze_scene_cuts(scene_cuts: List[List[float]]) -> Dict[str, float]:
    """Calculate statistics about scene cuts from a list of [start, end] times.
    
    Args:
        scene_cuts: List of [start, end] times for each scene in seconds
        
    Returns:
        Dict containing statistics like:
        {
            'scene_count': Number of scenes,
            'total_duration': Total duration in seconds,
            'avg_scene_length': Average scene length in seconds,
            'min_scene_length': Minimum scene length in seconds,
            'max_scene_length': Maximum scene length in seconds,
            'scene_length_std': Standard deviation of scene lengths
        }
        
    Example:
        >>> analyze_scene_cuts([[0, 5.5], [5.5, 10.0], [10.0, 15.0]])
        {'scene_count': 3, 'total_duration': 15.0, 'avg_scene_length': 5.0, ...}
    """
    if not scene_cuts:
        return {}
    
    # Calculate durations for each scene
    durations = [end - start for start, end in scene_cuts]
    cut_intervals = [scene_cuts[i+1][0] - scene_cuts[i][1] 
                     for i in range(len(scene_cuts)-1)]
    
    return {
        'num_scenes': len(scene_cuts),
        'total_duration': sum(durations),
        'avg_scene_length': np.mean(durations) if durations else 0,
        'min_scene_length': min(durations) if durations else 0,
        'max_scene_length': max(durations) if durations else 0,
        'scene_length_std': np.std(durations) if durations else 0,
        'avg_cut_interval': np.mean(cut_intervals) if cut_intervals else 0,
        'min_cut_interval': min(cut_intervals) if cut_intervals else 0,
        'max_cut_interval': max(cut_intervals) if cut_intervals else 0,
    }

def analyze_transitions(transitions: List[str]) -> Dict[str, float]:
    """Calculate statistics about transition types in a video.
    
    Args:
        transitions: List of transition type strings (e.g., ['cut', 'fade', 'dissolve'])
        
    Returns:
        Dictionary with percentage of each transition type, e.g.:
        {
            'transition_pct_cut': 60.0,
            'transition_pct_fade': 30.0,
            'transition_pct_dissolve': 10.0
        }
    """
    if not transitions:
        return {}
    
    # Count occurrences of each transition type
    transition_counts = {}
    for t in transitions:
        transition_counts[t] = transition_counts.get(t, 0) + 1
    
    # Convert counts to percentages
    total = len(transitions)
    return {
        f'transition_pct_{t}': (count / total) * 100 
        for t, count in transition_counts.items()
    }

def process_video_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single video's features into a flat dictionary suitable for ML.
    
    This function takes the nested structure of video features and flattens it into
    a single-level dictionary with consistent key naming. It handles missing data
    gracefully by providing default values.
    
    Args:
        features: Dictionary containing video analysis results with nested structure
        
    Returns:
        Dict containing all features as key-value pairs with clean, standardized keys
        
    Example:
        >>> features = {
        ...     'video_id': 'video1',
        ...     'scene_cuts': [[0, 5], [5, 10]],
        ...     'audio_features': {'beat_density': 0.8},
        ...     'motion_analysis': {'motion_intensity': 0.6}
        ... }
        >>> process_video_features(features)
        {'video_id': 'video1', 'scene_count': 2, 'total_duration': 10.0, ...}
    """
    if not features or not isinstance(features, dict):
        return {}
    
    result = {}
    
    # Basic video info
    result['video_id'] = features.get('video_id', '')
    
    # Scene cut analysis - extract timing and statistics
    scene_cuts = features.get('scene_cuts', [])
    result.update(analyze_scene_cuts(scene_cuts))
    
    # Audio features - extract key audio metrics
    audio = features.get('audio_features', {})
    result['beat_density'] = safe_float(audio.get('beat_density', 0))
    
    # Motion features - analyze movement patterns
    motion = features.get('motion_analysis', {})
    result['motion_intensity'] = safe_float(motion.get('motion_intensity', 0))
    result['motion_entropy'] = safe_float(motion.get('motion_entropy', 0))
    
    # B-roll detection - identify non-primary content
    broll = features.get('broll_analysis', {})
    result['broll_ratio'] = safe_float(broll.get('broll_ratio', 0))
    
    # Clean up all keys to ensure consistent naming
    result = {clean_key(k): v for k, v in result.items()}
    
    # Add transition analysis if available
    transitions = features.get('transition_types', [])
    if transitions:
        result.update({
            clean_key(k): v for k, v in analyze_transitions(transitions).items()
        })
    
    return result

def process_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Process a single JSON file containing video features.
    
    Handles multiple JSON formats:
    - Single video object
    - Dictionary of videos (keyed by video_id)
    - List of video objects
    
    Args:
        file_path: Path to the JSON file to process
        
    Returns:
        List of processed video feature dictionaries
        
    Raises:
        json.JSONDecodeError: If the file contains invalid JSON
        IOError: If the file cannot be read
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict):
            if 'video_id' in data:  # Single video object
                return [process_video_features(data)]
            else:  # Dictionary of videos
                return [process_video_features(feat) for feat in data.values()]
        elif isinstance(data, list):  # List of videos
            return [process_video_features(feat) for feat in data]
        else:
            print(f"Warning: Unexpected JSON structure in {file_path}")
            return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {str(e)}")
        raise
    except IOError as e:
        print(f"Error reading {file_path}: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error processing {file_path}: {str(e)}")
        return []

def process_json_directory(
    input_path: str, 
    output_path: str, 
    output_format: str = 'csv',
    skip_existing: bool = False
) -> Optional[str]:
    """
    Process JSON files and save as flattened tabular data.
    
    Args:
        input_path: Directory containing JSON files or path to a single JSON file
        output_path: Path to save the output file (without extension)
        output_format: Output format ('csv', 'parquet', or 'json')
        skip_existing: Skip processing if output file already exists
        
    Returns:
        Path to the output file if successful, None otherwise
    """
    # Handle output file extension
    output_ext = output_format.lower()
    if not output_path.endswith(f'.{output_ext}'):
        output_path = f"{output_path}.{output_ext}"
    
    # Skip if output exists and skip_existing is True
    if skip_existing and os.path.exists(output_path):
        print(f"Skipping existing file: {output_path}")
        return output_path
    
    # Collect all JSON files
    json_files = []
    if os.path.isfile(input_path) and input_path.endswith('.json'):
        json_files = [input_path]
    elif os.path.isdir(input_path):
        json_files = [
            os.path.join(root, f) 
            for root, _, files in os.walk(input_path)
            for f in files 
            if f.endswith('.json') and not f.startswith('.')
        ]
    
    if not json_files:
        print(f"No JSON files found in {input_path}")
        return None
    
    # Process all JSON files
    all_features = []
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        all_features.extend(process_json_file(json_file))
    
    if not all_features:
        print("No valid video features found in the JSON files")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Ensure consistent column order
    cols = sorted(df.columns)
    if 'video_id' in cols:
        cols.remove('video_id')
        cols = ['video_id'] + cols
    if 'style_label' in cols:
        cols.remove('style_label')
        cols.append('style_label')
    if 'energy_level' in cols:
        cols.remove('energy_level')
        cols.append('energy_level')
            
    df = df[cols]
    
    # Save in the requested format
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    
    try:
        if output_ext == 'csv':
            df.to_csv(output_path, index=False)
        elif output_ext == 'parquet':
            df.to_parquet(output_path, index=False)
        elif output_ext == 'json':
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
            
        print(f"Successfully saved {len(df)} videos to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error saving output file: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description='Flatten video feature JSON files into a tabular format for machine learning.'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Input directory containing JSON files or path to a single JSON file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='data/processed/video_features',
        help='Output file path (without extension). Default: data/processed/video_features.csv'
    )
    parser.add_argument(
        '-f', '--format',
        type=str,
        choices=['csv', 'parquet', 'json'],
        default='csv',
        help='Output format (default: csv)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip processing if output file already exists'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate JSON files before processing'
    )
    
    args = parser.parse_args()
    
    # Validate JSON files if requested
    if args.validate:
        print("Validating JSON files...")
        try:
            from validate_schema import validate_json_file
            
            if os.path.isfile(args.input) and args.input.endswith('.json'):
                is_valid = validate_json_file(args.input)
                if not is_valid:
                    print("Validation failed. Please fix the JSON files before proceeding.")
                    exit(1)
            elif os.path.isdir(args.input):
                # Find all JSON files in the directory
                json_files = [
                    os.path.join(root, f) 
                    for root, _, files in os.walk(args.input)
                    for f in files 
                    if f.endswith('.json') and not f.startswith('.')
                ]
                
                all_valid = True
                for json_file in json_files:
                    if not validate_json_file(json_file):
                        all_valid = False
                        
                if not all_valid:
                    print("\nSome JSON files failed validation. Please fix them before proceeding.")
                    exit(1)
        except ImportError:
            print("Warning: Could not import validate_schema. Skipping validation.")
    
    # Process the files
    output_file = process_json_directory(
        input_path=args.input,
        output_path=args.output,
        output_format=args.format,
        skip_existing=args.skip_existing
    )
    
    if output_file:
        print(f"\nProcessing complete. Output saved to: {output_file}")
    else:
        print("\nProcessing completed with no output.")
        exit(1)

if __name__ == "__main__":
    main()
