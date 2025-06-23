#!/usr/bin/env python3
"""
Extract audio features from video files using librosa.

This script analyzes the audio track of video files to extract features
such as tempo, spectral features, and other audio characteristics.
"""

import os
import json
import argparse
import tempfile
import subprocess
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# Suppress librosa warnings
import warnings
warnings.filterwarnings('ignore')

def extract_audio_from_video(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract audio from a video file using ffmpeg.
    
    Args:
        video_path: Path to the input video file
        output_path: Optional path to save the extracted audio (default: temp file)
        
    Returns:
        Path to the extracted audio file (WAV format)
    """
    if output_path is None:
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, 'extracted_audio.wav')
    
    try:
        # Use ffmpeg to extract audio as WAV
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', video_path,  # Input file
            '-vn',  # Disable video
            '-acodec', 'pcm_s16le',  # Audio codec
            '-ar', '44100',  # Sample rate
            '-ac', '1',  # Mono audio
            '-f', 'wav',  # Output format
            output_path
        ]
        
        # Run ffmpeg and suppress output
        subprocess.run(cmd, check=True, 
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
        
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio from {video_path}: {e.stderr}")
        raise

def extract_audio_features(audio_path: str) -> Dict[str, Any]:
    """
    Extract audio features using librosa.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Dictionary containing extracted audio features
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)  # Load with native sample rate
        
        # Basic features
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Extract features
        features = {
            'sample_rate': sr,
            'duration_seconds': float(duration),
            'rms_energy': float(np.mean(librosa.feature.rms(y=y))),  # Root Mean Square energy
            'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
        }
        
        # Spectral features
        S = np.abs(librosa.stft(y))
        spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
        
        features.update({
            'spectral_centroid_mean': float(np.mean(spectral_centroid)),
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
        })
        
        # MFCCs (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i, coeff in enumerate(mfcc):
            features[f'mfcc_{i+1}_mean'] = float(np.mean(coeff))
            features[f'mfcc_{i+1}_std'] = float(np.std(coeff))
        
        # Tempo and beat tracking
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        features.update({
            'tempo': float(tempo),
            'n_beats': len(beat_frames),
            'beats_per_second': len(beat_frames) / duration if duration > 0 else 0,
        })
        
        # Harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        features.update({
            'harmonic_ratio': float(np.mean(y_harmonic**2) / (np.mean(y_harmonic**2) + np.mean(y_percussive**2) + 1e-15)),
        })
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.update({
            'chroma_stft_mean': float(np.mean(chroma)),
            'chroma_stft_std': float(np.std(chroma)),
        })
        
        return features
        
    except Exception as e:
        print(f"Error extracting audio features: {str(e)}")
        return {}

def analyze_video_audio(video_path: str) -> Dict[str, Any]:
    """
    Analyze audio from a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with audio analysis results
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Extract audio to a temporary file
            audio_path = os.path.join(temp_dir, 'temp_audio.wav')
            audio_path = extract_audio_from_video(video_path, audio_path)
            
            # Extract audio features
            audio_features = extract_audio_features(audio_path)
            
            return {
                'audio_analysis': audio_features
            }
            
        except Exception as e:
            print(f"Error analyzing audio for {os.path.basename(video_path)}: {str(e)}")
            return {}

def process_videos(video_dir: str, output_dir: str) -> List[Dict[str, Any]]:
    """
    Process all videos in a directory and extract audio features.
    
    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save results
        
    Returns:
        List of dictionaries with analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    # Get all video files
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    video_files = [f for f in os.listdir(video_dir) 
                  if f.lower().endswith(video_extensions)]
    
    print(f"Found {len(video_files)} video files to process")
    
    for i, filename in enumerate(sorted(video_files), 1):
        try:
            video_path = os.path.join(video_dir, filename)
            print(f"\nProcessing {i}/{len(video_files)}: {filename}")
            
            # Analyze audio
            result = {
                'filename': filename,
                'file_path': os.path.abspath(video_path)
            }
            
            audio_result = analyze_video_audio(video_path)
            result.update(audio_result)
            
            # Save individual results
            output_filename = f"{os.path.splitext(filename)[0]}_audio.json"
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            results.append(result)
            print(f"✓ Completed: {filename}")
            
        except Exception as e:
            print(f"✗ Error processing {filename}: {str(e)}")
    
    # Save combined results
    if results:
        combined_path = os.path.join(output_dir, "combined_audio_analysis.json")
        with open(combined_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved combined audio analysis to {combined_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Extract audio features from video files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments
    parser.add_argument(
        '--input_dir', 
        default='../data/raw_videos',
        help='Directory containing video files'
    )
    parser.add_argument(
        '--output_dir', 
        default='../data/audio_analysis',
        help='Directory to save analysis results'
    )
    
    args = parser.parse_args()
    
    # Process videos and extract audio features
    results = process_videos(args.input_dir, args.output_dir)
    
    if results:
        print(f"\nSuccessfully processed {len(results)} videos")
        print(f"Results saved to: {os.path.abspath(args.output_dir)}")
    else:
        print("No videos were processed successfully.")

if __name__ == "__main__":
    main()
