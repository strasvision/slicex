"""
Video Metadata Extraction Module

This module provides functionality to extract comprehensive metadata from video files,
including video properties, scene detection, audio features, and motion analysis.

Key Features:
- Video property extraction using ffprobe
- Scene detection with PySceneDetect
- Audio feature extraction with librosa
- Motion analysis using optical flow
- Support for batch processing of video files

Example:
    >>> from scripts.extract_metadata import process_videos
    >>> results = process_videos(
    ...     video_dir='input_videos/',
    ...     output_dir='metadata_output/',
    ...     analyze_scenes=True,
    ...     analyze_audio=True,
    ...     analyze_motion=True
    ... )
"""

import os
import json
import cv2
import numpy as np
import librosa
import soundfile as sf
from datetime import datetime
from scenedetect import VideoManager, SceneManager, StatsManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images, write_scene_list_html
from typing import Dict, List, Tuple, Optional, Union, Any
import subprocess
import tempfile
from pathlib import Path
import shutil
from tqdm import tqdm
import warnings

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=UserWarning)

def get_video_info_ffprobe(video_path: str) -> Dict[str, Any]:
    """Extract video stream metadata using ffprobe.
    
    This function uses ffprobe to extract detailed technical information
    about the video stream, including dimensions, frame rate, duration, and codec.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary containing video metadata with the following keys:
        - width: Video width in pixels
        - height: Video height in pixels
        - fps: Frames per second (calculated from r_frame_rate)
        - duration: Video duration in seconds
        - frame_count: Total number of video frames
        - codec: Video codec name (e.g., 'h264', 'vp9')
        - pixel_format: Pixel format (e.g., 'yuv420p')
        
    Raises:
        subprocess.CalledProcessError: If ffprobe command fails
        json.JSONDecodeError: If ffprobe output is not valid JSON
        KeyError: If required fields are missing from ffprobe output
        
    Example:
        >>> info = get_video_info_ffprobe('example.mp4')
        >>> print(f"Resolution: {info['width']}x{info['height']}")
        >>> print(f"Duration: {info['duration']:.2f} seconds")
    """
    cmd = [
        'ffprobe',
        '-v', 'error',  # Only show errors
        '-select_streams', 'v:0',  # Only process the first video stream
        '-show_entries', 'stream=width,height,r_frame_rate,duration,nb_frames,codec_name,pix_fmt',
        '-of', 'json',  # Output in JSON format
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        stream = info['streams'][0]  # Get first video stream
        
        # Calculate FPS from fraction string (e.g., '30000/1001')
        try:
            num, denom = map(float, stream['r_frame_rate'].split('/'))
            fps = num / denom if denom != 0 else 0
        except (KeyError, ValueError, ZeroDivisionError):
            fps = 0
        
        return {
            'width': stream.get('width', 0),
            'height': stream.get('height', 0),
            'fps': fps,
            'duration': float(stream.get('duration', 0)),
            'frame_count': int(stream.get('nb_frames', 0)),
            'codec': stream.get('codec_name', ''),
            'pixel_format': stream.get('pix_fmt', '')
        }
    except Exception as e:
        print(f"Error running ffprobe: {e}")
        return {}

def extract_video_metadata(video_path: str) -> Dict:
    """Extract comprehensive metadata from a video file."""
    try:
        # First try with ffprobe for more reliable metadata
        ffprobe_info = get_video_info_ffprobe(video_path)
        
        # Fall back to OpenCV if ffprobe fails
        if not ffprobe_info:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            video_info = {
                'width': width,
                'height': height,
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'codec': 'unknown',
                'pixel_format': 'unknown'
            }
        else:
            video_info = ffprobe_info
        
        # Get file stats
        file_size = os.path.getsize(video_path)
        
        # Basic metadata
        metadata = {
            'filename': os.path.basename(video_path),
            'file_path': os.path.abspath(video_path),
            'file_size': file_size,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'resolution': f"{video_info['width']}x{video_info['height']}",
            'width': video_info['width'],
            'height': video_info['height'],
            'aspect_ratio': round(video_info['width'] / video_info['height'], 2) if video_info['height'] > 0 else 0,
            'fps': round(video_info['fps'], 3),
            'frame_count': video_info['frame_count'],
            'duration_seconds': round(video_info['duration'] if 'duration' in video_info else 
                                    (video_info['frame_count'] / video_info['fps'] if video_info['fps'] > 0 else 0), 2),
            'codec': video_info['codec'],
            'pixel_format': video_info['pixel_format'],
            'created_at': datetime.now().isoformat(),
            'processing': {
                'scene_detection': {},
                'audio_analysis': {},
                'motion_analysis': {}
            }
        }
        
        return metadata
        
    except Exception as e:
        print(f"Error extracting metadata from {video_path}: {str(e)}")
        raise

def detect_scenes(video_path: str, threshold: float = 30.0) -> List[Dict[str, Union[int, float]]]:
    """Detect scene boundaries in a video using PySceneDetect's content-aware detection.
    
    This function uses PySceneDetect to automatically identify scene transitions in a video
    by analyzing changes in the visual content between frames. It's particularly useful for
    analyzing editing patterns and segmenting videos into logical scenes.
    
    Args:
        video_path: Path to the video file to analyze
        threshold: Detection threshold (default: 30.0). Higher values make the detector
                 more sensitive to changes, resulting in more scenes. Typical range is 20-40.
                 
    Returns:
        List of dictionaries, where each dictionary contains metadata about a detected scene:
        - scene_num: Sequential scene number (1-based index)
        - start_frame: Starting frame number of the scene
        - end_frame: Ending frame number of the scene (inclusive)
        - start_time: Start time in seconds
        - end_time: End time in seconds
        - duration: Duration of the scene in seconds
        
    Raises:
        FileNotFoundError: If the specified video file does not exist
        
    Example:
        >>> scenes = detect_scenes('example.mp4', threshold=27.5)
        >>> for scene in scenes:
        ...     print(f"Scene {scene['scene_num']}: {scene['duration']:.2f}s")
        ...     print(f"  Frames: {scene['start_frame']}-{scene['end_frame']}")
        ...     print(f"  Time: {scene['start_time']:.2f}s - {scene['end_time']:.2f}s")
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Initialize video manager and scene detector
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    
    # Add content-aware scene detector with threshold
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    
    try:
        # Set video manager options - downscale for faster processing
        video_manager.set_downscale_factor()
        
        # Start video manager
        video_manager.start()
        
        # Perform scene detection
        scene_manager.detect_scenes(frame_source=video_manager)
        
        # Get list of detected scenes as (start_frame, end_frame) tuples
        scene_list = scene_manager.get_scene_list()
        
        # Convert to list of dictionaries with detailed timing information
        scenes = []
        for i, (start_frame, end_frame) in enumerate(scene_list):
            start_time = video_manager.get_base_timecode() + (start_frame / video_manager.get_framerate())
            end_time = video_manager.get_base_timecode() + (end_frame / video_manager.get_framerate())
            
            scenes.append({
                'scene_num': i + 1,  # 1-based index for readability
                'start_frame': int(start_frame),
                'end_frame': int(end_frame - 1),  # Make end_frame inclusive
                'start_time': float(start_time.get_seconds()),
                'end_time': float(end_time.get_seconds()),
                'duration': float((end_time - start_time).get_seconds())
            })
        
        return scenes
        
    except Exception as e:
        print(f"Error detecting scenes: {e}")
        return []
    finally:
        video_manager.release()

def extract_audio_features(audio_path: str, sr: int = 22050) -> Dict[str, Union[float, List[float]]]:
    """Extract audio features from an audio file using librosa.
    
    This function extracts various acoustic features that can be used for analyzing
    the audio characteristics of a video, including tempo, energy, and spectral properties.
    
    Args:
        audio_path: Path to the audio file to analyze
        sr: Target sample rate for audio loading (default: 22050 Hz)
        
    Returns:
        Dictionary containing the following audio features:
        - tempo: Estimated tempo in beats per minute (BPM)
        - rms_energy: Root Mean Square energy (loudness)
        - zero_crossing_rate: Rate of sign changes in the audio signal
        - spectral_centroid: Center of mass of the spectrum (brightness)
        - spectral_bandwidth: Spectral bandwidth (frequency range)
        - spectral_rolloff: Frequency below which a specified percentage of total energy is contained
        - mfcc: Mel-frequency cepstral coefficients (list of 20 values)
        - duration: Duration of the audio in seconds
        
    Raises:
        FileNotFoundError: If the specified audio file does not exist
        
    Example:
        >>> features = extract_audio_features('audio.wav')
        >>> print(f"Tempo: {features['tempo']:.1f} BPM")
        >>> print(f"Duration: {features['duration']:.2f} seconds")
        >>> print(f"Spectral Centroid: {features['spectral_centroid']:.1f} Hz")
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
    try:
        # Load audio file with target sample rate
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Extract features
        features = {
            # Tempo estimation (BPM)
            'tempo': float(librosa.beat.tempo(y=y, sr=sr)[0]),
            
            # Energy and dynamics
            'rms_energy': float(np.mean(librosa.feature.rms(y=y))),
            'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
            
            # Spectral features
            'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
            'spectral_bandwidth': float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
            'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
            
            # Timbre features (MFCCs)
            'mfcc': [float(x) for x in np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)],
            
            # Temporal features
            'duration': librosa.get_duration(y=y, sr=sr)
        }
        
        return features
        
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return {}

def extract_audio(video_path: str, output_dir: str) -> Optional[str]:
    """Extract audio track from a video file and save as WAV format.
    
    This function uses ffmpeg to extract the audio stream from a video file and
    saves it as a high-quality WAV file. The output will be saved in the specified
    directory with the same base name as the input video but with a .wav extension.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory where the extracted audio file will be saved
        
    Returns:
        Path to the extracted audio file if successful, None otherwise
        
    Raises:
        FileNotFoundError: If the input video file does not exist
        NotADirectoryError: If the output directory is not a valid directory
        
    Example:
        >>> audio_path = extract_audio('video.mp4', 'audio_output')
        >>> if audio_path:
        ...     print(f"Audio extracted to: {audio_path}")
        ...     # Use the audio file for further processing
        ...     features = extract_audio_features(audio_path)
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.isdir(output_dir):
        raise NotADirectoryError(f"Output directory does not exist: {output_dir}")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output audio path with WAV extension
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{video_name}.wav")
        
        # Use ffmpeg to extract audio with high quality settings
        cmd = [
            'ffmpeg',
            '-i', video_path,  # Input file
            '-vn',  # Disable video stream
            '-acodec', 'pcm_s16le',  # 16-bit PCM audio codec
            '-ar', '44100',  # CD-quality sample rate (44.1 kHz)
            '-ac', '2',  # Stereo audio (2 channels)
            '-y',  # Overwrite output file if it exists
            output_path
        ]
        
        # Run the command with error handling
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Verify the output file was created
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            print("Error: Failed to create output audio file")
            return None
        
    except subprocess.CalledProcessError as e:
        error_msg = f"ffmpeg command failed with code {e.returncode}"
        if hasattr(e, 'stderr') and e.stderr:
            error_msg += f": {e.stderr.strip()}"
        print(f"Error extracting audio: {error_msg}")
        return None


def _analyze_audio(video_path: str, metadata: Dict[str, Any], temp_dir: str) -> Dict[str, Any]:
    """Analyze audio features of a video file.
    
    This internal function handles the audio analysis pipeline for a video file by:
    1. Extracting the audio track to a temporary WAV file
    2. Analyzing the audio features using librosa
    3. Cleaning up temporary files
    
    Args:
        video_path: Path to the input video file
        metadata: Dictionary containing existing video metadata
        temp_dir: Directory for storing temporary files during processing
        
    Returns:
        Dictionary containing audio features including:
        - tempo: Estimated tempo in BPM
        - rms_energy: Root mean square energy
        - zero_crossing_rate: Rate of sign changes in audio signal
        - spectral_centroid: Center of mass of the spectrum
        - spectral_bandwidth: Spectral bandwidth
        - spectral_rolloff: Spectral roll-off frequency
        - mfcc: Mel-frequency cepstral coefficients
        - duration: Audio duration in seconds
        
    Note:
        This function creates temporary files in the specified directory
        and cleans them up after analysis.
        
    Example:
        >>> video_meta = {'duration': 60.0, 'fps': 30.0, 'processing': {'audio_analysis': {}}}}
        >>> audio_features = _analyze_audio('video.mp4', video_meta, 'temp_dir')
        >>> if audio_features:
        ...     print(f"Detected tempo: {audio_features['tempo']:.1f} BPM")
        ...     print(f"Audio duration: {audio_features['duration']:.2f}s")
    """
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found: {video_path}")
        metadata['processing']['audio_analysis'].update({
            'status': 'failed',
            'message': 'Video file not found'
        })
        return metadata
    
    try:
        # Initialize audio analysis status
        metadata['processing']['audio_analysis'] = {
            'status': 'pending',
            'message': 'Starting audio analysis'
        }
        
        # Create temp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)
        
        # Extract audio to temporary WAV file
        audio_path = extract_audio(video_path, temp_dir)
        if not audio_path or not os.path.exists(audio_path):
            metadata['processing']['audio_analysis'].update({
                'status': 'failed',
                'message': 'Failed to extract audio track'
            })
            return metadata
        
        # Update status
        metadata['processing']['audio_analysis'].update({
            'status': 'processing',
            'message': 'Extracted audio, analyzing features'
        })
        
        # Extract audio features using librosa
        audio_features = extract_audio_features(audio_path)
        if not audio_features:
            metadata['processing']['audio_analysis'].update({
                'status': 'failed',
                'message': 'No audio features could be extracted'
            })
            return metadata
        
        # Add audio features to metadata
        metadata['audio'] = audio_features
        metadata['processing']['audio_analysis'].update({
            'status': 'success',
            'message': 'Successfully analyzed audio features'
        })
        
        return metadata
        
    except Exception as e:
        error_msg = f"Error analyzing audio: {str(e)}"
        print(error_msg)
        metadata['processing']['audio_analysis'].update({
            'status': 'error',
            'message': error_msg
        })
        return metadata
    
    finally:
        # Clean up temporary audio file if it exists
        if 'audio_path' in locals() and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError as e:
                print(f"Warning: Could not remove temporary audio file {audio_path}: {e}")

def _analyze_motion(video_path: str, metadata: Dict[str, Any], temp_dir: str) -> Dict[str, Any]:
    """Analyze motion characteristics from precomputed motion analysis data.
    
    This function loads and integrates precomputed motion analysis data into the video metadata.
    The motion analysis is expected to be precomputed and stored in a JSON file with the
    same base name as the video file in a 'motion_analysis' directory.
    
    The motion analysis provides insights into:
    - Overall motion intensity (mean, max, variation)
    - Motion energy distribution
    - Motion histogram for understanding motion patterns
    
    Args:
        video_path: Path to the input video file
        metadata: Dictionary containing existing video metadata
        temp_dir: Directory for storing temporary files (unused in this implementation)
        
    Returns:
        Updated metadata dictionary with motion analysis results. The motion data includes:
        - mean_motion: Average motion magnitude across all frames
        - std_motion: Standard deviation of motion magnitudes
        - max_motion: Maximum motion magnitude observed
        - mean_energy: Average motion energy
        - motion_variation: Coefficient of variation of motion
        - motion_hist: Histogram of motion magnitudes
        
    Example:
        >>> video_meta = {'duration_seconds': 30.0, 'processing': {}}
        >>> motion_meta = _analyze_motion('video.mp4', video_meta, 'temp_dir')
        >>> print(f"Mean motion: {motion_meta['motion']['mean_motion']:.2f}")
        >>> print(f"Motion variation: {motion_meta['motion']['motion_variation']:.2f}")
        
    Note:
        - This function expects precomputed motion analysis data in JSON format
        - If the motion analysis file is not found, the function will skip the analysis
        - The motion data is typically generated by a separate motion analysis pipeline
    """
    # Initialize motion analysis status
    if 'processing' not in metadata:
        metadata['processing'] = {}
    
    metadata['processing']['motion_analysis'] = {
        'status': 'pending',
        'message': 'Starting motion analysis',
        'source': 'precomputed',
        'file': None
    }
    
    try:
        # Create temp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)
        
        # Construct path to precomputed motion analysis file
        motion_file = os.path.join(
            os.path.dirname(os.path.dirname(temp_dir)),  # Go up two levels from temp_dir
            'motion_analysis',
            f"{os.path.splitext(os.path.basename(video_path))[0]}_motion.json"
        )
        
        if not os.path.exists(motion_file):
            metadata['processing']['motion_analysis'].update({
                'status': 'skipped',
                'message': 'Precomputed motion data not found',
                'expected_path': motion_file
            })
            return metadata
        
        # Load pre-computed motion analysis
        with open(motion_file, 'r', encoding='utf-8') as f:
            motion_data = json.load(f)
        
        # Add motion data to metadata with type hints
        metadata['motion'] = {
            'mean_motion': float(motion_data.get('mean_motion', 0)),
            'std_motion': float(motion_data.get('std_motion', 0)),
            'max_motion': float(motion_data.get('max_motion', 0)),
            'mean_energy': float(motion_data.get('mean_energy', 0)),
            'motion_variation': float(motion_data.get('motion_variation', 0)),
            'motion_hist': motion_data.get('motion_hist', []),
            'analysis_timestamp': motion_data.get('timestamp', '')
        }
        
        # Update processing status
        metadata['processing']['motion_analysis'].update({
            'status': 'success',
            'file': motion_file,
            'message': 'Successfully loaded precomputed motion analysis'
        })
        
        # Add motion statistics to metadata
        if 'statistics' not in metadata:
            metadata['statistics'] = {}
            
        metadata['statistics'].update({
            'motion_intensity': metadata['motion']['mean_motion'],
            'motion_variability': metadata['motion']['motion_variation']
        })
    
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in motion analysis file: {str(e)}"
        print(f"Error: {error_msg}")
        metadata['processing']['motion_analysis'].update({
            'status': 'error',
            'message': error_msg,
            'error_type': 'JSONDecodeError'
        })
    except IOError as e:
        error_msg = f"I/O error reading motion analysis file: {str(e)}"
        print(f"Error: {error_msg}")
        metadata['processing']['motion_analysis'].update({
            'status': 'error',
            'message': error_msg,
            'error_type': 'IOError'
        })
    except Exception as e:
        error_msg = f"Unexpected error during motion analysis: {str(e)}"
        print(f"Error: {error_msg}")
        metadata['processing']['motion_analysis'].update({
            'status': 'error',
            'message': error_msg,
            'error_type': type(e).__name__
        })
    
    return metadata

def analyze_video_editing(video_path: str, metadata: Dict[str, Any], temp_dir: str) -> Dict[str, Any]:
    """Analyze video editing patterns including scene cuts and pacing metrics.
    
    This function analyzes the editing patterns in a video by detecting scene changes
    and calculating various pacing metrics. It first attempts to load precomputed
    scene detection results, and falls back to on-the-fly detection if needed.
    
    The analysis includes:
    - Scene detection and segmentation
    - Scene duration statistics (mean, std, min, max)
    - Editing pace metrics (cuts per minute, shot length variability)
    - Scene transition analysis
    
    Args:
        video_path: Path to the input video file
        metadata: Dictionary containing existing video metadata
        temp_dir: Directory for storing temporary files during processing
        
    Returns:
        Updated metadata dictionary with editing analysis results, including:
        - scene_detection: Status and statistics of scene detection
        - editing: Pacing metrics and editing characteristics
        - scene_transitions: Analysis of transitions between scenes
        
    Example:
        >>> video_meta = {'duration_seconds': 30.0, 'processing': {}}
        >>> editing_meta = analyze_video_editing('video.mp4', video_meta, 'temp_dir')
        >>> print(f"Cuts per minute: {editing_meta['editing']['cuts_per_minute']:.1f}")
        >>> print(f"Average shot length: {editing_meta['editing']['avg_shot_length']:.2f}s")
        
    Note:
        - Prefers precomputed scene detection data if available
        - Falls back to on-the-fly detection using PySceneDetect
        - Requires 'duration_seconds' in metadata for pace calculations
    """
    # Initialize processing status
    if 'processing' not in metadata:
        metadata['processing'] = {}
    
    try:
        # Check if precomputed scene detection data exists
        scene_file = os.path.join(
            os.path.dirname(os.path.dirname(temp_dir)),  # Go up two levels from temp_dir
            'scene_detection',
            f"{os.path.splitext(os.path.basename(video_path))[0]}_scenes.json"
        )
        
        if os.path.exists(scene_file):
            # Load pre-computed scene detection
            with open(scene_file, 'r', encoding='utf-8') as f:
                scene_data = json.load(f)
            
            if scene_data.get('status') == 'success':
                scenes = scene_data.get('scene_list', [])
                scene_durations = [s['end_time'] - s['start_time'] for s in scenes]
                avg_scene_duration = np.mean(scene_durations) if scene_durations else 0
                std_scene_duration = np.std(scene_durations) if len(scene_durations) > 1 else 0
                
                # Update scene detection metadata
                metadata['processing']['scene_detection'] = {
                    'status': 'success',
                    'source': 'precomputed',
                    'file': scene_file,
                    'num_scenes': len(scenes),
                    'avg_scene_duration': round(avg_scene_duration, 2),
                    'std_scene_duration': round(std_scene_duration, 2),
                    'min_scene_duration': round(min(scene_durations, default=0), 2),
                    'max_scene_duration': round(max(scene_durations, default=0), 2),
                    'scene_list': scenes[:10]  # Include first 10 scenes as sample
                }
                
                # Calculate and add editing pace metrics
                duration_minutes = metadata.get('duration_seconds', 0) / 60
                cuts_per_minute = len(scenes) / duration_minutes if duration_minutes > 0 else 0
                scene_variability = std_scene_duration / avg_scene_duration if avg_scene_duration > 0 else 0
                
                metadata['editing'] = {
                    'cuts_per_minute': round(cuts_per_minute, 2),
                    'avg_shot_length': round(avg_scene_duration, 2),
                    'scene_variability': round(scene_variability, 3),
                    'has_rapid_cuts': cuts_per_minute > 6,  # More than 6 cuts per minute
                    'has_long_takes': avg_scene_duration > 10  # Average scene > 10 seconds
                }
                
                # Add scene transition analysis if available in scene_data
                if 'transitions' in scene_data:
                    metadata['editing']['transition_types'] = scene_data['transitions']
                
                return metadata
        
        # Fall back to direct scene detection if precomputed data not available
        print(f"Performing on-the-fly scene detection for {os.path.basename(video_path)}")
        scenes = detect_scenes(video_path)
        
        if not scenes:
            metadata['processing']['scene_detection'] = {
                'status': 'failed',
                'message': 'No scenes detected or error during detection',
                'source': 'computed',
                'attempted_method': 'PySceneDetect',
                'timestamp': datetime.now().isoformat()
            }
            return metadata
        
        # Calculate scene statistics
        scene_durations = [s['end_time'] - s['start_time'] for s in scenes]
        avg_scene_duration = np.mean(scene_durations) if scene_durations else 0
        std_scene_duration = np.std(scene_durations) if len(scene_durations) > 1 else 0
        duration_minutes = metadata.get('duration_seconds', 0) / 60
        cuts_per_minute = len(scenes) / duration_minutes if duration_minutes > 0 else 0
        scene_variability = std_scene_duration / avg_scene_duration if avg_scene_duration > 0 else 0
        
        # Add scene information to metadata
        metadata['processing']['scene_detection'] = {
            'status': 'success',
            'source': 'computed',
            'method': 'PySceneDetect',
            'num_scenes': len(scenes),
            'scene_list': scenes[:5],  # Include first 5 scenes as sample
            'avg_scene_duration': round(avg_scene_duration, 2),
            'std_scene_duration': round(std_scene_duration, 2),
            'min_scene_duration': round(min(scene_durations, default=0), 2),
            'max_scene_duration': round(max(scene_durations, default=0), 2),
            'detection_timestamp': datetime.now().isoformat()
        }
        
        # Add editing pace metrics
        metadata['editing'] = {
            'cuts_per_minute': cuts_per_minute,
            'avg_shot_length': avg_scene_duration,
            'scene_variability': scene_variability,
            'has_rapid_cuts': cuts_per_minute > 6,  # More than 6 cuts per minute
            'has_long_takes': avg_scene_duration > 10  # Average scene > 10 seconds
        }
        
    except Exception as e:
        metadata['processing']['scene_detection'] = {
            'status': 'error',
            'message': str(e)
        }
    
    return metadata

def process_videos(
    video_dir: str, 
    output_dir: str, 
    analyze_scenes: bool = True,
    analyze_audio: bool = True,
    analyze_motion: bool = True,
    temp_dir: str = None
) -> List[Dict]:
    """Process all videos in a directory and save metadata.
    
    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save metadata JSON files
        analyze_scenes: Whether to perform scene detection analysis
        analyze_audio: Whether to perform audio feature extraction
        analyze_motion: Whether to perform motion analysis
        temp_dir: Directory for temporary files (defaults to system temp)
        
    Returns:
        List of metadata dictionaries for all processed videos
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up temporary directory
    if temp_dir is None:
        temp_dir = os.path.join(tempfile.gettempdir(), 'video_analysis')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Find all video files in the directory
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    video_files = [f for f in os.listdir(video_dir) 
                  if f.lower().endswith(video_extensions)]
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return []
    
    all_metadata = []
    
    # Process each video file
    for video_file in video_files:
        try:
            video_path = os.path.join(video_dir, video_file)
            print(f"\nProcessing: {video_file}")
            
            # Create a subdirectory for this video's temporary files
            video_temp_dir = os.path.join(temp_dir, os.path.splitext(video_file)[0])
            os.makedirs(video_temp_dir, exist_ok=True)
            
            # Initialize metadata with processing info
            metadata = {
                'filename': os.path.basename(video_path),
                'file_path': os.path.abspath(video_path),
                'processing': {
                    'scene_detection': {'status': 'pending'},
                    'audio_analysis': {'status': 'pending'},
                    'motion_analysis': {'status': 'pending'}
                },
                'created_at': datetime.now().isoformat()
            }
            
            # Extract basic video metadata
            metadata.update(extract_video_metadata(video_path))
            
            # Analyze video editing patterns if requested
            if analyze_scenes:
                print("  - Analyzing scene cuts...")
                metadata = analyze_video_editing(video_path, metadata, video_temp_dir)
            else:
                metadata['processing']['scene_detection'] = {'status': 'skipped'}
            
            # Analyze audio features if requested
            if analyze_audio and 'audio' not in metadata:
                print("  - Extracting audio features...")
                metadata = _analyze_audio(video_path, metadata, video_temp_dir)
            elif not analyze_audio:
                metadata['processing']['audio_analysis'] = {'status': 'skipped'}
            
            # Analyze motion if requested
            if analyze_motion and 'motion' not in metadata:
                print("  - Analyzing motion...")
                metadata = _analyze_motion(video_path, metadata, video_temp_dir)
            elif not analyze_motion:
                metadata['processing']['motion_analysis'] = {'status': 'skipped'}
            
            # Clean up temporary files
            try:
                shutil.rmtree(video_temp_dir, ignore_errors=True)
            except Exception as e:
                print(f"  - Warning: Could not clean up temporary files: {e}")
            
            # Save metadata to JSON file
            output_file = os.path.splitext(video_file)[0] + "_metadata.json"
            output_path = os.path.join(output_dir, output_file)
            
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ Saved metadata to {output_path}")
            all_metadata.append(metadata)
            
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Clean up temporary directory if it's empty
    try:
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
    except Exception as e:
        print(f"Warning: Could not remove temporary directory {temp_dir}: {e}")
    
    return all_metadata

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract metadata, audio features, and motion analysis from video files"
    )
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="../data/raw_videos",
        help="Directory containing input video files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="../data/metadata",
        help="Directory to save metadata files"
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default=None,
        help="Directory for temporary files (defaults to system temp)"
    )
    parser.add_argument(
        "--no_scenes",
        action="store_false",
        dest="analyze_scenes",
        help="Disable scene detection analysis"
    )
    parser.add_argument(
        "--no_audio",
        action="store_false",
        dest="analyze_audio",
        help="Disable audio feature extraction"
    )
    parser.add_argument(
        "--no_motion",
        action="store_false",
        dest="analyze_motion",
        help="Disable motion analysis"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    print(f"Starting video analysis with scene detection {'enabled' if args.analyze_scenes else 'disabled'}")
    process_videos(args.input_dir, args.output_dir, args.analyze_scenes)
