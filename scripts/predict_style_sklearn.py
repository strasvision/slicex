#!/usr/bin/env python3
"""
Video Style Classification - Prediction Script

This script uses a pre-trained scikit-learn model to predict the editing style of input videos.
It handles both single video files and directories of videos, producing predictions with
confidence scores for each style class.

Features:
- Loads a trained scikit-learn model and associated artifacts
- Processes videos to extract the same features used during training
- Makes predictions with confidence scores
- Supports batch processing of multiple videos
- Saves results to CSV for further analysis

Example:
    # Predict style for a single video
    python predict_style_sklearn.py --model_dir models/style_classifier --video_path input.mp4
    
    # Process all videos in a directory
    python predict_style_sklearn.py --model_dir models/style_classifier --video_dir input_videos --output predictions.csv
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from scripts.feature_extraction import extract_features

def load_model(model_dir: str) -> tuple:
    """
    Load the trained model and related artifacts from disk.
    
    Args:
        model_dir: Directory containing the saved model artifacts
        
    Returns:
        Tuple containing:
            model: The trained scikit-learn model
            scaler: The feature scaler used during training
            class_names: List of target class names
            metadata: Dictionary containing model metadata
                (includes 'feature_columns' used during training)
                
    Raises:
        FileNotFoundError: If any required model artifacts are missing
        
    Example:
        >>> model, scaler, classes, meta = load_model('models/style_classifier')
    """
    # Define expected artifact paths
    model_path = os.path.join(model_dir, 'best_model.joblib')
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    features_path = os.path.join(model_dir, 'selected_features.txt')
    
    # Verify all required files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Selected features not found at {features_path}")
    
    # Load model and artifacts
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Load selected features
    with open(features_path, 'r', encoding='utf-8') as f:
        feature_columns = [line.strip() for line in f if line.strip()]
    
    # Class names based on the training data
    class_names = ['showcase', 'tutorial', 'vlog']
    
    return model, scaler, class_names, {'feature_columns': feature_columns}

def predict_video_style(
    video_path: str, 
    model, 
    scaler, 
    feature_columns: list
) -> dict:
    """
    Predict the editing style of a single video file.
    
    This function handles the complete prediction pipeline for a single video:
    1. Extracts features using the same process as training
    2. Ensures all required features are present
    3. Applies the same scaling as during training
    4. Makes a prediction with confidence scores
    
    Args:
        video_path: Path to the video file to analyze
        model: Trained scikit-learn model
        scaler: Fitted StandardScaler used during training
        feature_columns: List of feature names expected by the model
        
    Returns:
        Dictionary containing:
        - prediction: Predicted class index (int)
        - probabilities: List of probabilities for each class
        - success: Boolean indicating if prediction was successful
        - error: Error message if prediction failed (only if success=False)
        
    Example:
        >>> result = predict_video_style('video.mp4', model, scaler, ['feat1', 'feat2'])
        >>> if result['success']:
        ...     print(f"Predicted class: {result['prediction']}")
        ...     print(f"Probabilities: {result['probabilities']}")
    """
    try:
        # Extract features from the video
        features = extract_features(video_path)
        
        # Create a DataFrame with the same structure as training data
        features_df = pd.DataFrame([features])
        
        # Select only the features used during training
        missing_cols = set(feature_columns) - set(features_df.columns)
        for col in missing_cols:
            features_df[col] = 0  # Add missing columns with default value 0
        
        # Reorder columns to match training data
        features_df = features_df[feature_columns]
        
        # Scale the features using the same scaler as during training
        features_scaled = scaler.transform(features_df)
        
        # Make prediction and get probabilities
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        return {
            'prediction': int(prediction),
            'probabilities': probabilities.tolist(),
            'success': True
        }
        
    except Exception as e:
        return {
            'video_path': video_path,
            'error': str(e)
        }

def predict_videos_in_directory(
    video_dir: str,
    model,
    scaler,
    feature_columns: list,
    class_names: list = None,
    output_file: str = None
) -> list:
    """
    Process all videos in a directory and predict their editing styles.
    
    This function:
    1. Scans the specified directory for video files
    2. Processes each video using predict_video_style()
    3. Displays progress and results
    4. Optionally saves results to a CSV file
    
    Args:
        video_dir: Path to directory containing video files
        model: Trained scikit-learn model
        scaler: Fitted StandardScaler used during training
        feature_columns: List of feature names expected by the model
        class_names: Optional list of class names for display
        output_file: Optional path to save results as CSV
        
    Returns:
        List of result dictionaries, one per video, with keys:
        - video_path: Path to the video file
        - prediction: Predicted class index (if successful)
        - probabilities: List of class probabilities (if successful)
        - predicted_class_name: Class name (if class_names provided)
        - success: Boolean indicating if prediction was successful
        - error: Error message if prediction failed
        
    Example:
        >>> results = predict_videos_in_directory(
        ...     'input_videos/',
        ...     model,
        ...     scaler,
        ...     feature_columns,
        ...     class_names=['showcase', 'tutorial', 'vlog'],
        ...     output_file='predictions.csv'
        ... )
    """
    # Get list of video files
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [
        os.path.join(video_dir, f) for f in os.listdir(video_dir)
        if f.lower().endswith(video_extensions)
    ]
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return []
    
    results = []
    
    # Process each video
    for video_path in tqdm(video_files, desc="Processing videos"):
        try:
            # Predict style for the current video
            result = predict_video_style(video_path, model, scaler, feature_columns)
            
            # Add video path to results
            result['video_path'] = video_path
            
            # Add class name if class_names is provided
            if class_names and 'prediction' in result:
                result['predicted_class_name'] = class_names[result['prediction']]
            
            results.append(result)
            
            # Print the result
            if result['success']:
                print(f"\nVideo: {os.path.basename(video_path)}")
                print(f"Predicted class: {result.get('predicted_class_name', result.get('prediction', 'N/A'))}")
                
                # Print probabilities if available
                if 'probabilities' in result and class_names:
                    print("Probabilities:")
                    for cls, prob in zip(class_names, result['probabilities']):
                        print(f"  {cls}: {prob*100:.1f}%")
            else:
                print(f"\nError processing {os.path.basename(video_path)}: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            print(f"\nError processing {os.path.basename(video_path)}: {str(e)}")
            results.append({
                'video_path': video_path,
                'error': str(e),
                'success': False
            })
    
    # Save results to CSV if output file is specified
    if output_file and results:
        results_df = pd.DataFrame(results)
        
        # Explode probabilities if they exist
        if 'probabilities' in results_df.columns and class_names:
            prob_cols = [f'prob_{cls}' for cls in class_names]
            results_df[prob_cols] = pd.DataFrame(
                results_df['probabilities'].tolist(), 
                index=results_df.index
            )
            results_df = results_df.drop('probabilities', axis=1)
        
        # Save to CSV
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
    
    return results

def main() -> None:
    """
    Main entry point for the video style prediction script.
    
    This function:
    1. Parses command-line arguments
    2. Loads the trained model and associated artifacts
    3. Processes either a single video or directory of videos
    4. Displays and optionally saves the prediction results
    
    Command-line Arguments:
        --model_dir: Directory containing the trained model and artifacts
        --video_path: Path to a single video file to process
        --video_dir: Directory containing multiple videos to process
        --output: Optional path to save results as CSV
        
    Example:
        # Process a single video
        python predict_style_sklearn.py --model_dir models/style_classifier \
                                      --video_path input/video1.mp4 \
                                      --output predictions.csv
                                      
        # Process all videos in a directory
        python predict_style_sklearn.py --model_dir models/style_classifier \
                                      --video_dir input_videos/ \
                                      --output batch_predictions.csv
    """
    parser = argparse.ArgumentParser(description='Predict video editing styles using a trained model.')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing the trained model and artifacts')
    parser.add_argument('--video_path', type=str, default=None,
                        help='Path to a single video file')
    parser.add_argument('--video_dir', type=str, default=None,
                        help='Directory containing video files to process')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file for results')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.video_path and not args.video_dir:
        parser.error('Either --video_path or --video_dir must be specified')
    
    if args.video_path and args.video_dir:
        parser.error('Only one of --video_path or --video_dir can be specified')
    
    # Load the model and artifacts
    try:
        print(f"Loading model from {args.model_dir}...")
        model, scaler, class_names, metadata = load_model(args.model_dir)
        feature_columns = metadata['feature_columns']
        
        print(f"Model loaded successfully. Using {len(feature_columns)} features.")
        print(f"Available classes: {', '.join(class_names)}")
        
        # Process single video or directory of videos
        if args.video_path:
            # Process single video
            if not os.path.exists(args.video_path):
                print(f"Error: Video file not found: {args.video_path}")
                return
                
            print(f"\nProcessing video: {os.path.basename(args.video_path)}")
            result = predict_video_style(args.video_path, model, scaler, feature_columns)
            
            if result['success']:
                print("\nPrediction Results:")
                print(f"Video: {os.path.basename(args.video_path)}")
                print(f"Predicted class: {class_names[result['prediction']]} (class {result['prediction']})")
                print("\nClass Probabilities:")
                for i, prob in enumerate(result['probabilities']):
                    print(f"  {class_names[i]}: {prob*100:.1f}%")
                
                # Save single result to CSV if output specified
                if args.output:
                    df = pd.DataFrame([{
                        'video_path': args.video_path,
                        'predicted_class': result['prediction'],
                        'predicted_style': class_names[result['prediction']],
                        'confidence': result['probabilities'][result['prediction']]
                    }])
                    df.to_csv(args.output, index=False)
                    print(f"\nResults saved to {args.output}")
            else:
                print(f"\nError processing video: {result.get('error', 'Unknown error')}")
                
        elif args.video_dir:
            # Process directory of videos
            if not os.path.isdir(args.video_dir):
                print(f"Error: Directory not found: {args.video_dir}")
                return
                
            predict_videos_in_directory(
                video_dir=args.video_dir,
                model=model,
                scaler=scaler,
                feature_columns=feature_columns,
                class_names=class_names,
                output_file=args.output
            )
    
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_exc()
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
