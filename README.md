# Video Style Analysis and Classification

This project provides tools for analyzing and classifying video editing styles using machine learning. It includes utilities for extracting video features (scene cuts, audio features, motion analysis), training classification models, and making predictions on new videos.

The system is designed to work with small datasets and provides interpretable results about the editing style of input videos.

## Project Structure

```
video-style-project/
├── data/
│   ├── raw_videos/          # Input MP4 videos
│   ├── processed/           # Processed data and features
│   └── annotations/         # JSON annotations and metadata
├── models/                  # Trained models and scalers
│   └── style_classifier/    # Best model and artifacts
│       ├── best_model.joblib    # Trained model
│       ├── scaler.joblib        # Feature scaler
│       ├── selected_features.txt # Features used for training
│       └── training_summary.json # Training metrics
├── scripts/
│   ├── extract_metadata.py     # Extract video metadata
│   ├── flatten_json.py         # Process nested JSON data
│   ├── train_model.py          # Train style classification model
│   ├── predict_style_sklearn.py # Make predictions (recommended)
│   └── feature_extraction.py   # Feature extraction utilities
├── notebooks/               # Jupyter notebooks for exploration
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install system dependencies (macOS):
   ```bash
   # Install ffmpeg for video processing
   brew install ffmpeg
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   The main dependencies include:
   - scikit-learn: For machine learning models
   - opencv-python: For video processing
   - librosa: For audio feature extraction
   - pandas/numpy: For data manipulation
   - scenedetect: For scene boundary detection
   - joblib: For model serialization

## Usage

### 1. Prepare Your Data

Place your training videos in the `data/raw_videos` directory. The filenames should be descriptive as they'll be used as identifiers.

### 2. Extract Video Features

Run the feature extraction pipeline:

```bash
python scripts/extract_metadata.py \
    --input_dir data/raw_videos \
    --output_dir data/processed
```

This will create JSON files with extracted features for each video.

### 3. Flatten Features (if needed)

Convert the nested JSON features into a flat CSV format:

```bash
python scripts/flatten_json.py \
    --input_dir data/processed \
    --output_file data/processed/video_features.csv
```

### 4. Train the Model

Train a style classification model:

```bash
python scripts/train_model.py \
    --data_path data/processed/video_features.csv \
    --target_column style_label \
    --output_dir models/style_classifier
```

The script will:
- Perform feature selection
- Train multiple models with cross-validation
- Save the best model and related artifacts
- Generate a training summary

### 5. Make Predictions

#### Single Video Prediction:
```bash
python scripts/predict_style_sklearn.py \
    --model_dir models/style_classifier \
    --video_path data/raw_videos/your_video.mp4
```

#### Batch Processing:
```bash
python scripts/predict_style_sklearn.py \
    --model_dir models/style_classifier \
    --video_dir data/raw_videos \
    --output_file predictions.csv
```

The output will show the predicted style and confidence scores for each video.

## Features Extracted

The system extracts the following features from videos:

- **Scene Analysis**:
  - Number of scenes
  - Average scene length
  - Scene length statistics (min, max, std)
  - Cut frequencies

- **Audio Features**:
  - Beat density
  - Spectral features (centroid, bandwidth, rolloff)
  - Zero-crossing rate
  - MFCCs (Mel-frequency cepstral coefficients)

- **Motion Analysis**:
  - Motion intensity
  - Motion entropy
  - Motion standard deviation

## Model Performance

The current best model is a Logistic Regression classifier with the following performance metrics (from cross-validation):

- Accuracy: 0.778 ± 0.157
- Precision (macro): 0.667 ± 0.236
- Recall (macro): 0.778 ± 0.157
- F1-score (macro): 0.704 ± 0.210

## Requirements

- Python 3.8+
- scikit-learn
- OpenCV (opencv-python)
- librosa
- pandas
- numpy
- scenedetect
- joblib
- tqdm

## Notebooks

The `notebooks/` directory contains Jupyter notebooks for exploratory data analysis and model testing.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with PyTorch and torchvision
- Uses pre-trained models from torchvision.models
- Inspired by various computer vision research papers
