# SliceX - Advanced Video Style Analysis

![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

SliceX is a powerful tool for analyzing and extracting features from videos, designed for video editors, content creators, and researchers. It provides comprehensive analysis of video content including scene detection, audio features, visual characteristics, and motion analysis.

## ✨ Features

- **Scene Detection**: Automatically detect scene changes using advanced algorithms
- **Audio Analysis**: Extract detailed audio features including music/speech classification
- **Visual Features**: Analyze color distribution, brightness, contrast, and more
- **Motion Analysis**: Detect camera movements and motion patterns
- **Zoom Detection**: Identify zoom effects and their characteristics
- **Command Line Interface**: Easy-to-use CLI for processing videos
- **Batch Processing**: Process multiple videos or entire directories
- **CSV Export**: Save analysis results in a structured format

## 🚀 Installation

### Prerequisites

- Python 3.8, 3.9, or 3.10
- FFmpeg (for audio extraction)

### Using pip

```bash
pip install slicerx
```

### From Source

1. Clone the repository:
   ```bash
   git clone https://github.com/strasvision/slicex.git
   cd slicex
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install with development dependencies:
   ```bash
   pip install -e .[dev]
   ```

## 🛠️ Usage

### Basic Usage

```bash
# Extract features from a single video
slicerx-extract input_video.mp4 --output features.csv

# Process all videos in a directory
slicerx-extract path/to/videos/ --recursive --output output/
```

### Command Line Options

```
usage: slicerx-extract [-h] [--output OUTPUT] [--threshold THRESHOLD] [--sample-rate SAMPLE_RATE] [--recursive] video_path

Extract features from video files.

positional arguments:
  video_path            Path to the input video file or directory

options:
  -h, --help            show this help message and exit
  --output OUTPUT, -o OUTPUT
                        Output CSV file or directory (default: output/features.csv)
  --threshold THRESHOLD, -t THRESHOLD
                        Threshold for scene detection (default: 30.0)
  --sample-rate SAMPLE_RATE, -s SAMPLE_RATE
                        Frames per second to sample (default: 1.0)
  --recursive, -r       Process videos in subdirectories recursively
```

## 📊 Features Extracted

### Audio Features
- RMS Energy
- Zero Crossing Rate
- Spectral Features (Centroid, Bandwidth, Rolloff, Flatness)
- MFCCs (Mel-frequency cepstral coefficients)
- Chroma Features
- Tempo Estimation
- Music/Speech Classification

### Visual Features
- Brightness, Contrast, Sharpness
- Color Histograms (RGB, HSV, LAB)
- Edge and Corner Density
- Motion Intensity and Direction
- Zoom Detection
- Composition Analysis (Rule of Thirds)

### Scene Information
- Scene Boundaries
- Duration
- Shot Transitions

## 🏗️ Project Structure

```
slicex/
├── scripts/                  # Main Python scripts
│   ├── extract_features.py    # Feature extraction pipeline
│   ├── train_model.py         # Model training
│   └── predict_style.py       # Style prediction
├── tests/                     # Test files
├── models/                    # Pre-trained models
├── data/                      # Example data
├── requirements.txt           # Production dependencies
└── setup.py                   # Package configuration
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with Python and OpenCV
- Uses PySceneDetect for scene detection
- Audio analysis powered by LibROSA
- Inspired by video content analysis research

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
