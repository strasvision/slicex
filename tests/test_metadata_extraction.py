"""Test cases for metadata extraction functionality."""
import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

# Import the functions we want to test
from scripts.extract_metadata import (
    extract_metadata,
    _analyze_audio,
    _analyze_motion,
    analyze_video_editing
)

class TestMetadataExtraction(unittest.TestCase):
    """Test metadata extraction functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.test_video = os.path.join(self.test_dir, "test_video.mp4")
        
        # Create a dummy video file
        with open(self.test_video, 'wb') as f:
            f.write(b'Dummy video data')
        
        # Basic metadata structure
        self.basic_metadata = {
            'duration_seconds': 30.0,
            'processing': {}
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            for root, dirs, files in os.walk(self.test_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.test_dir)
    
    @patch('cv2.VideoCapture')
    def test_analyze_motion_with_precomputed(self, mock_video_capture):
        """Test motion analysis with precomputed data."""
        # Create a mock motion analysis file
        motion_dir = os.path.join(self.test_dir, 'motion_analysis')
        os.makedirs(motion_dir, exist_ok=True)
        motion_file = os.path.join(motion_dir, 'test_video_motion.json')
        
        motion_data = {
            'mean_motion': 10.5,
            'std_motion': 2.3,
            'max_motion': 15.7,
            'mean_energy': 45.2,
            'motion_variation': 0.22,
            'motion_hist': [1, 2, 3, 4, 5],
            'timestamp': '2023-01-01T00:00:00Z'
        }
        
        with open(motion_file, 'w') as f:
            json.dump(motion_data, f)
        
        # Call the function
        result = _analyze_motion(
            self.test_video, 
            self.basic_metadata.copy(), 
            os.path.join(self.test_dir, 'temp')
        )
        
        # Check the results
        self.assertEqual(result['processing']['motion_analysis']['status'], 'success')
        self.assertEqual(result['motion']['mean_motion'], 10.5)
        self.assertEqual(result['motion']['max_motion'], 15.7)
    
    @patch('librosa.load')
    def test_analyze_audio_success(self, mock_load):
        """Test audio analysis with successful processing."""
        # Mock librosa.load to return dummy audio data
        mock_load.return_value = (np.random.rand(16000), 16000)  # 1 second of random audio
        
        # Mock the temp file creation
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = os.path.join(self.test_dir, 'temp_audio.wav')
            
            # Call the function
            result = _analyze_audio(
                self.test_video,
                self.basic_metadata.copy(),
                self.test_dir
            )
            
            # Check the results
            self.assertEqual(result['processing']['audio_analysis']['status'], 'success')
            self.assertIn('audio', result)
            self.assertIn('tempo', result['audio'])

if __name__ == '__main__':
    unittest.main()
