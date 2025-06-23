import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import json

class VideoStylePredictor:
    def __init__(self, model_path, num_classes, device='cuda'):
        """Initialize the predictor with a trained model."""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path, num_classes)
        self.transform = self._get_transforms()
        
    def _load_model(self, model_path, num_classes):
        """Load the trained model."""
        model = VideoStyleModel(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model
    
    def _get_transforms(self):
        """Get the same transforms used during training."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_video(self, video_path, num_frames=16):
        """Preprocess video into tensor of frames."""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames
        frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        frames = []
        
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.transform(frame)
                frames.append(frame)
        
        cap.release()
        
        if not frames:
            raise ValueError(f"Could not extract frames from {video_path}")
            
        return torch.stack(frames).unsqueeze(0)  # Add batch dimension
    
    def predict(self, video_path, class_names=None):
        """Predict the style of a video."""
        try:
            # Preprocess video
            inputs = self.preprocess_video(video_path).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(inputs)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
            # Prepare result
            result = {
                'video_path': video_path,
                'predicted_class': int(predicted[0]),
                'confidence': float(confidence[0]),
                'class_probabilities': probabilities[0].cpu().numpy().tolist()
            }
            
            if class_names:
                result['predicted_style'] = class_names[result['predicted_class']]
                result['class_names'] = class_names
                
            return result
            
        except Exception as e:
            return {
                'video_path': video_path,
                'error': str(e)
            }
    
    def predict_batch(self, video_dir, class_names=None, output_file=None):
        """Predict styles for all videos in a directory."""
        video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        results = []
        for video_path in tqdm(video_paths, desc="Processing videos"):
            result = self.predict(video_path, class_names)
            results.append(result)
            print(f"Processed {os.path.basename(video_path)}: {result.get('predicted_style', 'Error')}")
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved predictions to {output_file}")
        
        return results

class VideoStyleModel(nn.Module):
    """Same model architecture as used in training."""
    def __init__(self, num_classes):
        super(VideoStyleModel, self).__init__()
        self.model = models.video.r3d_18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.model(x)

def main():
    parser = argparse.ArgumentParser(description='Predict video styles using a trained model')
    parser.add_argument('--model_path', required=True,
                       help='Path to the trained model weights')
    parser.add_argument('--video_path', help='Path to a single video file')
    parser.add_argument('--video_dir', help='Directory containing video files')
    parser.add_argument('--output_file', help='File to save predictions (JSON)')
    parser.add_argument('--num_classes', type=int, required=True,
                       help='Number of style classes')
    parser.add_argument('--class_names', nargs='+',
                       help='List of class names (must match training order)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = VideoStylePredictor(
        model_path=args.model_path,
        num_classes=args.num_classes
    )
    
    # Make predictions
    if args.video_path:
        result = predictor.predict(args.video_path, args.class_names)
        print("\nPrediction Result:")
        print(json.dumps(result, indent=2))
    elif args.video_dir:
        results = predictor.predict_batch(
            video_dir=args.video_dir,
            class_names=args.class_names,
            output_file=args.output_file
        )
    else:
        print("Please provide either --video_path or --video_dir")

if __name__ == '__main__':
    main()
