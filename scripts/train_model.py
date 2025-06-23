"""
Video Style Classification Model Training

This script trains and evaluates machine learning models for classifying video editing styles.
It handles the complete ML pipeline including data loading, preprocessing, feature selection,
model training with cross-validation, and model persistence.

Example:
    python train_model.py --data_path data/processed/video_features.csv \
                         --target_column style_label \
                         --output_dir models/style_classifier
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    make_scorer
)
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import argparse
import warnings
from tqdm import tqdm
from typing import Tuple, Dict, Any, List, Union, Optional

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_and_preprocess_data(file_path: str, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
    """
    Load and preprocess the video metadata for model training.
    
    This function handles:
    - Loading CSV data
    - Separating features and target
    - Handling missing values
    - Encoding categorical variables
    
    Args:
        file_path: Path to the CSV file with video metadata
        target_column: Name of the target column (if available)
        
    Returns:
        Tuple containing:
            X: Feature matrix (DataFrame)
            y: Target vector (Series or None if no target_column)
            feature_names: List of feature names
            
    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If the target column is specified but not found
        
    Example:
        >>> X, y, features = load_and_preprocess_data('data.csv', 'style_label')
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Basic info
    print(f"\nOriginal data shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Drop columns with too many missing values
    threshold = len(df) * 0.8  # Keep columns with at least 80% non-null values
    df = df.dropna(axis=1, thresh=threshold)
    
    # Drop columns that are not useful for modeling
    drop_cols = [
        'source_file', 'video_id', 'filename', 'file_path',
        'created_at', 'processing.scene_detection.status',
        'processing.audio_analysis.status', 'processing.motion_analysis.status'
    ]
    drop_cols = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=drop_cols, errors='ignore')
    
    # Handle target variable if provided
    y = None
    if target_column and target_column in df.columns:
        # Encode target variable
        le = LabelEncoder()
        y = le.fit_transform(df[target_column])
        df = df.drop(columns=[target_column])
        
        # Save the label encoder
        os.makedirs('models', exist_ok=True)
        joblib.dump(le, 'models/label_encoder.joblib')
        print(f"\nEncoded {len(le.classes_)} classes: {list(le.classes_)}")
    
    # Separate features and handle missing values
    X = df.select_dtypes(include=['number'])
    
    # Fill remaining missing values with column means
    X = X.fillna(X.mean())
    
    print(f"\nFinal feature matrix shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    
    return X, y, X.columns.tolist()

def select_features(X: pd.DataFrame, y: pd.Series, k: int = 20) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select top k most important features using ANOVA F-value.
    
    This function performs feature selection using univariate statistical tests (ANOVA F-value)
    to identify the k most relevant features for the classification task. It also generates
    a feature importance plot for visualization.
    
    Args:
        X: Feature matrix (DataFrame with shape [n_samples, n_features])
        y: Target vector (Series with shape [n_samples,])
        k: Number of top features to select. If None, all features are used.
        
    Returns:
        Tuple containing:
            X_selected: DataFrame with selected features (shape [n_samples, k])
            selected_features: List of selected feature names (length k)
            
    Raises:
        ValueError: If k is greater than the number of available features
        
    Example:
        >>> X = pd.DataFrame({'feat1': [1, 2, 3], 'feat2': [4, 5, 6]})
        >>> y = pd.Series([0, 1, 0])
        >>> X_sel, features = select_features(X, y, k=1)
    """
    if k > X.shape[1]:
        k = X.shape[1]
        
    print(f"\nSelecting top {k} features...")
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    mask = selector.get_support()
    selected_features = [X.columns[i] for i in range(len(mask)) if mask[i]]
    
    # Print feature scores
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_,
        'P-value': selector.pvalues_
    }).sort_values('Score', ascending=False)
    
    print("\nTop features:")
    print(feature_scores.head(10).to_string())
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Score', y='Feature', data=feature_scores.head(20))
    plt.title('Top 20 Most Important Features (ANOVA F-value)')
    plt.tight_layout()
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/feature_importance.png')
    plt.close()
    
    return X_selected, selected_features

def train_and_evaluate(
    X: pd.DataFrame, 
    y: pd.Series, 
    feature_names: List[str], 
    output_dir: str = 'models'
) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
    """
    Train and evaluate multiple models with cross-validation for video style classification.
    
    This function performs the following steps:
    1. Standardizes the features
    2. Trains multiple models using cross-validation
    3. Evaluates each model using multiple metrics
    4. Saves the best model and related artifacts
    5. Generates performance visualizations
    
    Args:
        X: Feature matrix (DataFrame with shape [n_samples, n_features])
        y: Target vector (Series with shape [n_samples,])
        feature_names: List of feature names (length n_features)
        output_dir: Directory to save models and results
        
    Returns:
        Tuple containing:
            results: Dictionary with detailed results for all models
            best_model: The best performing model (fitted)
            best_metrics: Dictionary with metrics of the best model
            
    Raises:
        ValueError: If input data is empty or invalid
        
    Example:
        >>> X = pd.DataFrame({'feat1': [1, 2, 3], 'feat2': [4, 5, 6]})
        >>> y = pd.Series([0, 1, 0])
        >>> results, model, metrics = train_and_evaluate(X, y, ['feat1', 'feat2'])
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    
    # Define models to evaluate - using simpler models for small datasets
    models = {
        'Decision Tree': DecisionTreeClassifier(
            max_depth=3,  # Limit depth to prevent overfitting
            random_state=RANDOM_SEED,
            class_weight='balanced'
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,  # Increase max iterations for convergence
            random_state=RANDOM_SEED,
            class_weight='balanced',
            multi_class='multinomial',
            solver='lbfgs'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=50,  # Fewer trees for speed
            max_depth=3,  # Limit depth
            random_state=RANDOM_SEED,
            class_weight='balanced'
        )
    }
    
    # Define scoring metrics for cross-validation
    scoring = {
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro',
    }
    
    # Use stratified k-fold cross-validation
    cv = StratifiedKFold(n_splits=min(5, len(np.unique(y))), 
                        shuffle=True, 
                        random_state=RANDOM_SEED)
    # Initialize variables for best model tracking
    best_metrics = {
        'f1': 0,
        'model': None,
        'name': None
    }
    
    results = {}
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X_scaled, y, 
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            return_estimator=True
        )
        
        # Calculate mean and std of metrics
        metrics = {}
        for metric in ['test_accuracy', 'test_precision_macro', 'test_recall_macro', 'test_f1_macro']:
            metrics[metric] = {
                'mean': np.mean(cv_results[metric]),
                'std': np.std(cv_results[metric])
            }
        
        # Store results
        results[name] = {
            'cv_results': cv_results,
            'metrics': metrics,
            'model': model,
            'best_estimator': cv_results['estimator'][np.argmax(cv_results['test_f1_macro'])]
        }
        
        # Print results
        print(f"\n{name} Cross-Validation Results:")
        for metric in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']:
            m = metrics[f'test_{metric}']
            print(f"{metric.capitalize()}: {m['mean']:.3f} ± {m['std']:.3f}")
        
        # Update best model based on F1 score
        if metrics['test_f1_macro']['mean'] > best_metrics['f1']:
            best_metrics = {
                'f1': metrics['test_f1_macro']['mean'],
                'model': results[name]['best_estimator'],  # This is the fitted estimator from CV
                'name': name
            }
    
    # Save the best model
    if best_metrics['model'] is not None:
        # Save the best model
        model_path = os.path.join(output_dir, 'best_model.joblib')
        joblib.dump(best_metrics['model'], model_path)
        
        # Save the feature names used for training
        feature_path = os.path.join(output_dir, 'selected_features.txt')
        with open(feature_path, 'w') as f:
            f.write('\n'.join(feature_names))
        
        print(f"\nBest model ({best_metrics['name']}) saved to {model_path}")
        print(f"Features saved to {feature_path}")
    
    return results, best_metrics['model'], best_metrics

def main():
    parser = argparse.ArgumentParser(description='Train video style classification model')
    parser.add_argument('--data_path', 
                        default='data/processed/video_metadata_*.csv',
                        help='Path pattern to the flattened metadata CSV file(s)')
    parser.add_argument('--output_dir', 
                        default='models',
                        help='Directory to save models and results')
    parser.add_argument('--target_column',
                        default=None,
                        help='Name of the target column for supervised learning')
    parser.add_argument('--num_features',
                        type=int,
                        default=20,
                        help='Number of top features to select')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)
    
    # Find the most recent metadata file if using wildcard
    if '*' in args.data_path:
        import glob
        files = glob.glob(args.data_path)
        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {args.data_path}")
        # Sort by modification time and get the most recent
        files.sort(key=os.path.getmtime, reverse=True)
        data_path = files[0]
        print(f"Using most recent file: {data_path}")
    else:
        data_path = args.data_path
    
    # Load and preprocess data
    X, y, feature_names = load_and_preprocess_data(
        file_path=data_path,
        target_column=args.target_column
    )
    
    # If no target column is provided, we can only do unsupervised learning
    if y is None:
        print("\nNo target column provided. Performing unsupervised learning...")
        # TODO: Add unsupervised learning (clustering) here
        return
    
    # Select top features
    X_selected, selected_features = select_features(X, y, k=args.num_features)
    
    # Save selected features
    with open(os.path.join(args.output_dir, 'selected_features.txt'), 'w') as f:
        f.write('\n'.join(selected_features))
    
    # Train and evaluate models
    results, best_model, best_metrics = train_and_evaluate(
        X_selected, 
        y, 
        selected_features,
        output_dir=args.output_dir
    )
    
    # Save results summary
    summary = {
        'best_model': best_metrics.get('name', 'None'),
        'num_samples': len(X),
        'num_features': X_selected.shape[1],
        'best_f1_score': best_metrics.get('f1', 0),
        'feature_columns': selected_features.tolist() if hasattr(selected_features, 'tolist') else list(selected_features)
    }
    
    # Add feature importances if available
    if hasattr(best_model, 'feature_importances_'):
        summary['feature_importance'] = dict(zip(selected_features, best_model.feature_importances_))
    elif hasattr(best_model, 'coef_'):
        # For linear models, use mean absolute coefficients as importance
        summary['feature_importance'] = dict(zip(selected_features, np.mean(np.abs(best_model.coef_), axis=0)))
    
    # Save the model using joblib
    model_path = os.path.join(args.output_dir, 'video_style_model.joblib')
    joblib.dump(best_model, model_path)
    
    # Save the summary
    with open(os.path.join(args.output_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nTraining complete!")
    print(f"Results and models saved to: {os.path.abspath(args.output_dir)}")
    print(f'Model saved to {model_path}')

if __name__ == '__main__':
    main()
