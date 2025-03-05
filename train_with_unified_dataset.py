#!/usr/bin/env python3
"""
Train a model using the unified dataset

This script demonstrates how to use the unified dataset to train a Random Forest model.
It loads the features from the unified dataset, splits them into training and testing sets,
trains the model, and evaluates its performance.
"""

import os
import sys
import json
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train_model.log"),
        logging.StreamHandler()
    ]
)

def load_unified_dataset(dataset_path='backend/data/features/unified/unified_dataset.npz'):
    """
    Load the unified dataset
    """
    logging.info(f"Loading unified dataset from {dataset_path}")
    
    try:
        data = np.load(dataset_path, allow_pickle=True)
        
        # Convert to dictionaries for easier access
        metadata = data['metadata'].item()
        features = data['features'].item()
        sound_ids = data['sound_ids']
        class_mapping = data['class_mapping'].item()
        classes = data['classes']
        
        logging.info(f"Loaded dataset with {len(sound_ids)} sounds and {len(classes)} classes")
        
        return {
            'metadata': metadata,
            'features': features,
            'sound_ids': sound_ids,
            'class_mapping': class_mapping,
            'classes': classes
        }
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        return None

def prepare_rf_features(dataset):
    """
    Prepare features for Random Forest model
    """
    logging.info("Preparing features for Random Forest model")
    
    X = []
    y = []
    sound_ids = []
    
    for sound_id in dataset['sound_ids']:
        # Get the class index
        class_name = dataset['metadata'][sound_id]['class']
        class_index = dataset['class_mapping'][class_name]
        
        # Get the statistical features
        if 'statistical' in dataset['features'][sound_id]:
            # Convert statistical features dictionary to a flat array
            features = list(dataset['features'][sound_id]['statistical'].values())
            
            X.append(features)
            y.append(class_index)
            sound_ids.append(sound_id)
    
    return np.array(X), np.array(y), sound_ids

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    """
    Train a Random Forest classifier
    """
    logging.info(f"Training Random Forest with {n_estimators} estimators")
    
    start_time = time.time()
    
    # Create and train the model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    logging.info(f"Training completed in {training_time:.2f} seconds")
    
    return model

def evaluate_model(model, X_test, y_test, class_names):
    """
    Evaluate the model performance
    """
    logging.info("Evaluating model performance")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {accuracy:.4f}")
    
    # Get unique classes in the test set
    unique_classes = np.unique(np.concatenate([y_test, y_pred]))
    used_class_names = [class_names[i] for i in unique_classes]
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=used_class_names)
    logging.info(f"Classification Report:\n{report}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'used_class_names': used_class_names
    }

def plot_confusion_matrix(cm, class_names, output_path):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_path)
    logging.info(f"Saved confusion matrix to {output_path}")

def plot_feature_importance(model, feature_names, output_path):
    """
    Plot feature importance
    """
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot the top 20 features
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(min(20, len(indices))), importances[indices[:20]], align='center')
    plt.xticks(range(min(20, len(indices))), [feature_names[i] for i in indices[:20]], rotation=90)
    plt.tight_layout()
    plt.savefig(output_path)
    logging.info(f"Saved feature importance plot to {output_path}")

def save_model(model, output_dir, model_name, metadata=None):
    """
    Save the trained model and metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(output_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    logging.info(f"Saved model to {model_path}")
    
    # Save metadata if provided
    if metadata:
        metadata_path = os.path.join(output_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Saved model metadata to {metadata_path}")

def main():
    """
    Main function
    """
    import argparse
    parser = argparse.ArgumentParser(description='Train a model using the unified dataset')
    parser.add_argument('--dataset', default='backend/data/features/unified/unified_dataset.npz',
                      help='Path to the unified dataset')
    parser.add_argument('--output_dir', default='backend/data/features/results',
                      help='Directory to save model and results')
    parser.add_argument('--n_estimators', type=int, default=100,
                      help='Number of estimators for Random Forest')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='Proportion of data to use for testing')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_unified_dataset(args.dataset)
    if not dataset:
        logging.error("Failed to load dataset. Exiting.")
        return
    
    # Prepare features for Random Forest
    X, y, sound_ids = prepare_rf_features(dataset)
    logging.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, sound_ids, test_size=args.test_size, random_state=42, stratify=y
    )
    
    logging.info(f"Training set: {X_train.shape[0]} samples")
    logging.info(f"Testing set: {X_test.shape[0]} samples")
    
    # Train the model
    model = train_random_forest(X_train, y_train, n_estimators=args.n_estimators)
    
    # Evaluate the model
    class_names = [dataset['classes'][i] for i in range(len(dataset['classes']))]
    evaluation = evaluate_model(model, X_test, y_test, class_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        evaluation['confusion_matrix'],
        evaluation['used_class_names'],
        os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # Plot feature importance
    # Create feature names (these are just placeholders since we don't have the actual names)
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    plot_feature_importance(
        model,
        feature_names,
        os.path.join(args.output_dir, 'feature_importance.png')
    )
    
    # Save the model
    model_name = f"rf_model_{time.strftime('%Y%m%d%H%M%S')}"
    model_metadata = {
        'accuracy': float(evaluation['accuracy']),
        'n_estimators': args.n_estimators,
        'n_features': X.shape[1],
        'n_classes': len(evaluation['used_class_names']),
        'classes': evaluation['used_class_names'],
        'training_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    save_model(model, args.output_dir, model_name, model_metadata)
    
    # Save evaluation report
    report_path = os.path.join(args.output_dir, f"{model_name}_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {evaluation['accuracy']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(evaluation['report'])
    
    logging.info(f"Saved evaluation report to {report_path}")
    logging.info("Training and evaluation completed successfully!")

if __name__ == "__main__":
    main() 