#!/usr/bin/env python3
import os
import sys
import numpy as np
import logging
import argparse
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)

def load_features(feature_dir):
    """
    Load features from the unified feature directory
    
    Args:
        feature_dir: Directory containing unified features
        
    Returns:
        Dictionary with features, labels, and metadata
    """
    # Add feature directory to path for importing load_features
    sys.path.insert(0, feature_dir)
    
    try:
        # Try to import the load_features module
        from load_features import load_dataset_arrays, load_features_for_file, get_feature_vectors
        
        # Load metadata
        metadata_path = os.path.join(feature_dir, "feature_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            logging.error(f"Metadata file not found: {metadata_path}")
            return None
        
        # Check if we have a unified dataset file
        unified_dataset_path = os.path.join(feature_dir, "unified_dataset.npz")
        if os.path.exists(unified_dataset_path):
            logging.info(f"Loading unified dataset from {unified_dataset_path}")
            dataset = np.load(unified_dataset_path, allow_pickle=True)
            return {
                'dataset': dataset,
                'metadata': metadata,
                'load_features_for_file': load_features_for_file,
                'get_feature_vectors': get_feature_vectors
            }
        
        # Otherwise, load individual files
        logging.info(f"Loading individual feature files from {feature_dir}")
        dataset = load_dataset_arrays()
        return {
            'dataset': dataset,
            'metadata': metadata,
            'load_features_for_file': load_features_for_file,
            'get_feature_vectors': get_feature_vectors
        }
    
    except ImportError as e:
        logging.error(f"Error importing load_features module: {e}")
        return None
    except Exception as e:
        logging.error(f"Error loading features: {e}")
        return None

def prepare_feature_vectors(feature_data, feature_types):
    """
    Prepare feature vectors for model training
    
    Args:
        feature_data: Feature data loaded from load_features
        feature_types: List of feature types to include
        
    Returns:
        X, y arrays for model training
    """
    dataset = feature_data['dataset']
    load_features_for_file = feature_data['load_features_for_file']
    get_feature_vectors = feature_data['get_feature_vectors']
    
    # Check if we have filenames and labels
    if 'filenames' not in dataset or 'labels' not in dataset:
        logging.error("Dataset missing required fields (filenames or labels)")
        return None, None
    
    filenames = dataset['filenames']
    labels = dataset['labels']
    
    # Prepare feature vectors
    X = []
    y = []
    
    for i, filename in enumerate(filenames):
        try:
            # Load features for this file
            features = load_features_for_file(filename)
            if features is None:
                logging.warning(f"Could not load features for {filename}")
                continue
            
            # Get feature vector with selected types
            feature_vector = get_feature_vectors(features, feature_types)
            if feature_vector is None or len(feature_vector) == 0:
                logging.warning(f"No features extracted for {filename} with types {feature_types}")
                continue
            
            X.append(feature_vector)
            y.append(labels[i])
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
    
    if len(X) == 0:
        logging.error("No valid feature vectors extracted")
        return None, None
    
    return np.array(X), np.array(y)

def train_and_evaluate(X, y, class_names, model_name="Random Forest", n_estimators=100):
    """
    Train and evaluate a model
    
    Args:
        X: Feature matrix
        y: Labels
        class_names: List of class names
        model_name: Name of the model
        n_estimators: Number of estimators for RF
        
    Returns:
        Trained model and evaluation metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logging.info(f"Training {model_name} with {X_train.shape[1]} features")
    logging.info(f"Training set: {X_train.shape[0]} samples")
    logging.info(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    if model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    else:
        logging.error(f"Unknown model type: {model_name}")
        return None, None
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    logging.info(f"Training accuracy: {train_score:.4f}")
    logging.info(f"Test accuracy: {test_score:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    logging.info(f"Cross-validation scores: {cv_scores}")
    logging.info(f"Mean CV score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Detailed evaluation on test set
    y_pred = model.predict(X_test)
    
    # Get the unique classes in the test set
    unique_classes = np.unique(np.concatenate((y_test, y_pred)))
    
    # Filter class_names to only include classes that are in the data
    filtered_class_names = [class_names[i] for i in unique_classes if i < len(class_names)]
    
    # Generate classification report with filtered class names
    report = classification_report(y_test, y_pred, target_names=filtered_class_names)
    logging.info(f"Classification report:\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Return model and metrics
    metrics = {
        'train_score': train_score,
        'test_score': test_score,
        'cv_scores': cv_scores,
        'classification_report': report,
        'confusion_matrix': cm,
        'filtered_class_names': filtered_class_names
    }
    
    return model, metrics

def plot_confusion_matrix(cm, class_names, title, output_file=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    
    if output_file:
        plt.savefig(output_file)
        logging.info(f"Saved confusion matrix to {output_file}")
    else:
        plt.show()

def plot_feature_importance(model, feature_names, title, output_file=None, top_n=20):
    """Plot feature importance"""
    if not hasattr(model, 'feature_importances_'):
        logging.warning("Model does not have feature_importances_ attribute")
        return
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame for plotting
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Plot top N features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(top_n))
    plt.title(title)
    
    if output_file:
        plt.savefig(output_file)
        logging.info(f"Saved feature importance plot to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Train models using unified features")
    parser.add_argument('--feature-dir', default='enhanced_features', 
                        help='Directory containing unified features')
    parser.add_argument('--output-dir', default='model_results', 
                        help='Directory to save model results')
    parser.add_argument('--feature-types', nargs='+', 
                        default=['rf', 'rhythm', 'spectral', 'tonal'],
                        help='Feature types to use (rf, rhythm, spectral, tonal)')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of estimators for Random Forest')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load features
    logging.info(f"Loading features from {args.feature_dir}")
    feature_data = load_features(args.feature_dir)
    
    if feature_data is None:
        logging.error("Failed to load features")
        return
    
    # Get class names
    metadata = feature_data['metadata']
    class_names = metadata.get('class_names', [])
    
    # Try different feature combinations
    feature_combinations = [
        ['rf'],  # Basic RF features only
        ['rf', 'rhythm'],  # RF + rhythm
        ['rf', 'spectral'],  # RF + spectral
        ['rf', 'tonal'],  # RF + tonal
        ['rf', 'rhythm', 'spectral'],  # RF + rhythm + spectral
        ['rf', 'rhythm', 'spectral', 'tonal']  # All features
    ]
    
    # Filter to only include requested feature types
    feature_combinations = [
        combo for combo in feature_combinations 
        if all(ft in args.feature_types for ft in combo)
    ]
    
    # If no valid combinations, use the requested types directly
    if not feature_combinations:
        feature_combinations = [args.feature_types]
    
    # Train models with different feature combinations
    results = {}
    
    for combo in feature_combinations:
        combo_name = '+'.join(combo)
        logging.info(f"\n{'='*20} Training with {combo_name} features {'='*20}")
        
        # Prepare feature vectors
        X, y = prepare_feature_vectors(feature_data, combo)
        
        if X is None or y is None:
            logging.error(f"Failed to prepare feature vectors for {combo_name}")
            continue
        
        # Train and evaluate model
        model, metrics = train_and_evaluate(
            X, y, class_names, 
            model_name="Random Forest", 
            n_estimators=args.n_estimators
        )
        
        if model is None:
            logging.error(f"Failed to train model for {combo_name}")
            continue
        
        # Save results
        results[combo_name] = {
            'model': model,
            'metrics': metrics,
            'feature_count': X.shape[1]
        }
        
        # Plot confusion matrix
        cm_file = os.path.join(args.output_dir, f"confusion_matrix_{combo_name}.png")
        plot_confusion_matrix(
            metrics['confusion_matrix'], 
            metrics['filtered_class_names'], 
            f"Confusion Matrix - {combo_name}",
            cm_file
        )
        
        # Plot feature importance
        # We need to get feature names for this combination
        feature_names = []
        if 'rf' in combo and 'rf_feature_names' in metadata:
            feature_names.extend(metadata['rf_feature_names'])
        if 'rhythm' in combo and 'rhythm_features' in metadata:
            feature_names.extend(metadata['rhythm_features'])
        if 'spectral' in combo and 'spectral_features' in metadata:
            feature_names.extend(metadata['spectral_features'])
        if 'tonal' in combo and 'tonal_features' in metadata:
            feature_names.extend(metadata['tonal_features'])
        
        if feature_names and len(feature_names) == X.shape[1]:
            fi_file = os.path.join(args.output_dir, f"feature_importance_{combo_name}.png")
            plot_feature_importance(
                model, 
                feature_names, 
                f"Feature Importance - {combo_name}",
                fi_file
            )
        else:
            logging.warning(f"Feature names list length ({len(feature_names) if feature_names else 0}) does not match feature count ({X.shape[1]}). Skipping feature importance plot.")
    
    # Compare results
    logging.info("\n\n" + "="*30 + " RESULTS SUMMARY " + "="*30)
    summary = []
    
    for combo_name, result in results.items():
        metrics = result['metrics']
        summary.append({
            'Feature Set': combo_name,
            'Feature Count': result['feature_count'],
            'Train Accuracy': metrics['train_score'],
            'Test Accuracy': metrics['test_score'],
            'CV Score': metrics['cv_scores'].mean(),
            'CV Std': metrics['cv_scores'].std()
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('Test Accuracy', ascending=False)
    
    # Print summary
    logging.info("\nModel Performance Summary:")
    logging.info("\n" + summary_df.to_string(index=False))
    
    # Save summary to CSV
    summary_file = os.path.join(args.output_dir, "model_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    logging.info(f"\nSaved summary to {summary_file}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Feature Set', y='Test Accuracy', data=summary_df)
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    comparison_file = os.path.join(args.output_dir, "model_comparison.png")
    plt.savefig(comparison_file)
    logging.info(f"Saved comparison plot to {comparison_file}")

if __name__ == "__main__":
    main() 