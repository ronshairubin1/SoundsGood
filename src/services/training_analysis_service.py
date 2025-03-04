import os
import logging
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

class TrainingAnalysisService:
    """
    Service for analyzing training data and model performance.
    
    This service provides methods for retrieving and analyzing training statistics,
    model accuracy, class distributions, and feature importance.
    """
    
    def __init__(self):
        """
        Initialize the training analysis service.
        """
        self.training_stats = {}  # Store training statistics by model type
        self.model_metrics = {}   # Store model metrics by model type
        self.class_names = {}     # Store class names by model type
        self.training_data = {}   # Store references to training data by model type

    def register_training_results(self, model_type, stats, metrics, class_names, X_train=None, X_val=None, y_train=None, y_val=None):
        """
        Register training results for later analysis.
        
        Args:
            model_type (str): Type of model ('cnn', 'rf', 'ensemble')
            stats (dict): Training statistics
            metrics (dict): Model performance metrics
            class_names (list): List of class names
            X_train (ndarray, optional): Training features
            X_val (ndarray, optional): Validation features
            y_train (ndarray, optional): Training labels
            y_val (ndarray, optional): Validation labels
        """
        self.training_stats[model_type] = stats
        self.model_metrics[model_type] = metrics
        self.class_names[model_type] = class_names
        
        # Store references to training and validation data
        if X_train is not None and y_train is not None:
            self.training_data[model_type] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val
            }
        
        logging.info(f"Registered training results for {model_type} model")
    
    def get_model_accuracy(self, model_type):
        """
        Get the accuracy of a trained model.
        
        Args:
            model_type (str): Type of model ('cnn', 'rf', 'ensemble')
            
        Returns:
            float: Model accuracy or None if not available
        """
        if model_type not in self.model_metrics:
            logging.warning(f"No metrics available for {model_type} model")
            return None
        
        metrics = self.model_metrics[model_type]
        
        # Different models might store accuracy in different ways
        if model_type == 'cnn':
            return metrics.get('val_accuracy')
        elif model_type == 'rf':
            return metrics.get('accuracy')
        elif model_type == 'ensemble':
            return metrics.get('ensemble_accuracy')
        else:
            return None
    
    def get_model_class_names(self, model_type):
        """
        Get the class names for a trained model.
        
        Args:
            model_type (str): Type of model ('cnn', 'rf', 'ensemble')
            
        Returns:
            list: List of class names or empty list if not available
        """
        return self.class_names.get(model_type, [])
    
    def get_training_data_stats(self, model_type):
        """
        Get statistics about the training data.
        
        Args:
            model_type (str): Type of model ('cnn', 'rf', 'ensemble')
            
        Returns:
            dict: Training data statistics or empty dict if not available
        """
        if model_type not in self.training_stats:
            logging.warning(f"No training stats available for {model_type} model")
            return {}
        
        return self.training_stats[model_type]
    
    def get_evaluation_metrics(self, model_type):
        """
        Get evaluation metrics for a trained model.
        
        Args:
            model_type (str): Type of model ('cnn', 'rf', 'ensemble')
            
        Returns:
            dict: Evaluation metrics or empty dict if not available
        """
        if model_type not in self.model_metrics:
            logging.warning(f"No metrics available for {model_type} model")
            return {}
        
        return self.model_metrics[model_type]
    
    def get_feature_importance(self, model_type):
        """
        Get feature importance for a trained model.
        Only applicable for models that provide feature importance (e.g., RF).
        
        Args:
            model_type (str): Type of model (only 'rf' supported)
            
        Returns:
            dict: Feature importance dictionary or empty dict if not available
        """
        if model_type != 'rf' or model_type not in self.model_metrics:
            logging.warning(f"Feature importance not available for {model_type} model")
            return {}
        
        metrics = self.model_metrics[model_type]
        return metrics.get('feature_importance', {})
    
    def get_training_data(self, model_type):
        """
        Get references to training and validation data.
        
        Args:
            model_type (str): Type of model ('cnn', 'rf', 'ensemble')
            
        Returns:
            tuple: (X_train, X_val, y_train, y_val) or (None, None, None, None) if not available
        """
        if model_type not in self.training_data:
            logging.warning(f"No training data available for {model_type} model")
            return None, None, None, None
        
        data = self.training_data[model_type]
        return data.get('X_train'), data.get('X_val'), data.get('y_train'), data.get('y_val')
    
    def get_confusion_matrix(self, model_type):
        """
        Get the confusion matrix for a trained model.
        
        Args:
            model_type (str): Type of model ('cnn', 'rf', 'ensemble')
            
        Returns:
            ndarray: Confusion matrix or None if not available
        """
        if model_type not in self.model_metrics:
            logging.warning(f"No metrics available for {model_type} model")
            return None
        
        metrics = self.model_metrics[model_type]
        return metrics.get('confusion_matrix')
    
    def get_class_distribution(self, model_type):
        """
        Get the class distribution in the training data.
        
        Args:
            model_type (str): Type of model ('cnn', 'rf', 'ensemble')
            
        Returns:
            dict: Class distribution or empty dict if not available
        """
        if model_type not in self.training_stats:
            logging.warning(f"No training stats available for {model_type} model")
            return {}
        
        stats = self.training_stats[model_type]
        return stats.get('original_counts', {})
    
    def compare_models(self, model_types=None):
        """
        Compare multiple trained models.
        
        Args:
            model_types (list, optional): List of model types to compare.
                                         If None, compare all available models.
            
        Returns:
            dict: Comparison results
        """
        if model_types is None:
            model_types = list(self.model_metrics.keys())
        
        comparison = {
            'model_types': model_types,
            'accuracies': {},
            'training_times': {},
            'class_counts': {}
        }
        
        for model_type in model_types:
            comparison['accuracies'][model_type] = self.get_model_accuracy(model_type)
            
            # Get training time if available
            stats = self.get_training_data_stats(model_type)
            metrics = self.get_evaluation_metrics(model_type)
            
            if stats:
                comparison['class_counts'][model_type] = stats.get('original_counts', {})
            
            # Different models might store training time differently
            if model_type == 'cnn' and stats:
                comparison['training_times'][model_type] = stats.get('training_time')
            elif model_type == 'rf' and metrics:
                comparison['training_times'][model_type] = metrics.get('training_duration')
            else:
                comparison['training_times'][model_type] = None
        
        return comparison
    
    def generate_training_report(self, model_type):
        """
        Generate a comprehensive training report.
        
        Args:
            model_type (str): Type of model ('cnn', 'rf', 'ensemble')
            
        Returns:
            dict: Comprehensive training report
        """
        report = {
            'model_type': model_type,
            'accuracy': self.get_model_accuracy(model_type),
            'class_names': self.get_model_class_names(model_type),
            'class_distribution': self.get_class_distribution(model_type),
            'metrics': self.get_evaluation_metrics(model_type)
        }
        
        # Add model-specific details
        if model_type == 'cnn':
            stats = self.get_training_data_stats(model_type)
            if 'history' in stats:
                report['training_history'] = stats['history']
        
        elif model_type == 'rf':
            report['feature_importance'] = self.get_feature_importance(model_type)
        
        return report 