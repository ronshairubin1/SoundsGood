import os
import logging
import numpy as np
from collections import defaultdict
import time
import json
from datetime import datetime

class InferenceAnalysisService:
    """
    Service for analyzing inference results.
    
    This service provides methods for tracking and analyzing model predictions,
    calculating accuracy metrics, and detecting performance degradation.
    """
    
    def __init__(self, max_history=1000):
        """
        Initialize the inference analysis service.
        
        Args:
            max_history (int): Maximum number of predictions to keep in history
        """
        self.max_history = max_history
        self.prediction_history = []  # List of prediction dictionaries
        self.class_predictions = defaultdict(int)  # Count of predictions by class
        self.class_accuracies = defaultdict(lambda: {'correct': 0, 'total': 0})  # Accuracy by class
        self.feedback_history = []  # List of feedback dictionaries
        self.inference_stats = {
            'total_predictions': 0,
            'total_correct': 0,
            'total_feedback': 0,
            'avg_confidence': 0.0,
            'avg_response_time': 0.0,
            'top_classes': [],
            'misclassifications': []
        }
    
    def record_prediction(self, prediction_data):
        """
        Record a new prediction for analysis.
        
        Args:
            prediction_data (dict): Dictionary containing prediction details
                Required keys: 'predicted_class', 'confidence'
                Optional keys: 'true_class', 'response_time', 'model_type'
                
        Returns:
            dict: Updated inference statistics
        """
        # Add timestamp if not present
        if 'timestamp' not in prediction_data:
            prediction_data['timestamp'] = datetime.now().isoformat()
        
        # Add to history
        self.prediction_history.append(prediction_data)
        if len(self.prediction_history) > self.max_history:
            self.prediction_history.pop(0)
        
        # Update class prediction counts
        predicted_class = prediction_data.get('predicted_class')
        if predicted_class:
            self.class_predictions[predicted_class] += 1
        
        # Update accuracy stats if true class is provided
        true_class = prediction_data.get('true_class')
        if true_class is not None:
            self.class_accuracies[true_class]['total'] += 1
            if true_class == predicted_class:
                self.class_accuracies[true_class]['correct'] += 1
            else:
                # Record misclassification
                misclass_data = {
                    'true_class': true_class,
                    'predicted_class': predicted_class,
                    'confidence': prediction_data.get('confidence', 0.0),
                    'timestamp': prediction_data.get('timestamp')
                }
                self.inference_stats['misclassifications'].append(misclass_data)
                if len(self.inference_stats['misclassifications']) > self.max_history:
                    self.inference_stats['misclassifications'].pop(0)
        
        # Update overall stats
        self.inference_stats['total_predictions'] += 1
        if true_class is not None and true_class == predicted_class:
            self.inference_stats['total_correct'] += 1
        
        # Update average confidence
        confidence = prediction_data.get('confidence', 0.0)
        old_avg = self.inference_stats['avg_confidence']
        count = self.inference_stats['total_predictions']
        self.inference_stats['avg_confidence'] = (old_avg * (count - 1) + confidence) / count
        
        # Update average response time if provided
        response_time = prediction_data.get('response_time')
        if response_time is not None:
            old_avg = self.inference_stats['avg_response_time']
            count = self.inference_stats['total_predictions']
            self.inference_stats['avg_response_time'] = (old_avg * (count - 1) + response_time) / count
        
        # Update top classes
        top_classes = sorted(self.class_predictions.items(), key=lambda x: x[1], reverse=True)[:5]
        self.inference_stats['top_classes'] = [{'class': c, 'count': count} for c, count in top_classes]
        
        return self.inference_stats
    
    def record_feedback(self, feedback_data):
        """
        Record user feedback about a prediction.
        
        Args:
            feedback_data (dict): Dictionary containing feedback details
                Required keys: 'predicted_class', 'actual_class', 'is_correct'
                Optional keys: 'confidence', 'timestamp', 'model_type'
                
        Returns:
            dict: Updated inference statistics
        """
        # Add timestamp if not present
        if 'timestamp' not in feedback_data:
            feedback_data['timestamp'] = datetime.now().isoformat()
        
        # Add to feedback history
        self.feedback_history.append(feedback_data)
        if len(self.feedback_history) > self.max_history:
            self.feedback_history.pop(0)
        
        # Update stats
        self.inference_stats['total_feedback'] += 1
        
        # Record as true class for accuracy calculation
        predicted_class = feedback_data.get('predicted_class')
        actual_class = feedback_data.get('actual_class')
        is_correct = feedback_data.get('is_correct', False)
        
        if actual_class:
            self.class_accuracies[actual_class]['total'] += 1
            if is_correct:
                self.class_accuracies[actual_class]['correct'] += 1
            elif predicted_class:
                # Record misclassification
                misclass_data = {
                    'true_class': actual_class,
                    'predicted_class': predicted_class,
                    'confidence': feedback_data.get('confidence', 0.0),
                    'timestamp': feedback_data.get('timestamp'),
                    'from_feedback': True
                }
                self.inference_stats['misclassifications'].append(misclass_data)
                if len(self.inference_stats['misclassifications']) > self.max_history:
                    self.inference_stats['misclassifications'].pop(0)
        
        return self.inference_stats
    
    def get_inference_stats(self):
        """
        Get comprehensive inference statistics.
        
        Returns:
            dict: Inference statistics
        """
        # Calculate overall accuracy
        total_correct = self.inference_stats['total_correct']
        total_preds = self.inference_stats['total_predictions']
        overall_accuracy = total_correct / total_preds if total_preds > 0 else 0.0
        
        # Calculate class-specific accuracies
        class_accuracy = {}
        for class_name, stats in self.class_accuracies.items():
            if stats['total'] > 0:
                class_accuracy[class_name] = stats['correct'] / stats['total']
            else:
                class_accuracy[class_name] = 0.0
        
        # Add to stats
        full_stats = self.inference_stats.copy()
        full_stats['overall_accuracy'] = overall_accuracy
        full_stats['class_accuracy'] = class_accuracy
        full_stats['prediction_counts'] = dict(self.class_predictions)
        
        # Calculate confusion sources (classes often confused with each other)
        confusion_sources = defaultdict(lambda: defaultdict(int))
        for misclass in self.inference_stats['misclassifications']:
            true_class = misclass.get('true_class')
            pred_class = misclass.get('predicted_class')
            if true_class and pred_class:
                confusion_sources[true_class][pred_class] += 1
        
        # Convert to regular dict for JSON serialization
        full_stats['confusion_sources'] = {
            true_class: {
                pred_class: count for pred_class, count in sources.items()
            } for true_class, sources in confusion_sources.items()
        }
        
        return full_stats
    
    def get_recent_predictions(self, count=10):
        """
        Get the most recent predictions.
        
        Args:
            count (int): Number of predictions to return
            
        Returns:
            list: List of recent prediction dictionaries
        """
        return self.prediction_history[-count:]
    
    def get_class_accuracy(self, class_name):
        """
        Get the accuracy for a specific class.
        
        Args:
            class_name (str): Name of the class
            
        Returns:
            float: Accuracy for the specified class
        """
        stats = self.class_accuracies.get(class_name, {'correct': 0, 'total': 0})
        return stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
    
    def get_confusion_matrix(self):
        """
        Calculate a confusion matrix from the prediction history.
        
        Returns:
            dict: Confusion matrix as a nested dictionary
        """
        class_names = list(self.class_accuracies.keys())
        confusion = {true_class: {pred_class: 0 for pred_class in class_names} 
                    for true_class in class_names}
        
        # Count occurrences in prediction history
        for pred in self.prediction_history:
            true_class = pred.get('true_class')
            pred_class = pred.get('predicted_class')
            if true_class in class_names and pred_class in class_names:
                confusion[true_class][pred_class] += 1
        
        return confusion
    
    def detect_drift(self, window_size=100):
        """
        Detect potential model drift based on recent accuracy.
        
        Args:
            window_size (int): Number of recent predictions to analyze
            
        Returns:
            dict: Drift analysis results
        """
        if len(self.prediction_history) < window_size * 2:
            return {'drift_detected': False, 'message': 'Not enough data for drift analysis'}
        
        # Get accuracy for earlier and recent windows
        earlier = self.prediction_history[-window_size*2:-window_size]
        recent = self.prediction_history[-window_size:]
        
        # Calculate accuracies
        earlier_correct = sum(1 for p in earlier if p.get('true_class') == p.get('predicted_class') 
                            and p.get('true_class') is not None)
        recent_correct = sum(1 for p in recent if p.get('true_class') == p.get('predicted_class')
                           and p.get('true_class') is not None)
        
        earlier_total = sum(1 for p in earlier if p.get('true_class') is not None)
        recent_total = sum(1 for p in recent if p.get('true_class') is not None)
        
        if earlier_total == 0 or recent_total == 0:
            return {'drift_detected': False, 'message': 'Not enough labeled data for drift analysis'}
        
        earlier_acc = earlier_correct / earlier_total
        recent_acc = recent_correct / recent_total
        
        # Check for significant drop
        threshold = 0.1  # 10% drop in accuracy
        drift_detected = (earlier_acc - recent_acc) > threshold
        
        return {
            'drift_detected': drift_detected,
            'earlier_accuracy': earlier_acc,
            'recent_accuracy': recent_acc,
            'accuracy_delta': earlier_acc - recent_acc,
            'threshold': threshold,
            'window_size': window_size
        }
    
    def get_confidence_distribution(self):
        """
        Calculate confidence distribution for predictions.
        
        Returns:
            dict: Confidence distribution statistics
        """
        if not self.prediction_history:
            return {'message': 'No prediction data available'}
        
        # Extract confidence values
        confidences = [p.get('confidence', 0.0) for p in self.prediction_history]
        
        # Calculate bins (0.0-0.1, 0.1-0.2, etc.)
        bins = {}
        for i in range(10):
            lower = i / 10
            upper = (i + 1) / 10
            count = sum(1 for c in confidences if lower <= c < upper)
            bins[f"{lower:.1f}-{upper:.1f}"] = count
        
        # Count confidences exactly 1.0
        bins["1.0"] = sum(1 for c in confidences if c == 1.0)
        
        # Calculate statistics
        return {
            'distribution': bins,
            'average': sum(confidences) / len(confidences),
            'min': min(confidences),
            'max': max(confidences),
            'median': sorted(confidences)[len(confidences)//2],
            'total_predictions': len(confidences)
        }
    
    def reset_stats(self):
        """
        Reset all inference statistics.
        
        Returns:
            dict: Fresh inference statistics
        """
        self.prediction_history = []
        self.class_predictions = defaultdict(int)
        self.class_accuracies = defaultdict(lambda: {'correct': 0, 'total': 0})
        self.feedback_history = []
        self.inference_stats = {
            'total_predictions': 0,
            'total_correct': 0,
            'total_feedback': 0,
            'avg_confidence': 0.0,
            'avg_response_time': 0.0,
            'top_classes': [],
            'misclassifications': []
        }
        
        return self.inference_stats 