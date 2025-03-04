import os
import logging
import numpy as np
import json

from .base import BaseModel
from .cnn import CNNModel
from .rf import RandomForestModel

class EnsembleModel(BaseModel):
    """
    Ensemble model implementation that combines CNN and RandomForest models.
    Provides a weighted combination of predictions from both models.
    """
    
    def __init__(self, model_dir='models'):
        """
        Initialize the Ensemble model.
        
        Args:
            model_dir (str): Directory for model storage
        """
        super().__init__(model_dir)
        self.cnn_model = CNNModel(model_dir)
        self.rf_model = RandomForestModel(model_dir)
        self.rf_weight = 0.5  # Default weight for RF model (CNN weight = 1 - rf_weight)
    
    def build(self, cnn_params=None, rf_params=None, rf_weight=0.5, **kwargs):
        """
        Build the Ensemble model by building both CNN and RF models.
        
        Args:
            cnn_params (dict): Parameters for CNN model building
            rf_params (dict): Parameters for RF model building
            rf_weight (float): Weight for RF model predictions (0-1)
            **kwargs: Additional parameters
            
        Returns:
            self: The ensemble model
        """
        self.rf_weight = rf_weight
        
        # Build CNN model if params provided
        if cnn_params is not None:
            self.cnn_model.build(**cnn_params)
        
        # Build RF model if params provided
        if rf_params is not None:
            self.rf_model.build(**rf_params)
        
        return self
    
    def train(self, X_train=None, y_train=None, X_val=None, y_val=None, **kwargs):
        """
        Train the Ensemble model by training both CNN and RF models.
        
        Args:
            X_train: If provided, should be a dict with 'cnn' and 'rf' keys for respective inputs
            y_train: Training labels (same for both models)
            X_val: If provided, should be a dict with 'cnn' and 'rf' keys for respective inputs
            y_val: Validation labels (same for both models)
            **kwargs: Additional parameters including:
                - cnn_params: Parameters for CNN training
                - rf_params: Parameters for RF training
            
        Returns:
            dict: Training metrics for both models
        """
        cnn_metrics = None
        rf_metrics = None
        ensemble_metrics = {}
        
        # Extract model-specific parameters
        cnn_params = kwargs.get('cnn_params', {})
        rf_params = kwargs.get('rf_params', {})
        
        # Train CNN model if data provided
        if X_train is not None and 'cnn' in X_train:
            cnn_X_train = X_train['cnn']
            cnn_X_val = X_val['cnn'] if X_val and 'cnn' in X_val else None
            
            cnn_history = self.cnn_model.train(
                cnn_X_train, y_train, 
                cnn_X_val, y_val,
                **cnn_params
            )
            
            # Store class names in CNN model
            self.cnn_model.set_class_names(self.class_names)
            
            # Extract metrics from CNN training
            if hasattr(cnn_history, 'history'):
                cnn_metrics = {
                    'train_accuracy': cnn_history.history.get('accuracy', [])[-1],
                    'val_accuracy': cnn_history.history.get('val_accuracy', [])[-1] if 'val_accuracy' in cnn_history.history else None,
                    'epochs': len(cnn_history.history.get('accuracy', []))
                }
            else:
                cnn_metrics = {'trained': True}
        
        # Train RF model if data provided
        if X_train is not None and 'rf' in X_train:
            rf_X_train = X_train['rf']
            rf_X_val = X_val['rf'] if X_val and 'rf' in X_val else None
            
            rf_metrics = self.rf_model.train(
                rf_X_train, y_train,
                rf_X_val, y_val,
                **rf_params
            )
            
            # Store class names in RF model
            self.rf_model.set_class_names(self.class_names)
        
        # Combine metrics
        ensemble_metrics = {
            'cnn_metrics': cnn_metrics,
            'rf_metrics': rf_metrics,
            'rf_weight': self.rf_weight
        }
        
        return ensemble_metrics
    
    def predict(self, X, **kwargs):
        """
        Make ensemble predictions by combining CNN and RF predictions.
        
        Args:
            X: Should be a dict with 'cnn' and 'rf' keys for respective inputs
            **kwargs: Additional prediction parameters
            
        Returns:
            tuple: (predicted_classes, ensemble_probabilities)
        """
        # Validate input
        if not isinstance(X, dict) or 'cnn' not in X or 'rf' not in X:
            raise ValueError("Input X must be a dictionary with 'cnn' and 'rf' keys")
        
        # Get RF predictions
        rf_predictions, rf_probabilities = self.rf_model.predict(X['rf'])
        
        # Get CNN predictions
        cnn_predictions, cnn_probabilities = self.cnn_model.predict(X['cnn'])
        
        # Combine probabilities using weighted average
        ensemble_probabilities = (self.rf_weight * rf_probabilities + 
                                 (1 - self.rf_weight) * cnn_probabilities)
        
        # Get the final class predictions
        ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)
        
        return ensemble_predictions, ensemble_probabilities
    
    def get_individual_predictions(self, X):
        """
        Get predictions from individual models without combining them.
        
        Args:
            X: Should be a dict with 'cnn' and 'rf' keys for respective inputs
            
        Returns:
            dict: Individual model predictions and probabilities
        """
        # Get RF predictions
        rf_predictions, rf_probabilities = self.rf_model.predict(X['rf'])
        
        # Get CNN predictions
        cnn_predictions, cnn_probabilities = self.cnn_model.predict(X['cnn'])
        
        return {
            'rf': {
                'predictions': rf_predictions,
                'probabilities': rf_probabilities
            },
            'cnn': {
                'predictions': cnn_predictions,
                'probabilities': cnn_probabilities
            }
        }
    
    def get_top_predictions(self, rf_input, cnn_input, top_n=1):
        """
        Get top N predictions with probabilities for an audio sample.
        
        Args:
            rf_input: Features for RF model
            cnn_input: Features for CNN model
            top_n (int): Number of top predictions to return
            
        Returns:
            list: Top N predictions with sound class and probability
        """
        # Make ensemble prediction
        _, ensemble_probabilities = self.predict({'rf': rf_input, 'cnn': cnn_input})
        
        # Get the probability for each class
        result = []
        for i, probs in enumerate(ensemble_probabilities):
            # Get indices of top N probabilities
            top_indices = np.argsort(probs)[-top_n:][::-1]
            
            # Create list of top predictions
            top_preds = []
            for idx in top_indices:
                if idx < len(self.class_names):
                    top_preds.append({
                        'sound': self.class_names[idx],
                        'probability': float(probs[idx])
                    })
                else:
                    logging.warning(f"Index {idx} out of range for class_names (length: {len(self.class_names)})")
            
            result.append(top_preds)
        
        return result
    
    def evaluate(self, X_test, y_test, **kwargs):
        """
        Evaluate the Ensemble model on test data.
        
        Args:
            X_test: Should be a dict with 'cnn' and 'rf' keys for respective inputs
            y_test: Test labels
            **kwargs: Additional evaluation parameters
            
        Returns:
            dict: Evaluation metrics
        """
        # Get ensemble predictions
        ensemble_predictions, _ = self.predict(X_test)
        
        # Get individual model evaluations
        cnn_metrics = self.cnn_model.evaluate(X_test['cnn'], y_test)
        rf_metrics = self.rf_model.evaluate(X_test['rf'], y_test)
        
        # Calculate ensemble accuracy
        ensemble_accuracy = np.mean(ensemble_predictions == y_test)
        
        # Calculate per-class accuracy for ensemble
        class_accuracy = {}
        for class_idx in range(len(self.class_names)):
            class_mask = (y_test == class_idx)
            if np.sum(class_mask) > 0:
                class_acc = np.mean(ensemble_predictions[class_mask] == class_idx)
                class_accuracy[class_idx] = float(class_acc)
        
        # Build confusion matrix for ensemble
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, ensemble_predictions)
        
        # Return metrics
        metrics = {
            'ensemble_accuracy': float(ensemble_accuracy),
            'class_accuracy': class_accuracy,
            'confusion_matrix': cm.tolist(),
            'cnn_metrics': cnn_metrics,
            'rf_metrics': rf_metrics,
            'rf_weight': self.rf_weight
        }
        
        return metrics
    
    def save(self, filename=None, **kwargs):
        """
        Save the Ensemble model (both CNN and RF) to disk.
        
        Args:
            filename (str, optional): Base filename to save as
            **kwargs: Additional saving parameters
            
        Returns:
            bool: Success status
        """
        if not self._ensure_directory():
            return False
        
        if filename is None:
            filename = "ensemble"
        
        try:
            # Save CNN model with derived filename
            cnn_success = self.cnn_model.save(f"{filename}_cnn.h5")
            
            # Save RF model with derived filename
            rf_success = self.rf_model.save(f"{filename}_rf.joblib")
            
            # Save ensemble metadata
            metadata_path = os.path.join(self.model_dir, f"{filename}_metadata.json")
            metadata = {
                'class_names': self.class_names,
                'rf_weight': self.rf_weight
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            if cnn_success and rf_success:
                logging.info(f"Ensemble model saved to {self.model_dir}/{filename}*")
                return True
            else:
                logging.error("Failed to save one or more components of the ensemble model")
                return False
                
        except Exception as e:
            logging.error(f"Failed to save ensemble model: {e}")
            return False
    
    def load(self, filename=None, **kwargs):
        """
        Load the Ensemble model (both CNN and RF) from disk.
        
        Args:
            filename (str, optional): Base filename to load from
            **kwargs: Additional loading parameters
            
        Returns:
            bool: Success status
        """
        if filename is None:
            filename = "ensemble"
        
        try:
            # Load CNN model with derived filename
            cnn_success = self.cnn_model.load(f"{filename}_cnn.h5")
            
            # Load RF model with derived filename
            rf_success = self.rf_model.load(f"{filename}_rf.joblib")
            
            # Load ensemble metadata
            metadata_path = os.path.join(self.model_dir, f"{filename}_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.class_names = metadata.get('class_names', [])
                self.rf_weight = metadata.get('rf_weight', 0.5)
            
            # Get class names from individual models if not in metadata
            if not self.class_names:
                if self.cnn_model.class_names:
                    self.class_names = self.cnn_model.class_names
                elif self.rf_model.class_names:
                    self.class_names = self.rf_model.class_names
            
            if cnn_success and rf_success:
                logging.info(f"Ensemble model loaded from {self.model_dir}/{filename}*")
                return True
            else:
                logging.warning("Failed to load one or more components of the ensemble model")
                return False
                
        except Exception as e:
            logging.error(f"Failed to load ensemble model: {e}")
            return False 