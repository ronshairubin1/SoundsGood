import os
import logging
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .base import BaseModel

class RandomForestModel(BaseModel):
    """
    RandomForest model implementation that inherits from BaseModel.
    Used for audio classification based on classical audio features.
    """
    
    def __init__(self, model_dir='models'):
        """
        Initialize the RandomForest model.
        
        Args:
            model_dir (str): Directory for model storage
        """
        super().__init__(model_dir)
        self.feature_names = None
    
    def build(self, n_estimators=100, max_depth=None, **kwargs):
        """
        Build the RandomForest model.
        
        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int, optional): Maximum depth of the trees
            **kwargs: Additional model parameters
            
        Returns:
            SklearnRF: The constructed model
        """
        self.model = SklearnRF(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,  # Use all available cores
            random_state=42,
            **kwargs
        )
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train the RandomForest model.
        
        Args:
            X_train: Training features (classical audio features)
            y_train: Training labels
            X_val: Validation features (optional, used for monitoring only)
            y_val: Validation labels (optional, used for monitoring only)
            **kwargs: Additional training parameters
                - feature_names: Names of the features
            
        Returns:
            dict: Training metrics
        """
        if self.model is None:
            self.build()
        
        # Save feature names if provided
        self.feature_names = kwargs.get('feature_names', None)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate training accuracy
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Calculate validation accuracy if provided
        val_accuracy = None
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
        
        # Feature importance
        feature_importance = None
        if self.feature_names is not None:
            feature_importance = dict(zip(
                self.feature_names, 
                self.model.feature_importances_
            ))
        
        # Return training metrics
        metrics = {
            'train_accuracy': float(train_accuracy),
            'val_accuracy': float(val_accuracy) if val_accuracy is not None else None,
            'feature_importance': feature_importance
        }
        
        return metrics
    
    def predict(self, X, **kwargs):
        """
        Make predictions using the RandomForest model.
        
        Args:
            X: Input features (classical audio features)
            **kwargs: Additional prediction parameters
            
        Returns:
            tuple: (predicted_classes, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() or train() first.")
        
        # Get the predicted classes
        predicted_classes = self.model.predict(X)
        
        # Get the raw probabilities
        probabilities = self.model.predict_proba(X)
        
        return predicted_classes, probabilities
    
    def evaluate(self, X_test, y_test, **kwargs):
        """
        Evaluate the RandomForest model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            **kwargs: Additional evaluation parameters
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() or train() first.")
        
        # Predict on test data
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate per-class accuracy
        class_accuracy = {}
        for class_idx in range(len(self.class_names)):
            class_mask = (y_test == class_idx)
            if np.sum(class_mask) > 0:
                class_acc = np.mean(y_pred[class_mask] == class_idx)
                class_accuracy[class_idx] = float(class_acc)
        
        # Return metrics
        metrics = {
            'accuracy': float(accuracy),
            'class_accuracy': class_accuracy,
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def save(self, filename=None, **kwargs):
        """
        Save the RandomForest model to disk.
        
        Args:
            filename (str, optional): Filename to save as
            **kwargs: Additional saving parameters
            
        Returns:
            bool: Success status
        """
        if self.model is None:
            logging.error("No model to save.")
            return False
        
        if not self._ensure_directory():
            return False
        
        if filename is None:
            filename = "rf_model.joblib"
        
        try:
            # Get the full path
            model_path = self.get_model_path(filename)
            
            # Save the model
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save metadata (class names, feature names, etc.)
            metadata_path = model_path.replace('.joblib', '_metadata.json')
            
            # Get scikit-learn version
            import sklearn
            from datetime import datetime
            
            # Enhanced metadata with more information
            metadata = {
                'class_names': self.class_names,
                'feature_names': self.feature_names,
                'model_type': 'random_forest',
                'creation_date': datetime.now().isoformat(),
                'sklearn_version': sklearn.__version__,
                'n_estimators': self.model.n_estimators,
                'max_depth': self.model.max_depth,
                'n_classes': len(self.class_names),
                'feature_importances': self.model.feature_importances_.tolist() if hasattr(self.model, 'feature_importances_') else None,
                'training_params': getattr(self, 'training_params', {}),
                'metrics': getattr(self, 'metrics_history', {})
            }
            
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logging.info(f"Model saved to {model_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to save model: {e}")
            return False
    
    def load(self, filename=None, **kwargs):
        """
        Load the RandomForest model from disk.
        
        Args:
            filename (str, optional): Filename to load from
            **kwargs: Additional loading parameters
            
        Returns:
            bool: Success status
        """
        if filename is None:
            filename = "rf_model.joblib"
        
        try:
            # Get the full path
            model_path = self.get_model_path(filename)
            
            # Check if the model exists
            if not os.path.exists(model_path):
                logging.error(f"Model file {model_path} not found.")
                return False
            
            # Load the model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Try to load metadata
            metadata_path = model_path.replace('.joblib', '_metadata.json')
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.class_names = metadata.get('class_names', [])
                self.feature_names = metadata.get('feature_names', None)
            
            logging.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return False
    
    def get_feature_importance(self):
        """
        Get feature importance from the RandomForest model.
        
        Returns:
            dict: Feature names and their importance scores
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() or train() first.")
        
        if self.feature_names is None:
            return dict(enumerate(self.model.feature_importances_))
        
        return dict(zip(self.feature_names, self.model.feature_importances_)) 