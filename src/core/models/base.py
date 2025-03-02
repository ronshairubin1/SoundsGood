import os
import logging
import numpy as np
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for all models.
    Defines a standard interface that all model implementations must follow.
    """
    
    def __init__(self, model_dir='models'):
        """
        Initialize the base model.
        
        Args:
            model_dir (str): Directory for model storage
        """
        self.model_dir = model_dir
        self.model = None
        self.class_names = []
    
    @abstractmethod
    def build(self, **kwargs):
        """
        Build the model architecture.
        
        Args:
            **kwargs: Model-specific parameters
            
        Returns:
            The constructed model
        """
        pass
    
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Training history
        """
        pass
    
    @abstractmethod
    def predict(self, X, **kwargs):
        """
        Make predictions using the model.
        
        Args:
            X: Input features
            **kwargs: Additional prediction parameters
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, X_test, y_test, **kwargs):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            **kwargs: Additional evaluation parameters
            
        Returns:
            Evaluation metrics
        """
        pass
    
    @abstractmethod
    def save(self, filename, **kwargs):
        """
        Save the model to disk.
        
        Args:
            filename (str): Filename to save as
            **kwargs: Additional saving parameters
            
        Returns:
            bool: Success status
        """
        pass
    
    @abstractmethod
    def load(self, filename, **kwargs):
        """
        Load the model from disk.
        
        Args:
            filename (str): Filename to load from
            **kwargs: Additional loading parameters
            
        Returns:
            bool: Success status
        """
        pass
    
    def get_class_names(self):
        """
        Get the class names for this model.
        
        Returns:
            list: List of class names
        """
        return self.class_names
    
    def set_class_names(self, class_names):
        """
        Set the class names for this model.
        
        Args:
            class_names (list): List of class names
        """
        self.class_names = class_names
    
    def get_model_path(self, filename=None):
        """
        Get the full path for a model file.
        
        Args:
            filename (str, optional): Specific filename or use default
            
        Returns:
            str: Full path to the model file
        """
        if filename is None:
            filename = f"model_{self.__class__.__name__}.h5"
        return os.path.join(self.model_dir, filename)
    
    def _ensure_directory(self):
        """
        Ensure the model directory exists.
        
        Returns:
            bool: True if directory exists or was created
        """
        if not os.path.exists(self.model_dir):
            try:
                os.makedirs(self.model_dir, exist_ok=True)
                return True
            except Exception as e:
                logging.error(f"Failed to create model directory {self.model_dir}: {e}")
                return False
        return True 