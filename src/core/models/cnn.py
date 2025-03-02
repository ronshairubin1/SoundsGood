import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, callbacks
import io  # Add this import for StringIO

from .base import BaseModel

class CNNModel(BaseModel):
    """
    CNN model implementation that inherits from BaseModel.
    Used for audio classification based on mel spectrogram features.
    """
    
    def __init__(self, model_dir='models'):
        """
        Initialize the CNN model.
        
        Args:
            model_dir (str): Directory for model storage
        """
        super().__init__(model_dir)
        self.input_shape = None
        self.num_classes = None
        self.model_summary = None
    
    def build(self, input_shape, num_classes, **kwargs):
        """
        Build the CNN model architecture.
        
        Args:
            input_shape (tuple): Shape of input features (height, width, channels)
            num_classes (int): Number of output classes
            **kwargs: Additional model parameters
            
        Returns:
            tf.keras.Model: The constructed model
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Build a sequential model
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block (optional for more complex tasks)
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=kwargs.get('learning_rate', 0.001)),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Store the model
        self.model = model
        
        # Save the model summary as a string
        string_io = io.StringIO()  # Use Python's built-in StringIO instead of tf.io.StringIO
        model.summary(print_fn=lambda x: string_io.write(x + '\n'))
        self.model_summary = string_io.getvalue()
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train the CNN model.
        
        Args:
            X_train: Training features (mel spectrograms)
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training parameters
                - epochs: Number of training epochs
                - batch_size: Batch size for training
                - class_weights: Optional class weights for imbalanced data
            
        Returns:
            tf.keras.callbacks.History: Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        # Set up training parameters
        epochs = kwargs.get('epochs', 50)
        batch_size = kwargs.get('batch_size', 32)
        class_weights = kwargs.get('class_weights', None)
        
        # Set up callbacks
        training_callbacks = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                min_delta=0.001,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.2,
                patience=5,
                min_lr=1e-5
            )
        ]
        
        # Add ModelCheckpoint if save_best is True
        if kwargs.get('save_best', False):
            if not self._ensure_directory():
                logging.warning("Could not create model directory, skipping checkpoint callback")
            else:
                checkpoint_path = os.path.join(self.model_dir, "best_cnn_model.h5")
                checkpoint_callback = callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
                training_callbacks.append(checkpoint_callback)
        
        # Train the model
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=training_callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X, **kwargs):
        """
        Make predictions using the CNN model.
        
        Args:
            X: Input features (mel spectrograms)
            **kwargs: Additional prediction parameters
            
        Returns:
            tuple: (predicted_classes, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() or train() first.")
        
        # Get the raw probabilities
        probabilities = self.model.predict(X)
        
        # Get the predicted class indices
        predicted_classes = np.argmax(probabilities, axis=1)
        
        return predicted_classes, probabilities
    
    def evaluate(self, X_test, y_test, **kwargs):
        """
        Evaluate the CNN model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            **kwargs: Additional evaluation parameters
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() or train() first.")
        
        # Evaluate the model
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Make predictions for confusion matrix
        y_pred, _ = self.predict(X_test)
        
        # Calculate per-class accuracy
        class_accuracy = {}
        for class_idx in range(self.num_classes):
            class_mask = (y_test == class_idx)
            if np.sum(class_mask) > 0:
                class_acc = np.mean(y_pred[class_mask] == class_idx)
                class_accuracy[class_idx] = float(class_acc)
        
        # Return metrics
        metrics = {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'class_accuracy': class_accuracy
        }
        
        return metrics
    
    def save(self, filename=None, **kwargs):
        """
        Save the CNN model to disk.
        
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
            filename = "cnn_model.h5"
        
        try:
            # Get the full path
            model_path = self.get_model_path(filename)
            
            # Save the model
            self.model.save(model_path)
            
            # Save metadata (class names, etc.)
            metadata_path = model_path.replace('.h5', '_metadata.json')
            import json
            metadata = {
                'class_names': self.class_names,
                'input_shape': self.input_shape,
                'num_classes': self.num_classes
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            logging.info(f"Model saved to {model_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to save model: {e}")
            return False
    
    def load(self, filename=None, **kwargs):
        """
        Load the CNN model from disk.
        
        Args:
            filename (str, optional): Filename to load from
            **kwargs: Additional loading parameters
            
        Returns:
            bool: Success status
        """
        if filename is None:
            filename = "cnn_model.h5"
        
        try:
            # Get the full path
            model_path = self.get_model_path(filename)
            
            # Check if the model exists
            if not os.path.exists(model_path):
                logging.error(f"Model file {model_path} not found.")
                return False
            
            # Custom objects might be needed for loading
            custom_objects = kwargs.get('custom_objects', {})
            
            # Load the model
            self.model = models.load_model(model_path, custom_objects=custom_objects)
            
            # Try to load metadata
            metadata_path = model_path.replace('.h5', '_metadata.json')
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.class_names = metadata.get('class_names', [])
                self.input_shape = metadata.get('input_shape')
                self.num_classes = metadata.get('num_classes')
            
            logging.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return False
    
    def get_model_summary(self):
        """
        Get the model summary as a string.
        
        Returns:
            str: Model summary
        """
        return self.model_summary 