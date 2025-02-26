# File: src/ml/rf_classifier.py

import joblib
import os
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier as SKRF

class RandomForestClassifier:
    """
    A RandomForest-based classifier adapted from old_code/model.py
    """

    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.classes_ = None
        self.feature_count = None
        os.makedirs(model_dir, exist_ok=True)

    def train(self, X, y):
        """
        Train a Random Forest classifier
        """
        try:
            self.model = SKRF(
                n_estimators=200,
                max_depth=None,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X, y)
            self.classes_ = self.model.classes_
            self.feature_count = X.shape[1]
            logging.info(f"RF model trained with {len(self.classes_)} classes.")
            return True
        except Exception as e:
            logging.error(f"Error training RandomForest: {str(e)}")
            return False

    def predict(self, X):
        if self.model is None:
            logging.error("RandomForest model not trained or loaded.")
            return None, None
        try:
            X = np.array(X)
            if X.shape[1] != self.feature_count:
                logging.error("Feature count mismatch for RF.")
                return None, None
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            return predictions, probabilities
        except Exception as e:
            logging.error(f"Error in RF prediction: {str(e)}")
            return None, None

    def get_top_predictions(self, X, top_n=3):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        probs = self.model.predict_proba(X)
        top_indices = np.argsort(probs, axis=1)[:, -top_n:][:, ::-1]
        top_probabilities = np.take_along_axis(probs, top_indices, axis=1)
        top_labels = self.model.classes_[top_indices]
        results = []
        for labels, pvals in zip(top_labels, top_probabilities):
            pred_list = []
            for label, val in zip(labels, pvals):
                pred_list.append({"sound": label, "probability": float(val)})
            results.append(pred_list)
        return results

    def save(self, filename='rf_sound_classifier.joblib'):
        if self.model is None:
            logging.error("No RF model to save.")
            return False
        try:
            path = os.path.join(self.model_dir, filename)
            joblib.dump({
                'model': self.model,
                'classes': self.classes_,
                'feature_count': self.feature_count
            }, path)
            logging.info(f"RF model saved to {path}")
            return True
        except Exception as e:
            logging.error(f"Error saving RF model: {str(e)}")
            return False

    def load(self, filename='rf_sound_classifier.joblib'):
        try:
            path = os.path.join(self.model_dir, filename)
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.classes_ = model_data['classes']
            self.feature_count = model_data['feature_count']
            logging.info(f"RF model loaded from {path}")
            return True
        except Exception as e:
            logging.error(f"Error loading RF model: {str(e)}")
            return False
