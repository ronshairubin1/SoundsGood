# File: src/ml/ensemble_classifier.py

import numpy as np

class EnsembleClassifier:
    """
    A simple ensemble that merges predictions from a CNN model 
    and from a RandomForest model, taking either an average 
    or a weighted average of probabilities.
    """
    def __init__(self, rf_classifier, cnn_model, class_names, rf_weight=0.5):
        """
        Args:
            rf_classifier: An instance of RandomForestClassifier
            cnn_model: A loaded Keras model
            class_names: The list of classes in both approaches
            rf_weight: The weight to assign to RF predictions, 
                       the CNN gets (1 - rf_weight)
        """
        self.rf_classifier = rf_classifier
        self.cnn_model = cnn_model
        self.class_names = class_names
        self.rf_weight = rf_weight

    def predict(self, X_rf, X_cnn):
        """
        X_rf: feature vectors for RF 
        X_cnn: CNN input (mel-spectrograms)
        
        Returns final predicted class and combined confidence
        """
        # 1) Get RF probabilities
        _, rf_probs = self.rf_classifier.predict(X_rf)

        # 2) Get CNN probabilities
        cnn_probs = self.cnn_model.predict(X_cnn)

        # 3) Weighted average
        combined_probs = self.rf_weight * rf_probs + (1 - self.rf_weight) * cnn_probs

        # 4) Final prediction
        preds = []
        for row in combined_probs:
            idx = np.argmax(row)
            confidence = row[idx]
            preds.append((self.class_names[idx], confidence))
        return preds

    def get_top_predictions(self, X_rf, X_cnn, top_n=3):
        _, rf_probs = self.rf_classifier.predict(X_rf)
        cnn_probs = self.cnn_model.predict(X_cnn)
        combined_probs = self.rf_weight * rf_probs + (1 - self.rf_weight) * cnn_probs

        results = []
        for row in combined_probs:
            top_indices = np.argsort(row)[-top_n:][::-1]
            pred_list = []
            for i in top_indices:
                pred_list.append({
                    'sound': self.class_names[i],
                    'probability': float(row[i])
                })
            results.append(pred_list)
        return results
