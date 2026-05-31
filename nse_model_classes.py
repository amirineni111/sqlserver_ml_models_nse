"""
Shared model class definitions for NSE ML pipeline.

Keeping these in a separate module ensures joblib/pickle can find the
classes when loading saved models from any script context (not just __main__).
"""

import numpy as np


class IsotonicCalibratedModel:
    """Wrapper for isotonic-calibrated model (scikit-learn 1.6+ compatible)"""

    def __init__(self, base_model, calibrators):
        self.base_model = base_model
        self.calibrators = calibrators
        self.classes_ = base_model.classes_

    def predict_proba(self, X):
        base_proba = self.base_model.predict_proba(X)
        calibrated_proba = np.column_stack([
            cal.transform(base_proba[:, i])
            for i, cal in enumerate(self.calibrators)
        ])
        # Normalize to sum to 1
        calibrated_proba = calibrated_proba / calibrated_proba.sum(axis=1, keepdims=True)
        return calibrated_proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class SigmoidCalibratedModel:
    """Wrapper for sigmoid-calibrated model (Platt scaling, scikit-learn 1.6+ compatible)"""

    def __init__(self, base_model, platt_scaler):
        self.base_model = base_model
        self.platt_scaler = platt_scaler
        self.classes_ = base_model.classes_

    def predict_proba(self, X):
        base_proba = self.base_model.predict_proba(X)
        return self.platt_scaler.predict_proba(base_proba)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
