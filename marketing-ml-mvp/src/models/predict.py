"""
Prediction module for marketing campaign response model with Pipeline Integration
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Union, Dict, Any, Optional, List, Tuple
import logging

from ..config.settings import MODELS_PATH

logger = logging.getLogger(__name__)


class MarketingPredictor:
    """
    Marketing campaign response predictor with calibration support.
    """
    
    def __init__(self, models_path: Optional[Path] = None):
        """
        Initialize predictor.
        
        Args:
            models_path: Path to load models from
        """
        self.models_path = models_path or MODELS_PATH
        
        self.model = None
        self.calibrator = None
        self.metadata = None
        self.is_loaded = False
        
    def load_model(self) -> None:
        """Load trained model and metadata."""
        logger.info("Loading trained model and metadata")
        
        # Load model
        model_path = self.models_path / MODEL_NAME
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load metadata
        metadata_path = self.models_path / MODEL_METADATA_NAME
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"Metadata loaded from {metadata_path}")
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")
            self.metadata = {}
        
        # Load calibrator if available
        calibrator_path = self.models_path / CALIBRATOR_NAME
        if calibrator_path.exists():
            self.calibrator = TemperatureScaling.load(calibrator_path)
            logger.info("Temperature scaling calibrator loaded")
        else:
            logger.info("No calibrator found - using uncalibrated predictions")
        
        self.is_loaded = True
        logger.info("Model loading completed")
    
    def _validate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Validate input features against expected model features.
        
        Args:
            X: Input features
            
        Returns:
            Validated features DataFrame
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before making predictions")
        
        expected_features = self.metadata.get('feature_names')
        if expected_features is None:
            logger.warning("No feature names found in metadata - skipping validation")
            return X
        
        # Check for missing features
        missing_features = set(expected_features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Check for extra features
        extra_features = set(X.columns) - set(expected_features)
        if extra_features:
            logger.warning(f"Extra features will be ignored: {extra_features}")
        
        # Return features in correct order
        return X[expected_features]
    
    def predict_proba(self, X: pd.DataFrame, use_calibration: bool = True) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            use_calibration: Whether to apply calibration if available
            
        Returns:
            Predicted probabilities
        """
        if not self.is_loaded:
            self.load_model()
        
        # Validate features
        X_validated = self._validate_features(X)
        
        # Get raw predictions
        raw_probs = self.model.predict_proba(X_validated)[:, 1]  # Probability of class 1
        
        # Apply calibration if available and requested
        if use_calibration and self.calibrator is not None:
            calibrated_probs = self.calibrator.predict_proba(raw_probs)
            logger.debug(f"Applied temperature scaling calibration")
            return calibrated_probs
        
        return raw_probs
    
    def predict(self, X: pd.DataFrame, use_calibration: bool = True, 
                threshold: Optional[float] = None) -> np.ndarray:
        """
        Make binary predictions.
        
        Args:
            X: Input features
            use_calibration: Whether to apply calibration
            threshold: Classification threshold. If None, uses optimal threshold from metadata
            
        Returns:
            Binary predictions (0 or 1)
        """
        # Get probabilities
        probs = self.predict_proba(X, use_calibration)
        
        # Determine threshold
        if threshold is None:
            threshold = self.metadata.get('optimal_threshold', 0.5)
            logger.debug(f"Using optimal threshold: {threshold}")
        else:
            logger.debug(f"Using provided threshold: {threshold}")
        
        # Apply threshold
        predictions = (probs >= threshold).astype(int)
        
        return predictions
    
    def predict_single(self, customer_data: Dict[str, Any], 
                      use_calibration: bool = True,
                      threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Make prediction for a single customer.
        
        Args:
            customer_data: Customer features as dictionary
            use_calibration: Whether to apply calibration
            threshold: Classification threshold
            
        Returns:
            Prediction result with probability and class
        """
        # Convert to DataFrame
        X = pd.DataFrame([customer_data])
        
        # Get probability and prediction
        prob = self.predict_proba(X, use_calibration)[0]
        pred = self.predict(X, use_calibration, threshold)[0]
        
        # Determine threshold used
        final_threshold = threshold if threshold is not None else self.metadata.get('optimal_threshold', 0.5)
        
        result = {
            'probability': float(prob),
            'prediction': int(pred),
            'threshold_used': final_threshold,
            'will_respond': bool(pred),
            'confidence': 'high' if abs(prob - 0.5) > 0.3 else 'medium' if abs(prob - 0.5) > 0.1 else 'low'
        }
        
        return result
    
    def predict_batch(self, customers_data: List[Dict[str, Any]], 
                     use_calibration: bool = True,
                     threshold: Optional[float] = None,
                     include_details: bool = False) -> List[Dict[str, Any]]:
        """
        Make predictions for a batch of customers.
        
        Args:
            customers_data: List of customer feature dictionaries
            use_calibration: Whether to apply calibration
            threshold: Classification threshold
            include_details: Whether to include detailed prediction info
            
        Returns:
            List of prediction results
        """
        if not customers_data:
            return []
        
        # Convert to DataFrame
        X = pd.DataFrame(customers_data)
        
        # Get predictions
        probs = self.predict_proba(X, use_calibration)
        preds = self.predict(X, use_calibration, threshold)
        
        # Determine threshold used
        final_threshold = threshold if threshold is not None else self.metadata.get('optimal_threshold', 0.5)
        
        # Build results
        results = []
        for i, (prob, pred) in enumerate(zip(probs, preds)):
            result = {
                'customer_id': i,
                'probability': float(prob),
                'prediction': int(pred),
                'will_respond': bool(pred)
            }
            
            if include_details:
                result.update({
                    'threshold_used': final_threshold,
                    'confidence': 'high' if abs(prob - 0.5) > 0.3 else 'medium' if abs(prob - 0.5) > 0.1 else 'low'
                })
            
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        if not self.is_loaded:
            self.load_model()
        
        info = {
            'model_loaded': self.is_loaded,
            'model_type': type(self.model).__name__ if self.model else None,
            'has_calibrator': self.calibrator is not None,
            'calibrator_type': 'TemperatureScaling' if self.calibrator else None
        }
        
        if self.metadata:
            info.update({
                'optimal_threshold': self.metadata.get('optimal_threshold'),
                'model_metrics': self.metadata.get('metrics'),
                'feature_count': len(self.metadata.get('feature_names', [])),
                'training_samples': self.metadata.get('training_samples'),
                'validation_samples': self.metadata.get('validation_samples')
            })
        
        return info
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> Optional[Dict[str, float]]:
        """
        Get feature importance from the model.
        
        Args:
            top_n: Number of top features to return. If None, returns all
            
        Returns:
            Dictionary of feature importances
        """
        if not self.is_loaded:
            self.load_model()
        
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not support feature importance")
            return None
        
        feature_names = self.metadata.get('feature_names', [])
        if not feature_names:
            logger.warning("Feature names not available in metadata")
            return None
        
        # Get importances
        importances = self.model.feature_importances_
        
        # Create dictionary
        feature_importance = dict(zip(feature_names, importances))
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Return top_n if specified
        if top_n:
            sorted_features = sorted_features[:top_n]
        
        return dict(sorted_features)
    
    def explain_prediction(self, customer_data: Dict[str, Any], 
                          top_features: int = 5) -> Dict[str, Any]:
        """
        Provide basic explanation for a prediction.
        
        Args:
            customer_data: Customer features
            top_features: Number of top features to include in explanation
            
        Returns:
            Explanation dictionary
        """
        # Get prediction
        prediction_result = self.predict_single(customer_data)
        
        # Get feature importance
        feature_importance = self.get_feature_importance(top_features)
        
        # Get customer feature values for top features
        customer_features = {}
        if feature_importance:
            for feature in feature_importance.keys():
                if feature in customer_data:
                    customer_features[feature] = customer_data[feature]
        
        explanation = {
            'prediction': prediction_result,
            'top_influential_features': feature_importance,
            'customer_feature_values': customer_features,
            'explanation_note': f"Based on {top_features} most important features from model training"
        }
        
        return explanation


def load_marketing_predictor(models_path: Optional[Path] = None) -> MarketingPredictor:
    """
    Convenience function to load a marketing predictor.
    
    Args:
        models_path: Path to load models from
        
    Returns:
        Loaded MarketingPredictor instance
    """
    predictor = MarketingPredictor(models_path)
    predictor.load_model()
    return predictor


def predict_customer_response(customer_data: Union[Dict[str, Any], List[Dict[str, Any]]],
                            models_path: Optional[Path] = None,
                            use_calibration: bool = True,
                            threshold: Optional[float] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Convenience function to predict customer response.
    
    Args:
        customer_data: Single customer dict or list of customer dicts
        models_path: Path to models
        use_calibration: Whether to use calibration
        threshold: Classification threshold
        
    Returns:
        Prediction result(s)
    """
    predictor = load_marketing_predictor(models_path)
    
    if isinstance(customer_data, dict):
        return predictor.predict_single(customer_data, use_calibration, threshold)
    else:
        return predictor.predict_batch(customer_data, use_calibration, threshold)