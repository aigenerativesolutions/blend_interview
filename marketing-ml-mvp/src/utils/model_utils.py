"""
Utility functions for model operations
"""
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import logging

from ..config.settings import MODELS_PATH

logger = logging.getLogger(__name__)


def save_model_artifacts(model: Any, metadata: Dict[str, Any],
                        model_path: Path, metadata_path: Path) -> None:
    """
    Save model and metadata to files.
    
    Args:
        model: Trained model to save
        metadata: Model metadata dictionary
        model_path: Path to save model
        metadata_path: Path to save metadata
    """
    try:
        # Save model
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Metadata saved to {metadata_path}")
        
    except Exception as e:
        logger.error(f"Error saving model artifacts: {str(e)}")
        raise


def load_model_artifacts(model_path: Path, metadata_path: Path) -> Tuple[Any, Dict[str, Any]]:
    """
    Load model and metadata from files.
    
    Args:
        model_path: Path to model file
        metadata_path: Path to metadata file
        
    Returns:
        Tuple of (model, metadata)
    """
    try:
        # Load model
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load metadata
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Metadata loaded from {metadata_path}")
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")
        
        return model, metadata
        
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        raise


def validate_input_data(data: Union[Dict[str, Any], pd.DataFrame], 
                       expected_features: Optional[list] = None) -> pd.DataFrame:
    """
    Validate and prepare input data for model prediction.
    
    Args:
        data: Input data as dictionary or DataFrame
        expected_features: List of expected feature names
        
    Returns:
        Validated DataFrame
    """
    # Convert to DataFrame if needed
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Data must be dictionary or DataFrame")
    
    # Check for expected features if provided
    if expected_features:
        missing_features = set(expected_features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Reorder columns to match expected features
        df = df[expected_features]
    
    # Check for missing values
    if df.isnull().any().any():
        logger.warning("Input data contains missing values")
        # Could implement imputation strategy here if needed
    
    return df


def calculate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                          y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive model evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score, classification_report
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary')
    }
    
    # Add ROC AUC if probabilities are provided
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    
    # Add confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0
        })
    
    return metrics


def create_model_summary(model: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a comprehensive model summary.
    
    Args:
        model: Trained model
        metadata: Model metadata
        
    Returns:
        Model summary dictionary
    """
    summary = {
        'model_type': type(model).__name__,
        'model_parameters': getattr(model, 'get_params', lambda: {})(),
        'feature_count': len(metadata.get('feature_names', [])),
        'training_samples': metadata.get('training_samples', 'Unknown'),
        'validation_samples': metadata.get('validation_samples', 'Unknown')
    }
    
    # Add metrics if available
    if 'metrics' in metadata:
        summary['performance_metrics'] = metadata['metrics']
    
    # Add threshold if available
    if 'optimal_threshold' in metadata:
        summary['optimal_threshold'] = metadata['optimal_threshold']
    
    # Add feature importance if available
    if hasattr(model, 'feature_importances_') and 'feature_names' in metadata:
        feature_names = metadata['feature_names']
        importances = model.feature_importances_
        feature_importance = dict(zip(feature_names, importances))
        # Sort by importance
        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        summary['feature_importance'] = dict(list(sorted_importance.items())[:10])  # Top 10
    
    return summary


def export_model_for_deployment(models_path: Path, 
                               export_path: Path,
                               include_metadata: bool = True) -> None:
    """
    Export model artifacts for deployment.
    
    Args:
        models_path: Path to model directory
        export_path: Path to export directory
        include_metadata: Whether to include metadata files
    """
    from shutil import copy2
    
    export_path.mkdir(parents=True, exist_ok=True)
    
    # Copy model files
    for file_path in models_path.glob("*.pkl"):
        copy2(file_path, export_path)
        logger.info(f"Copied {file_path.name} to export directory")
    
    # Copy metadata if requested
    if include_metadata:
        for file_path in models_path.glob("*.json"):
            copy2(file_path, export_path)
            logger.info(f"Copied {file_path.name} to export directory")
    
    logger.info(f"Model artifacts exported to {export_path}")


def check_model_drift(reference_data: pd.DataFrame, 
                     current_data: pd.DataFrame,
                     threshold: float = 0.1) -> Dict[str, Any]:
    """
    Simple model drift detection based on feature distributions.
    
    Args:
        reference_data: Reference (training) data
        current_data: Current production data
        threshold: Drift threshold for alerting
        
    Returns:
        Drift analysis results
    """
    drift_results = {
        'has_drift': False,
        'drift_score': 0.0,
        'feature_drifts': {},
        'threshold': threshold
    }
    
    # Calculate feature-wise drift using simple statistical tests
    from scipy.stats import ks_2samp
    
    feature_drifts = {}
    drift_scores = []
    
    for feature in reference_data.columns:
        if feature in current_data.columns:
            # Use Kolmogorov-Smirnov test for drift detection
            ref_values = reference_data[feature].dropna()
            cur_values = current_data[feature].dropna()
            
            if len(ref_values) > 0 and len(cur_values) > 0:
                # For numerical features
                if pd.api.types.is_numeric_dtype(ref_values):
                    statistic, p_value = ks_2samp(ref_values, cur_values)
                    drift_score = 1 - p_value  # Higher score means more drift
                    
                    feature_drifts[feature] = {
                        'drift_score': drift_score,
                        'p_value': p_value,
                        'has_drift': drift_score > threshold,
                        'method': 'ks_test'
                    }
                    
                    drift_scores.append(drift_score)
                
                # For categorical features (simple frequency comparison)
                else:
                    ref_freq = ref_values.value_counts(normalize=True)
                    cur_freq = cur_values.value_counts(normalize=True)
                    
                    # Calculate total variation distance
                    all_categories = set(ref_freq.index) | set(cur_freq.index)
                    tvd = 0.5 * sum(abs(ref_freq.get(cat, 0) - cur_freq.get(cat, 0)) 
                                  for cat in all_categories)
                    
                    feature_drifts[feature] = {
                        'drift_score': tvd,
                        'has_drift': tvd > threshold,
                        'method': 'total_variation_distance'
                    }
                    
                    drift_scores.append(tvd)
    
    # Overall drift assessment
    if drift_scores:
        overall_drift_score = np.mean(drift_scores)
        drift_results.update({
            'drift_score': overall_drift_score,
            'has_drift': overall_drift_score > threshold,
            'feature_drifts': feature_drifts
        })
    
    return drift_results


def generate_prediction_batch_id() -> str:
    """
    Generate unique batch ID for prediction tracking.
    
    Returns:
        Unique batch identifier
    """
    from datetime import datetime
    import uuid
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_id = f"batch_{timestamp}_{str(uuid.uuid4())[:8]}"
    
    return batch_id


def log_prediction_metrics(predictions: np.ndarray, 
                         probabilities: Optional[np.ndarray] = None,
                         batch_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Log prediction metrics for monitoring.
    
    Args:
        predictions: Binary predictions
        probabilities: Prediction probabilities  
        batch_id: Batch identifier
        
    Returns:
        Logged metrics dictionary
    """
    metrics = {
        'batch_id': batch_id or generate_prediction_batch_id(),
        'timestamp': pd.Timestamp.now().isoformat(),
        'total_predictions': len(predictions),
        'positive_predictions': int(np.sum(predictions)),
        'negative_predictions': int(len(predictions) - np.sum(predictions)),
        'positive_rate': float(np.mean(predictions))
    }
    
    if probabilities is not None:
        metrics.update({
            'mean_probability': float(np.mean(probabilities)),
            'std_probability': float(np.std(probabilities)),
            'min_probability': float(np.min(probabilities)),
            'max_probability': float(np.max(probabilities))
        })
    
    logger.info(f"Prediction metrics logged: {metrics}")
    
    return metrics