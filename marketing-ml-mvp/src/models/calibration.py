"""
Model calibration module using Temperature Scaling
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy.optimize import minimize_scalar

from ..config.settings import MODELS_PATH, CALIBRATOR_NAME

logger = logging.getLogger(__name__)


class TemperatureScaling:
    """
    Temperature Scaling calibration method.
    
    Temperature scaling is a simple post-processing technique that applies
    a single parameter T to the logits before the softmax function.
    For binary classification: p_calibrated = sigmoid(logit / T)
    """
    
    def __init__(self):
        """Initialize Temperature Scaling calibrator."""
        self.temperature = 1.0
        self.is_fitted = False
        
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray, 
            method: str = 'minimize') -> 'TemperatureScaling':
        """
        Fit the temperature parameter.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities (uncalibrated)
            method: Optimization method ('minimize' or 'grid_search')
            
        Returns:
            Self
        """
        logger.info("Fitting temperature scaling parameter")
        
        if method == 'minimize':
            self.temperature = self._fit_temperature_minimize(y_true, y_prob)
        elif method == 'grid_search':
            self.temperature = self._fit_temperature_grid_search(y_true, y_prob)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.is_fitted = True
        logger.info(f"Fitted temperature: {self.temperature:.4f}")
        
        return self
    
    def _fit_temperature_minimize(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Fit temperature using scipy minimize_scalar.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            Optimal temperature
        """
        def objective(temperature):
            # Convert probabilities to logits
            eps = 1e-7
            y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
            logits = np.log(y_prob_clipped / (1 - y_prob_clipped))
            
            # Apply temperature scaling
            calibrated_logits = logits / temperature
            calibrated_probs = 1 / (1 + np.exp(-calibrated_logits))
            
            # Calculate negative log-likelihood (minimize)
            return log_loss(y_true, calibrated_probs)
        
        # Optimize temperature
        result = minimize_scalar(objective, bounds=(0.1, 10.0), method='bounded')
        
        return result.x
    
    def _fit_temperature_grid_search(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Fit temperature using grid search.
        
        Args:
            y_true: True labels  
            y_prob: Predicted probabilities
            
        Returns:
            Optimal temperature
        """
        temperatures = np.linspace(0.1, 5.0, 100)
        best_temp = 1.0
        best_score = float('inf')
        
        for temp in temperatures:
            # Convert to logits and apply temperature
            eps = 1e-7
            y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
            logits = np.log(y_prob_clipped / (1 - y_prob_clipped))
            
            calibrated_logits = logits / temp
            calibrated_probs = 1 / (1 + np.exp(-calibrated_logits))
            
            score = log_loss(y_true, calibrated_probs)
            
            if score < best_score:
                best_score = score
                best_temp = temp
        
        return best_temp
    
    def predict_proba(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to probabilities.
        
        Args:
            y_prob: Uncalibrated probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("TemperatureScaling must be fitted before making predictions")
        
        # Convert to logits
        eps = 1e-7
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped))
        
        # Apply temperature scaling
        calibrated_logits = logits / self.temperature
        calibrated_probs = 1 / (1 + np.exp(-calibrated_logits))
        
        return calibrated_probs
    
    def save(self, filepath: Path) -> None:
        """Save the temperature scaler."""
        joblib.dump({'temperature': self.temperature, 'is_fitted': self.is_fitted}, filepath)
        logger.info(f"Temperature scaler saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'TemperatureScaling':
        """Load a temperature scaler."""
        data = joblib.load(filepath)
        scaler = cls()
        scaler.temperature = data['temperature']
        scaler.is_fitted = data['is_fitted']
        logger.info(f"Temperature scaler loaded from {filepath}")
        return scaler


class CalibrationEvaluator:
    """
    Evaluate calibration quality of probabilistic classifiers.
    """
    
    @staticmethod
    def reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, 
                           n_bins: int = 10, title: str = "Reliability Diagram") -> Dict[str, Any]:
        """
        Create reliability diagram data.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration
            title: Plot title
            
        Returns:
            Dictionary with calibration data
        """
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0  # Expected Calibration Error
        bin_data = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                bin_data.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'accuracy': accuracy_in_bin,
                    'confidence': avg_confidence_in_bin,
                    'count': in_bin.sum(),
                    'prop_in_bin': prop_in_bin
                })
        
        return {
            'ece': ece,
            'bin_data': bin_data,
            'title': title
        }
    
    @staticmethod
    def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """
        Calculate various calibration metrics.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary with calibration metrics
        """
        # Expected Calibration Error
        rel_data = CalibrationEvaluator.reliability_diagram(y_true, y_prob)
        ece = rel_data['ece']
        
        # Brier Score
        brier_score = brier_score_loss(y_true, y_prob)
        
        # Log Loss
        logloss = log_loss(y_true, y_prob)
        
        return {
            'expected_calibration_error': ece,
            'brier_score': brier_score,
            'log_loss': logloss
        }
    
    @staticmethod
    def plot_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, 
                                n_bins: int = 10, title: str = "Reliability Diagram",
                                save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot reliability diagram.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities  
            n_bins: Number of bins
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        rel_data = CalibrationEvaluator.reliability_diagram(y_true, y_prob, n_bins, title)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Extract data for plotting
        bin_data = rel_data['bin_data']
        if bin_data:
            confidences = [item['confidence'] for item in bin_data]
            accuracies = [item['accuracy'] for item in bin_data]
            counts = [item['count'] for item in bin_data]
            
            # Plot bars
            bars = ax.bar(confidences, accuracies, width=0.08, alpha=0.7, 
                         edgecolor='black', linewidth=1)
            
            # Color bars by count
            norm = plt.Normalize(vmin=min(counts), vmax=max(counts))
            colors = plt.cm.Blues(norm(counts))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.8, label='Perfect calibration')
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f"{title}\\nECE: {rel_data['ece']:.4f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Reliability diagram saved to {save_path}")
        
        return fig


class ModelCalibrator:
    """
    Main calibration class that handles different calibration methods.
    """
    
    def __init__(self, models_path: Optional[Path] = None):
        """
        Initialize calibrator.
        
        Args:
            models_path: Path to save calibrators
        """
        self.models_path = models_path or MODELS_PATH
        self.models_path.mkdir(exist_ok=True)
        
        self.temperature_scaler = None
        self.isotonic_calibrator = None
        self.sigmoid_calibrator = None
    
    def fit_temperature_scaling(self, y_true: np.ndarray, y_prob: np.ndarray) -> TemperatureScaling:
        """
        Fit Temperature Scaling calibrator.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            Fitted temperature scaler
        """
        logger.info("Fitting temperature scaling calibrator")
        
        self.temperature_scaler = TemperatureScaling()
        self.temperature_scaler.fit(y_true, y_prob)
        
        return self.temperature_scaler
    
    def fit_isotonic_regression(self, y_true: np.ndarray, y_prob: np.ndarray) -> IsotonicRegression:
        """
        Fit isotonic regression calibrator.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            Fitted isotonic calibrator
        """
        logger.info("Fitting isotonic regression calibrator")
        
        self.isotonic_calibrator = IsotonicRegression(out_of_bounds='clip')
        self.isotonic_calibrator.fit(y_prob, y_true)
        
        return self.isotonic_calibrator
    
    def compare_calibration_methods(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """
        Compare different calibration methods.
        
        Args:
            y_true: True labels
            y_prob: Uncalibrated probabilities
            
        Returns:
            Comparison results
        """
        logger.info("Comparing calibration methods")
        
        results = {}
        
        # Uncalibrated baseline
        uncal_metrics = CalibrationEvaluator.calibration_metrics(y_true, y_prob)
        results['uncalibrated'] = uncal_metrics
        
        # Temperature Scaling
        temp_scaler = self.fit_temperature_scaling(y_true, y_prob)
        temp_probs = temp_scaler.predict_proba(y_prob)
        temp_metrics = CalibrationEvaluator.calibration_metrics(y_true, temp_probs)
        results['temperature_scaling'] = temp_metrics
        
        # Isotonic Regression
        iso_calibrator = self.fit_isotonic_regression(y_true, y_prob)
        iso_probs = iso_calibrator.predict(y_prob)
        iso_metrics = CalibrationEvaluator.calibration_metrics(y_true, iso_probs)
        results['isotonic_regression'] = iso_metrics
        
        return results
    
    def save_calibrators(self) -> None:
        """Save all fitted calibrators."""
        if self.temperature_scaler:
            temp_path = self.models_path / CALIBRATOR_NAME
            self.temperature_scaler.save(temp_path)
        
        if self.isotonic_calibrator:
            iso_path = self.models_path / "isotonic_calibrator.pkl"
            joblib.dump(self.isotonic_calibrator, iso_path)
            logger.info(f"Isotonic calibrator saved to {iso_path}")
    
    def load_temperature_scaler(self) -> Optional[TemperatureScaling]:
        """Load temperature scaler if exists."""
        temp_path = self.models_path / CALIBRATOR_NAME
        if temp_path.exists():
            self.temperature_scaler = TemperatureScaling.load(temp_path)
            return self.temperature_scaler
        return None


def calibrate_model_predictions(y_true: np.ndarray, y_prob: np.ndarray,
                               models_path: Optional[Path] = None,
                               method: str = 'temperature_scaling',
                               save_calibrator: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function to calibrate model predictions.
    
    Args:
        y_true: True labels
        y_prob: Uncalibrated probabilities
        models_path: Path to save calibrators
        method: Calibration method ('temperature_scaling', 'isotonic', 'compare')
        save_calibrator: Whether to save the fitted calibrator
        
    Returns:
        Tuple of (calibrated_probabilities, calibration_metrics)
    """
    calibrator = ModelCalibrator(models_path)
    
    if method == 'temperature_scaling':
        temp_scaler = calibrator.fit_temperature_scaling(y_true, y_prob)
        calibrated_probs = temp_scaler.predict_proba(y_prob)
        metrics = CalibrationEvaluator.calibration_metrics(y_true, calibrated_probs)
        
        if save_calibrator:
            calibrator.save_calibrators()
        
        return calibrated_probs, metrics
        
    elif method == 'isotonic':
        iso_calibrator = calibrator.fit_isotonic_regression(y_true, y_prob)
        calibrated_probs = iso_calibrator.predict(y_prob)
        metrics = CalibrationEvaluator.calibration_metrics(y_true, calibrated_probs)
        
        if save_calibrator:
            calibrator.save_calibrators()
        
        return calibrated_probs, metrics
        
    elif method == 'compare':
        comparison_results = calibrator.compare_calibration_methods(y_true, y_prob)
        
        # Return temperature scaling as default
        temp_scaler = calibrator.temperature_scaler
        calibrated_probs = temp_scaler.predict_proba(y_prob)
        
        if save_calibrator:
            calibrator.save_calibrators()
        
        return calibrated_probs, comparison_results
    
    else:
        raise ValueError(f"Unknown calibration method: {method}")