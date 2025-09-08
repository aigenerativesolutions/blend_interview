"""
Temperature Scaling Calibration - Robust Implementation from Notebook
Implementación robusta de calibración con Temperature Scaling usando scipy.optimize
"""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from scipy.special import expit, logit
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path
import logging
from typing import Dict, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class TemperatureScalingCalibrator:
    """
    Temperature Scaling robusto para calibrar probabilidades del modelo
    Implementación exacta del notebook usando scipy.optimize
    """
    
    def __init__(self):
        self.temperature = 1.0
        self.is_fitted = False
        self.calibration_metrics = {}
        self.y_true = None
        self.p_uncalibrated = None
    
    def logit_clip(self, p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Clip probabilities and compute logit to avoid log(0) - from notebook
        """
        p = np.asarray(p).clip(eps, 1-eps)
        return np.log(p/(1-p))
    
    def calibrate_temp(self, p: np.ndarray, T: float) -> np.ndarray:
        """
        Apply temperature scaling - from notebook
        """
        return expit(self.logit_clip(p) / T)
    
    def _objective_T(self, T: float) -> float:
        """
        Objective function for temperature optimization - from notebook
        """
        pc = self.calibrate_temp(self.p_uncalibrated, T)
        return log_loss(self.y_true, pc, labels=[0, 1])
    
    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> 'TemperatureScalingCalibrator':
        """
        Entrenar el calibrador de temperatura usando scipy.optimize (implementación del notebook)
        """
        logger.info("🌡️ Entrenando Temperature Scaling calibrator (robust version)...")
        
        # Store data for optimization
        self.p_uncalibrated = np.asarray(probabilities).ravel()
        self.y_true = np.asarray(y_true).ravel()
        
        # Optimize temperature using scipy.optimize (from notebook)
        logger.info("🔧 Optimizando temperatura con minimize_scalar...")
        res = minimize_scalar(
            self._objective_T, 
            bounds=(0.2, 5.0), 
            method='bounded'
        )
        
        self.temperature = float(res.x)
        self.is_fitted = True
        
        # Calculate calibration metrics
        calibrated_probs = self.predict_proba(probabilities)
        self._calculate_calibration_metrics(probabilities, calibrated_probs, y_true)
        
        logger.info(f"✅ Calibrador entrenado. Temperatura óptima: {self.temperature:.4f}")
        logger.info(f"📊 Log-loss optimizado: {res.fun:.6f}")
        logger.info(f"📊 Brier Score antes: {self.calibration_metrics['brier_before']:.4f}")
        logger.info(f"📊 Brier Score después: {self.calibration_metrics['brier_after']:.4f}")
        
        return self
    
    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Aplicar calibración de temperatura usando implementación robusta del notebook
        """
        if not self.is_fitted:
            raise ValueError("Calibrador debe ser fitted antes de predict_proba")
        
        # Use robust implementation from notebook
        return self.calibrate_temp(probabilities, self.temperature)
    
    def _calculate_calibration_metrics(self, original_probs: np.ndarray, 
                                     calibrated_probs: np.ndarray, 
                                     y_true: np.ndarray) -> None:
        """
        Calcular métricas de calibración
        """
        self.calibration_metrics = {
            'temperature': float(self.temperature),
            'brier_before': float(brier_score_loss(y_true, original_probs)),
            'brier_after': float(brier_score_loss(y_true, calibrated_probs)),
            'log_loss_before': float(log_loss(y_true, original_probs)),
            'log_loss_after': float(log_loss(y_true, calibrated_probs)),
            'n_samples': len(y_true),
            'timestamp': datetime.now().isoformat()
        }
        
        # Calcular mejora
        self.calibration_metrics['brier_improvement'] = (
            self.calibration_metrics['brier_before'] - self.calibration_metrics['brier_after']
        )
        self.calibration_metrics['log_loss_improvement'] = (
            self.calibration_metrics['log_loss_before'] - self.calibration_metrics['log_loss_after']
        )
    
    def generate_reliability_diagram(self, original_probs: np.ndarray, 
                                   calibrated_probs: np.ndarray,
                                   y_true: np.ndarray, 
                                   artifacts_dir: Path) -> None:
        """
        Generar diagrama de confiabilidad (reliability diagram)
        """
        logger.info("📈 Generando reliability diagram...")
        
        plots_dir = artifacts_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Reliability diagram - Antes de calibración
            fraction_of_positives, mean_predicted_value = self._reliability_curve(
                y_true, original_probs, n_bins=10
            )
            ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="Original", linewidth=2)
            ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            ax1.set_xlabel("Mean Predicted Probability")
            ax1.set_ylabel("Fraction of Positives")
            ax1.set_title("Reliability Diagram - Before Calibration")
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # Reliability diagram - Después de calibración
            fraction_of_positives_cal, mean_predicted_value_cal = self._reliability_curve(
                y_true, calibrated_probs, n_bins=10
            )
            ax2.plot(mean_predicted_value_cal, fraction_of_positives_cal, "s-", label="Calibrated", linewidth=2)
            ax2.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            ax2.set_xlabel("Mean Predicted Probability")
            ax2.set_ylabel("Fraction of Positives")
            ax2.set_title("Reliability Diagram - After Calibration")
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "reliability_diagram.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Histograma de probabilidades
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Antes
            ax1.hist(original_probs, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax1.set_xlabel("Predicted Probability")
            ax1.set_ylabel("Frequency")
            ax1.set_title("Probability Distribution - Before Calibration")
            ax1.grid(alpha=0.3)
            
            # Después
            ax2.hist(calibrated_probs, bins=50, alpha=0.7, color='green', edgecolor='black')
            ax2.set_xlabel("Predicted Probability")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Probability Distribution - After Calibration")
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "probability_distributions.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📈 Reliability diagrams guardados en {plots_dir}")
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
    
    def _reliability_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                         n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcular curva de confiabilidad
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        fraction_of_positives = []
        mean_predicted_values = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.sum() / len(y_prob)
            
            if prop_in_bin > 0 and in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                
                fraction_of_positives.append(accuracy_in_bin)
                mean_predicted_values.append(avg_confidence_in_bin)
            else:
                fraction_of_positives.append(0)
                mean_predicted_values.append((bin_lower + bin_upper) / 2)
        
        return np.array(fraction_of_positives), np.array(mean_predicted_values)
    
    def save(self, filepath: Path) -> None:
        """
        Serializar calibrador
        """
        calibrator_data = {
            'temperature': self.temperature,
            'is_fitted': self.is_fitted,
            'calibration_metrics': self.calibration_metrics,
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.0_robust'
        }
        
        joblib.dump(calibrator_data, filepath)
        logger.info(f"💾 Calibrador guardado en {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'TemperatureScalingCalibrator':
        """
        Cargar calibrador serializado
        """
        calibrator_data = joblib.load(filepath)
        
        calibrator = cls()
        calibrator.temperature = calibrator_data['temperature']
        calibrator.is_fitted = calibrator_data['is_fitted']
        calibrator.calibration_metrics = calibrator_data.get('calibration_metrics', {})
        
        logger.info(f"📂 Calibrador cargado desde {filepath}")
        logger.info(f"🌡️ Temperatura: {calibrator.temperature:.4f}")
        
        return calibrator

def calibrate_model_probabilities(model, X_val: np.ndarray, y_val: np.ndarray,
                                 artifacts_dir: Path) -> TemperatureScalingCalibrator:
    """
    Pipeline completo de calibración de probabilidades
    """
    logger.info("🌡️ Iniciando calibración de probabilidades...")
    
    # Obtener probabilidades del modelo
    original_probabilities = model.predict_proba(X_val)[:, 1]
    
    # Crear y entrenar calibrador
    calibrator = TemperatureScalingCalibrator()
    calibrator.fit(original_probabilities, y_val)
    
    # Obtener probabilidades calibradas
    calibrated_probabilities = calibrator.predict_proba(original_probabilities)
    
    # Generar plots
    calibrator.generate_reliability_diagram(
        original_probabilities, calibrated_probabilities, y_val, artifacts_dir
    )
    
    # Guardar calibrador
    calibrator_path = artifacts_dir / "temperature_calibrator.pkl"
    calibrator.save(calibrator_path)
    
    # Guardar métricas de calibración
    metrics_path = artifacts_dir / "calibration_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(calibrator.calibration_metrics, f, indent=2)
    
    logger.info("✅ Calibración completada y guardada")
    
    return calibrator

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🌡️ Script de Temperature Scaling Calibration")
    print("Usar run_pipeline.py para ejecutar el pipeline completo")