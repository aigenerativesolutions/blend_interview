"""
Clase de inferencia para producción - Mejores prácticas 2025
Carga modelo base XGBoost y calibrador por separado
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
from typing import Union, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ProductionMarketingModel:
    """
    Modelo de inferencia para producción siguiendo mejores prácticas 2025:
    - Modelo base XGBoost en formato nativo JSON
    - Calibrador de temperatura separado
    - Versionado independiente de componentes
    """
    
    def __init__(self, model_path: Union[str, Path], 
                 calibrator_path: Union[str, Path] = None,
                 threshold_path: Union[str, Path] = None):
        """
        Inicializar modelo de producción
        
        Args:
            model_path: Ruta al modelo XGBoost (.json)
            calibrator_path: Ruta al calibrador de temperatura (.pkl)
            threshold_path: Ruta al threshold óptimo (.json)
        """
        self.model_path = Path(model_path)
        self.calibrator_path = Path(calibrator_path) if calibrator_path else None
        self.threshold_path = Path(threshold_path) if threshold_path else None
        
        self.base_model = None
        self.calibrator = None
        self.optimal_threshold = 0.5
        
        self._load_components()
    
    def _load_components(self) -> None:
        """Cargar todos los componentes del modelo"""
        
        # 1. Cargar modelo base XGBoost (formato nativo)
        logger.info(f"Cargando modelo XGBoost desde {self.model_path}")
        self.base_model = xgb.XGBClassifier()
        self.base_model.load_model(self.model_path)
        logger.info("✓ Modelo base XGBoost cargado")
        
        # 2. Cargar calibrador de temperatura (si existe)
        if self.calibrator_path and self.calibrator_path.exists():
            logger.info(f"Cargando calibrador desde {self.calibrator_path}")
            self.calibrator = joblib.load(self.calibrator_path)
            logger.info("✓ Calibrador de temperatura cargado")
        else:
            logger.warning("No se encontró calibrador - usando probabilidades sin calibrar")
        
        # 3. Cargar threshold óptimo (si existe)
        if self.threshold_path and self.threshold_path.exists():
            import json
            with open(self.threshold_path, 'r') as f:
                threshold_data = json.load(f)
                self.optimal_threshold = threshold_data.get('optimal_threshold', 0.5)
            logger.info(f"✓ Threshold óptimo cargado: {self.optimal_threshold:.4f}")
        else:
            logger.warning("No se encontró threshold óptimo - usando 0.5")
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predecir probabilidades calibradas
        
        Args:
            X: Features de entrada
            
        Returns:
            Array de probabilidades calibradas [prob_clase_0, prob_clase_1]
        """
        # 1. Predicciones del modelo base
        raw_probs = self.base_model.predict_proba(X)
        
        # 2. Aplicar calibración de temperatura (si está disponible)
        if self.calibrator is not None:
            try:
                calibrated_probs = self.calibrator.predict_proba(raw_probs)
                logger.debug("Probabilidades calibradas aplicadas")
                return calibrated_probs
            except Exception as e:
                logger.warning(f"Error aplicando calibración: {e} - usando probabilidades sin calibrar")
                return raw_probs
        
        return raw_probs
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predecir clases usando threshold óptimo
        
        Args:
            X: Features de entrada
            
        Returns:
            Array de predicciones binarias [0, 1]
        """
        probs = self.predict_proba(X)
        # Usar threshold óptimo para clase positiva
        predictions = (probs[:, 1] >= self.optimal_threshold).astype(int)
        return predictions
    
    def predict_with_confidence(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Predecir con métricas de confianza
        
        Args:
            X: Features de entrada
            
        Returns:
            Diccionario con predicciones, probabilidades y confianza
        """
        probs = self.predict_proba(X)
        predictions = self.predict(X)
        
        # Calcular métricas de confianza
        confidence_scores = np.max(probs, axis=1)  # Máxima probabilidad
        uncertainty = 1 - confidence_scores  # Incertidumbre
        
        return {
            'predictions': predictions,
            'probabilities': probs,
            'confidence_scores': confidence_scores,
            'uncertainty': uncertainty,
            'threshold_used': self.optimal_threshold
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Información del modelo cargado"""
        return {
            'model_path': str(self.model_path),
            'calibrator_path': str(self.calibrator_path) if self.calibrator_path else None,
            'has_calibrator': self.calibrator is not None,
            'optimal_threshold': self.optimal_threshold,
            'model_type': 'XGBClassifier',
            'calibration_method': 'temperature_scaling' if self.calibrator else None
        }

# Ejemplo de uso
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Configurar rutas
    artifacts_dir = Path("artifacts")
    
    try:
        # Inicializar modelo de producción
        model = ProductionMarketingModel(
            model_path=artifacts_dir / "xgb_base_model.json",
            calibrator_path=artifacts_dir / "temperature_calibrator.pkl",
            threshold_path=artifacts_dir / "optimal_threshold_optuna.json"
        )
        
        print("✓ Modelo de producción inicializado")
        print("Información del modelo:")
        info = model.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error inicializando modelo: {e}")