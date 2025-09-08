"""
Final Model Training - FIXED VERSION
Entrenar modelo final con los mejores parámetros del 3er tuning de Optuna
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, roc_curve
)
import joblib
import json
from pathlib import Path
import logging
from typing import Dict, Any, Tuple
from datetime import datetime
from .json_utils import safe_json_dump
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class FinalModelTrainer:
    """
    Entrenar modelo final con mejores parámetros del 3er tuning
    """
    
    def __init__(self):
        self.model = None
        self.best_params = None
        self.training_metrics = {}
        self.test_metrics = {}
        self.optimal_threshold = 0.5
        
    def train_final_model(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         feature_names: list) -> xgb.XGBClassifier:
        """
        Entrenar modelo final con mejores parámetros
        """
        if self.best_params is None:
            raise ValueError("Debe proporcionar los mejores parámetros")
        
        logger.info("🎯 Entrenando modelo final con mejores parámetros de Optuna...")
        
        # Crear modelo con mejores parámetros
        self.model = xgb.XGBClassifier(**self.best_params)
        
        # Entrenar
        start_time = datetime.now()
        try:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=50,
                verbose=False
            )
        except TypeError:
            # Versión nueva de XGBoost
            self.model.set_params(early_stopping_rounds=50)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
        end_time = datetime.now()
        
        training_duration = end_time - start_time
        logger.info(f"Modelo entrenado en {training_duration}")
        
        # Calcular métricas
        self._calculate_metrics(X_train, y_train, X_test, y_test, feature_names)
        
        return self.model
    
    def _calculate_metrics(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          feature_names: list) -> None:
        """
        Calcular métricas completas del modelo
        """
        logger.info("📊 Calculando métricas del modelo...")
        
        # Predicciones
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        y_train_proba = self.model.predict_proba(X_train)[:, 1]
        y_test_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Predicciones con threshold óptimo
        y_train_pred_opt = (y_train_proba >= self.optimal_threshold).astype(int)
        y_test_pred_opt = (y_test_proba >= self.optimal_threshold).astype(int)
        
        # Métricas de entrenamiento
        self.training_metrics = {
            'accuracy': float(accuracy_score(y_train, y_train_pred)),
            'accuracy_optimal': float(accuracy_score(y_train, y_train_pred_opt)),
            'precision': float(precision_score(y_train, y_train_pred)),
            'precision_optimal': float(precision_score(y_train, y_train_pred_opt)),
            'recall': float(recall_score(y_train, y_train_pred)),
            'recall_optimal': float(recall_score(y_train, y_train_pred_opt)),
            'f1': float(f1_score(y_train, y_train_pred)),
            'f1_optimal': float(f1_score(y_train, y_train_pred_opt)),
            'roc_auc': float(roc_auc_score(y_train, y_train_proba))
        }
        
        # Métricas de test
        self.test_metrics = {
            'accuracy': float(accuracy_score(y_test, y_test_pred)),
            'accuracy_optimal': float(accuracy_score(y_test, y_test_pred_opt)),
            'precision': float(precision_score(y_test, y_test_pred)),
            'precision_optimal': float(precision_score(y_test, y_test_pred_opt)),
            'recall': float(recall_score(y_test, y_test_pred)),
            'recall_optimal': float(recall_score(y_test, y_test_pred_opt)),
            'f1': float(f1_score(y_test, y_test_pred)),
            'f1_optimal': float(f1_score(y_test, y_test_pred_opt)),
            'roc_auc': float(roc_auc_score(y_test, y_test_proba))
        }
        
        # Feature importance
        feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        logger.info(f"🎯 Test ROC-AUC: {self.test_metrics['roc_auc']:.4f}")
        logger.info(f"🎯 Test F1 (threshold=0.5): {self.test_metrics['f1']:.4f}")
        logger.info(f"🎯 Test F1 (threshold óptimo): {self.test_metrics['f1_optimal']:.4f}")
        logger.info(f"📈 Top 3 features: {list(sorted_importance.keys())[:3]}")
        
        return sorted_importance
    
    def save_final_model(self, artifacts_dir: Path, feature_names: list) -> None:
        """
        Guardar modelo final y metadatos
        """
        if self.model is None:
            raise ValueError("Debe entrenar el modelo primero")
        
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar modelo base XGBoost (formato nativo JSON - mejor práctica)
        model_file = artifacts_dir / "xgb_base_model.json"
        self.model.save_model(model_file)
        logger.info(f"💾 Modelo base XGBoost guardado en {model_file}")
        
        # También mantener compatibilidad con pickle para casos específicos
        model_file_pkl = artifacts_dir / "final_model.pkl"
        joblib.dump(self.model, model_file_pkl)
        logger.info(f"💾 Modelo también guardado como pickle en {model_file_pkl}")
        
        # Feature importance
        feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        # Metadatos completos
        metadata = {
            'model_type': 'XGBClassifier',
            'best_params': self.best_params,
            'optimal_threshold': self.optimal_threshold,
            'training_metrics': self.training_metrics,
            'test_metrics': self.test_metrics,
            'feature_importance': sorted_importance,
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'methodology': 'optuna_third_tuning_winner'
        }
        
        # Guardar metadatos
        metadata_file = artifacts_dir / "final_model_metadata.json"
        with open(metadata_file, 'w') as f:
            safe_json_dump(metadata, f, indent=2)
        logger.info(f"📊 Metadatos guardados en {metadata_file}")

def train_final_model_pipeline(X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray,
                              feature_names: list, artifacts_dir: Path,
                              best_params: Dict[str, Any] = None,
                              optimal_threshold: float = None) -> Dict[str, Any]:
    """
    Pipeline completo de entrenamiento del modelo final - VERSIÓN CORREGIDA
    Recibe parámetros directamente de Optuna en lugar de cargar archivos
    """
    logger.info("🚀 Iniciando entrenamiento del modelo final...")
    
    # Crear trainer
    trainer = FinalModelTrainer()
    
    # Usar parámetros pasados directamente
    if best_params is not None:
        trainer.best_params = best_params
        logger.info(f"Usando parámetros de Optuna: {len(best_params)} parámetros")
    else:
        raise ValueError("Se requieren best_params de Optuna")
    
    if optimal_threshold is not None:
        trainer.optimal_threshold = optimal_threshold
        logger.info(f"Usando threshold de Optuna: {optimal_threshold:.4f}")
    else:
        trainer.optimal_threshold = 0.5
        logger.info("Usando threshold por defecto: 0.5")
    
    # Entrenar modelo
    model = trainer.train_final_model(X_train, y_train, X_test, y_test, feature_names)
    
    # Guardar modelo y metadatos
    trainer.save_final_model(artifacts_dir, feature_names)
    
    logger.info("Pipeline de entrenamiento final completado")
    
    # Retornar resumen
    return {
        'model': model,
        'best_params': trainer.best_params,
        'optimal_threshold': trainer.optimal_threshold,
        'test_metrics': trainer.test_metrics,
        'training_metrics': trainer.training_metrics
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🎯 Script de Entrenamiento Final - VERSIÓN CORREGIDA")
    print("Usar run_pipeline.py para ejecutar el pipeline completo")