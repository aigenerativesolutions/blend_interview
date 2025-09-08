"""
Final Model Training
Entrenar modelo final con los mejores parámetros del 3er tuning
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
        
    def load_best_params(self, artifacts_dir: Path) -> Dict[str, Any]:
        """
        Cargar mejores parámetros del 3er tuning
        """
        best_params_file = artifacts_dir / "best_params_tuning3.json"
        
        if not best_params_file.exists():
            raise FileNotFoundError(f"Archivo de mejores parámetros no encontrado: {best_params_file}")
        
        with open(best_params_file, 'r') as f:
            self.best_params = json.load(f)
        
        logger.info(f"📂 Mejores parámetros cargados: {self.best_params}")
        return self.best_params
    
    def load_optimal_threshold(self, artifacts_dir: Path) -> float:
        """
        Cargar threshold óptimo
        """
        threshold_file = artifacts_dir / "optimal_threshold.json"
        
        if threshold_file.exists():
            with open(threshold_file, 'r') as f:
                threshold_data = json.load(f)
                self.optimal_threshold = threshold_data['threshold']
                logger.info(f"🎯 Threshold óptimo cargado: {self.optimal_threshold:.4f}")
        else:
            logger.warning("⚠️ Threshold óptimo no encontrado, usando 0.5")
            self.optimal_threshold = 0.5
        
        return self.optimal_threshold
    
    def train_final_model(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         feature_names: list) -> xgb.XGBClassifier:
        """
        Entrenar modelo final con mejores parámetros
        """
        if self.best_params is None:
            raise ValueError("Debe cargar los mejores parámetros primero")
        
        logger.info("🎯 Entrenando modelo final con mejores parámetros del 3er tuning...")
        
        # Crear modelo con mejores parámetros
        self.model = xgb.XGBClassifier(
            **self.best_params,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # Entrenar
        start_time = datetime.now()
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=50,
            verbose=False
        )
        end_time = datetime.now()
        
        training_duration = end_time - start_time
        logger.info(f"✅ Modelo entrenado en {training_duration}")
        
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
        }\n        \n        # Métricas de test\n        self.test_metrics = {\n            'accuracy': float(accuracy_score(y_test, y_test_pred)),\n            'accuracy_optimal': float(accuracy_score(y_test, y_test_pred_opt)),\n            'precision': float(precision_score(y_test, y_test_pred)),\n            'precision_optimal': float(precision_score(y_test, y_test_pred_opt)),\n            'recall': float(recall_score(y_test, y_test_pred)),\n            'recall_optimal': float(recall_score(y_test, y_test_pred_opt)),\n            'f1': float(f1_score(y_test, y_test_pred)),\n            'f1_optimal': float(f1_score(y_test, y_test_pred_opt)),\n            'roc_auc': float(roc_auc_score(y_test, y_test_proba))\n        }\n        \n        # Feature importance\n        feature_importance = dict(zip(feature_names, self.model.feature_importances_))\n        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))\n        \n        logger.info(f\"🎯 Test ROC-AUC: {self.test_metrics['roc_auc']:.4f}\")\n        logger.info(f\"🎯 Test F1 (threshold=0.5): {self.test_metrics['f1']:.4f}\")\n        logger.info(f\"🎯 Test F1 (threshold óptimo): {self.test_metrics['f1_optimal']:.4f}\")\n        logger.info(f\"📈 Top 3 features: {list(sorted_importance.keys())[:3]}\")\n        \n        return sorted_importance\n    \n    def generate_plots(self, X_test: np.ndarray, y_test: np.ndarray, \n                      artifacts_dir: Path) -> None:\n        \"\"\"\n        Generar plots de evaluación del modelo\n        \"\"\"\n        logger.info(\"📈 Generando plots de evaluación...\")\n        \n        # Crear directorio para plots\n        plots_dir = artifacts_dir / \"plots\"\n        plots_dir.mkdir(exist_ok=True)\n        \n        y_test_proba = self.model.predict_proba(X_test)[:, 1]\n        \n        # 1. ROC Curve\n        plt.figure(figsize=(8, 6))\n        fpr, tpr, _ = roc_curve(y_test, y_test_proba)\n        plt.plot(fpr, tpr, color='darkorange', lw=2, \n                label=f'ROC curve (AUC = {self.test_metrics[\"roc_auc\"]:.3f})')\n        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\n        plt.xlim([0.0, 1.0])\n        plt.ylim([0.0, 1.05])\n        plt.xlabel('False Positive Rate')\n        plt.ylabel('True Positive Rate')\n        plt.title('ROC Curve - Final Model')\n        plt.legend(loc=\"lower right\")\n        plt.grid(alpha=0.3)\n        plt.tight_layout()\n        plt.savefig(plots_dir / \"roc_curve.png\", dpi=300, bbox_inches='tight')\n        plt.close()\n        \n        # 2. Precision-Recall Curve\n        plt.figure(figsize=(8, 6))\n        precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)\n        f1_scores = 2 * (precision * recall) / (precision + recall)\n        f1_scores = np.nan_to_num(f1_scores)\n        \n        plt.plot(recall, precision, color='blue', lw=2, label='PR curve')\n        plt.axvline(x=self.test_metrics['recall_optimal'], color='red', linestyle='--', \n                   label=f'Optimal threshold ({self.optimal_threshold:.3f})')\n        plt.axhline(y=self.test_metrics['precision_optimal'], color='red', linestyle='--')\n        plt.xlabel('Recall')\n        plt.ylabel('Precision')\n        plt.title('Precision-Recall Curve - Final Model')\n        plt.legend()\n        plt.grid(alpha=0.3)\n        plt.tight_layout()\n        plt.savefig(plots_dir / \"precision_recall_curve.png\", dpi=300, bbox_inches='tight')\n        plt.close()\n        \n        # 3. Confusion Matrix\n        plt.figure(figsize=(8, 6))\n        y_test_pred_opt = (y_test_proba >= self.optimal_threshold).astype(int)\n        cm = confusion_matrix(y_test, y_test_pred_opt)\n        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', \n                   xticklabels=['No Response', 'Response'],\n                   yticklabels=['No Response', 'Response'])\n        plt.title(f'Confusion Matrix (threshold={self.optimal_threshold:.3f})')\n        plt.ylabel('Actual')\n        plt.xlabel('Predicted')\n        plt.tight_layout()\n        plt.savefig(plots_dir / \"confusion_matrix.png\", dpi=300, bbox_inches='tight')\n        plt.close()\n        \n        logger.info(f\"📈 Plots guardados en {plots_dir}\")\n    \n    def save_final_model(self, artifacts_dir: Path, feature_names: list) -> None:\n        \"\"\"\n        Guardar modelo final y metadatos\n        \"\"\"\n        if self.model is None:\n            raise ValueError(\"Debe entrenar el modelo primero\")\n        \n        artifacts_dir.mkdir(parents=True, exist_ok=True)\n        \n        # Guardar modelo\n        model_file = artifacts_dir / \"final_model.pkl\"\n        joblib.dump(self.model, model_file)\n        logger.info(f\"💾 Modelo final guardado en {model_file}\")\n        \n        # Feature importance\n        feature_importance = dict(zip(feature_names, self.model.feature_importances_))\n        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))\n        \n        # Metadatos completos\n        metadata = {\n            'model_type': 'XGBClassifier',\n            'best_params': self.best_params,\n            'optimal_threshold': self.optimal_threshold,\n            'training_metrics': self.training_metrics,\n            'test_metrics': self.test_metrics,\n            'feature_importance': sorted_importance,\n            'feature_names': feature_names,\n            'n_features': len(feature_names),\n            'timestamp': datetime.now().isoformat(),\n            'version': '1.0.0',\n            'methodology': 'third_tuning_winner'\n        }\n        \n        # Guardar metadatos\n        metadata_file = artifacts_dir / \"final_model_metadata.json\"\n        with open(metadata_file, 'w') as f:\n            json.dump(metadata, f, indent=2)\n        logger.info(f\"📊 Metadatos guardados en {metadata_file}\")\n        \n        # Guardar feature importance separado\n        importance_file = artifacts_dir / \"feature_importance.json\"\n        with open(importance_file, 'w') as f:\n            json.dump(sorted_importance, f, indent=2)\n        logger.info(f\"📈 Feature importance guardado en {importance_file}\")\n\ndef train_final_model_pipeline(X_train: np.ndarray, y_train: np.ndarray,\n                              X_test: np.ndarray, y_test: np.ndarray,\n                              feature_names: list, artifacts_dir: Path) -> Dict[str, Any]:\n    \"\"\"\n    Pipeline completo de entrenamiento del modelo final\n    \"\"\"\n    logger.info(\"🚀 Iniciando entrenamiento del modelo final...\")\n    \n    # Crear trainer\n    trainer = FinalModelTrainer()\n    \n    # Cargar mejores parámetros y threshold\n    trainer.load_best_params(artifacts_dir)\n    trainer.load_optimal_threshold(artifacts_dir)\n    \n    # Entrenar modelo\n    model = trainer.train_final_model(X_train, y_train, X_test, y_test, feature_names)\n    \n    # Generar plots\n    trainer.generate_plots(X_test, y_test, artifacts_dir)\n    \n    # Guardar modelo y metadatos\n    trainer.save_final_model(artifacts_dir, feature_names)\n    \n    logger.info(\"✅ Pipeline de entrenamiento final completado\")\n    \n    # Retornar resumen\n    return {\n        'model': model,\n        'best_params': trainer.best_params,\n        'optimal_threshold': trainer.optimal_threshold,\n        'test_metrics': trainer.test_metrics,\n        'training_metrics': trainer.training_metrics\n    }\n\nif __name__ == \"__main__\":\n    logging.basicConfig(level=logging.INFO)\n    \n    print(\"🎯 Script de Entrenamiento Final\")\n    print(\"Usar run_pipeline.py para ejecutar el pipeline completo\")