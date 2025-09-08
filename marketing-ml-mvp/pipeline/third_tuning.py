"""
3er Hyperparameter Tuning - Metodología Ganadora con Optuna
Implementación exacta del tercer tuning que dio mejores resultados usando optimización bayesiana
"""
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
from xgboost import XGBClassifier
import json
import joblib
from pathlib import Path
import logging
from typing import Dict, Any, Tuple
from datetime import datetime
from .json_utils import safe_json_dump

logger = logging.getLogger(__name__)

class OptunaXGBoostOptimizer:
    """
    Implementación del 3er tuning usando Optuna - metodología ganadora del notebook
    """
    
    def __init__(self):
        self.best_params = None
        self.best_score = None
        self.study = None
        self.scale_pos_weight = None
        
        # Configuración de cross-validation
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.random_state = 42
        
    def calculate_scale_pos_weight(self, y_train: np.ndarray) -> float:
        """
        Calcular scale_pos_weight para datos desbalanceados
        """
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        logger.info(f"📊 Scale pos weight calculado: {scale_pos_weight:.4f}")
        return scale_pos_weight
    
    def objective(self, trial, X_train: pd.DataFrame, y_train: pd.DataFrame) -> float:
        """
        Función objetivo para Optuna - reproduce exactamente el notebook
        """
        # Definir parámetros a optimizar con rangos amplios
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500),  # más largo
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
            'scale_pos_weight': self.scale_pos_weight,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Crear modelo con optimizaciones del notebook
        model = XGBClassifier(
            tree_method='hist',
            enable_categorical=True,
            **param
        )
        
        # Cross-validation estratificado
        val_f1_scores = []
        
        for train_idx, val_idx in self.cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Entrenar con early stopping
            try:
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=100,
                    verbose=False
                )
            except TypeError:
                # Versión nueva de XGBoost
                model.set_params(early_stopping_rounds=100)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            
            # Predicciones y F1-score
            y_val_pred = model.predict(X_val)
            val_f1_scores.append(f1_score(y_val, y_val_pred))
        
        # Retornar F1-score promedio (SIN penalización por overfitting)
        return np.mean(val_f1_scores)
    
    def optimize(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Dict[str, Any]:
        """
        Ejecutar optimización con Optuna exactamente como en el notebook
        """
        logger.info("🎯 Iniciando 3er Tuning con Optuna - Metodología Ganadora")
        logger.info(f"🔄 Optimizando con 100 trials usando TPESampler")
        
        # Calcular scale_pos_weight
        y_train_values = y_train.values if hasattr(y_train, 'values') else y_train
        self.scale_pos_weight = self.calculate_scale_pos_weight(y_train_values)
        
        # Configurar sampler exacto del notebook
        sampler = TPESampler(seed=42, multivariate=True, n_startup_trials=20)
        
        # Crear estudio
        self.study = optuna.create_study(direction='maximize', sampler=sampler)
        
        start_time = datetime.now()
        logger.info("🚀 Iniciando optimización bayesiana...")
        
        # Optimizar (reproduce exacto del notebook)
        self.study.optimize(
            lambda trial: self.objective(trial, X_train, y_train),
            n_trials=100,
            callbacks=[self._log_callback]
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Extraer mejores resultados
        self.best_score = self.study.best_value
        self.best_params = self.study.best_params.copy()
        
        # Agregar parámetros fijos
        self.best_params.update({
            'scale_pos_weight': self.scale_pos_weight,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',
            'enable_categorical': True
        })
        
        logger.info(f"Optimización completada en {duration}")
        logger.info(f"🏆 Mejor F1-Score: {self.best_score:.4f}")
        logger.info(f"🎯 Mejores parámetros: {self.best_params}")
        
        # Crear resumen de resultados
        results = {
            'best_params': self.best_params,
            'best_score': float(self.best_score),
            'n_trials': len(self.study.trials),
            'tuning_type': 'optuna_bayesian',
            'methodology': 'winner_notebook',
            'duration_seconds': duration.total_seconds(),
            'scale_pos_weight': float(self.scale_pos_weight),
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def _log_callback(self, study, trial):
        """Callback para logging del progreso"""
        if trial.number % 10 == 0:
            logger.info(f"Trial {trial.number}: F1-Score = {trial.value:.4f}")
    
    def get_best_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> XGBClassifier:
        """
        Entrenar modelo final con mejores parámetros
        """
        if self.best_params is None:
            raise ValueError("Debe ejecutar optimize() primero")
        
        logger.info("🤖 Entrenando modelo final con mejores parámetros...")
        
        # Crear modelo con mejores parámetros
        best_model = XGBClassifier(**self.best_params)
        
        # Entrenar en todo el conjunto
        best_model.fit(X_train, y_train)
        
        logger.info("Modelo final entrenado")
        return best_model
    
    def save_results(self, artifacts_dir: Path) -> None:
        """
        Guardar resultados completos del tuning Optuna
        """
        if self.best_params is None:
            raise ValueError("Debe ejecutar optimize() primero")
        
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Guardar mejores parámetros
        best_params_file = artifacts_dir / "best_params_optuna.json"
        # Convert numpy types to Python types for JSON serialization
        serializable_params = {}
        for key, value in self.best_params.items():
            if hasattr(value, 'item'):  # numpy scalar
                serializable_params[key] = value.item()
            else:
                serializable_params[key] = value
        
        with open(best_params_file, 'w') as f:
            safe_json_dump(serializable_params, f, indent=2)
        logger.info(f"💾 Mejores parámetros guardados en {best_params_file}")
        
        # 2. Guardar estudio completo de Optuna
        study_file = artifacts_dir / "optuna_study.pkl"
        joblib.dump(self.study, study_file)
        logger.info(f"📊 Estudio Optuna guardado en {study_file}")
        
        # 3. Guardar resumen de resultados
        results_summary = {
            'best_params': self.best_params,
            'best_f1_score': float(self.best_score),
            'n_trials': len(self.study.trials),
            'scale_pos_weight': float(self.scale_pos_weight),
            'sampler': 'TPESampler',
            'methodology': 'notebook_winner',
            'cv_folds': self.cv.n_splits,
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = artifacts_dir / "optuna_results.json"
        with open(results_file, 'w') as f:
            safe_json_dump(results_summary, f, indent=2)
        logger.info(f"📈 Resumen completo guardado en {results_file}")
        
        # 4. Guardar histórico de trials
        trials_df = self.study.trials_dataframe()
        trials_file = artifacts_dir / "optuna_trials.csv"
        trials_df.to_csv(trials_file, index=False)
        logger.info(f"📋 Histórico de trials guardado en {trials_file}")

def find_optimal_threshold(model, X_val: pd.DataFrame, y_val: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """
    Encontrar threshold óptimo usando precision-recall curve
    """
    logger.info("🎯 Buscando threshold óptimo...")
    
    # Obtener probabilidades
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
    
    # F1-score para cada threshold
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)
    
    # Threshold óptimo
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    metrics = {
        'optimal_threshold': float(optimal_threshold),
        'optimal_f1': float(f1_scores[optimal_idx]),
        'optimal_precision': float(precision[optimal_idx]),
        'optimal_recall': float(recall[optimal_idx]),
        'roc_auc': float(roc_auc_score(y_val, y_proba))
    }
    
    logger.info(f"Threshold óptimo: {optimal_threshold:.4f}")
    logger.info(f"📊 F1-score óptimo: {metrics['optimal_f1']:.4f}")
    logger.info(f"🎯 ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return optimal_threshold, metrics

def run_third_tuning(X_train: pd.DataFrame, y_train: pd.DataFrame, 
                    X_val: pd.DataFrame, y_val: pd.DataFrame,
                    artifacts_dir: Path) -> Dict[str, Any]:
    """
    Ejecutar pipeline completo del 3er tuning con Optuna
    """
    logger.info("🚀 Iniciando pipeline completo del 3er tuning con Optuna...")
    
    # 1. Optimización de hiperparámetros con Optuna
    optimizer = OptunaXGBoostOptimizer()
    tuning_results = optimizer.optimize(X_train, y_train)
    
    # 2. Entrenar modelo final con mejores parámetros
    best_model = optimizer.get_best_model(X_train, y_train)
    
    # 3. Encontrar threshold óptimo
    optimal_threshold, threshold_metrics = find_optimal_threshold(
        best_model, X_val, y_val
    )
    
    # 4. Combinar resultados
    complete_results = {
        **tuning_results,
        'optimal_threshold': optimal_threshold,
        'threshold_metrics': threshold_metrics,
        'model': best_model
    }
    
    # 5. Guardar todo
    optimizer.save_results(artifacts_dir)
    
    # 6. Guardar modelo final
    model_file = artifacts_dir / "best_model_optuna.pkl"
    joblib.dump(best_model, model_file)
    logger.info(f"🤖 Modelo final guardado en {model_file}")
    
    # 7. Guardar threshold óptimo
    threshold_file = artifacts_dir / "optimal_threshold_optuna.json"
    with open(threshold_file, 'w') as f:
        safe_json_dump({
            'threshold': optimal_threshold,
            'metrics': threshold_metrics,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    logger.info(f"🎯 Threshold óptimo guardado en {threshold_file}")
    
    logger.info("Pipeline completo del 3er tuning con Optuna completado")
    
    return complete_results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🎯 Script del 3er Tuning con Optuna - Metodología Ganadora")
    print("Para ejecutar, usar run_pipeline.py que prepara los datos")
    print("Este script reproduce exactamente la optimización del notebook")