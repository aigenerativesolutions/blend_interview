"""
XGBoost model training module for marketing campaign prediction
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_recall_curve
from xgboost import XGBClassifier
import logging

from ..config.settings import (
    MODELS_PATH, MODEL_NAME, MODEL_METADATA_NAME, 
    XGBOOST_PARAMS, CV_FOLDS, STRATIFIED_CV
)

logger = logging.getLogger(__name__)


class XGBoostTrainer:
    """
    XGBoost model trainer with hyperparameter tuning and threshold optimization.
    Based on the notebook's 3-stage tuning approach.
    """
    
    def __init__(self, models_path: Optional[Path] = None):
        """
        Initialize trainer.
        
        Args:
            models_path: Path to save models. If None, uses default from settings.
        """
        self.models_path = models_path or MODELS_PATH
        self.models_path.mkdir(exist_ok=True)
        
        self.model = None
        self.best_params = None
        self.best_threshold = None
        self.cv_scores = None
        self.feature_names = None
        
    def train_base_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_val: pd.DataFrame, y_val: pd.Series) -> XGBClassifier:
        """
        Train base XGBoost model without hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features  
            y_val: Validation target
            
        Returns:
            Trained XGBClassifier
        """
        logger.info("Training base XGBoost model")
        
        # Store feature names for later use
        self.feature_names = list(X_train.columns)
        
        # Initialize model with base parameters
        model = XGBClassifier(**XGBOOST_PARAMS)
        
        # Train model
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        # Evaluate
        val_pred = model.predict(X_val)
        val_prob = model.predict_proba(X_val)[:, 1]
        
        roc_auc = roc_auc_score(y_val, val_prob)
        f1 = f1_score(y_val, val_pred)
        
        logger.info(f"Base model - ROC AUC: {roc_auc:.4f}, F1: {f1:.4f}")
        
        return model
    
    def first_tuning(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        First hyperparameter tuning round - focusing on tree structure.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Best parameters from first tuning
        """
        logger.info("Starting first hyperparameter tuning round")
        
        # Parameter grid for first tuning (based on notebook)
        param_grid_1 = {
            'max_depth': [4, 5, 6, 7],
            'min_child_weight': [1, 2, 3],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        best_params = self._grid_search_cv(X_train, y_train, param_grid_1)
        logger.info(f"First tuning best params: {best_params}")
        
        return best_params
    
    def second_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, 
                     base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Second hyperparameter tuning round - focusing on regularization.
        
        Args:
            X_train: Training features
            y_train: Training target
            base_params: Parameters from first tuning
            
        Returns:
            Best parameters from second tuning
        """
        logger.info("Starting second hyperparameter tuning round")
        
        # Parameter grid for second tuning (based on notebook)
        param_grid_2 = {
            'learning_rate': [0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        }
        
        # Combine with best params from first tuning
        combined_params = {**base_params, **XGBOOST_PARAMS}
        
        best_params = self._grid_search_cv(X_train, y_train, param_grid_2, combined_params)
        logger.info(f"Second tuning best params: {best_params}")
        
        return best_params
    
    def third_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Third hyperparameter tuning round - fine-tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            base_params: Parameters from previous tuning rounds
            
        Returns:
            Best parameters from third tuning
        """
        logger.info("Starting third hyperparameter tuning round")
        
        # Parameter grid for third tuning (based on notebook)
        param_grid_3 = {
            'max_depth': [base_params.get('max_depth', 6) - 1, 
                         base_params.get('max_depth', 6),
                         base_params.get('max_depth', 6) + 1],
            'learning_rate': [base_params.get('learning_rate', 0.1) * 0.8,
                            base_params.get('learning_rate', 0.1),
                            base_params.get('learning_rate', 0.1) * 1.2],
            'n_estimators': [base_params.get('n_estimators', 100) - 50,
                           base_params.get('n_estimators', 100),
                           base_params.get('n_estimators', 100) + 50]
        }
        
        # Clean negative values
        for key, values in param_grid_3.items():
            param_grid_3[key] = [max(1, v) if isinstance(v, (int, float)) else v for v in values]
        
        best_params = self._grid_search_cv(X_train, y_train, param_grid_3, base_params)
        logger.info(f"Third tuning best params: {best_params}")
        
        return best_params
    
    def _grid_search_cv(self, X: pd.DataFrame, y: pd.Series, param_grid: Dict[str, list], 
                       base_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform grid search with cross-validation.
        
        Args:
            X: Features
            y: Target
            param_grid: Parameter grid to search
            base_params: Base parameters to include
            
        Returns:
            Best parameters
        """
        from itertools import product
        
        if base_params is None:
            base_params = XGBOOST_PARAMS.copy()
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        best_score = -np.inf
        best_params = base_params.copy()
        
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        
        for combination in product(*param_values):
            params = base_params.copy()
            for name, value in zip(param_names, combination):
                params[name] = value
            
            # Cross-validation
            scores = []
            for train_idx, val_idx in cv.split(X, y):
                X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
                y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
                
                model = XGBClassifier(**params)
                model.fit(X_train_cv, y_train_cv, verbose=False)
                
                y_prob = model.predict_proba(X_val_cv)[:, 1]
                score = roc_auc_score(y_val_cv, y_prob)
                scores.append(score)
            
            avg_score = np.mean(scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = params.copy()
        
        logger.info(f"Best CV score: {best_score:.4f}")
        return best_params
    
    def optimize_threshold(self, X_val: pd.DataFrame, y_val: pd.Series, 
                         model: XGBClassifier) -> float:
        """
        Optimize classification threshold using precision-recall curve and F1 score.
        
        Args:
            X_val: Validation features
            y_val: Validation target
            model: Trained model
            
        Returns:
            Optimal threshold
        """
        logger.info("Optimizing classification threshold")
        
        # Get prediction probabilities
        y_prob = model.predict_proba(X_val)[:, 1]
        
        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
        
        # Calculate F1 scores for each threshold
        f1_scores = []
        for precision, recall in zip(precisions, recalls):
            if precision + recall == 0:
                f1_scores.append(0)
            else:
                f1_scores.append(2 * precision * recall / (precision + recall))
        
        # Find threshold that maximizes F1 score
        best_f1_idx = np.argmax(f1_scores)
        
        # Handle case where we have one fewer threshold than precision/recall values
        if best_f1_idx < len(thresholds):
            optimal_threshold = thresholds[best_f1_idx]
        else:
            optimal_threshold = 0.5  # Default fallback
        
        best_f1 = f1_scores[best_f1_idx]
        
        logger.info(f"Optimal threshold: {optimal_threshold:.4f}, F1 score: {best_f1:.4f}")
        
        return optimal_threshold
    
    def train_complete_pipeline(self, X: pd.DataFrame, y: pd.Series, 
                              test_size: float = 0.2) -> Tuple[XGBClassifier, Dict[str, Any]]:
        """
        Complete training pipeline with all tuning rounds and threshold optimization.
        
        Args:
            X: Features
            y: Target
            test_size: Size of validation set
            
        Returns:
            Tuple of (trained_model, metadata)
        """
        logger.info("Starting complete training pipeline")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
        
        # Train base model
        base_model = self.train_base_model(X_train, y_train, X_val, y_val)
        
        # First tuning
        params_1 = self.first_tuning(X_train, y_train)
        
        # Second tuning
        params_2 = self.second_tuning(X_train, y_train, params_1)
        
        # Third tuning
        final_params = self.third_tuning(X_train, y_train, params_2)
        
        # Train final model with best parameters
        logger.info("Training final model with optimized parameters")
        final_model = XGBClassifier(**final_params)
        final_model.fit(X_train, y_train, verbose=False)
        
        # Optimize threshold
        optimal_threshold = self.optimize_threshold(X_val, y_val, final_model)
        
        # Final evaluation
        val_prob = final_model.predict_proba(X_val)[:, 1]
        val_pred = (val_prob >= optimal_threshold).astype(int)
        
        final_metrics = {
            'roc_auc': roc_auc_score(y_val, val_prob),
            'f1_score': f1_score(y_val, val_pred),
            'classification_report': classification_report(y_val, val_pred, output_dict=True)
        }
        
        # Store results
        self.model = final_model
        self.best_params = final_params
        self.best_threshold = optimal_threshold
        
        # Create metadata
        metadata = {
            'model_type': 'XGBClassifier',
            'hyperparameters': final_params,
            'optimal_threshold': optimal_threshold,
            'metrics': final_metrics,
            'feature_names': self.feature_names,
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }
        
        logger.info(f"Training completed. ROC AUC: {final_metrics['roc_auc']:.4f}, F1: {final_metrics['f1_score']:.4f}")
        
        return final_model, metadata
    
    def save_model(self, model: XGBClassifier, metadata: Dict[str, Any]) -> None:
        """
        Save trained model and metadata.
        
        Args:
            model: Trained model
            metadata: Model metadata
        """
        # Save model
        model_path = self.models_path / MODEL_NAME
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metadata
        metadata_path = self.models_path / MODEL_METADATA_NAME
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Metadata saved to {metadata_path}")
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      params: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Target
            params: Model parameters
            
        Returns:
            Cross-validation scores
        """
        if params is None:
            params = self.best_params or XGBOOST_PARAMS
        
        logger.info(f"Performing {CV_FOLDS}-fold cross-validation")
        
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        
        scores = {
            'roc_auc': [],
            'f1_score': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            model = XGBClassifier(**params)
            model.fit(X_train_cv, y_train_cv, verbose=False)
            
            y_prob = model.predict_proba(X_val_cv)[:, 1]
            y_pred = model.predict(X_val_cv)
            
            roc_auc = roc_auc_score(y_val_cv, y_prob)
            f1 = f1_score(y_val_cv, y_pred)
            
            scores['roc_auc'].append(roc_auc)
            scores['f1_score'].append(f1)
            
            logger.info(f"Fold {fold + 1}: ROC AUC: {roc_auc:.4f}, F1: {f1:.4f}")
        
        # Calculate mean and std
        cv_results = {}
        for metric, values in scores.items():
            cv_results[f'{metric}_mean'] = np.mean(values)
            cv_results[f'{metric}_std'] = np.std(values)
        
        logger.info(f"CV Results: ROC AUC: {cv_results['roc_auc_mean']:.4f} ± {cv_results['roc_auc_std']:.4f}")
        logger.info(f"CV Results: F1: {cv_results['f1_score_mean']:.4f} ± {cv_results['f1_score_std']:.4f}")
        
        self.cv_scores = cv_results
        return cv_results


def train_marketing_model(X: pd.DataFrame, y: pd.Series, 
                         models_path: Optional[Path] = None,
                         save_model: bool = True) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """
    Convenience function to train marketing response prediction model.
    
    Args:
        X: Features
        y: Target
        models_path: Path to save models
        save_model: Whether to save the trained model
        
    Returns:
        Tuple of (trained_model, metadata)
    """
    trainer = XGBoostTrainer(models_path)
    model, metadata = trainer.train_complete_pipeline(X, y)
    
    if save_model:
        trainer.save_model(model, metadata)
    
    return model, metadata