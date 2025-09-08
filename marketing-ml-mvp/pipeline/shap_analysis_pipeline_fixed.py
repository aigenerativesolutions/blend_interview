"""
SHAP Analysis Pipeline - VERSIÓN MEJORADA
Análisis completo de SHAP con top valores e interdependencias
"""
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from .json_utils import safe_json_dump

logger = logging.getLogger(__name__)

class SHAPAnalysisPipeline:
    """
    Pipeline completo de análisis SHAP con análisis detallado
    Incluye top valores positivos/negativos e interdependencias
    """
    
    def __init__(self):
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        self.feature_names = []
        self.is_fitted = False
        self.background_data = None
        self._current_X = None
    
    def create_explainer(self, model, background_data: np.ndarray, 
                        feature_names: List[str]) -> None:
        """
        Crear explainer SHAP para el modelo
        """
        logger.info("🔍 Creando SHAP explainer...")
        
        self.feature_names = feature_names
        self.background_data = background_data
        
        # Usar TreeExplainer para XGBoost (más rápido y preciso)
        self.explainer = shap.TreeExplainer(model)
        self.expected_value = self.explainer.expected_value
        
        # Si expected_value es array, tomar el valor para clase positiva
        if isinstance(self.expected_value, (list, np.ndarray)):
            self.expected_value = self.expected_value[1] if len(self.expected_value) > 1 else self.expected_value[0]
        
        self.is_fitted = True
        logger.info(f"SHAP explainer creado. Expected value: {self.expected_value:.4f}")
    
    def calculate_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Calcular SHAP values para el dataset
        """
        if not self.is_fitted:
            raise ValueError("Debe crear el explainer primero")
        
        logger.info(f"🔄 Calculando SHAP values para {X.shape[0]} samples...")
        
        # Guardar X para análisis posteriores
        self._current_X = X
        
        # Calcular SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # TreeExplainer puede retornar lista o array
        if isinstance(shap_values, list):
            # Para clasificación binaria, tomar clase positiva (índice 1)
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        self.shap_values = shap_values
        logger.info(f"SHAP values calculados. Shape: {shap_values.shape}")
        
        return shap_values
    
    def get_global_feature_importance(self) -> Dict[str, float]:
        """
        Obtener importancia global de features usando SHAP
        """
        if self.shap_values is None:
            raise ValueError("Debe calcular SHAP values primero")
        
        # Importancia basada en valor absoluto medio
        importance = np.abs(self.shap_values).mean(axis=0)
        
        # Crear diccionario con nombres de features
        feature_importance = dict(zip(self.feature_names, importance))
        
        # Ordenar por importancia
        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        logger.info(f"📊 Top 5 features: {list(sorted_importance.keys())[:5]}")
        
        return sorted_importance
    
    def extract_top_shap_values(self, X: np.ndarray, top_n: int = 20) -> Dict[str, Dict]:
        """
        Extraer top N valores positivos y negativos de SHAP para cada feature
        """
        logger.info(f"📊 Extrayendo top {top_n} valores SHAP por feature...")
        
        feature_analysis = {}
        
        for i, feature_name in enumerate(self.feature_names):
            feature_shap_values = self.shap_values[:, i]
            feature_input_values = X[:, i]
            
            # Valores positivos (ordenados de mayor a menor)
            positive_mask = feature_shap_values > 0
            if np.any(positive_mask):
                pos_indices = np.where(positive_mask)[0]
                pos_shap = feature_shap_values[pos_indices]
                pos_inputs = feature_input_values[pos_indices]
                
                # Ordenar por valor SHAP descendente
                sorted_pos_idx = np.argsort(pos_shap)[::-1]
                top_pos_count = min(len(sorted_pos_idx), top_n)
                sorted_idx = sorted_pos_idx[:top_pos_count]
                
                top_positive = [
                    {
                        'sample_idx': int(pos_indices[idx]),
                        'shap_value': float(pos_shap[idx]),
                        'input_value': float(pos_inputs[idx])
                    }
                    for idx in sorted_idx
                ]
            else:
                top_positive = []
            
            # Valores negativos (ordenados de menor a mayor, más negativos primero)
            negative_mask = feature_shap_values < 0
            if np.any(negative_mask):
                neg_indices = np.where(negative_mask)[0]
                neg_shap = feature_shap_values[neg_indices]
                neg_inputs = feature_input_values[neg_indices]
                
                # Ordenar por valor SHAP ascendente (más negativo primero)
                sorted_neg_idx = np.argsort(neg_shap)
                top_neg_count = min(len(sorted_neg_idx), top_n)
                sorted_idx = sorted_neg_idx[:top_neg_count]
                
                top_negative = [
                    {
                        'sample_idx': int(neg_indices[idx]),
                        'shap_value': float(neg_shap[idx]),
                        'input_value': float(neg_inputs[idx])
                    }
                    for idx in sorted_idx
                ]
            else:
                top_negative = []
            
            feature_analysis[feature_name] = {
                'top_positive_impacts': top_positive,
                'top_negative_impacts': top_negative,
                'total_positive': len(top_positive),
                'total_negative': len(top_negative),
                'mean_positive_shap': float(np.mean(feature_shap_values[positive_mask])) if np.any(positive_mask) else 0.0,
                'mean_negative_shap': float(np.mean(feature_shap_values[negative_mask])) if np.any(negative_mask) else 0.0
            }
        
        logger.info(f"Top valores SHAP extraídos para {len(self.feature_names)} features")
        return feature_analysis
    
    def analyze_feature_interactions(self, X: np.ndarray, top_features: int = 7) -> Dict[str, Dict]:
        """
        Analizar interdependencias entre las top N features más importantes
        """
        logger.info(f"🔗 Analizando interdependencias entre top {top_features} features...")
        
        # Obtener top features por importancia
        feature_importance = self.get_global_feature_importance()
        top_feature_names = list(feature_importance.keys())[:top_features]
        top_feature_indices = [self.feature_names.index(name) for name in top_feature_names]
        
        interactions = {}
        
        for i, feature_i in enumerate(top_feature_names):
            feature_i_idx = top_feature_indices[i]
            feature_i_shap = self.shap_values[:, feature_i_idx]
            feature_i_values = X[:, feature_i_idx]
            
            interactions[feature_i] = {}
            
            for j, feature_j in enumerate(top_feature_names):
                if i != j:  # No comparar feature consigo misma
                    feature_j_idx = top_feature_indices[j]
                    feature_j_values = X[:, feature_j_idx]
                    
                    # Calcular correlación entre SHAP values
                    shap_correlation = np.corrcoef(feature_i_shap, self.shap_values[:, feature_j_idx])[0, 1]
                    
                    # Calcular correlación entre input values
                    input_correlation = np.corrcoef(feature_i_values, feature_j_values)[0, 1]
                    
                    # Análisis por cuadrantes
                    # Cuadrante 1: Ambos features altos
                    high_i = feature_i_values > np.median(feature_i_values)
                    high_j = feature_j_values > np.median(feature_j_values)
                    q1_mask = high_i & high_j
                    
                    # Cuadrante 2: Feature i alto, j bajo
                    low_j = feature_j_values <= np.median(feature_j_values)
                    q2_mask = high_i & low_j
                    
                    # Cuadrante 3: Ambos features bajos
                    low_i = feature_i_values <= np.median(feature_i_values)
                    q3_mask = low_i & low_j
                    
                    # Cuadrante 4: Feature i bajo, j alto
                    q4_mask = low_i & high_j
                    
                    quadrant_analysis = {}
                    for q_name, q_mask in [('both_high', q1_mask), ('i_high_j_low', q2_mask), 
                                          ('both_low', q3_mask), ('i_low_j_high', q4_mask)]:
                        if np.any(q_mask):
                            quadrant_analysis[q_name] = {
                                'count': int(np.sum(q_mask)),
                                'mean_shap_i': float(np.mean(feature_i_shap[q_mask])),
                                'mean_shap_j': float(np.mean(self.shap_values[:, feature_j_idx][q_mask])),
                                'mean_input_i': float(np.mean(feature_i_values[q_mask])),
                                'mean_input_j': float(np.mean(feature_j_values[q_mask]))
                            }
                    
                    interactions[feature_i][feature_j] = {
                        'shap_correlation': float(shap_correlation) if not np.isnan(shap_correlation) else 0.0,
                        'input_correlation': float(input_correlation) if not np.isnan(input_correlation) else 0.0,
                        'quadrant_analysis': quadrant_analysis
                    }
        
        logger.info(f"Análisis de interdependencias completado para {len(top_feature_names)} features")
        return interactions
    
    def save_analysis_results(self, artifacts_dir: Path, 
                             feature_importance: Dict[str, float]) -> None:
        """
        Guardar todos los resultados del análisis SHAP incluyendo top valores e interdependencias
        """
        logger.info("💾 Guardando resultados de análisis SHAP...")
        
        shap_dir = artifacts_dir / "shap_values"
        shap_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar SHAP explainer
        explainer_path = shap_dir / "shap_explainer.pkl"
        joblib.dump(self.explainer, explainer_path)
        
        # Guardar SHAP values
        values_path = shap_dir / "shap_values.npz"
        np.savez_compressed(values_path, 
                           shap_values=self.shap_values,
                           expected_value=self.expected_value,
                           feature_names=self.feature_names)
        
        # Guardar feature importance
        importance_path = shap_dir / "global_feature_importance.json"
        with open(importance_path, 'w') as f:
            safe_json_dump(feature_importance, f, indent=2)
        
        # NUEVO: Extraer y guardar top 20 valores positivos/negativos por feature
        logger.info("📊 Extrayendo top valores SHAP por feature...")
        if self._current_X is not None:
            top_values_analysis = self.extract_top_shap_values(self._current_X, top_n=20)
            top_values_path = shap_dir / "top_shap_values_by_feature.json"
            with open(top_values_path, 'w') as f:
                safe_json_dump(top_values_analysis, f, indent=2)
            logger.info(f"💾 Top valores SHAP guardados en {top_values_path}")
            
            # NUEVO: Analizar interdependencias entre top 7 features
            logger.info("🔗 Analizando interdependencias entre features...")
            interactions_analysis = self.analyze_feature_interactions(self._current_X, top_features=7)
            interactions_path = shap_dir / "feature_interactions_top7.json"
            with open(interactions_path, 'w') as f:
                safe_json_dump(interactions_analysis, f, indent=2)
            logger.info(f"💾 Análisis de interdependencias guardado en {interactions_path}")
        
        # Guardar metadatos
        metadata = {
            'expected_value': float(self.expected_value),
            'n_samples': int(self.shap_values.shape[0]),
            'n_features': int(self.shap_values.shape[1]),
            'feature_names': self.feature_names,
            'top_values_extracted': True,
            'interactions_analyzed': True,
            'timestamp': datetime.now().isoformat(),
            'version': '1.1.0'
        }
        
        metadata_path = shap_dir / "shap_metadata.json"
        with open(metadata_path, 'w') as f:
            safe_json_dump(metadata, f, indent=2)
        
        logger.info(f"💾 SHAP analysis completo guardado en {shap_dir}")

def run_complete_shap_analysis(model, X: np.ndarray, feature_names: List[str],
                              artifacts_dir: Path, background_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Ejecutar análisis completo de SHAP con top valores e interdependencias
    """
    logger.info("🔍 Iniciando análisis completo de SHAP...")
    
    # Usar subset para background si no se proporciona
    if background_data is None:
        background_data = X[:100] if len(X) > 100 else X
    
    # Crear pipeline
    shap_pipeline = SHAPAnalysisPipeline()
    
    # 1. Crear explainer
    shap_pipeline.create_explainer(model, background_data, feature_names)
    
    # 2. Calcular SHAP values
    shap_values = shap_pipeline.calculate_shap_values(X)
    
    # 3. Obtener feature importance
    feature_importance = shap_pipeline.get_global_feature_importance()
    
    # 4. Guardar resultados (incluyendo top valores e interdependencias)
    shap_pipeline.save_analysis_results(artifacts_dir, feature_importance)
    
    logger.info("Análisis completo de SHAP finalizado")
    
    return {
        'shap_pipeline': shap_pipeline,
        'feature_importance': feature_importance,
        'expected_value': shap_pipeline.expected_value
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🔍 Script de SHAP Analysis Pipeline - VERSIÓN MEJORADA")
    print("Incluye top 20 valores positivos/negativos e interdependencias de top 7 features")
    print("Usar run_pipeline.py para ejecutar el pipeline completo")