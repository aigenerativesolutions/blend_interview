#!/usr/bin/env python3
"""
Script para validar el modelo entrenado en la nube usando el 20% de test data local
Este script descarga el modelo más reciente y lo evalúa con datos que NUNCA vio durante entrenamiento
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add marketing-ml-mvp to path for imports
sys.path.append(str(Path(__file__).parent / "marketing-ml-mvp"))

try:
    from model_sync import ModelSyncManager
    from src.data.preprocessor import MarketingDataPreprocessor
    HAS_MODEL_SYNC = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import model sync: {e}")
    HAS_MODEL_SYNC = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalModelValidator:
    """Validador de modelos usando el test set local (20%)"""
    
    def __init__(self):
        self.test_data_path = Path("data/test_data_local.csv")
        self.models_dir = Path("marketing-ml-mvp/models")
        self.sync_manager = ModelSyncManager() if HAS_MODEL_SYNC else None
        self.preprocessor = MarketingDataPreprocessor()
    
    def load_test_data(self):
        """Carga el test set local (20% que nunca vio el modelo)"""
        if not self.test_data_path.exists():
            logger.error(f"❌ Test data not found: {self.test_data_path}")
            logger.info("🔄 Please run split_data.py first to generate test_data_local.csv")
            return None, None
        
        logger.info(f"📊 Loading test data from: {self.test_data_path}")
        
        # Use the same preprocessor as training
        X_test, y_test, feature_names = self.preprocessor.load_and_preprocess(self.test_data_path)
        
        logger.info(f"✅ Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        return X_test, y_test, feature_names
    
    def download_latest_model(self):
        """Descarga el modelo más reciente del bucket GCP"""
        if not self.sync_manager:
            logger.error("❌ Model sync not available. Please check your GCP credentials.")
            return None
        
        try:
            logger.info("📥 Downloading latest model from GCP...")
            model_path = self.sync_manager.download_latest_model()
            
            if model_path and Path(model_path).exists():
                logger.info(f"✅ Model downloaded: {model_path}")
                return model_path
            else:
                logger.error("❌ No model found in bucket or download failed")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error downloading model: {e}")
            return None
    
    def load_local_model(self):
        """Carga un modelo local si está disponible"""
        local_model_path = self.models_dir / "xgboost_model.pkl"
        
        if local_model_path.exists():
            logger.info(f"📂 Loading local model: {local_model_path}")
            try:
                model = joblib.load(local_model_path)
                return str(local_model_path), model
            except Exception as e:
                logger.error(f"❌ Error loading local model: {e}")
                return None, None
        else:
            logger.warning(f"⚠️ No local model found at: {local_model_path}")
            return None, None
    
    def evaluate_model(self, model, X_test, y_test, model_path):
        """Evalúa el modelo con métricas completas"""
        logger.info("🧪 Evaluating model performance...")
        
        # Predicciones
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Métricas básicas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # AUC si tenemos probabilidades
        auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        
        # Resultados
        results = {
            'model_path': model_path,
            'test_samples': len(y_test),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred)
        }
        
        return results
    
    def print_results(self, results):
        """Imprime los resultados de la validación"""
        print("\n" + "="*70)
        print("📊 RESULTADOS DE VALIDACIÓN DEL MODELO")
        print("="*70)
        
        print(f"🏷️  Modelo evaluado: {Path(results['model_path']).name}")
        print(f"🧪 Muestras de test: {results['test_samples']}")
        print(f"📈 Precisión (Accuracy): {results['accuracy']:.3f}")
        print(f"🎯 Precision: {results['precision']:.3f}")
        print(f"🔍 Recall: {results['recall']:.3f}")
        print(f"⚖️  F1-Score: {results['f1_score']:.3f}")
        
        if results['auc_score']:
            print(f"📊 AUC-ROC: {results['auc_score']:.3f}")
        
        print(f"\n📊 Matriz de Confusión:")
        print(f"   True Negative:  {results['confusion_matrix'][0,0]}")
        print(f"   False Positive: {results['confusion_matrix'][0,1]}")
        print(f"   False Negative: {results['confusion_matrix'][1,0]}")
        print(f"   True Positive:  {results['confusion_matrix'][1,1]}")
        
        print(f"\n📋 Reporte Detallado:")
        print(results['classification_report'])
        
        # Interpretación
        print("\n💡 INTERPRETACIÓN:")
        if results['accuracy'] > 0.8:
            print("✅ Excelente performance del modelo")
        elif results['accuracy'] > 0.7:
            print("👍 Buena performance del modelo")
        elif results['accuracy'] > 0.6:
            print("⚠️ Performance moderada del modelo")
        else:
            print("❌ Performance baja del modelo - necesita mejoras")
        
        if results['f1_score'] < 0.5:
            print("⚠️ F1-Score bajo - considera ajustar el threshold o reentrenar")
    
    def save_results(self, results):
        """Guarda los resultados en un archivo"""
        results_dir = Path("validation_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"validation_results_{timestamp}.json"
        
        # Convertir numpy arrays a listas para JSON
        json_results = results.copy()
        json_results['confusion_matrix'] = results['confusion_matrix'].tolist()
        
        import json
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to: {results_file}")

def main():
    """Función principal de validación"""
    print("🧪 VALIDACIÓN LOCAL DEL MODELO MLOps")
    print("="*50)
    
    validator = LocalModelValidator()
    
    # 1. Cargar datos de test
    X_test, y_test, feature_names = validator.load_test_data()
    if X_test is None:
        return
    
    # 2. Intentar descargar modelo más reciente de GCP
    model_path = None
    model = None
    
    if HAS_MODEL_SYNC:
        logger.info("🔄 Attempting to download latest model from GCP...")
        downloaded_path = validator.download_latest_model()
        if downloaded_path:
            model_path = downloaded_path
            model = joblib.load(model_path)
    
    # 3. Si no hay modelo de GCP, usar modelo local
    if model is None:
        logger.info("🔄 Trying local model...")
        local_path, local_model = validator.load_local_model()
        if local_model is not None:
            model_path = local_path
            model = local_model
    
    # 4. Si no hay modelo disponible
    if model is None:
        logger.error("❌ No model available for validation")
        logger.info("Please either:")
        logger.info("  1. Run the training pipeline to generate a model in GCP")
        logger.info("  2. Train a model locally first")
        return
    
    # 5. Evaluar modelo
    results = validator.evaluate_model(model, X_test, y_test, model_path)
    
    # 6. Mostrar resultados
    validator.print_results(results)
    
    # 7. Guardar resultados
    validator.save_results(results)
    
    print("\n✅ Validación completada!")
    print("🚀 Ready to use the validated model in production!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Validation failed: {str(e)}")
        sys.exit(1)