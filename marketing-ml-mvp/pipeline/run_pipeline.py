#!/usr/bin/env python3
"""
MLOps Pipeline Orchestrator
Ejecuta todo el pipeline de ML: Features -> 3er Tuning -> Training -> Calibration -> SHAP
"""
import sys
from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import argparse

# Agregar directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

# Importar módulos del pipeline
from pipeline.feature_engineering import MarketingFeaturePipeline, prepare_data_for_training
from pipeline.third_tuning import run_third_tuning
from pipeline.train_final_fixed import train_final_model_pipeline
from pipeline.temperature_calibration import calibrate_model_probabilities
from pipeline.shap_analysis_pipeline_fixed import run_complete_shap_analysis

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

class MLOpsPipelineOrchestrator:
    """
    Orquestador principal del pipeline MLOps
    Ejecuta todos los pasos en secuencia y maneja errores
    """
    
    def __init__(self, data_path: Path, artifacts_dir: Path):
        self.data_path = data_path
        self.artifacts_dir = artifacts_dir
        self.pipeline_results = {}
        self.start_time = None
        self.end_time = None
        
        # Crear directorio de artifacts
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    def run_complete_pipeline(self, test_size: float = 0.2, 
                             val_split: float = 0.2) -> Dict[str, Any]:
        """
        Ejecutar pipeline completo de MLOps
        """
        self.start_time = datetime.now()
        logger.info("🚀 INICIANDO PIPELINE MLOPS COMPLETO")
        logger.info("=" * 60)
        
        try:
            # Paso 1: Feature Engineering
            logger.info("📊 PASO 1: FEATURE ENGINEERING")
            data = self._run_feature_engineering(test_size)
            
            # Paso 2: Crear validation split
            logger.info("📊 PASO 2: CREANDO VALIDATION SPLIT")
            train_val_data = self._create_validation_split(data, val_split)
            
            # Paso 3: 3er Tuning (Metodología Ganadora)
            logger.info("🎯 PASO 3: 3ER TUNING - METODOLOGÍA GANADORA")
            tuning_results = self._run_third_tuning(train_val_data)
            
            # Paso 4: Entrenamiento Final
            logger.info("🏋️ PASO 4: ENTRENAMIENTO MODELO FINAL")
            final_model_results = self._run_final_training(data, tuning_results)
            
            # Paso 5: Calibración de Temperatura
            logger.info("🌡️ PASO 5: CALIBRACIÓN DE TEMPERATURA")
            calibration_results = self._run_temperature_calibration(
                final_model_results['model'], train_val_data
            )
            
            # Paso 6: Análisis SHAP
            logger.info("🔍 PASO 6: ANÁLISIS SHAP COMPLETO")
            shap_results = self._run_shap_analysis(
                final_model_results['model'], data
            )
            
            # Paso 7: Generar resumen final
            logger.info("📋 PASO 7: GENERANDO RESUMEN FINAL")
            final_summary = self._generate_final_summary(
                data, tuning_results, final_model_results, 
                calibration_results, shap_results
            )
            
            self.end_time = datetime.now()
            duration = self.end_time - self.start_time
            
            logger.info("=" * 60)
            logger.info(f"✅ PIPELINE COMPLETADO EN {duration}")
            logger.info(f"🏆 ROC-AUC Final: {final_model_results['test_metrics']['roc_auc']:.4f}")
            logger.info(f"🎯 Threshold Óptimo: {final_model_results['optimal_threshold']:.4f}")
            logger.info(f"🌡️ Temperatura: {calibration_results.temperature:.4f}")
            logger.info("=" * 60)
            
            return final_summary
            
        except Exception as e:
            logger.error(f"❌ ERROR EN PIPELINE: {str(e)}")
            raise
    
    def _run_feature_engineering(self, test_size: float) -> Dict[str, Any]:
        """
        Ejecutar feature engineering pipeline
        """
        logger.info(f"Procesando datos desde: {self.data_path}")
        
        # Preparar datos
        data = prepare_data_for_training(self.data_path, test_size=test_size)
        
        # Guardar pipeline
        pipeline_path = self.artifacts_dir / "feature_pipeline.pkl"
        data['pipeline'].save(pipeline_path)
        
        logger.info(f"✅ Features: {len(data['feature_names'])}")
        logger.info(f"✅ Train: {data['X_train'].shape}, Test: {data['X_test'].shape}")
        
        self.pipeline_results['feature_engineering'] = {
            'n_features': len(data['feature_names']),
            'train_shape': data['X_train'].shape,
            'test_shape': data['X_test'].shape,
            'feature_names': data['feature_names']
        }
        
        return data
    
    def _create_validation_split(self, data: Dict[str, Any], 
                               val_split: float) -> Dict[str, Any]:
        """
        Crear split de validación desde datos de entrenamiento
        """
        from sklearn.model_selection import train_test_split
        
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            data['X_train'], data['y_train'], 
            test_size=val_split, 
            random_state=42, 
            stratify=data['y_train']
        )
        
        logger.info(f"✅ Train: {X_train_split.shape}, Val: {X_val.shape}")
        
        return {
            'X_train': pd.DataFrame(X_train_split, columns=data['feature_names']),
            'X_val': pd.DataFrame(X_val, columns=data['feature_names']),
            'y_train': pd.Series(y_train_split),
            'y_val': pd.Series(y_val),
            'feature_names': data['feature_names']
        }
    
    def _run_third_tuning(self, train_val_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecutar 3er tuning (metodología ganadora)
        """
        tuning_results = run_third_tuning(
            train_val_data['X_train'], train_val_data['y_train'],
            train_val_data['X_val'], train_val_data['y_val'],
            self.artifacts_dir
        )
        
        logger.info(f"✅ Mejor ROC-AUC: {tuning_results['best_score']:.4f}")
        logger.info(f"✅ Threshold óptimo: {tuning_results['optimal_threshold']:.4f}")
        
        self.pipeline_results['third_tuning'] = tuning_results
        
        return tuning_results
    
    def _run_final_training(self, data: Dict[str, Any], tuning_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecutar entrenamiento del modelo final con parámetros de Optuna
        """
        final_results = train_final_model_pipeline(
            data['X_train'], data['y_train'],
            data['X_test'], data['y_test'],
            data['feature_names'],
            self.artifacts_dir,
            best_params=tuning_results['best_params'],
            optimal_threshold=tuning_results['optimal_threshold']
        )
        
        logger.info(f"✅ ROC-AUC Test: {final_results['test_metrics']['roc_auc']:.4f}")
        logger.info(f"✅ F1 Score (óptimo): {final_results['test_metrics']['f1_optimal']:.4f}")
        
        self.pipeline_results['final_training'] = final_results
        
        return final_results
    
    def _run_temperature_calibration(self, model, train_val_data: Dict[str, Any]):
        """
        Ejecutar calibración de temperatura
        """
        calibrator = calibrate_model_probabilities(
            model, 
            train_val_data['X_val'], 
            train_val_data['y_val'],
            self.artifacts_dir
        )
        
        logger.info(f"✅ Temperatura: {calibrator.temperature:.4f}")
        logger.info(f"✅ Mejora Brier: {calibrator.calibration_metrics['brier_improvement']:.4f}")
        
        self.pipeline_results['calibration'] = {
            'temperature': calibrator.temperature,
            'metrics': calibrator.calibration_metrics
        }
        
        return calibrator
    
    def _run_shap_analysis(self, model, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecutar análisis SHAP completo
        """
        # Usar subset para SHAP (más rápido)
        X_shap = data['X_test'][:500] if len(data['X_test']) > 500 else data['X_test']
        
        shap_results = run_complete_shap_analysis(
            model, X_shap, data['feature_names'], self.artifacts_dir
        )
        
        top_5_features = list(shap_results['feature_importance'].keys())[:5]
        logger.info(f"✅ Top 5 features: {top_5_features}")
        logger.info(f"✅ Expected value: {shap_results['expected_value']:.4f}")
        
        self.pipeline_results['shap_analysis'] = {
            'top_features': top_5_features,
            'expected_value': shap_results['expected_value'],
            'n_dependence_plots': len(shap_results['dependence_plots']),
            'n_waterfall_plots': len(shap_results['waterfall_plots'])
        }
        
        return shap_results
    
    def _generate_final_summary(self, data: Dict[str, Any], 
                              tuning_results: Dict[str, Any],
                              final_model_results: Dict[str, Any],
                              calibration_results, 
                              shap_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generar resumen final del pipeline
        """
        duration = self.end_time - self.start_time
        
        summary = {
            'pipeline_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'duration_formatted': str(duration),
                'version': '1.0.0',
                'methodology': 'third_tuning_winner'
            },
            'data_info': {
                'original_shape': data['original_shape'],
                'train_shape': data['X_train'].shape,
                'test_shape': data['X_test'].shape,
                'n_features': len(data['feature_names']),
                'feature_names': data['feature_names']
            },
            'model_performance': {
                'best_cv_score': tuning_results['best_score'],
                'optimal_threshold': tuning_results['optimal_threshold'],
                'test_metrics': final_model_results['test_metrics'],
                'training_metrics': final_model_results['training_metrics']
            },
            'calibration': {
                'temperature': calibration_results.temperature,
                'brier_improvement': calibration_results.calibration_metrics['brier_improvement'],
                'calibration_metrics': calibration_results.calibration_metrics
            },
            'interpretability': {
                'expected_value': shap_results['expected_value'],
                'top_5_features': list(shap_results['feature_importance'].keys())[:5],
                'feature_importance': shap_results['feature_importance']
            },
            'artifacts': {
                'feature_pipeline': 'feature_pipeline.pkl',
                'final_model': 'final_model.pkl',
                'temperature_calibrator': 'temperature_calibrator.pkl',
                'shap_explainer': 'shap_values/shap_explainer.pkl',
                'plots_generated': True
            }
        }
        
        # Guardar resumen
        summary_path = self.artifacts_dir / "pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"📋 Resumen guardado en: {summary_path}")
        
        return summary

def main():
    """
    Función principal para ejecutar desde línea de comandos
    """
    parser = argparse.ArgumentParser(description='MLOps Pipeline Orchestrator')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Ruta al archivo CSV de datos')
    parser.add_argument('--artifacts-dir', type=str, default='artifacts',
                       help='Directorio para guardar artifacts')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proporción para test set')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Proporción para validation split')
    
    args = parser.parse_args()
    
    # Convertir a Path
    data_path = Path(args.data_path)
    artifacts_dir = Path(args.artifacts_dir)
    
    # Validar archivo de datos
    if not data_path.exists():
        logger.error(f"❌ Archivo de datos no encontrado: {data_path}")
        sys.exit(1)
    
    # Crear y ejecutar orquestador
    orchestrator = MLOpsPipelineOrchestrator(data_path, artifacts_dir)
    
    try:
        summary = orchestrator.run_complete_pipeline(
            test_size=args.test_size,
            val_split=args.val_split
        )
        
        print("\n" + "=" * 60)
        print("🎉 PIPELINE MLOPS COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        print(f"🏆 ROC-AUC: {summary['model_performance']['test_metrics']['roc_auc']:.4f}")
        print(f"🎯 F1 Score: {summary['model_performance']['test_metrics']['f1_optimal']:.4f}")
        print(f"🌡️ Temperatura: {summary['calibration']['temperature']:.4f}")
        print(f"📁 Artifacts: {artifacts_dir}")
        print("=" * 60)
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Pipeline falló: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()