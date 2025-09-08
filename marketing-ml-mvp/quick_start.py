#!/usr/bin/env python3
"""
Quick Start Script - MLOps Pipeline
Ejecuta el pipeline completo con configuración óptima
"""
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Quick start para ejecutar el pipeline completo"""
    
    print("🚀 MLOps Pipeline - Quick Start")
    print("=" * 50)
    
    # Verificar que los archivos existan
    project_root = Path(__file__).parent
    data_path = project_root / "data" / "marketing_campaign.csv"
    
    if not data_path.exists():
        logger.error(f" Dataset not found: {data_path}")
        logger.info("💡 Make sure the dataset is in the correct location:")
        logger.info("   data/marketing_campaign.csv")
        return False
    
    # Add project to Python path
    sys.path.append(str(project_root))
    
    try:
        # Import the pipeline
        from pipeline.run_pipeline import MLOpsPipelineOrchestrator
        
        # Setup paths
        artifacts_dir = project_root / "artifacts"
        
        logger.info(f"📊 Dataset: {data_path}")
        logger.info(f"📁 Artifacts: {artifacts_dir}")
        
        # Create orchestrator
        orchestrator = MLOpsPipelineOrchestrator(data_path, artifacts_dir)
        
        # Run complete pipeline
        logger.info("🎯 Starting complete MLOps pipeline...")
        
        summary = orchestrator.run_complete_pipeline(
            test_size=0.2,      # 20% for test set
            val_split=0.2       # 20% of training for validation
        )
        
        # Display results
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        
        # Performance metrics
        test_metrics = summary['model_performance']['test_metrics']
        print(f"🏆 ROC-AUC: {test_metrics['roc_auc']:.4f}")
        print(f"🎯 F1 Score: {test_metrics['f1_optimal']:.4f}")
        print(f"📊 Precision: {test_metrics['precision_optimal']:.4f}")
        print(f"📊 Recall: {test_metrics['recall_optimal']:.4f}")
        
        # Calibration
        calibration = summary['calibration']
        print(f"🌡️ Temperatura: {calibration['temperature']:.4f}")
        print(f"📈 Mejora Brier Score: {calibration['brier_improvement']:.4f}")
        
        # Interpretability
        interp = summary['interpretability']
        print(f"🔍 Expected SHAP Value: {interp['expected_value']:.4f}")
        print(f"🔝 Top 5 Features:")
        for i, feature in enumerate(interp['top_5_features'], 1):
            importance = interp['feature_importance'][feature]
            print(f"   {i}. {feature}: {importance:.4f}")
        
        # Artifacts
        print(f"\n📁 Artifacts guardados en: {artifacts_dir}")
        artifacts = summary['artifacts']
        for artifact_name, filename in artifacts.items():
            if isinstance(filename, str):
                artifact_path = artifacts_dir / filename
                if artifact_path.exists():
                    print(f"    {artifact_name}: {filename}")
                else:
                    print(f"    {artifact_name}: {filename} (not found)")
        
        # Duration
        duration = summary['pipeline_info']['duration_formatted']
        print(f"\n⏱️ Duración total: {duration}")
        
        print("=" * 60)
        print("💡 Next steps:")
        print("   1. Review artifacts in the 'artifacts' folder")
        print("   2. Check plots in 'artifacts/plots'")
        print("   3. Review SHAP analysis in 'artifacts/shap_values'")
        print("   4. Use FastAPI to serve the model: python src/app.py")
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        logger.error(f" Import error: {str(e)}")
        logger.info("💡 Try running: pip install -r requirements.txt")
        return False
    except Exception as e:
        logger.error(f" Pipeline error: {str(e)}")
        logger.info("💡 Check the logs above for detailed error information")
        logger.info("💡 You can run validate_pipeline.py first to check components")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 For debugging, try:")
        print("   python validate_pipeline.py")
        print("   python debug_test.py")
    sys.exit(0 if success else 1)