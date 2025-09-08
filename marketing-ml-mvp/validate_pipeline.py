#!/usr/bin/env python3
"""
Script de validación simple para el MLOps pipeline
Prueba los componentes críticos paso a paso
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def validate_data_loading():
    """Validate data loading with original dataset"""
    logger.info("🧪 Validating data loading...")
    
    data_path = Path("data/marketing_campaign.csv")
    
    if not data_path.exists():
        logger.error(f"❌ Data file not found: {data_path}")
        return False
    
    try:
        # Load with semicolon separator (original dataset format)
        df = pd.read_csv(data_path, sep=';')
        logger.info(f"✅ Data loaded: {df.shape}")
        logger.info(f"📊 Columns: {list(df.columns)}")
        
        # Check target column
        if 'Response' not in df.columns:
            logger.error("❌ Target column 'Response' not found")
            return False
        
        logger.info(f"🎯 Target distribution: {df['Response'].value_counts().to_dict()}")
        
        # Check for expected columns from notebook
        expected_cols = ['Year_Birth', 'Education', 'Marital_Status', 'Income', 
                        'MntWines', 'MntFruits', 'MntMeatProducts', 'Dt_Customer']
        
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"⚠️ Missing expected columns: {missing_cols}")
        else:
            logger.info("✅ All expected columns present")
        
        # Clean data (remove columns that should be dropped)
        columns_to_drop = ['ID', 'Z_CostContact', 'Z_Revenue']
        df_clean = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        logger.info(f"🧹 Data after cleaning: {df_clean.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {str(e)}")
        return False

def validate_feature_engineering():
    """Validate feature engineering pipeline"""
    logger.info("🧪 Validating feature engineering...")
    
    try:
        from pipeline.feature_engineering import MarketingFeaturePipeline, prepare_data_for_training
        
        # Test with small data sample
        data_path = Path("data/marketing_campaign.csv")
        if not data_path.exists():
            logger.error(f"❌ Data file not found: {data_path}")
            return False
        
        # Load small sample
        df = pd.read_csv(data_path, sep=';', nrows=100)  # Only 100 rows for testing
        logger.info(f"📊 Test data loaded: {df.shape}")
        
        # Test pipeline creation
        pipeline = MarketingFeaturePipeline()
        logger.info("✅ Feature pipeline created")
        
        # Test feature creation
        df_with_features = pipeline.create_features(df)
        expected_new_features = ['Age', 'Total_Spent', 'Customer_Days']
        
        for feature in expected_new_features:
            if feature in df_with_features.columns:
                logger.info(f"✅ {feature} created successfully")
            else:
                logger.warning(f"⚠️ {feature} not found in features")
        
        # Test full pipeline preparation (small sample)
        try:
            data = prepare_data_for_training(data_path, test_size=0.2, random_state=42)
            logger.info(f"✅ Pipeline preparation successful")
            logger.info(f"📊 Train: {data['X_train'].shape}, Test: {data['X_test'].shape}")
            logger.info(f"🔧 Features: {len(data['feature_names'])}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Error in pipeline preparation: {str(e)}")
            return False
        
    except ImportError as e:
        logger.error(f"❌ Import error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"❌ Error in feature engineering: {str(e)}")
        return False

def validate_third_tuning_import():
    """Validate third tuning module can be imported"""
    logger.info("🧪 Validating third tuning import...")
    
    try:
        from pipeline.third_tuning import run_third_tuning
        logger.info("✅ Third tuning module imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Third tuning import error: {str(e)}")
        return False

def validate_temperature_calibration_import():
    """Validate temperature calibration module"""
    logger.info("🧪 Validating temperature calibration...")
    
    try:
        from pipeline.temperature_calibration import TemperatureScalingCalibrator
        
        # Test calibrator creation
        calibrator = TemperatureScalingCalibrator()
        logger.info("✅ Temperature calibrator created successfully")
        
        # Test with dummy data
        np.random.seed(42)
        dummy_probs = np.random.uniform(0.1, 0.9, 100)
        dummy_labels = np.random.binomial(1, dummy_probs, 100)
        
        # Test fitting
        calibrator.fit(dummy_probs, dummy_labels)
        logger.info(f"✅ Calibrator fitted. Temperature: {calibrator.temperature:.4f}")
        
        # Test prediction
        calibrated_probs = calibrator.predict_proba(dummy_probs)
        logger.info(f"✅ Calibration predictions generated: {calibrated_probs.shape}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Temperature calibration import error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"❌ Temperature calibration error: {str(e)}")
        return False

def validate_pipeline_orchestrator():
    """Validate main pipeline orchestrator"""
    logger.info("🧪 Validating pipeline orchestrator...")
    
    try:
        from pipeline.run_pipeline import MLOpsPipelineOrchestrator
        
        # Test orchestrator creation
        data_path = Path("data/marketing_campaign.csv")
        artifacts_dir = Path("test_artifacts")
        
        orchestrator = MLOpsPipelineOrchestrator(data_path, artifacts_dir)
        logger.info("✅ Pipeline orchestrator created successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Pipeline orchestrator import error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"❌ Pipeline orchestrator error: {str(e)}")
        return False

def main():
    """Run validation tests"""
    logger.info("🚀 Starting pipeline validation tests...")
    
    tests = [
        ("Data Loading", validate_data_loading),
        ("Feature Engineering", validate_feature_engineering),
        ("Third Tuning Import", validate_third_tuning_import),
        ("Temperature Calibration", validate_temperature_calibration_import),
        ("Pipeline Orchestrator", validate_pipeline_orchestrator),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results[test_name] = result
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {str(e)}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All validation tests passed! Pipeline components are working.")
        logger.info("💡 You can now run the full pipeline with:")
        logger.info("   python pipeline/run_pipeline.py --data-path data/marketing_campaign.csv")
    else:
        logger.error("⚠️ Some validation tests failed. Review errors before running full pipeline.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)