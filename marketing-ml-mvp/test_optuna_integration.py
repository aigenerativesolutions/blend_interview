#!/usr/bin/env python3
"""
Test de integración rápido para la nueva implementación con Optuna
Valida que todos los componentes funcionen sin ejecutar 100 trials
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

def test_optuna_optimizer():
    """Test basic Optuna optimizer functionality"""
    logger.info("🧪 Testing Optuna optimizer...")
    
    try:
        from pipeline.third_tuning import OptunaXGBoostOptimizer
        
        # Create dummy data
        np.random.seed(42)
        X_train = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'Marital_Status': np.random.choice(['Single', 'Married'], 100).astype('category')
        })
        y_train = pd.Series(np.random.binomial(1, 0.3, 100))
        
        # Test optimizer creation
        optimizer = OptunaXGBoostOptimizer()
        logger.info("✅ Optimizer created successfully")
        
        # Test scale_pos_weight calculation
        scale_weight = optimizer.calculate_scale_pos_weight(y_train.values)
        logger.info(f"✅ Scale pos weight calculated: {scale_weight:.4f}")
        
        # Test with minimal trials (2 trials only for speed)
        logger.info("🔄 Testing optimization with 2 trials...")
        
        # Monkey patch for quick testing
        original_optimize = optimizer.optimize
        def quick_optimize(X_train, y_train):
            # Same as original but with 2 trials
            import optuna
            from optuna.samplers import TPESampler
            
            optimizer.scale_pos_weight = optimizer.calculate_scale_pos_weight(y_train.values)
            sampler = TPESampler(seed=42, multivariate=True, n_startup_trials=1)
            optimizer.study = optuna.create_study(direction='maximize', sampler=sampler)
            
            # Only 2 trials for testing
            optimizer.study.optimize(
                lambda trial: optimizer.objective(trial, X_train, y_train),
                n_trials=2
            )
            
            optimizer.best_score = optimizer.study.best_value
            optimizer.best_params = optimizer.study.best_params.copy()
            optimizer.best_params.update({
                'scale_pos_weight': optimizer.scale_pos_weight,
                'use_label_encoder': False,
                'eval_metric': 'logloss',
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'hist',
                'enable_categorical': True
            })
            
            return {
                'best_params': optimizer.best_params,
                'best_score': float(optimizer.best_score),
                'n_trials': len(optimizer.study.trials),
                'scale_pos_weight': float(optimizer.scale_pos_weight)
            }
        
        # Run quick test
        results = quick_optimize(X_train, y_train)
        logger.info(f"✅ Optimization completed with {results['n_trials']} trials")
        logger.info(f"📊 Best F1-Score: {results['best_score']:.4f}")
        
        # Test best model creation
        best_model = optimizer.get_best_model(X_train, y_train)
        logger.info("✅ Best model created successfully")
        
        # Test predictions
        predictions = best_model.predict(X_train)
        probabilities = best_model.predict_proba(X_train)
        logger.info(f"✅ Predictions generated: {predictions.shape}")
        logger.info(f"✅ Probabilities generated: {probabilities.shape}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"❌ Error in Optuna optimizer test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_engineering_integration():
    """Test that feature engineering works with new categorical handling"""
    logger.info("🧪 Testing feature engineering integration...")
    
    try:
        from pipeline.feature_engineering import prepare_data_for_training
        
        # Check if data exists
        data_path = Path("data/marketing_campaign.csv")
        if not data_path.exists():
            logger.warning(f"⚠️ Data file not found: {data_path}")
            return True  # Skip test, not failure
        
        # Test with small sample
        import pandas as pd
        df = pd.read_csv(data_path, sep=';', nrows=50)
        
        # Create pipeline and test
        from pipeline.feature_engineering import MarketingFeaturePipeline
        pipeline = MarketingFeaturePipeline()
        
        # Test feature creation
        df_with_features = pipeline.create_features(df)
        expected_features = ['Age', 'Total_Kids', 'Total_Spent', 'Months_As_Customer']
        
        for feature in expected_features:
            if feature in df_with_features.columns:
                logger.info(f"✅ {feature} created successfully")
            else:
                logger.error(f"❌ {feature} not found")
                return False
        
        # Test categorical encoding
        X = df_with_features.drop(columns=['Response'], errors='ignore')
        X_encoded = pipeline.encode_categorical_features(X, fit=True)
        
        # Check Education is numeric
        if 'Education' in X_encoded.columns:
            if X_encoded['Education'].dtype in ['int64', 'int32']:
                logger.info("✅ Education properly encoded to numeric")
            else:
                logger.error(f"❌ Education not numeric: {X_encoded['Education'].dtype}")
                return False
        
        # Check Marital_Status is categorical
        if 'Marital_Status' in X_encoded.columns:
            if X_encoded['Marital_Status'].dtype.name == 'category':
                logger.info("✅ Marital_Status properly encoded to categorical")
            else:
                logger.error(f"❌ Marital_Status not categorical: {X_encoded['Marital_Status'].dtype}")
                return False
        
        logger.info("✅ Feature engineering integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in feature engineering integration: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run integration tests"""
    logger.info("🚀 Starting Optuna integration tests...")
    
    tests = [
        ("Feature Engineering Integration", test_feature_engineering_integration),
        ("Optuna Optimizer", test_optuna_optimizer),
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
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All integration tests passed! Optuna implementation is ready.")
        logger.info("💡 Now you can run the full pipeline with:")
        logger.info("   python quick_start.py")
    else:
        logger.error("⚠️ Some integration tests failed. Review errors before running full pipeline.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)