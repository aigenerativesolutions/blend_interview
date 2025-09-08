#!/usr/bin/env python3
"""
Script de debug para probar componentes del pipeline individualmente
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

def test_data_loading():
    """Test basic data loading"""
    logger.info("🧪 Testing data loading...")
    
    data_path = Path("data/marketing_campaign_data.csv")
    
    if not data_path.exists():
        logger.error(f" Data file not found: {data_path}")
        return False
    
    try:
        # Try loading with semicolon separator
        df = pd.read_csv(data_path, sep=';')
        logger.info(f" Data loaded successfully: {df.shape}")
        logger.info(f"📊 Columns: {list(df.columns)}")
        
        # Check target column
        if 'Response' not in df.columns:
            logger.error(" Target column 'Response' not found")
            return False
        
        logger.info(f"🎯 Target distribution: {df['Response'].value_counts().to_dict()}")
        
        # Clean data
        columns_to_drop = ['ID', 'Z_CostContact', 'Z_Revenue']
        df_clean = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        logger.info(f"🧹 Data after cleaning: {df_clean.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f" Error loading data: {str(e)}")
        return False

def test_basic_feature_engineering():
    """Test basic feature engineering"""
    logger.info("🧪 Testing feature engineering...")
    
    try:
        from pipeline.feature_engineering import MarketingFeaturePipeline
        
        # Load data
        data_path = Path("data/marketing_campaign_data.csv")
        df = pd.read_csv(data_path, sep=';')
        
        # Clean data
        columns_to_drop = ['ID', 'Z_CostContact', 'Z_Revenue']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        # Create pipeline
        pipeline = MarketingFeaturePipeline()
        
        # Test feature creation
        df_with_features = pipeline.create_features(df)
        logger.info(f" Features created: {df_with_features.columns.tolist()}")
        
        # Check for new features
        expected_features = ['Age', 'Total_Spent', 'Customer_Days']
        for feature in expected_features:
            if feature in df_with_features.columns:
                logger.info(f" {feature}: {df_with_features[feature].describe()}")
            else:
                logger.warning(f" {feature} not found")
        
        return True
        
    except Exception as e:
        logger.error(f" Error in feature engineering: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_dependencies():
    """Test required dependencies"""
    logger.info("🧪 Testing dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'xgboost', 'shap', 
        'matplotlib', 'seaborn', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f" {package}: OK")
        except ImportError:
            logger.error(f" {package}: MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f" Missing packages: {missing_packages}")
        logger.info("💡 Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def main():
    """Run all debug tests"""
    logger.info("🚀 Starting debug tests...")
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Data Loading", test_data_loading),
        ("Feature Engineering", test_basic_feature_engineering),
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
                logger.info(f" {test_name}: PASSED")
            else:
                logger.error(f" {test_name}: FAILED")
        except Exception as e:
            logger.error(f" {test_name}: ERROR - {str(e)}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    for test_name, result in results.items():
        status = " PASS" if result else " FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        logger.info("🎉 All tests passed! Pipeline should work.")
    else:
        logger.error(" Some tests failed. Fix issues before running pipeline.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)