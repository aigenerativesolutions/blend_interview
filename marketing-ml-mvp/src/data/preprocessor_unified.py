"""
Data preprocessing module for marketing campaign data
Unified implementation following exact notebook methodology
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

def prepare_data_for_training(data_path: Path, test_size: float = 0.2, random_state: int = 42, sample_fraction: float = 1.0) -> Dict[str, Any]:
    """
    Prepare data with exact notebook methodology
    Unified implementation that replaces both preprocessors
    """
    logger.info(f"Loading data from {data_path}")
    
    # Load data with semicolon separator
    df = pd.read_csv(data_path, sep=';')
    logger.info(f"Data loaded: {df.shape}")
    
    # Sample data for faster testing
    if sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=random_state)
        logger.info(f"Sampled to {sample_fraction*100}% of data: {df.shape}")
    
    # EXACT NOTEBOOK IMPLEMENTATION
    
    # Convert Dt_Customer to datetime
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')
    reference_date = pd.Timestamp('2014-09-01')
    
    months = (
        (reference_date.year - df['Dt_Customer'].dt.year) * 12 +
        (reference_date.month - df['Dt_Customer'].dt.month)
    )
    
    # Adjust if day doesn't match with reference_date
    adjust = (reference_date.day < df['Dt_Customer'].dt.day).astype(int)
    df['Months_As_Customer'] = (months - adjust).astype(int)
    
    # Calculate Age using fixed year 2014
    df['Age'] = 2014 - df['Year_Birth']
    
    # Combine Kidhome and Teenhome to reduce columns
    df['Total_Kids'] = df['Kidhome'] + df['Teenhome']
    
    # Calculate Total_Spent to see if total amount tells something 
    spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts',
                     'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    df['Total_Spent'] = df[spending_cols].sum(axis=1)
    
    # Drop irrelevant columns, age already calculated
    df.drop(columns=['ID', 'Year_Birth', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer'], inplace=True)
    
    # Map Education to ordered cardinality for XGBoost categorical efficiency
    education_order = {
        'Basic': 0,
        '2n Cycle': 1,
        'Graduation': 2,
        'Master': 3,
        'PhD': 4
    }
    df['Education'] = df['Education'].map(education_order).astype('int')
    
    # Convert Marital_Status to categorical for native XGBoost handling
    df['Marital_Status'] = df['Marital_Status'].astype('category')
    
    logger.info(f"Feature engineering completed: {df.shape}")
    
    # Define X and y
    X = df.drop(columns=['Response'])
    y = df['Response']
    
    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Feature names
    feature_names = X.columns.tolist()
    
    return {
        'X_train': X_train,
        'X_test': X_test, 
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'original_shape': df.shape
    }

class MarketingDataPreprocessor:
    """
    Unified data preprocessor following exact notebook methodology
    Maintains compatibility with existing code structure
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        
    def process_pipeline(self, target_col: str = 'Response') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete preprocessing pipeline using exact notebook methodology
        """
        logger.info("Starting preprocessing pipeline")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Use the unified function
        data = prepare_data_for_training(self.data_path)
        
        # Combine X_train and X_test for full dataset
        X_full = pd.concat([data['X_train'], data['X_test']], ignore_index=True)
        y_full = pd.concat([data['y_train'], data['y_test']], ignore_index=True)
        
        self.processed_data = X_full.copy()
        self.processed_data[target_col] = y_full
        
        logger.info("Preprocessing pipeline completed")
        
        return X_full, y_full

def load_and_preprocess_data(data_path: Optional[Path] = None, 
                           target_col: str = 'Response') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function using unified methodology
    """
    if data_path is None:
        raise ValueError("data_path is required")
    
    preprocessor = MarketingDataPreprocessor(data_path)
    return preprocessor.process_pipeline(target_col)