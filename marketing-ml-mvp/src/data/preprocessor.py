"""
Data preprocessing module for marketing campaign data
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

from ..config.settings import (
    DATA_PATH, MARKETING_DATA_FILE, CURRENT_YEAR, 
    SPENDING_COLUMNS, COLUMNS_TO_DROP
)

logger = logging.getLogger(__name__)


class MarketingDataPreprocessor:
    """
    Data preprocessor for marketing campaign data.
    Handles loading, feature engineering, and data preparation.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize preprocessor.
        
        Args:
            data_path: Path to data directory. If None, uses default from settings.
        """
        self.data_path = data_path or DATA_PATH
        self.data_file = self.data_path / MARKETING_DATA_FILE
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load raw marketing campaign data.
        
        Returns:
            DataFrame with raw data
        """
        logger.info(f"Loading data from {self.data_file}")
        
        try:
            # Based on notebook: df = pd.read_csv("marketing_campaign.csv", sep=";")
            self.raw_data = pd.read_csv(self.data_file, sep=";")
            logger.info(f"Loaded {len(self.raw_data)} records with {len(self.raw_data.columns)} columns")
            
            return self.raw_data.copy()
            
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_file}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from raw data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            DataFrame with new features
        """
        logger.info("Creating new features")
        
        df_processed = df.copy()
        
        # Calculate Age (from notebook: df['Age'] = 2014 - df['Year_Birth'])
        df_processed['Age'] = CURRENT_YEAR - df_processed['Year_Birth']
        logger.info("Created Age feature")
        
        # Calculate Total_Spent (from notebook: df['Total_Spent'] = df[spending_cols].sum(axis=1))
        df_processed['Total_Spent'] = df_processed[SPENDING_COLUMNS].sum(axis=1)
        logger.info("Created Total_Spent feature")
        
        # Calculate Customer_Days if Dt_Customer exists
        if 'Dt_Customer' in df_processed.columns:
            try:
                df_processed['Dt_Customer'] = pd.to_datetime(df_processed['Dt_Customer'])
                reference_date = pd.to_datetime('2014-12-31')  # Assuming end of 2014
                df_processed['Customer_Days'] = (reference_date - df_processed['Dt_Customer']).dt.days
                logger.info("Created Customer_Days feature")
            except Exception as e:
                logger.warning(f"Could not create Customer_Days feature: {str(e)}")
        
        return df_processed
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by removing unnecessary columns and handling missing values.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data")
        
        df_clean = df.copy()
        
        # Drop columns that are not needed for modeling
        # Based on notebook: df.drop(columns=['ID', 'Year_Birth', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer'], inplace=True)
        columns_to_drop = [col for col in COLUMNS_TO_DROP if col in df_clean.columns]
        if columns_to_drop:
            df_clean = df_clean.drop(columns=columns_to_drop)
            logger.info(f"Dropped columns: {columns_to_drop}")
        
        # Handle missing values
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna()
        final_rows = len(df_clean)
        
        if initial_rows != final_rows:
            logger.info(f"Dropped {initial_rows - final_rows} rows with missing values")
        
        return df_clean
    
    def prepare_for_training(self, df: pd.DataFrame, target_col: str = 'Response') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for model training.
        
        Args:
            df: Processed DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of (features_df, target_series)
        """
        logger.info(f"Preparing data for training with target: {target_col}")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        # Separate features and target
        # Based on notebook: X = df.drop(columns=['Response']) and y = df['Response']
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def get_categorical_columns(self, df: pd.DataFrame) -> list:
        """
        Get list of categorical columns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of categorical column names
        """
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        logger.info(f"Categorical columns: {categorical_cols}")
        return categorical_cols
    
    def get_numerical_columns(self, df: pd.DataFrame) -> list:
        """
        Get list of numerical columns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of numerical column names
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"Numerical columns: {numerical_cols}")
        return numerical_cols
    
    def process_pipeline(self, target_col: str = 'Response') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete preprocessing pipeline.
        
        Args:
            target_col: Name of target column
            
        Returns:
            Tuple of (features_df, target_series)
        """
        logger.info("Starting complete preprocessing pipeline")
        
        # Load data
        raw_df = self.load_data()
        
        # Create features
        featured_df = self.create_features(raw_df)
        
        # Clean data
        clean_df = self.clean_data(featured_df)
        
        # Prepare for training
        X, y = self.prepare_for_training(clean_df, target_col)
        
        self.processed_data = clean_df
        
        logger.info("Preprocessing pipeline completed successfully")
        
        return X, y


def load_and_preprocess_data(data_path: Optional[Path] = None, 
                           target_col: str = 'Response') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function to load and preprocess marketing data.
    
    Args:
        data_path: Path to data directory
        target_col: Name of target column
        
    Returns:
        Tuple of (features_df, target_series)
    """
    preprocessor = MarketingDataPreprocessor(data_path)
    return preprocessor.process_pipeline(target_col)