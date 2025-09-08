"""
SHAP (SHapley Additive exPlanations) analysis module for model explainability
"""
import pandas as pd
import numpy as np
import shap
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import matplotlib.pyplot as plt
import json

from ..models.predict import MarketingPredictor
from ..config.settings import MODELS_PATH

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """
    SHAP analyzer for marketing campaign response model.
    """
    
    def __init__(self, predictor: Optional[MarketingPredictor] = None, 
                 models_path: Optional[Path] = None):
        """
        Initialize SHAP analyzer.
        
        Args:
            predictor: MarketingPredictor instance. If None, will create one.
            models_path: Path to models directory
        """
        self.models_path = models_path or MODELS_PATH
        
        if predictor is None:
            from ..models.predict import load_marketing_predictor
            self.predictor = load_marketing_predictor(self.models_path)
        else:
            self.predictor = predictor
        
        self.explainer = None
        self.shap_values_cache = None
        self.background_data = None
        
    def create_explainer(self, background_data: pd.DataFrame, 
                        explainer_type: str = 'tree') -> None:
        """
        Create SHAP explainer.
        
        Args:
            background_data: Background dataset for SHAP explainer
            explainer_type: Type of explainer ('tree', 'kernel', 'linear')
        """
        logger.info(f"Creating SHAP explainer of type: {explainer_type}")
        
        # Validate features
        validated_data = self.predictor._validate_features(background_data)
        self.background_data = validated_data
        
        if explainer_type == 'tree':
            # TreeExplainer for tree-based models (XGBoost)
            self.explainer = shap.TreeExplainer(self.predictor.model)
            
        elif explainer_type == 'kernel':
            # KernelExplainer - model agnostic but slower
            def model_predict(X):
                return self.predictor.model.predict_proba(X)[:, 1]
            
            # Use a sample of background data for efficiency
            background_sample = validated_data.sample(
                min(100, len(validated_data)), 
                random_state=42
            )
            self.explainer = shap.KernelExplainer(model_predict, background_sample)
            
        elif explainer_type == 'linear':
            # LinearExplainer for linear models
            self.explainer = shap.LinearExplainer(
                self.predictor.model, 
                validated_data
            )
            
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
        
        logger.info(f"SHAP explainer created successfully")
    
    def calculate_shap_values(self, X: pd.DataFrame, 
                            cache_results: bool = True) -> np.ndarray:
        """
        Calculate SHAP values for given data.
        
        Args:
            X: Input features
            cache_results: Whether to cache the results
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer not created. Call create_explainer() first.")
        
        logger.info(f"Calculating SHAP values for {len(X)} samples")
        
        # Validate features
        X_validated = self.predictor._validate_features(X)
        
        # Calculate SHAP values
        if isinstance(self.explainer, shap.TreeExplainer):
            shap_values = self.explainer.shap_values(X_validated)
            
            # TreeExplainer returns different formats depending on model
            if isinstance(shap_values, list):
                # Binary classification - take positive class
                shap_values = shap_values[1]
                
        else:
            shap_values = self.explainer.shap_values(X_validated)
        
        if cache_results:
            self.shap_values_cache = shap_values
        
        logger.info(f"SHAP values calculated. Shape: {shap_values.shape}")
        
        return shap_values
    
    def get_feature_importance(self, X: pd.DataFrame, 
                             shap_values: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Get feature importance based on mean absolute SHAP values.
        
        Args:
            X: Input features
            shap_values: Pre-calculated SHAP values. If None, will calculate.
            
        Returns:
            Dictionary of feature importances
        """
        if shap_values is None:
            shap_values = self.calculate_shap_values(X)
        
        # Calculate mean absolute SHAP values
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Get feature names
        feature_names = list(X.columns)
        
        # Create dictionary and sort by importance
        importance_dict = dict(zip(feature_names, feature_importance))
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        return sorted_importance
    
    def explain_single_prediction(self, customer_data: Dict[str, Any],
                                shap_values: Optional[np.ndarray] = None,
                                customer_index: int = 0) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP values.
        
        Args:
            customer_data: Customer features as dictionary
            shap_values: Pre-calculated SHAP values
            customer_index: Index of customer in SHAP values array
            
        Returns:
            Explanation dictionary
        """
        # Convert to DataFrame
        X = pd.DataFrame([customer_data])
        
        # Get prediction
        prediction_result = self.predictor.predict_single(customer_data)
        
        # Calculate SHAP values if not provided
        if shap_values is None:
            shap_values = self.calculate_shap_values(X)
        
        # Get SHAP values for this customer
        customer_shap = shap_values[customer_index]
        feature_names = list(X.columns)
        
        # Create feature contributions
        contributions = []
        for feature, shap_val, actual_val in zip(feature_names, customer_shap, X.iloc[0]):
            contributions.append({
                'feature': feature,
                'value': actual_val,
                'shap_value': float(shap_val),
                'contribution': 'positive' if shap_val > 0 else 'negative',
                'abs_contribution': abs(float(shap_val))
            })
        
        # Sort by absolute contribution
        contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
        
        # Calculate base value (expected value)
        if hasattr(self.explainer, 'expected_value'):
            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        else:
            base_value = 0.5  # Default for binary classification
        
        explanation = {
            'prediction': prediction_result,
            'base_value': float(base_value),
            'feature_contributions': contributions,
            'top_positive_features': [
                c for c in contributions if c['contribution'] == 'positive'
            ][:5],
            'top_negative_features': [
                c for c in contributions if c['contribution'] == 'negative'  
            ][:5],
            'explanation_summary': self._generate_explanation_summary(
                prediction_result, contributions[:5]
            )
        }
        
        return explanation
    
    def _generate_explanation_summary(self, prediction_result: Dict[str, Any],
                                    top_contributions: List[Dict[str, Any]]) -> str:
        """
        Generate human-readable explanation summary.
        
        Args:
            prediction_result: Prediction result
            top_contributions: Top feature contributions
            
        Returns:
            Explanation summary string
        """
        will_respond = prediction_result['will_respond']
        probability = prediction_result['probability']
        confidence = prediction_result.get('confidence', 'medium')
        
        # Base summary
        response_text = "will respond" if will_respond else "will not respond"
        summary = f"The model predicts this customer {response_text} to the campaign "
        summary += f"with {probability:.1%} probability ({confidence} confidence). "
        
        # Top contributing factors
        if top_contributions:
            summary += "Key factors: "
            factors = []
            for contrib in top_contributions[:3]:
                factor_text = f"{contrib['feature']} ({contrib['contribution']} impact)"
                factors.append(factor_text)
            summary += ", ".join(factors) + "."
        
        return summary
    
    def create_summary_plot(self, X: pd.DataFrame, 
                          shap_values: Optional[np.ndarray] = None,
                          plot_type: str = 'bar',
                          max_features: int = 10,
                          save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create SHAP summary plot.
        
        Args:
            X: Input features
            shap_values: Pre-calculated SHAP values
            plot_type: Type of plot ('bar', 'dot', 'violin')
            max_features: Maximum number of features to show
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        if shap_values is None:
            shap_values = self.calculate_shap_values(X)
        
        # Validate features
        X_validated = self.predictor._validate_features(X)
        
        plt.figure(figsize=(10, 8))
        
        if plot_type == 'bar':
            shap.summary_plot(
                shap_values, X_validated, 
                plot_type='bar', 
                max_display=max_features,
                show=False
            )
        elif plot_type == 'dot':
            shap.summary_plot(
                shap_values, X_validated,
                max_display=max_features, 
                show=False
            )
        elif plot_type == 'violin':
            shap.summary_plot(
                shap_values, X_validated,
                plot_type='violin',
                max_display=max_features,
                show=False
            )
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
        fig = plt.gcf()
        plt.title(f'SHAP Feature Importance ({plot_type.title()} Plot)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP plot saved to {save_path}")
        
        return fig
    
    def create_waterfall_plot(self, customer_data: Dict[str, Any],
                            shap_values: Optional[np.ndarray] = None,
                            customer_index: int = 0,
                            max_features: int = 10,
                            save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create SHAP waterfall plot for a single prediction.
        
        Args:
            customer_data: Customer features
            shap_values: Pre-calculated SHAP values
            customer_index: Index in SHAP values array
            max_features: Maximum features to show
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        # Convert to DataFrame
        X = pd.DataFrame([customer_data])
        
        if shap_values is None:
            shap_values = self.calculate_shap_values(X)
        
        # Validate features
        X_validated = self.predictor._validate_features(X)
        
        # Create waterfall plot
        plt.figure(figsize=(10, 8))
        
        if hasattr(shap, 'plots'):
            # SHAP v0.40+ API
            try:
                shap.plots.waterfall(
                    shap.Explanation(
                        values=shap_values[customer_index],
                        base_values=getattr(self.explainer, 'expected_value', 0),
                        data=X_validated.iloc[customer_index],
                        feature_names=list(X_validated.columns)
                    ),
                    max_display=max_features,
                    show=False
                )
            except Exception as e:
                logger.warning(f"New SHAP API failed, using legacy: {e}")
                # Fallback to creating custom waterfall
                self._create_custom_waterfall(
                    shap_values[customer_index], 
                    X_validated.iloc[customer_index],
                    max_features
                )
        else:
            # Create custom waterfall for older SHAP versions
            self._create_custom_waterfall(
                shap_values[customer_index],
                X_validated.iloc[customer_index], 
                max_features
            )
        
        fig = plt.gcf()
        plt.title('SHAP Waterfall Plot - Individual Prediction Explanation')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP waterfall plot saved to {save_path}")
        
        return fig
    
    def _create_custom_waterfall(self, shap_values: np.ndarray, 
                               customer_data: pd.Series,
                               max_features: int = 10) -> None:
        """
        Create custom waterfall plot.
        
        Args:
            shap_values: SHAP values for single instance
            customer_data: Customer feature values
            max_features: Maximum features to display
        """
        # Get feature names and SHAP values
        feature_names = list(customer_data.index)
        values = shap_values
        
        # Sort by absolute SHAP value
        sorted_indices = np.argsort(np.abs(values))[::-1][:max_features]
        
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_values = values[sorted_indices]
        
        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Base value
        base_value = getattr(self.explainer, 'expected_value', 0.5)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        
        # Calculate cumulative values
        cumsum = base_value
        y_pos = range(len(sorted_features))
        
        colors = ['green' if val > 0 else 'red' for val in sorted_values]
        
        bars = ax.barh(y_pos, sorted_values, color=colors, alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('SHAP Value')
        ax.set_title('Feature Contributions to Prediction')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, sorted_values):
            width = bar.get_width()
            ax.text(
                width + 0.01 if width >= 0 else width - 0.01,
                bar.get_y() + bar.get_height()/2,
                f'{value:.3f}',
                ha='left' if width >= 0 else 'right',
                va='center'
            )
    
    def batch_explain(self, customers_data: List[Dict[str, Any]],
                     include_plots: bool = False) -> List[Dict[str, Any]]:
        """
        Explain predictions for a batch of customers.
        
        Args:
            customers_data: List of customer feature dictionaries
            include_plots: Whether to include plot data (slower)
            
        Returns:
            List of explanation dictionaries
        """
        if not customers_data:
            return []
        
        logger.info(f"Explaining predictions for {len(customers_data)} customers")
        
        # Convert to DataFrame
        X = pd.DataFrame(customers_data)
        
        # Calculate SHAP values once for all customers
        shap_values = self.calculate_shap_values(X)
        
        explanations = []
        for i, customer_data in enumerate(customers_data):
            explanation = self.explain_single_prediction(
                customer_data, shap_values, i
            )
            explanation['customer_id'] = i
            explanations.append(explanation)
        
        return explanations
    
    def save_analysis_results(self, results: Dict[str, Any], 
                            filepath: Path) -> None:
        """
        Save SHAP analysis results to file.
        
        Args:
            results: Analysis results dictionary
            filepath: File path to save results
        """
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.number):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"SHAP analysis results saved to {filepath}")


def create_shap_analyzer(models_path: Optional[Path] = None) -> SHAPAnalyzer:
    """
    Convenience function to create a SHAP analyzer.
    
    Args:
        models_path: Path to models directory
        
    Returns:
        SHAPAnalyzer instance
    """
    return SHAPAnalyzer(models_path=models_path)


def explain_customer_prediction(customer_data: Dict[str, Any],
                               background_data: pd.DataFrame,
                               models_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Convenience function to explain a single customer prediction.
    
    Args:
        customer_data: Customer features
        background_data: Background dataset for SHAP
        models_path: Path to models
        
    Returns:
        Explanation dictionary
    """
    analyzer = create_shap_analyzer(models_path)
    analyzer.create_explainer(background_data)
    return analyzer.explain_single_prediction(customer_data)