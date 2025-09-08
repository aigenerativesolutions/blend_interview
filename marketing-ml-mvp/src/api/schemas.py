"""
Pydantic schemas for API request/response models
"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from enum import Enum


class EducationLevel(str, Enum):
    """Education level enumeration"""
    BASIC = "Basic"
    GRADUATION = "Graduation" 
    MASTER = "Master"
    PHD = "PhD"


class MaritalStatus(str, Enum):
    """Marital status enumeration"""
    SINGLE = "Single"
    TOGETHER = "Together"
    MARRIED = "Married"
    DIVORCED = "Divorced"
    WIDOW = "Widow"
    ALONE = "Alone"
    ABSURD = "Absurd"
    YOLO = "YOLO"


class CustomerFeatures(BaseModel):
    """Customer features input model"""
    # Demographics
    Education: EducationLevel = Field(..., description="Customer education level")
    Marital_Status: MaritalStatus = Field(..., description="Customer marital status")
    Income: float = Field(..., ge=0, description="Customer annual income")
    Kidhome: int = Field(..., ge=0, le=10, description="Number of kids at home")
    Teenhome: int = Field(..., ge=0, le=10, description="Number of teenagers at home")
    
    # Spending on different products
    MntWines: float = Field(..., ge=0, description="Amount spent on wines")
    MntFruits: float = Field(..., ge=0, description="Amount spent on fruits")
    MntMeatProducts: float = Field(..., ge=0, description="Amount spent on meat products")
    MntFishProducts: float = Field(..., ge=0, description="Amount spent on fish products") 
    MntSweetProducts: float = Field(..., ge=0, description="Amount spent on sweet products")
    MntGoldProds: float = Field(..., ge=0, description="Amount spent on gold products")
    
    # Purchases through different channels
    NumDealsPurchases: int = Field(..., ge=0, description="Number of deals purchases")
    NumWebPurchases: int = Field(..., ge=0, description="Number of web purchases")
    NumCatalogPurchases: int = Field(..., ge=0, description="Number of catalog purchases")
    NumStorePurchases: int = Field(..., ge=0, description="Number of store purchases")
    NumWebVisitsMonth: int = Field(..., ge=0, description="Number of web visits per month")
    
    # Campaign responses
    AcceptedCmp1: int = Field(..., ge=0, le=1, description="Accepted campaign 1 (0/1)")
    AcceptedCmp2: int = Field(..., ge=0, le=1, description="Accepted campaign 2 (0/1)")
    AcceptedCmp3: int = Field(..., ge=0, le=1, description="Accepted campaign 3 (0/1)")
    AcceptedCmp4: int = Field(..., ge=0, le=1, description="Accepted campaign 4 (0/1)")
    AcceptedCmp5: int = Field(..., ge=0, le=1, description="Accepted campaign 5 (0/1)")
    
    # Recency and complaints
    Recency: int = Field(..., ge=0, description="Days since last purchase")
    Complain: int = Field(..., ge=0, le=1, description="Customer complained (0/1)")
    
    # Derived features (will be calculated if not provided)
    Age: Optional[float] = Field(None, ge=18, le=120, description="Customer age")
    Total_Spent: Optional[float] = Field(None, ge=0, description="Total amount spent")
    Customer_Days: Optional[int] = Field(None, ge=0, description="Days as customer")
    
    @validator('Age', pre=True, always=True)
    def set_age_if_missing(cls, v):
        # Age will be calculated in preprocessing if not provided
        return v
    
    @validator('Total_Spent', pre=True, always=True) 
    def calculate_total_spent(cls, v, values):
        if v is None and all(key in values for key in ['MntWines', 'MntFruits', 'MntMeatProducts', 
                                                       'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']):
            return (values['MntWines'] + values['MntFruits'] + values['MntMeatProducts'] +
                   values['MntFishProducts'] + values['MntSweetProducts'] + values['MntGoldProds'])
        return v

    class Config:
        schema_extra = {
            "example": {
                "Education": "Graduation",
                "Marital_Status": "Married",
                "Income": 58138.0,
                "Kidhome": 0,
                "Teenhome": 0,
                "MntWines": 635,
                "MntFruits": 88,
                "MntMeatProducts": 546,
                "MntFishProducts": 172,
                "MntSweetProducts": 88,
                "MntGoldProds": 88,
                "NumDealsPurchases": 3,
                "NumWebPurchases": 8,
                "NumCatalogPurchases": 10,
                "NumStorePurchases": 4,
                "NumWebVisitsMonth": 7,
                "AcceptedCmp1": 0,
                "AcceptedCmp2": 0,
                "AcceptedCmp3": 0,
                "AcceptedCmp4": 0,
                "AcceptedCmp5": 0,
                "Recency": 58,
                "Complain": 0
            }
        }


class PredictionRequest(BaseModel):
    """Single prediction request model"""
    customer: CustomerFeatures = Field(..., description="Customer features")
    use_calibration: bool = Field(True, description="Use probability calibration")
    threshold: Optional[float] = Field(None, ge=0, le=1, description="Classification threshold")
    explain: bool = Field(False, description="Include SHAP explanation")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model"""
    customers: List[CustomerFeatures] = Field(..., min_items=1, max_items=1000, 
                                             description="List of customers")
    use_calibration: bool = Field(True, description="Use probability calibration")
    threshold: Optional[float] = Field(None, ge=0, le=1, description="Classification threshold")
    include_details: bool = Field(False, description="Include detailed prediction info")


class PredictionResponse(BaseModel):
    """Single prediction response model"""
    probability: float = Field(..., ge=0, le=1, description="Probability of response")
    prediction: int = Field(..., ge=0, le=1, description="Binary prediction (0/1)")
    will_respond: bool = Field(..., description="Will customer respond to campaign")
    threshold_used: float = Field(..., description="Threshold used for classification")
    confidence: str = Field(..., description="Confidence level (low/medium/high)")
    
    class Config:
        schema_extra = {
            "example": {
                "probability": 0.75,
                "prediction": 1,
                "will_respond": True,
                "threshold_used": 0.5,
                "confidence": "high"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model"""
    predictions: List[Dict[str, Any]] = Field(..., description="List of predictions")
    summary: Dict[str, Any] = Field(..., description="Batch prediction summary")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "customer_id": 0,
                        "probability": 0.75,
                        "prediction": 1,
                        "will_respond": True
                    }
                ],
                "summary": {
                    "total_customers": 1,
                    "predicted_responders": 1,
                    "predicted_non_responders": 0,
                    "average_probability": 0.75
                }
            }
        }


class FeatureContribution(BaseModel):
    """SHAP feature contribution model"""
    feature: str = Field(..., description="Feature name")
    value: Union[str, float, int] = Field(..., description="Feature value")
    shap_value: float = Field(..., description="SHAP contribution value")
    contribution: str = Field(..., description="Positive or negative contribution")
    abs_contribution: float = Field(..., ge=0, description="Absolute contribution value")


class ExplanationResponse(BaseModel):
    """SHAP explanation response model"""
    prediction: PredictionResponse = Field(..., description="Prediction details")
    base_value: float = Field(..., description="Model base value (expected value)")
    feature_contributions: List[FeatureContribution] = Field(..., 
                                                            description="All feature contributions")
    top_positive_features: List[FeatureContribution] = Field(...,
                                                           description="Top positive contributors")
    top_negative_features: List[FeatureContribution] = Field(...,
                                                           description="Top negative contributors")
    explanation_summary: str = Field(..., description="Human-readable explanation")


class ModelInfo(BaseModel):
    """Model information response model"""
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_type: str = Field(..., description="Type of model")
    has_calibrator: bool = Field(..., description="Whether calibrator is available")
    calibrator_type: Optional[str] = Field(None, description="Type of calibrator")
    optimal_threshold: Optional[float] = Field(None, description="Optimal classification threshold")
    feature_count: Optional[int] = Field(None, description="Number of features")
    training_samples: Optional[int] = Field(None, description="Number of training samples")
    validation_samples: Optional[int] = Field(None, description="Number of validation samples")
    model_metrics: Optional[Dict[str, Any]] = Field(None, description="Model performance metrics")


class FeatureImportance(BaseModel):
    """Feature importance response model"""
    feature_importance: Dict[str, float] = Field(..., description="Feature importance scores")
    top_n: int = Field(..., description="Number of top features returned")
    
    class Config:
        schema_extra = {
            "example": {
                "feature_importance": {
                    "Total_Spent": 0.25,
                    "Income": 0.18,
                    "Age": 0.15,
                    "MntWines": 0.12,
                    "NumWebPurchases": 0.10
                },
                "top_n": 5
            }
        }


class HealthResponse(BaseModel):
    """API health check response"""
    status: str = Field(..., description="API status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    model_status: str = Field(..., description="Model loading status")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input data",
                "details": {
                    "field": "Income",
                    "issue": "Must be greater than or equal to 0"
                }
            }
        }