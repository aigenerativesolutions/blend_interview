"""
FastAPI application for Marketing Campaign Response Prediction
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
import traceback

from .schemas import (
    PredictionRequest, BatchPredictionRequest,
    PredictionResponse, BatchPredictionResponse,
    ExplanationResponse, ModelInfo, FeatureImportance,
    HealthResponse, ErrorResponse, CustomerFeatures
)
from ..models.predict import MarketingPredictor, load_marketing_predictor
from ..utils.model_utils import log_prediction_metrics, generate_prediction_batch_id
from ..config.settings import API_TITLE, API_VERSION, API_HOST, API_PORT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
predictor: Optional[MarketingPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup
    logger.info("Starting up Marketing ML API...")
    
    global predictor
    
    try:
        # Load predictor with all pipeline artifacts
        predictor = load_marketing_predictor()
        logger.info(" Pipeline predictor loaded successfully")
        
    except Exception as e:
        logger.error(f" Failed to load pipeline predictor during startup: {str(e)}")
        # Could set predictor = None and handle gracefully in endpoints
    
    logger.info("API startup completed")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Marketing ML API...")


# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="API for predicting customer response to marketing campaigns using XGBoost with SHAP explanations",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_predictor() -> MarketingPredictor:
    """Dependency to get the predictor instance."""
    if predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    return predictor




@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            details={"exception_type": type(exc).__name__}
        ).dict()
    )


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "title": API_TITLE,
        "version": API_VERSION,
        "description": "Marketing Campaign Response Prediction API",
        "docs_url": "/docs",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_status = "loaded" if predictor is not None else "not_loaded"
    
    return HealthResponse(
        status="healthy" if predictor is not None else "degraded",
        timestamp=datetime.now().isoformat(),
        version=API_VERSION,
        model_status=model_status
    )


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info(predictor: MarketingPredictor = Depends(get_predictor)):
    """Get model information and metadata."""
    try:
        info = predictor.get_model_info()
        return ModelInfo(**info)
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")


@app.get("/model/feature-importance", response_model=FeatureImportance)
async def get_feature_importance(
    top_n: Optional[int] = 10,
    predictor: MarketingPredictor = Depends(get_predictor)
):
    """Get model feature importance."""
    try:
        importance = predictor.get_feature_importance(top_n)
        
        if importance is None:
            raise HTTPException(
                status_code=404,
                detail="Feature importance not available for this model"
            )
        
        return FeatureImportance(
            feature_importance=importance,
            top_n=len(importance)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving feature importance: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    predictor: MarketingPredictor = Depends(get_predictor)
):
    """Make a single prediction."""
    try:
        # Convert customer features to dict
        customer_data = request.customer.dict()
        
        # Make prediction
        result = predictor.predict_single(
            customer_data,
            use_calibration=request.use_calibration,
            threshold=request.threshold
        )
        
        # Log prediction metrics in background
        background_tasks.add_task(
            log_prediction_metrics,
            np.array([result['prediction']]),
            np.array([result['probability']]),
            generate_prediction_batch_id()
        )
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in single prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    predictor: MarketingPredictor = Depends(get_predictor)
):
    """Make batch predictions."""
    try:
        # Convert customers to list of dicts
        customers_data = [customer.dict() for customer in request.customers]
        
        # Make batch predictions
        results = predictor.predict_batch(
            customers_data,
            use_calibration=request.use_calibration,
            threshold=request.threshold,
            include_details=request.include_details
        )
        
        # Calculate summary statistics
        predictions = [r['prediction'] for r in results]
        probabilities = [r['probability'] for r in results]
        
        summary = {
            'total_customers': len(results),
            'predicted_responders': sum(predictions),
            'predicted_non_responders': len(predictions) - sum(predictions),
            'average_probability': np.mean(probabilities),
            'response_rate': np.mean(predictions)
        }
        
        # Log metrics in background
        background_tasks.add_task(
            log_prediction_metrics,
            np.array(predictions),
            np.array(probabilities),
            generate_prediction_batch_id()
        )
        
        return BatchPredictionResponse(
            predictions=results,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(
    request: PredictionRequest,
    predictor: MarketingPredictor = Depends(get_predictor)
):
    """Explain a single prediction using integrated SHAP."""
    try:
        customer_data = request.customer.dict()
        
        # Get prediction
        prediction_result = predictor.predict_single(
            customer_data,
            use_calibration=request.use_calibration,
            threshold=request.threshold
        )
        
        # Get SHAP explanation
        shap_explanation = predictor.get_shap_explanation(customer_data)
        
        if shap_explanation is None:
            # Fallback to basic feature importance explanation
            basic_explanation = predictor.explain_prediction(customer_data, top_features=10)
            
            # Convert to SHAP format for compatibility
            feature_contributions = []
            if 'top_influential_features' in basic_explanation:
                for feature, importance in basic_explanation['top_influential_features'].items():
                    feature_contributions.append({
                        'feature': feature,
                        'value': basic_explanation['customer_feature_values'].get(feature, 0),
                        'shap_value': float(importance),  # Using feature importance as proxy
                        'contribution': 'positive' if importance > 0 else 'negative',
                        'abs_contribution': abs(float(importance))
                    })
            
            return ExplanationResponse(
                prediction=PredictionResponse(**prediction_result),
                base_value=0.5,  # Default base value
                feature_contributions=feature_contributions,
                top_positive_features=[c for c in feature_contributions if c['contribution'] == 'positive'][:5],
                top_negative_features=[c for c in feature_contributions if c['contribution'] == 'negative'][:5],
                explanation_summary=f"Basic feature importance explanation (SHAP not available)"
            )
        
        # Convert SHAP explanation to response format
        feature_contributions = []
        for contrib in shap_explanation['feature_contributions']:
            feature_contributions.append({
                'feature': contrib['feature'],
                'value': 0,  # We don't have original feature values in SHAP format
                'shap_value': contrib['shap_value'],
                'contribution': contrib['contribution'],
                'abs_contribution': contrib['abs_contribution']
            })
        
        # Generate explanation summary
        will_respond = prediction_result['will_respond']
        probability = prediction_result['probability']
        confidence = prediction_result['confidence']
        
        response_text = "will respond" if will_respond else "will not respond"
        summary = f"The model predicts this customer {response_text} to the campaign "
        summary += f"with {probability:.1%} probability ({confidence} confidence). "
        
        if feature_contributions:
            top_factors = [c['feature'] for c in feature_contributions[:3]]
            summary += f"Key factors: {', '.join(top_factors)}."
        
        return ExplanationResponse(
            prediction=PredictionResponse(**prediction_result),
            base_value=shap_explanation['expected_value'],
            feature_contributions=feature_contributions,
            top_positive_features=shap_explanation.get('top_positive', [])[:5],
            top_negative_features=shap_explanation.get('top_negative', [])[:5],
            explanation_summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error in explanation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.post("/predict/explain", response_model=ExplanationResponse)
async def predict_and_explain(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    predictor: MarketingPredictor = Depends(get_predictor)
):
    """Make a prediction and provide SHAP explanation in one call."""
    try:
        # This endpoint combines prediction and explanation
        # Reuse the explain_prediction logic
        return await explain_prediction(request, predictor)
        
    except Exception as e:
        logger.error(f"Error in predict and explain: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Predict and explain failed: {str(e)}")


@app.get("/model/sample-input", response_model=CustomerFeatures)
async def get_sample_input():
    """Get a sample customer input for testing."""
    return CustomerFeatures(
        Education="Graduation",
        Marital_Status="Married",
        Income=58138.0,
        Kidhome=0,
        Teenhome=0,
        MntWines=635,
        MntFruits=88,
        MntMeatProducts=546,
        MntFishProducts=172,
        MntSweetProducts=88,
        MntGoldProds=88,
        NumDealsPurchases=3,
        NumWebPurchases=8,
        NumCatalogPurchases=10,
        NumStorePurchases=4,
        NumWebVisitsMonth=7,
        AcceptedCmp1=0,
        AcceptedCmp2=0,
        AcceptedCmp3=0,
        AcceptedCmp4=0,
        AcceptedCmp5=0,
        Recency=58,
        Complain=0
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )