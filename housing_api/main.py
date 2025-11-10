from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from typing import Dict, Optional
import uvicorn
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="California Housing Price Prediction API",
    description="API for predicting California housing prices using trained ML model",
    version="2.0.0"
)

# Global variables to store model bundle
model = None
scaler = None
metadata = None

# Define request schema matching your feature set
class PredictionRequest(BaseModel):
    MedInc: float = Field(..., description="Median income in block group")
    HouseAge: float = Field(..., description="Median house age in block group")
    AveRooms: float = Field(..., description="Average number of rooms per household")
    AveBedrms: float = Field(..., description="Average number of bedrooms per household")
    Population: float = Field(..., description="Block group population")
    AveOccup: float = Field(..., description="Average number of household members")
    Latitude: float = Field(..., description="Block group latitude")
    Longitude: float = Field(..., description="Block group longitude")
    
    class Config:
        json_schema_extra = {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984127,
                "AveBedrms": 1.023810,
                "Population": 322.0,
                "AveOccup": 2.555556,
                "Latitude": 37.88,
                "Longitude": -122.23
            }
        }

# Define response schema
class PredictionResponse(BaseModel):
    predicted_price: float
    predicted_price_formatted: str
    model_info: Dict[str, str]
    features_used: Dict[str, float]

class ModelInfoResponse(BaseModel):
    model_loaded: bool
    model_name: Optional[str] = None
    model_type: Optional[str] = None
    training_date: Optional[str] = None
    test_r2: Optional[float] = None
    test_rmse: Optional[float] = None
    test_mae: Optional[float] = None
    n_features: Optional[int] = None
    features: Optional[list] = None

@app.on_event("startup")
async def load_model():
    """Load the trained model bundle when the app starts"""
    global model, scaler, metadata
    try:
        # Load the complete model bundle
        model_bundle = joblib.load("housing_model_v2.pkl")
        
        model = model_bundle['model']
        scaler = model_bundle['scaler']
        metadata = model_bundle['metadata']
        
        print("=" * 70)
        print("✓ Model Bundle Loaded Successfully!")
        print("=" * 70)
        print(f"  Model Name:     {metadata.get('model_name', 'N/A')}")
        print(f"  Model Type:     {metadata.get('model_type', 'N/A')}")
        print(f"  Training Date:  {metadata.get('training_date', 'N/A')[:10]}")
        print(f"  Test R²:        {metadata.get('test_r2', 'N/A'):.4f}")
        print(f"  Test RMSE:      {metadata.get('test_rmse', 'N/A'):.4f}")
        print(f"  Features:       {metadata.get('n_features', 'N/A')}")
        print("=" * 70)
        
    except FileNotFoundError:
        print("=" * 70)
        print("✗ Error: housing_model_v2.pkl not found!")
        print("=" * 70)
        print("Make sure the model file is in the same directory as main.py")
    except Exception as e:
        print("=" * 70)
        print(f"✗ Error loading model: {str(e)}")
        print("=" * 70)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "California Housing Price Prediction API",
        "status": "running",
        "model_loaded": model is not None,
        "version": "2.0.0",
        "endpoints": {
            "predict": "/predict (POST) - Make price predictions",
            "model_info": "/model-info (GET) - Get model details",
            "health": "/health (GET) - Health check",
            "docs": "/docs (GET) - Interactive API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get detailed information about the loaded model"""
    if model is None or metadata is None:
        return ModelInfoResponse(model_loaded=False)
    
    return ModelInfoResponse(
        model_loaded=True,
        model_name=metadata.get('model_name'),
        model_type=metadata.get('model_type'),
        training_date=metadata.get('training_date'),
        test_r2=metadata.get('test_r2'),
        test_rmse=metadata.get('test_rmse'),
        test_mae=metadata.get('test_mae'),
        n_features=metadata.get('n_features'),
        features=metadata.get('features')
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a house price prediction
    
    Returns price in hundred thousands (e.g., 2.5 = $250,000)
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Convert request to dictionary
        features_dict = request.dict()
        
        # Create DataFrame with input features
        features_df = pd.DataFrame([features_dict])
        
        # Add engineered features (same as training pipeline)
        features_df['RoomsPerHousehold'] = features_df['AveRooms'] / features_df['AveOccup']
        features_df['BedroomsPerRoom'] = features_df['AveBedrms'] / features_df['AveRooms']
        features_df['PopulationPerHousehold'] = features_df['Population'] / features_df['HouseAge']
        
        # Store engineered features for response
        engineered_features = {
            "RoomsPerHousehold": float(features_df['RoomsPerHousehold'].iloc[0]),
            "BedroomsPerRoom": float(features_df['BedroomsPerRoom'].iloc[0]),
            "PopulationPerHousehold": float(features_df['PopulationPerHousehold'].iloc[0])
        }
        
        # Ensure correct feature order
        feature_order = metadata.get('features', features_df.columns.tolist())
        features_df = features_df[feature_order]
        
        # Scale features (keep as DataFrame to preserve feature names)
        features_scaled = scaler.transform(features_df)
        features_scaled_df = pd.DataFrame(
            features_scaled, 
            columns=features_df.columns,
            index=features_df.index
        )
        
        # Make prediction
        prediction = model.predict(features_scaled_df)
        predicted_value = float(prediction[0])
        
        # Format as dollar amount
        formatted_price = f"${predicted_value * 100_000:,.2f}"
        
        return PredictionResponse(
            predicted_price=predicted_value,
            predicted_price_formatted=formatted_price,
            model_info={
                "model_name": metadata.get('model_name', 'Unknown'),
                "model_type": metadata.get('model_type', 'Unknown'),
                "test_r2": f"{metadata.get('test_r2', 0):.4f}"
            },
            features_used={
                **features_dict,
                **engineered_features
            }
        )
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print("=" * 70)
        print("PREDICTION ERROR:")
        print(error_details)
        print("=" * 70)
        raise HTTPException(
            status_code=400,
            detail=f"Prediction error: {str(e)}"
        )

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Starting California Housing Price Prediction API")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000)