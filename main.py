from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import uvicorn
import os
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
classifier = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global classifier
    try:
        logger.info("🔄 Loading fake news detection model...")
        classifier = pipeline(
            "text-classification",
            model="abutair1/fake-news-detector-bert"
        )
        logger.info("✅ Model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise e
    
    yield
    
    # Shutdown (cleanup if needed)
    logger.info("🔄 Shutting down...")

# Initialize FastAPI with lifespan
app = FastAPI(
    title="Fake News Detector API",
    description="BERT-based fake news detection API deployed on Azure",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response models
class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    prediction: str
    confidence: float

@app.get("/")
def root():
    """Health check"""
    return {
        "message": "Fake News Detector API is running on Azure!",
        "status": "healthy",
        "model": "abutair1/fake-news-detector-bert",
        "version": "1.0.0"
    }

@app.get("/health")
def health():
    """Detailed health check"""
    return {
        "status": "healthy" if classifier else "model_not_loaded",
        "model_loaded": classifier is not None,
        "platform": "Azure App Service"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: TextInput):
    """Predict if news is fake or real"""
    if not classifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = classifier(input_data.text)
        prediction = result[0]
        
        logger.info(f"Prediction made: {prediction['label']} with confidence {prediction['score']:.4f}")
        
        return PredictionResponse(
            text=input_data.text,
            prediction=prediction['label'],
            confidence=prediction['score']
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)