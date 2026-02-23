"""
FastAPI application entry point for the Sarcasm Detection API.
Exposes endpoints to interact with the SarcasmPredictor.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference import SarcasmPredictor

app = FastAPI(
    title="Sarcasm Detection API",
    description="API for detecting sarcasm in text (News Headlines) using DistilBERT.",
    version="1.0.0"
)

predictor = SarcasmPredictor()

class TextInput(BaseModel):
    """Schema for the incoming text prediction request."""
    text: str

@app.get("/")
def home():
    """Root endpoint providing basic API information and documentation link."""
    return {
        "message": "Welcome to Sarcasm Detection API!",
        "docs_url": "/docs" 
    }

@app.post("/predict")
def predict_sarcasm(input_data: TextInput):
    """
    Accepts a text string and returns the sarcasm prediction result.
    """
    try:
        result = predictor.predict(input_data.text)
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))