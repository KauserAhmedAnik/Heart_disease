from fastapi import FastAPI
from functools import lru_cache
from transformers import pipeline
import time, redis, json, hashlib
from app.schemas import HeartInput, PredictionOutput
import joblib
import numpy as np


app = FastAPI()


# Load the model once at startup
model = joblib.load("model/Heart_model.joblib")

# Iris class names for readability
class_names = [0,1]


@app.get("/info")
def model_info():
    """Basic model info"""
    return {
        "model_type": "RandomForestClassifier",
        "classes": class_names
    }

@app.post("/predict", response_model=PredictionOutput)
def predict_survival(data:HeartInput):
    """Make prediction from input features"""
    features = np.array([[data.age, data.sex, data.ca, data.cp,data.trestbps, data.chol, data.fbs, data.restecg, data.thalach, data.exang, data.oldpeak, data.slope, data.thal]])
    prediction = model.predict(features)[0]
    predicted_class = class_names[prediction]
    return {"predicted_class": predicted_class}


@app.get("/health")
def health_check():
    """
    Example API:
    - Simple health check endpoint.
    - Load balancers like Nginx, AWS ELB, or Kubernetes call this to check
      if the instance is alive and healthy.
    ⚙️ Use when: You deploy multiple instances and need auto-failover.
    """
    return {"status": "healthy"}