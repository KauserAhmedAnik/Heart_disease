from pydantic import BaseModel
from typing import List

# Input schema for /predict
class HeartInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol:int
    fbs:int
    restecg:int
    thalach:int
    exang:int
    oldpeak:float
    slope:int
    ca:int
    thal:int

# Output schema for /predict
class PredictionOutput(BaseModel):
    predicted_class: int