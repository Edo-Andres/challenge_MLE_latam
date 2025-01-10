from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from typing import List
import pandas as pd
from challenge.model import DelayModel

# Initialize the application and the model
app = FastAPI()
delay_model = DelayModel()

# Load the pre-trained model
delay_model.load("challenge/reg_model_2.pkl")

# Define the input structure
class PredictionInput(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    @field_validator('TIPOVUELO')
    def validate_tipovuelo(cls, v):
        if v not in ['I', 'N']:
            raise ValueError('TIPOVUELO must be either "I" or "N"')
        return v

    @field_validator('MES')
    def validate_mes(cls, v):
        if not 1 <= v <= 12:
            raise ValueError('MES must be between 1 and 12')
        return v

# Health endpoint
@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}

# Prediction endpoint
@app.post("/predict", status_code=200)
async def post_predict(inputs: List[PredictionInput]) -> dict:
    try:
        # Convert input data to a DataFrame
        input_data = pd.DataFrame([input.model_dump() for input in inputs])

        # Preprocess the data
        features = delay_model.preprocess(input_data)

        # Make predictions
        predictions = delay_model.predict(features)

        return {"predictions": predictions}

    except Exception as e:
        return {"error": str(e)}