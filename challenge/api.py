from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, ValidationError
from typing import List, Dict
import pandas as pd
from challenge.model import DelayModel

from fastapi.exceptions import RequestValidationError


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

class PredictionRequest(BaseModel):
    flights: List[PredictionInput]

# Health endpoint
@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}



@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

@app.post("/predict", status_code=200)
async def post_predict(request: PredictionRequest) -> dict:
    try:
        # Convert input data to a DataFrame
        input_data = pd.DataFrame([flight.model_dump() for flight in request.flights])

        # Preprocess the data
        features = delay_model.preprocess(input_data)

        # Make predictions
        predictions = delay_model.predict(features)

        return {"predict": predictions}
    except ValidationError as e:
        # Captura y lanza errores de validación
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Otros errores genéricos
        raise HTTPException(status_code=400, detail=str(e))

