from src.pipeline import predict_pipeline
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from matplotlib.pyplot import title
from src.pipeline.predict_pipeline import PredictPipeline

app = FastAPI(title="Flood Risk Advisory API", version="0.1.0")

predict_pipeline = PredictPipeline()

class WeatherRecord(BaseModel):
    date: str
    district: str
    climatic_zone: str
    precipitation_sum: float = Field(ge=0)
    rain_48h: float = Field(ge=0)     
    rain_72h: float = Field(ge=0)    
    soil_moisture_0_to_7cm_mean: float = Field(ge=0, le=1)
    soil_moisture_7_to_28cm_mean: float = Field(ge=0, le=1)
    soil_saturation_index: float = Field(ge=0, le=1)
    temperature_2m_max: float
    wind_speed_10m_max: float = Field(ge=0)

class PredictRequest(BaseModel):
    records: List[WeatherRecord]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictRequest):
    if len(request.records) == 0:
        raise HTTPException(status_code=422, detail="No records provided")
    districts = {r.district for r in request.records}
    if len(districts) > 1:
        raise HTTPException(status_code=422, detail="/predict accepts one district at a time; use /predict/batch for multiple")
    try:
        return predict_pipeline.predict_from_records([r.model_dump() for r in request.records])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(request: PredictRequest):
    try:
        return predict_pipeline.predict_from_records([r.model_dump() for r in request.records])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))