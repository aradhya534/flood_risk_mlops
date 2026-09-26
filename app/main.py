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
    try:
        df = pd.DataFrame([r.model_dump() for r in request.records])
        df["date"] = pd.to_datetime(df["date"])
        result = predict_pipeline.predict(df)
        return result.to_dict(orient="records")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

