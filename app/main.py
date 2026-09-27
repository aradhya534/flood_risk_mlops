from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
import pandas as pd 


from src.pipeline.weather_fetch import fetch_weather_history, find_nearest_district
from src.pipeline.predict_pipeline import PredictPipeline
from src.components.data_transformation import add_input_features


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
        raise HTTPException(
            status_code=422,
            detail="/predict accepts one district at a time; use /predict/batch for multiple",
        )
    try:
        return predict_pipeline.predict_from_records(
            [r.model_dump() for r in request.records]
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(request: PredictRequest):
    try:
        return predict_pipeline.predict_from_records(
            [r.model_dump() for r in request.records]
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/predict/live")
def predict_live(lat: float, lon: float):
    try:
        district, centroid_lat, centroid_lon, zone = find_nearest_district(lat, lon)
        history_df = fetch_weather_history(centroid_lat, centroid_lon, district, zone, days=21)
        records = history_df.to_dict(orient="records")
        result = predict_pipeline.predict_from_records(records)
        for r in result:
            r["matched_district"] = district  # transparency: tell the caller which district was used
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/districts")
def list_districts():
    raw = pd.read_csv("artifacts/raw.csv")
    districts = (
        raw.drop_duplicates("district")[["district", "latitude", "longitude"]]
        .sort_values("district")
        .to_dict(orient="records")
    )
    return districts


@app.get("/predict/live")
def predict_live(lat: float, lon: float):
    try:
        district, centroid_lat, centroid_lon, zone = find_nearest_district(lat, lon)
        history_df = fetch_weather_history(centroid_lat, centroid_lon, district, zone, days=21)
        records = history_df.to_dict(orient="records")
        result = predict_pipeline.predict_from_records(records)

        processed = add_input_features(history_df)
        latest = processed.sort_values("date").iloc[-1]
        for r in result:
            r["matched_district"] = district
            r["data_as_of"] = str(latest["date"].date())
            r["rain_72h_mm"] = round(float(latest["rain_72h"]), 1)
            r["soil_saturation_pct"] = round(float(latest["soil_saturation_index"]) * 100, 1)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/", StaticFiles(directory="static", html=True), name="static")