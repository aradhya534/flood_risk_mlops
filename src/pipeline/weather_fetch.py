import sys
from datetime import date, timedelta
import pandas as pd 
import numpy as np
import requests

from src.exception import CustomException
from src.logger import logging

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

VARIABLES = [
    "precipitation_sum",
    "soil_moisture_0_to_7cm_mean",
    "soil_moisture_7_to_28cm_mean",
    "temperature_2m_max",
    "wind_speed_10m_max",
]

def fetch_weather_history(
    latitude: float,
    longitude: float,
    district: str,
    climatic_zone: str,
    days: int = 21,
) -> pd.DataFrame:
    """Fetch the last `days` days of daily weather for one location from
    Open-Meteo's historical archive, shaped to match predict_pipeline's input.
    Open-Meteo has a short reporting lag, so we end the window 3 days before
    today rather than today itself, to avoid requesting data not yet published."""

    try: 
        end = date.today() - timedelta(days=3)
        start = end - timedelta(days=days-1)

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": ",".join(VARIABLES),
            "timezone": "auto",
        }

        response = requests.get(ARCHIVE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()["daily"]

        df = pd.DataFrame(data)
        df = df.rename(columns={"time":"date"})
        df["date"] = pd.to_datetime(df["date"])
        df["district"] = district
        df["climatic_zone"] = climatic_zone

        logging.info(f"fetched {len(df)} days of weather for {district} ({start} to {end})")
        return df
    except Exception as e:
        raise CustomException(e, sys)



def find_nearest_district(latitude: float, longitude: float) -> tuple[str, float, float, str]:
    """Approximate which district a coordinate belongs to, by nearest centroid.
    Imprecise near district borders, since we only have one point per district
    (its centroid), not its true administrative boundary."""
    try:
        raw = pd.read_csv("artifacts/raw.csv")
        centroids = raw.drop_duplicates("district")[
            ["district", "latitude", "longitude", "climatic_zone"]
        ]

        dists = np.sqrt(
            (centroids["latitude"] - latitude) ** 2
            + (centroids["longitude"] - longitude) ** 2
        )
        nearest = centroids.loc[dists.idxmin()]
        return (
            str(nearest["district"]),
            float(nearest["latitude"]),
            float(nearest["longitude"]),
            str(nearest["climatic_zone"]),
        )
    except Exception as e:
        raise CustomException(e, sys)