import sys
from datetime import date, timedelta
import pandas as pd
import numpy as np
import requests

from src.exception import CustomException
from src.logger import logging

from functools import lru_cache
from pathlib import Path

DISTRICTS_CSV = Path(__file__).resolve().parents[2] / "configs" / "districts.csv"


@lru_cache(maxsize=1)
def load_districts() -> pd.DataFrame:
    """Read the 25-row district lookup once and reuse it for every request."""
    return pd.read_csv(DISTRICTS_CSV)


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
        start = end - timedelta(days=days - 1)

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
        df = df.rename(columns={"time": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df["district"] = district
        df["climatic_zone"] = climatic_zone

        logging.info(
            f"fetched {len(df)} days of weather for {district} ({start} to {end})"
        )
        return df
    except Exception as e:
        raise CustomException(e, sys)


def get_district_coordinates(district: str) -> tuple[float, float, str]:
    """Look up a district's lat/lon/climatic_zone."""
    try:
        districts = load_districts()
        row = districts[districts["district"] == district]
        if row.empty:
            raise ValueError(f"Unknown district: {district}")
        row = row.iloc[0]
        return (
            float(row["latitude"]),
            float(row["longitude"]),
            str(row["climatic_zone"]),
        )
    except Exception as e:
        raise CustomException(e, sys)


def find_nearest_district(
    latitude: float, longitude: float
) -> tuple[str, float, float, str]:
    """Approximate which district a coordinate belongs to, by nearest centroid."""
    try:
        centroids = load_districts()
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
