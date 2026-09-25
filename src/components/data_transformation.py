import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging

TARGET = "advisory_48h"
HORIZON_DAYS = 2

NUM_FEATURES = [
    "precipitation_sum", "rain_48h", "rain_72h", "rain_7d",
    "rain_lag1", "rain_lag2", "rain_lag3",
    "soil_moisture_0_to_7cm_mean", "soil_moisture_7_to_28cm_mean",
    "soil_saturation_index", "soil_change_3d",
    "temperature_2m_max", "wind_speed_10m_max",
    "month_sin", "month_cos", "rain_3d", "rain_max_7d", "wet_days_7d", "soil_mean_7d",
]
CAT_FEATURES = ["district", "climatic_zone"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        """Add lag/rolling features and the target. Must run on the FULL sorted
        series, before any split — see the earlier discussion on why."""
        df = df.sort_values(["district", "date"]).reset_index(drop=True).copy()
        g = df.groupby("district")

        df["is_advisory"] = (df["flood_category"] != "Low Risk (Normal)").astype(int)
        df[TARGET] = g["is_advisory"].shift(-HORIZON_DAYS)

        for lag in [1, 2, 3]:
            df[f"rain_lag{lag}"] = g["precipitation_sum"].shift(lag)

        df["rain_3d"] = g["precipitation_sum"].transform(lambda s: s.rolling(3).sum())
        df["rain_7d"] = g["precipitation_sum"].transform(lambda s: s.rolling(7).sum())
        df["rain_max_7d"] = g["precipitation_sum"].transform(lambda s: s.rolling(7).max())
        df["wet_days_7d"] = g["precipitation_sum"].transform(lambda s: (s >= 1).rolling(7).sum())
        df["soil_change_3d"] = df["soil_saturation_index"] - g["soil_saturation_index"].shift(3)
        df["soil_mean_7d"] = g["soil_saturation_index"].transform(lambda s: s.rolling(7).mean())

        df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)

        df = df.dropna(subset=[TARGET] + NUM_FEATURES).reset_index(drop=True)
        df[TARGET] = df[TARGET].astype(int)
        return df
    except Exception as e:
        raise CustomException(e, sys)

def split_by_time(df: pd.DataFrame,
                train_end_year: int = 2021,
                val_year: int = 2022,
                test_start_year: int = 2023) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split — train on the past, validate and test on later years.
    No shuffling: row order in time is what makes this a fair simulation of
    predicting the future from the past."""
    train = df[df["date"].dt.year <= train_end_year].reset_index(drop=True)
    val = df[df["date"].dt.year == val_year].reset_index(drop=True)
    test = df[df["date"].dt.year >= test_start_year].reset_index(drop=True)
    return train, val, test
        

class DataTransformation:
    def __init__(self):
        
        self.output_dir = Path("artifacts")

    def initiate(self, raw_path: Path) -> tuple[Path, Path, Path]:
        try:
            raw = pd.read_csv(raw_path, parse_dates=["date"])
            logging.info(f"Loaded raw data: {raw.shape}")

            df = build_features(raw)
            logging.info(f"Features built: {df.shape}, positive rate {df['advisory_48h'].mean():.4f}")

            train, val, test = split_by_time(df)
            logging.info(f"Split -> train {len(train)}, val {len(val)}, test {len(test)}")

            paths = {}
            for name, part in [("train", train), ("val", val), ("test", test)]:
                out_path = self.output_dir / f"{name}.csv"
                part.to_csv(out_path, index=False)
                paths[name] = out_path
                logging.info(f"Saved {name} -> {out_path}")

            return paths["train"], paths["val"], paths["test"]

        except Exception as e:
            raise CustomException(e, sys)

