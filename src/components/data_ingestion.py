import sys
from pathlib import Path

import pandas as pd
from src.exception import CustomException
from src.logger import logging


class DataIngestion:
    def __init__(self, raw_data_dir: str = "data"):
        self.raw_data_dir = Path(raw_data_dir)

    def initiate(self) -> Path:
        try:
            csv_path = next(self.raw_data_dir.rglob("*csv"))
            logging.info(f"found the dataset at {csv_path}")

            df = pd.read_csv(csv_path, parse_dates=["date"])
            df = df.sort_values(["district", "date"]).reset_index(drop=True)

            output_path = Path("artifacts") / "raw.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logging.info(
                f"Ingestion complete. {df.shape} \n saved raw data to {output_path}"
            )
            return output_path

        except Exception as e:
            raise CustomException(e, sys)
