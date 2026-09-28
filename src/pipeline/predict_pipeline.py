from src.components.model_trainer import MODELS_DIR
import sys

import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, load_json
from src.components.data_transformation import add_input_features


class PredictPipeline:
    def __init__(self):
        try:
            self.model = load_object(MODELS_DIR / "xgboost_flood_advisory.joblib")
            config = load_json(MODELS_DIR / "model_config.json")
            self.threshold = config["threshold"]
            self.num_features = config["num_features"]
            self.cat_features = config["cat_features"]
            logging.info(f"Loaded model, threshold={self.threshold:.3f}")

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, history_df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = add_input_features(history_df)
            df = df.dropna(subset=self.num_features)
            if df.empty:
                raise ValueError(
                    "Not enough history to compute all features (need 14+ days per district)"
                )

            latest = df.sort_values("date").groupby("district").tail(1)
            X = latest[self.num_features + self.cat_features]
            scores = self.model.predict_proba(X)[:, 1]
            preds = (scores >= self.threshold).astype(int)

            result = latest[["date", "district"]].copy()
            result["flood_risk_score"] = scores
            result["advisory_predicted"] = preds
            return result.reset_index(drop=True)
        except Exception as e:
            raise CustomException(e, sys)

    def predict_from_records(self, records: list[dict]) -> list[dict]:
        """Takes a list of dicts (already-validated Pydantic records, via .model_dump()),
        returns a list of dicts ready for a JSON response. Used by both /predict and
        /predict/batch, since the underlying logic is identical either way."""
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        result = self.predict(df)
        return result.to_dict(orient="records")
