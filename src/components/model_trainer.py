import sys
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve,
)
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import load_file
from src.components.data_transformation import NUM_FEATURES, CAT_FEATURES, TARGET
from src.utils import save_json
from src.utils import save_object


ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = Path("models")

def evaluate(y_true, y_pred, y_score=None) -> dict:
    return{
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":recall_score(y_true, y_pred, zero_division=0),
        "f1":f1_score(y_true,y_pred),
        "pr_auc": average_precision_score(y_true, y_score if y_score is not None else y_pred)

    }

def best_threshold(y_true, y_score, beta: float = 1.0):
    prec, rec, thresh = precision_recall_curve(y_true, y_score)
    f_scores = (1+beta**2) * (prec*rec) / (beta**2*prec+rec+1e-9)
    idx = np.nanargmax(f_scores[:-1])
    return float(thresh[idx])


class ModelTrainer:
    
    def initiate(self, train_path: Path, val_path: Path):
        try:
            train = load_file(train_path)
            val = load_file(val_path)
            logging.info(f"Loaded data: train {len(train)}, val {len(val)}")

            X_train = train[NUM_FEATURES + CAT_FEATURES]
            y_train = train[TARGET]

            X_val = val[NUM_FEATURES + CAT_FEATURES]
            y_val = val[TARGET]

            xgb_pipe = Pipeline([
                ("pre", ColumnTransformer([
                    ("num", "passthrough", NUM_FEATURES),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)
                ])),
                ("model", XGBClassifier(eval_metric="aucpr", random_state=42, n_jobs=1)),

            ])

            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            param_dist = {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [3, 4, 6, 8],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__subsample": [0.7, 0.9, 1.0],
                "model__colsample_bytree": [0.7, 0.9, 1.0],
                "model__min_child_weight": [1, 5, 10],
                "model__scale_pos_weight": [1, scale_pos_weight],
            }

            tscv = TimeSeriesSplit(n_splits=4)
            search = RandomizedSearchCV(
                xgb_pipe, param_distributions=param_dist, n_iter=20,
                scoring="average_precision", cv=tscv, random_state=42, n_jobs=-1,
            )
            logging.info("Starting hyperparameter search")
            search.fit(X_train, y_train)
            logging.info(f"Best params: {search.best_params_}")
            logging.info(f"Best CV PR-AUC: {search.best_score_:.4f}")

            best_model = search.best_estimator_
            y_score = best_model.predict_proba(X_val)[:, 1]
            threshold = best_threshold(y_val, y_score, beta=1.0)
            y_pred = (y_score >= threshold).astype(int)
            val_metrics = evaluate(y_val, y_pred, y_score)
            logging.info(f"Val metrics at threshold={threshold:.3f}: {val_metrics}")

            save_object(MODELS_DIR / "xgboost_flood_advisory.joblib", best_model)
            save_json(MODELS_DIR / "model_config.json", {
                "threshold": threshold,
                "num_features": NUM_FEATURES,
                "cat_features": CAT_FEATURES,
                "best_params": search.best_params_,
                "cv_pr_auc": search.best_score_,
                "val_metrics": val_metrics,
            })
            logging.info(f"Saved model and config to {MODELS_DIR}")

            return val_metrics

            
        except Exception as e:
            raise CustomException(e, sys)
        
    