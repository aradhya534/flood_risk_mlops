from pathlib import Path
import pandas as pd
from src.exception import CustomException
import sys
import joblib
import json
import os
from src.logger import logging


def load_file(file_path: Path):
    try:
        return pd.read_csv(file_path, parse_dates=["date"])

    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
        logging.info("object saved successfully")
    except Exception as e:
        raise CustomException(e, sys)


def save_json(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(obj, f, indent=2)
        logging.info("json object saved successfully")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(e, sys)


def load_json(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise CustomException(e, sys)
