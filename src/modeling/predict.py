"""Inference helpers for the trained housing price model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path("models/best_model.joblib")
META_PATH = Path("models/model_metadata.json")


def load_artifacts(model_path: Path = MODEL_PATH, metadata_path: Path = META_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata artifact: {metadata_path}")

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def prepare_input(payload: dict, metadata: dict) -> pd.DataFrame:
    feature_cols = metadata["feature_columns"]
    defaults = metadata.get("defaults", {})

    row = {col: payload.get(col, defaults.get(col)) for col in feature_cols}
    return pd.DataFrame([row], columns=feature_cols)


def predict_price(payload: dict) -> float:
    model, metadata = load_artifacts()
    X = prepare_input(payload, metadata)
    pred_log = model.predict(X)
    return float(np.expm1(pred_log)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Malaysian housing price from JSON payload")
    parser.add_argument(
        "--payload",
        required=True,
        help="JSON string with feature values, e.g. '{\"Bedroom\": 3, \"Bathroom\": 2}'",
    )
    args = parser.parse_args()

    payload = json.loads(args.payload)
    prediction = predict_price(payload)
    print(json.dumps({"predicted_price_rm": prediction}, indent=2))


if __name__ == "__main__":
    main()
