"""
Training pipeline — Ames Housing dataset, individual house sale prices.
Reads ml/config.json, trains model, calibrates conformal wrapper, saves wrapper.pkl.

Split: 60% train | 20% calibration | 20% test
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.wrapper import UncertaintyWrapper

ML_DIR = Path(__file__).parent
CONFIG_PATH = ML_DIR / "config.json"
SCALER_PATH = ML_DIR / "scaler.pkl"

# The 5 features a house buyer naturally knows
FEATURE_NAMES = ["GrLivArea", "BedroomAbvGr", "FullBath", "OverallQual", "YearBuilt"]

REGRESSION_MODELS = {
    "random_forest":     RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "linear_regression": LinearRegression,
    "ridge":             Ridge,
}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def build_model(cfg: dict):
    cls = REGRESSION_MODELS.get(cfg["model_type"])
    if cls is None:
        raise ValueError(f"Unknown model: {cfg['model_type']}")
    try:
        return cls(random_state=cfg.get("random_state", 42), **cfg.get("model_params", {}))
    except TypeError:
        return cls(**cfg.get("model_params", {}))


def load_dataset():
    print("Fetching Ames Housing dataset...")
    data = fetch_openml("house_prices", version=1, as_frame=True, parser="auto")
    df = data.data[FEATURE_NAMES].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(df.median())
    y = data.target.astype(float).values
    return df.values, y


def train_and_save() -> dict:
    cfg = load_config()
    print(f"Config: task={cfg['task_type']} model={cfg['model_type']} "
          f"confidence={cfg['confidence_level']}")

    X, y = load_dataset()
    print(f"Dataset: {X.shape[0]} houses, features: {FEATURE_NAMES}")

    X_train_cal, X_test, y_train_cal, y_test = train_test_split(
        X, y, test_size=0.20, random_state=cfg.get("random_state", 42)
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_cal, y_train_cal, test_size=0.25, random_state=cfg.get("random_state", 42)
    )
    print(f"Train: {len(X_train)} | Calibration: {len(X_cal)} | Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_cal_s   = scaler.transform(X_cal)
    X_test_s  = scaler.transform(X_test)

    model = build_model(cfg)
    model.fit(X_train_s, y_train)

    alpha = 1.0 - cfg["confidence_level"]
    # Embed scaler in wrapper so inference always scales raw input automatically
    wrapper = UncertaintyWrapper(model, task_type="regression", alpha=alpha, scaler=scaler)
    q = wrapper.calibrate(X_cal_s, y_cal)
    print(f"Conformal margin q = ${q:,.0f}  (covers {cfg['confidence_level']:.0%} of outcomes)")

    preds = model.predict(X_test_s)
    rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
    coverage = float(np.mean((y_test >= preds - q) & (y_test <= preds + q)))
    print(f"Test RMSE: ${rmse:,.0f} | Empirical coverage: {coverage:.3f} (target {cfg['confidence_level']:.2f})")

    wrapper.save()  # scaler is now inside wrapper.pkl — no separate scaler.pkl needed at inference

    print(f"Artifacts saved to {ML_DIR}")
    return {"rmse": rmse, "coverage": coverage, "margin": q}


if __name__ == "__main__":
    train_and_save()
