"""
API integration tests.

These tests run against the real trained model (wrapper.pkl must exist).
Run: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# A typical mid-range house used as a baseline across tests
TYPICAL_HOUSE = {
    "GrLivArea":    1500.0,
    "BedroomAbvGr": 3.0,
    "FullBath":     2.0,
    "OverallQual":  6.0,
    "YearBuilt":    1990.0,
}

SMALL_OLD_HOUSE = {
    "GrLivArea":    700.0,
    "BedroomAbvGr": 2.0,
    "FullBath":     1.0,
    "OverallQual":  3.0,
    "YearBuilt":    1945.0,
}

LARGE_NEW_HOUSE = {
    "GrLivArea":    3500.0,
    "BedroomAbvGr": 5.0,
    "FullBath":     4.0,
    "OverallQual":  9.0,
    "YearBuilt":    2005.0,
}


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["calibrated"] is True
    assert body["task_type"] == "regression"


def test_health_includes_model_metadata():
    r = client.get("/health")
    body = r.json()
    assert "features" in body
    assert "margin" in body
    assert "confidence_level" in body
    assert len(body["features"]) == 5


# ── /predict ──────────────────────────────────────────────────────────────────

def test_predict_returns_positive_price():
    r = client.post("/predict", json={"features": TYPICAL_HOUSE})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] > 0
    assert body["prediction_usd"].startswith("$")


def test_predict_larger_house_costs_more():
    small = client.post("/predict", json={"features": SMALL_OLD_HOUSE}).json()
    large = client.post("/predict", json={"features": LARGE_NEW_HOUSE}).json()
    assert large["prediction"] > small["prediction"]


def test_predict_higher_quality_costs_more():
    low = TYPICAL_HOUSE | {"OverallQual": 2.0}
    high = TYPICAL_HOUSE | {"OverallQual": 9.0}
    r_low  = client.post("/predict", json={"features": low}).json()
    r_high = client.post("/predict", json={"features": high}).json()
    assert r_high["prediction"] > r_low["prediction"]


def test_predict_rejects_missing_field():
    bad = {k: v for k, v in TYPICAL_HOUSE.items() if k != "GrLivArea"}
    r = client.post("/predict", json={"features": bad})
    assert r.status_code == 422


def test_predict_rejects_out_of_range_sqft():
    bad = TYPICAL_HOUSE | {"GrLivArea": 99999.0}
    r = client.post("/predict", json={"features": bad})
    assert r.status_code == 422


# ── /predict_with_uncertainty ─────────────────────────────────────────────────

def test_uncertainty_response_shape():
    r = client.post("/predict_with_uncertainty", json={"features": TYPICAL_HOUSE})
    assert r.status_code == 200
    body = r.json()
    for key in ("prediction", "lower_bound", "upper_bound", "margin", "confidence_level",
                "prediction_usd", "lower_bound_usd", "upper_bound_usd", "interval_width"):
        assert key in body, f"Missing key: {key}"


def test_uncertainty_interval_contains_prediction():
    body = client.post("/predict_with_uncertainty", json={"features": TYPICAL_HOUSE}).json()
    assert body["lower_bound"] <= body["prediction"] <= body["upper_bound"]


def test_uncertainty_interval_width_equals_twice_margin():
    body = client.post("/predict_with_uncertainty", json={"features": TYPICAL_HOUSE}).json()
    assert abs(body["interval_width"] - 2 * body["margin"]) < 1.0


def test_uncertainty_confidence_level_is_95():
    body = client.post("/predict_with_uncertainty", json={"features": TYPICAL_HOUSE}).json()
    assert body["confidence_level"] == pytest.approx(0.95)


def test_uncertainty_larger_house_higher_price():
    small = client.post("/predict_with_uncertainty", json={"features": SMALL_OLD_HOUSE}).json()
    large = client.post("/predict_with_uncertainty", json={"features": LARGE_NEW_HOUSE}).json()
    assert large["prediction"] > small["prediction"]


def test_uncertainty_margin_is_consistent():
    """Conformal margin is fixed at calibration — must be identical across predictions."""
    r1 = client.post("/predict_with_uncertainty", json={"features": TYPICAL_HOUSE}).json()
    r2 = client.post("/predict_with_uncertainty", json={"features": LARGE_NEW_HOUSE}).json()
    assert r1["margin"] == r2["margin"]
