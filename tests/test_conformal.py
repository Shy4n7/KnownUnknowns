"""
Unit tests for UncertaintyWrapper — conformal prediction logic.
These run without a trained model by constructing a minimal mock.
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from ml.wrapper import UncertaintyWrapper


def _make_wrapper(alpha: float = 0.05) -> tuple[UncertaintyWrapper, np.ndarray, np.ndarray]:
    """Train a tiny LinearRegression and return a calibrated wrapper + test data."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 3))
    y = 3 * X[:, 0] + 2 * X[:, 1] + rng.standard_normal(200) * 0.5

    X_train, X_cal, X_test = X[:100], X[100:150], X[150:]
    y_train, y_cal, y_test = y[:100], y[100:150], y[150:]

    model = LinearRegression().fit(X_train, y_train)
    wrapper = UncertaintyWrapper(model, task_type="regression", alpha=alpha)
    wrapper.calibrate(X_cal, y_cal)
    return wrapper, X_test, y_test


# ── Calibration ───────────────────────────────────────────────────────────────

def test_calibrate_sets_q():
    wrapper, _, _ = _make_wrapper()
    assert wrapper._q is not None
    assert wrapper._q > 0


def test_calibrate_q_grows_with_confidence():
    w90, _, _ = _make_wrapper(alpha=0.10)
    w99, _, _ = _make_wrapper(alpha=0.01)
    assert w99._q > w90._q


# ── predict ───────────────────────────────────────────────────────────────────

def test_predict_returns_scalar():
    wrapper, X_test, _ = _make_wrapper()
    result = wrapper.predict(X_test[0])
    assert "prediction" in result
    assert isinstance(result["prediction"], float)


# ── predict_with_uncertainty ──────────────────────────────────────────────────

def test_interval_contains_prediction():
    wrapper, X_test, _ = _make_wrapper()
    for x in X_test[:10]:
        r = wrapper.predict_with_uncertainty(x)
        assert r["lower_bound"] <= r["prediction"] <= r["upper_bound"]


def test_interval_width_equals_twice_margin():
    wrapper, X_test, _ = _make_wrapper()
    r = wrapper.predict_with_uncertainty(X_test[0])
    assert abs(r["interval_width"] - 2 * r["margin"]) < 1e-3


def test_empirical_coverage_meets_guarantee():
    """
    The core conformal guarantee: coverage on test set >= (1 - alpha).
    We use a generous tolerance (3%) to account for finite sample size.
    """
    alpha = 0.05
    wrapper, X_test, y_test = _make_wrapper(alpha=alpha)

    covered = sum(
        wrapper.predict_with_uncertainty(x)["lower_bound"] <= y
        <= wrapper.predict_with_uncertainty(x)["upper_bound"]
        for x, y in zip(X_test, y_test)
    )
    coverage = covered / len(y_test)
    assert coverage >= (1 - alpha) - 0.03, f"Coverage {coverage:.3f} below target {1-alpha}"


def test_uncalibrated_wrapper_raises():
    model = LinearRegression().fit([[1], [2]], [1, 2])
    wrapper = UncertaintyWrapper(model, task_type="regression")
    with pytest.raises(RuntimeError, match="not calibrated"):
        wrapper.predict_with_uncertainty(np.array([1.0]))
