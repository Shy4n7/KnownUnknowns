"""
UncertaintyWrapper — model-agnostic conformal prediction layer.

Works with any sklearn-style model:
  - regression:     predict(X) → scalar
  - classification: predict_proba(X) → class probabilities

Conformal prediction primer
───────────────────────────
The idea: use a held-out calibration set (never seen during training) to
learn how wrong the model typically is, then turn that into a guarantee.

Regression (split conformal):
  1. Run model on calibration set; compute residuals |y_true - y_pred|.
  2. Find the (1-alpha) quantile of those residuals → margin q.
  3. At test time: interval = [pred - q, pred + q].
     This interval contains the true value with probability >= (1 - alpha).

Classification (RAPS-style):
  1. Compute nonconformity score = 1 - P(true_class | x) on calibration set.
  2. Find the (1-alpha) quantile of those scores → threshold tau.
  3. At test time: include class c if 1 - P(c|x) <= tau.
     The resulting set contains the true class with probability >= (1-alpha).

Both methods share the same finite-sample correction:
  quantile level = ceil((n+1)(1-alpha)) / n
which ensures exact (not just asymptotic) coverage on finite datasets.
"""

import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np

ML_DIR = Path(__file__).parent
WRAPPER_PATH = ML_DIR / "wrapper.pkl"


class UncertaintyWrapper:
    """
    Drop-in conformal prediction wrapper for any sklearn-style model.

    Parameters
    ----------
    model : any object implementing predict(X), and optionally predict_proba(X)
    task_type : "regression" or "classification"
    alpha : miscoverage level (default 0.05 → 95% coverage guarantee)
    """

    def __init__(
        self,
        model: Any,
        task_type: Literal["regression", "classification"],
        alpha: float = 0.05,
        scaler: Any = None,
    ):
        self.model = model
        self.task_type = task_type
        self.alpha = alpha
        self.scaler = scaler  # optional StandardScaler; applied before every predict
        self._q: float | None = None  # conformal quantile (regression margin or clf threshold)

    # ------------------------------------------------------------------
    # Calibration — must be called once before predict_with_uncertainty
    # ------------------------------------------------------------------

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> float:
        """
        Fit the conformal quantile on the calibration set.
        Returns q so callers can log/inspect it.
        """
        if self.task_type == "regression":
            self._q = self._calibrate_regression(X_cal, y_cal)
        else:
            self._q = self._calibrate_classification(X_cal, y_cal)
        return self._q

    def _calibrate_regression(self, X_cal: np.ndarray, y_cal: np.ndarray) -> float:
        preds = self.model.predict(X_cal)
        # Nonconformity score: absolute residual
        scores = np.abs(y_cal - preds)
        return float(self._conformal_quantile(scores))

    def _calibrate_classification(self, X_cal: np.ndarray, y_cal: np.ndarray) -> float:
        proba = self.model.predict_proba(X_cal)
        # Nonconformity score: 1 - P(true class)
        scores = 1.0 - proba[np.arange(len(y_cal)), y_cal.astype(int)]
        return float(self._conformal_quantile(scores))

    def _conformal_quantile(self, scores: np.ndarray) -> float:
        """Finite-sample corrected quantile: ceil((n+1)(1-alpha)) / n."""
        n = len(scores)
        level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        return float(np.quantile(scores, level))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _scale(self, x2d: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            return self.scaler.transform(x2d)
        return x2d

    def predict(self, x: np.ndarray) -> dict:
        """Point prediction — no uncertainty."""
        x2d = self._scale(x.reshape(1, -1))
        if self.task_type == "regression":
            return self._predict_regression(x2d)
        return self._predict_classification(x2d)

    def predict_with_uncertainty(self, x: np.ndarray) -> dict:
        """Point prediction + conformal interval/set."""
        if self._q is None:
            raise RuntimeError("Wrapper not calibrated. Call calibrate() first.")
        x2d = self._scale(x.reshape(1, -1))
        if self.task_type == "regression":
            return self._predict_regression_with_interval(x2d)
        return self._predict_classification_with_set(x2d)

    # --- Regression helpers -------------------------------------------

    def _predict_regression(self, x2d: np.ndarray) -> dict:
        pred = float(self.model.predict(x2d)[0])
        return {"prediction": round(pred, 4)}

    def _predict_regression_with_interval(self, x2d: np.ndarray) -> dict:
        pred = float(self.model.predict(x2d)[0])
        # Symmetric interval: prediction ± q
        lower = pred - self._q
        upper = pred + self._q
        return {
            "prediction": round(pred, 4),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "interval_width": round(upper - lower, 4),
            "margin": round(self._q, 4),
            "confidence_level": round(1.0 - self.alpha, 4),
        }

    # --- Classification helpers ---------------------------------------

    def _predict_classification(self, x2d: np.ndarray) -> dict:
        proba = self.model.predict_proba(x2d)[0]
        label = int(np.argmax(proba))
        return {
            "prediction": label,
            "probability": round(float(proba[label]), 4),
            "probabilities": [round(float(p), 4) for p in proba],
        }

    def _predict_classification_with_set(self, x2d: np.ndarray) -> dict:
        proba = self.model.predict_proba(x2d)[0]
        label = int(np.argmax(proba))

        # Include class c if its nonconformity score <= tau
        prediction_set = [int(c) for c, p in enumerate(proba) if (1.0 - p) <= self._q]
        if not prediction_set:
            prediction_set = [label]

        return {
            "prediction": label,
            "prediction_set": prediction_set,
            "probabilities": [round(float(p), 4) for p in proba],
            "uncertainty": round(float(1.0 - proba[label]), 4),
            "tau": round(self._q, 4),
            "confidence_level": round(1.0 - self.alpha, 4),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path = WRAPPER_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path = WRAPPER_PATH) -> "UncertaintyWrapper":
        with open(path, "rb") as f:
            return pickle.load(f)
