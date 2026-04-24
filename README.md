# KnownUnknowns

**Model-Agnostic Uncertainty-Aware Prediction System**

---

## The Problem: A Number Without a Margin is a Guess

A model that predicts "$420,000" for a house gives you an answer, but not the honest one. Two neighbourhoods can both receive "$420,000" — one where the model has seen thousands of similar blocks and is quietly confident to within $40k, another where the features are unusual and the true value could be anywhere between $250k and $600k.

Accuracy metrics measure average performance across a test set. They say nothing about the uncertainty of *any individual prediction*. In real estate, finance, medicine, and safety-critical systems, knowing *how much to trust* a prediction is as important as the prediction itself.

---

## The Solution: Conformal Prediction

Conformal prediction wraps any trained model and produces a **coverage-guaranteed interval** (regression) or **prediction set** (classification) — a range of outcomes that is statistically guaranteed to contain the true value with a user-specified probability (default: 95%).

The guarantee is **non-parametric and distribution-free**: no assumptions about the data distribution are needed.

### How it works — Regression

1. **Train** a model on 60% of the data (model never sees calibration or test splits).
2. **Calibrate** on a separate 20% hold-out:
   - Compute absolute residuals: `score = |true_price − predicted_price|`
   - Find the 95th-percentile residual → **margin q**
3. **At test time**: interval = `[prediction − q, prediction + q]`

Across all future predictions, at least 95% of intervals will contain the true value. This is a hard statistical guarantee.

### How it works — Classification (kept from v1)

1. Calibration score = `1 − P(true_class | x)` per calibration sample.
2. Find (1−α) quantile → threshold **τ**.
3. At test time: include class `c` in prediction set if `1 − P(c|x) ≤ τ`.

### Interval interpretation

| Interval width | Meaning |
|---------------|---------|
| Narrow (< $80k) | Model is confident — neighbourhood is well-represented in training data |
| Moderate ($80k–$160k) | Typical uncertainty — use the point estimate with normal caution |
| Wide (> $160k) | Model is uncertain — features are unusual or conflicting; seek additional signals |

---

## Architecture

```
KnownUnknowns/
├── ml/
│   ├── config.json       ← task_type, model_type, confidence_level, model_params
│   ├── wrapper.py        ← UncertaintyWrapper (model-agnostic conformal layer)
│   ├── train.py          ← reads config, trains model, calibrates wrapper, saves wrapper.pkl
│   ├── conformal.py      ← re-exports UncertaintyWrapper for backwards compatibility
│   ├── wrapper.pkl       ← serialised trained + calibrated wrapper (generated)
│   └── scaler.pkl        ← feature scaler (generated)
│
├── app/
│   ├── main.py           ← FastAPI routes
│   ├── models.py         ← business logic (thin translation layer)
│   ├── schemas.py        ← Pydantic request / response models
│   └── utils.py          ← singleton wrapper loader, logging
│
├── dashboard/
│   └── streamlit_app.py  ← UI: sliders → API → prediction + interval visualisation
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Train + calibrate

```bash
python -m ml.train
```

Downloads California Housing (sklearn), trains a RandomForest, calibrates the conformal margin, and saves `ml/wrapper.pkl` + `ml/scaler.pkl`.

Sample output:
```
Train: 12384 | Calibration: 4128 | Test: 4129
Conformal margin q = 0.4812  (covers 95% of outcomes)
Test RMSE: 0.4601 | Empirical coverage: 0.9504 (target 0.95)
```

### 3. Start the API

```bash
uvicorn app.main:app --reload
```

→ `http://localhost:8000`  
→ Interactive docs: `http://localhost:8000/docs`

### 4. Start the dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

→ `http://localhost:8501`

---

## Docker

```bash
docker build -t known-unknowns .
docker run -p 8000:8000 known-unknowns
```

---

## API Reference

### `POST /predict`

Point prediction only.

**Request:**
```json
{
  "features": {
    "MedInc": 3.5, "HouseAge": 25, "AveRooms": 5.5,
    "AveBedrms": 1.1, "Population": 900, "AveOccup": 2.8,
    "Latitude": 37.77, "Longitude": -122.42
  }
}
```

**Response:**
```json
{
  "prediction": 2.4731,
  "prediction_usd": "$247,310"
}
```

---

### `POST /predict_with_uncertainty`

Point prediction + conformal interval.

**Response:**
```json
{
  "prediction": 2.4731,
  "prediction_usd": "$247,310",
  "lower_bound": 1.9919,
  "upper_bound": 2.9543,
  "lower_bound_usd": "$199,190",
  "upper_bound_usd": "$295,430",
  "interval_width": 0.9624,
  "margin": 0.4812,
  "confidence_level": 0.95
}
```

The `[lower_bound, upper_bound]` interval is guaranteed to contain the true median house value ≥ 95% of the time.

---

## Swapping the Model

Edit `ml/config.json` and re-run `python -m ml.train`. No other changes needed.

```json
{
  "task_type": "regression",
  "model_type": "gradient_boosting",
  "confidence_level": 0.95,
  "model_params": {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05
  }
}
```

Available `model_type` values: `random_forest`, `gradient_boosting`, `linear_regression`, `ridge`.

---

## Bring Your Own Model

Train any sklearn-compatible model externally, then wrap it:

```python
import pickle
import numpy as np
from ml.wrapper import UncertaintyWrapper

# Load your own model
with open("my_model.pkl", "rb") as f:
    my_model = pickle.load(f)

# Load calibration data (X_cal must be already scaled)
X_cal = np.load("my_calibration_X.npy")
y_cal = np.load("my_calibration_y.npy")

# Wrap, calibrate, save
wrapper = UncertaintyWrapper(my_model, task_type="regression", alpha=0.05)
q = wrapper.calibrate(X_cal, y_cal)
print(f"Conformal margin: {q:.4f}")
wrapper.save()  # writes ml/wrapper.pkl — API picks it up automatically
```

Requirements for the custom model:
- Implements `predict(X)` → array of shape `(n,)`
- For classification: also implements `predict_proba(X)` → array of shape `(n, n_classes)`

---

## Features (California Housing)

| Feature | Description |
|---------|-------------|
| MedInc | Median household income in block group ($10k units) |
| HouseAge | Median house age in years |
| AveRooms | Average rooms per household |
| AveBedrms | Average bedrooms per household |
| Population | Block group population |
| AveOccup | Average household size |
| Latitude | Block group latitude |
| Longitude | Block group longitude |

**Target:** Median house value in $100,000 units (so `2.5` = $250,000).

---

*For educational and research purposes only. Not financial or real estate advice.*
