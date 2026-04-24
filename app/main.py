"""
KnownUnknowns — Model-Agnostic Uncertainty-Aware Prediction API
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import run_predict, run_predict_with_uncertainty
from app.schemas import HealthResponse, PredictRequest, PredictResponse, UncertaintyResponse
from app.utils import get_wrapper, setup_logging

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_wrapper()
        logger.info("UncertaintyWrapper loaded successfully.")
    except RuntimeError as e:
        logger.error("Startup failed: %s", e)
    yield


app = FastAPI(
    title="KnownUnknowns — Uncertainty-Aware Prediction API",
    description=(
        "House price prediction with conformal prediction intervals. "
        "Every /predict_with_uncertainty response carries a statistically "
        "valid interval guaranteed to contain the true value ≥ 95% of the time."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    try:
        w = get_wrapper()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            calibrated=w._q is not None,
            task_type=w.task_type,
        )
    except RuntimeError:
        return HealthResponse(status="degraded", model_loaded=False, calibrated=False, task_type="unknown")


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(request: PredictRequest):
    """Point prediction — returns the model's best single estimate."""
    try:
        return run_predict(request.features)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        logger.exception("Unexpected error in /predict")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/predict_with_uncertainty", response_model=UncertaintyResponse, tags=["prediction"])
def predict_with_uncertainty(request: PredictRequest):
    """
    Point prediction + conformal prediction interval.

    The [lower_bound, upper_bound] interval is guaranteed to contain the true
    house value with probability >= confidence_level (default 95%).

    A narrow interval means the model is confident about this neighbourhood's
    price. A wide interval means the features are unusual or under-represented
    in the training data — treat the point estimate with more caution.
    """
    try:
        return run_predict_with_uncertainty(request.features)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        logger.exception("Unexpected error in /predict_with_uncertainty")
        raise HTTPException(status_code=500, detail="Internal server error")
