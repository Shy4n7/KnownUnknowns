import logging
import numpy as np

from app.utils import get_wrapper
from app.schemas import HouseFeatures, PredictResponse, UncertaintyResponse

logger = logging.getLogger("known_unknowns")


def _fmt(value: float) -> str:
    return f"${value:,.0f}"


def run_predict(features: HouseFeatures) -> PredictResponse:
    wrapper = get_wrapper()
    result = wrapper.predict(np.array(features.to_array()))
    pred = result["prediction"]
    logger.info("predict | $%.0f", pred)
    return PredictResponse(prediction=round(pred, 2), prediction_usd=_fmt(pred))


def run_predict_with_uncertainty(features: HouseFeatures) -> UncertaintyResponse:
    wrapper = get_wrapper()
    result = wrapper.predict_with_uncertainty(np.array(features.to_array()))
    pred, lower, upper = result["prediction"], result["lower_bound"], result["upper_bound"]
    logger.info("predict_with_uncertainty | $%.0f  [$%.0f - $%.0f]", pred, lower, upper)
    return UncertaintyResponse(
        prediction=round(pred, 2),
        prediction_usd=_fmt(pred),
        lower_bound=round(lower, 2),
        upper_bound=round(upper, 2),
        lower_bound_usd=_fmt(lower),
        upper_bound_usd=_fmt(upper),
        interval_width=result["interval_width"],
        margin=result["margin"],
        confidence_level=result["confidence_level"],
    )
