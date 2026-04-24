"""Shared utilities: wrapper loading, logging setup."""

import logging
import sys
from functools import lru_cache
from pathlib import Path

from ml.wrapper import UncertaintyWrapper

ML_DIR = Path(__file__).parent.parent / "ml"


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("known_unknowns")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@lru_cache(maxsize=1)
def get_wrapper() -> UncertaintyWrapper:
    """Load the calibrated wrapper once and cache it for the process lifetime."""
    wrapper_path = ML_DIR / "wrapper.pkl"
    if not wrapper_path.exists():
        raise RuntimeError(
            "wrapper.pkl not found. Run `python -m ml.train` to train and calibrate."
        )
    return UncertaintyWrapper.load(wrapper_path)
