"""
conformal.py — thin re-export kept for backwards compatibility.

The full conformal logic now lives in ml/wrapper.py (UncertaintyWrapper).
Import from there directly; this module is only here so any external
scripts that did `from ml.conformal import ...` don't break immediately.
"""

from ml.wrapper import UncertaintyWrapper

__all__ = ["UncertaintyWrapper"]
