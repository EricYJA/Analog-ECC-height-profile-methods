"""Argument checks shared by the public Python and native interfaces."""

from numbers import Integral, Real

import numpy as np


def matrix(value, name="G", *, parity=False):
    """Normalize real, finite matrix input without changing caller data."""
    try:
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raise ValueError(f"{name} must be a real matrix.")
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real numeric matrix.") from exc
    if result.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    rows, cols = result.shape
    if cols == 0 or (rows == 0 and not parity):
        raise ValueError(f"{name} must be nonempty.")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values.")
    return result


def height_index(value, n, *, allow_none=False, jiang=False):
    if value is None and allow_none:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("m must be an integer" + (" or None." if allow_none else "."))
    value = int(value)
    lower, upper = (1, min(30, n - 1)) if jiang else (0, n - 1)
    if not lower <= value <= upper:
        raise ValueError(f"m must satisfy {lower} <= m <= {upper}.")
    return value


def threshold(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("early_quit_threshold must be a real number.")
    value = float(value)
    if np.isnan(value):
        raise ValueError("early_quit_threshold must not be NaN.")
    return value


def tolerance(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("tol must be a real number.")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError("tol must be finite and nonnegative.")
    return value


def thread_count(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("num_threads must be a positive integer or None.")
    if value < 1:
        raise ValueError("num_threads must be a positive integer or None.")
    return int(value)
