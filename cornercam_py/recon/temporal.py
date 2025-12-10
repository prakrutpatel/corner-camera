from __future__ import annotations

import numpy as np
from typing import Literal

from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter


TemporalMethod = Literal["none", "gaussian", "median", "savgol", "ewma", "adaptive"]


def _ensure_3d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x[:, :, None]
    return x


def _gaussian_time(x: np.ndarray, sigma: float) -> np.ndarray:
    sigma = float(sigma)
    if sigma <= 0:
        return x
    return gaussian_filter1d(x, sigma=sigma, axis=0, mode="nearest")


def _median_time(x: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    window = max(window, 3)
    if window % 2 == 0:
        window += 1
    # size=(time, angle, chan)
    return median_filter(x, size=(window, 1, 1), mode="nearest")


def _savgol_time(x: np.ndarray, window: int, poly: int) -> np.ndarray:
    window = int(window)
    window = max(window, 5)
    if window % 2 == 0:
        window += 1

    poly = int(poly)
    poly = max(1, min(poly, window - 1))

    T, K, C = x.shape
    y = np.empty_like(x)
    for k in range(K):
        for c in range(C):
            y[:, k, c] = savgol_filter(
                x[:, k, c],
                window_length=window,
                polyorder=poly,
                mode="interp",
            )
    return y


def _ewma_time(x: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.01, 0.99))
    T = x.shape[0]
    y = np.empty_like(x)
    y[0] = x[0]
    for t in range(1, T):
        y[t] = alpha * x[t] + (1 - alpha) * y[t - 1]
    return y


def _base_smooth(x: np.ndarray, params: dict, base_method: str) -> np.ndarray:
    """
    Apply one of the existing fixed temporal filters as the 'base' smoother.
    """
    base_method = (base_method or "gaussian").lower()

    if base_method == "gaussian":
        sigma = float(params.get("temporal_sigma", 1.0))
        return _gaussian_time(x, sigma)

    if base_method == "median":
        window = int(params.get("temporal_window", 7))
        return _median_time(x, window)

    if base_method == "savgol":
        window = int(params.get("temporal_window", 7))
        poly = int(params.get("temporal_savgol_poly", 2))
        return _savgol_time(x, window, poly)

    if base_method == "ewma":
        alpha = float(params.get("temporal_ewma_alpha", 0.25))
        return _ewma_time(x, alpha)

    # Unknown base => no smoothing
    return x


def _motion_proxy(x: np.ndarray) -> np.ndarray:
    """
    Per-time motion proxy based on mean abs temporal difference.
    Returns m shape: (T,)
    """
    T = x.shape[0]
    if T < 2:
        return np.zeros((T,), dtype=np.float64)

    d = np.abs(x[1:] - x[:-1]).mean(axis=(1, 2))
    m = np.empty((T,), dtype=np.float64)
    m[0] = d[0]
    m[1:] = d
    return m


def _robust_norm_to_01(v: np.ndarray) -> np.ndarray:
    """
    Robustly normalize to ~[0,1] using median/MAD.
    Falls back to min-max when MAD is tiny.
    """
    v = v.astype(np.float64, copy=False)
    eps = 1e-12

    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))

    if mad < eps:
        # fallback min-max
        vmin = float(v.min())
        vmax = float(v.max())
        if abs(vmax - vmin) < eps:
            return np.full_like(v, 0.5, dtype=np.float64)
        return np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0)

    # z-score using robust sigma
    z = (v - med) / (1.4826 * mad + eps)
    # clamp to a reasonable range
    z = np.clip(z, -1.5, 1.5)
    return (z + 1.5) / 3.0


def _apply_adaptive(x: np.ndarray, params: dict) -> np.ndarray:
    """
    Adaptive temporal regularization via per-time blending:

        high motion -> keep raw
        low motion  -> use base-smoothed

    Params:
        temporal_adapt_base: "gaussian"|"median"|"savgol"|"ewma" (default "gaussian")
        temporal_adapt_strength: float (default 1.0)
            >1 strengthens adaptiveness; <1 weakens it.
    """
    base = params.get("temporal_adapt_base", "gaussian")
    strength = float(params.get("temporal_adapt_strength", 1.0))

    xs = _base_smooth(x, params, base_method=base)

    m = _motion_proxy(x)
    w = _robust_norm_to_01(m)

    # Strengthen/soften adaptiveness by pushing away/toward 0.5
    if strength != 1.0:
        w = np.clip(0.5 + (w - 0.5) * strength, 0.0, 1.0)

    # Blend (broadcast over K,C)
    w3 = w[:, None, None]
    y = w3 * x + (1.0 - w3) * xs
    return y


def temporal_denoise(outframes: np.ndarray, params: dict) -> np.ndarray:
    """
    Apply temporal denoising to outframes.

    outframes shape: (T, K, C) or (T, K)

    Params:
        params["temporal_denoise"] : bool (default False)
        params["temporal_method"] : str in {"none","gaussian","median","savgol","ewma","adaptive"}

        gaussian:
            params["temporal_sigma"] : float (default 1.0)

        median / savgol:
            params["temporal_window"] : int (default 7)
            params["temporal_savgol_poly"] : int (default 2)

        ewma:
            params["temporal_ewma_alpha"] : float (default 0.25)

        adaptive:
            params["temporal_adapt_base"] : str base method
            params["temporal_adapt_strength"] : float (default 1.0)

    Returns denoised outframes with same shape dimensionality as input.
    """
    if not bool(params.get("temporal_denoise", False)):
        return outframes

    method: TemporalMethod = params.get("temporal_method", "gaussian")
    method = method.lower()

    if method == "none":
        return outframes

    x = outframes.astype(np.float64, copy=False)
    x3 = _ensure_3d(x)

    if method == "gaussian":
        sigma = float(params.get("temporal_sigma", 1.0))
        y = _gaussian_time(x3, sigma)

    elif method == "median":
        window = int(params.get("temporal_window", 5))
        y = _median_time(x3, window)

    elif method == "savgol":
        window = int(params.get("temporal_window", 7))
        poly = int(params.get("temporal_savgol_poly", 2))
        y = _savgol_time(x3, window, poly)

    elif method == "ewma":
        alpha = float(params.get("temporal_ewma_alpha", 0.25))
        y = _ewma_time(x3, alpha)

    elif method == "adaptive":
        y = _apply_adaptive(x3, params)

    else:
        raise ValueError(f"Unknown temporal_method: {method}")

    # restore original dimensionality
    if x.ndim == 2:
        return y[:, :, 0]
    return y
