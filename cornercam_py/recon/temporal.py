from __future__ import annotations

import numpy as np

from typing import Literal, Optional

from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter


TemporalMethod = Literal["none", "gaussian", "median", "savgol", "ewma"]


def temporal_denoise(outframes: np.ndarray, params: dict) -> np.ndarray:
    """
    Apply temporal denoising to outframes.

    outframes shape: (T, K, C) where:
        T = time index
        K = hidden angle bins (nsamples)
        C = channels (1 or 3)

    Params:
        params["temporal_denoise"] : bool (default False)
        params["temporal_method"] : str in {"none","gaussian","median","savgol","ewma"}
        params["temporal_sigma"] : float (gaussian default 1.0)
        params["temporal_window"] : int (median/savgol default 5)
        params["temporal_savgol_poly"] : int (default 2)
        params["temporal_ewma_alpha"] : float (default 0.25)

    Returns denoised outframes with same shape.
    """
    if not bool(params.get("temporal_denoise", False)):
        return outframes

    method: TemporalMethod = params.get("temporal_method", "gaussian")
    method = method.lower()

    if method == "none":
        return outframes

    x = outframes.astype(np.float64, copy=False)

    # Ensure 3D (T,K,C)
    if x.ndim == 2:
        x = x[:, :, None]

    T, K, C = x.shape

    if method == "gaussian":
        sigma = float(params.get("temporal_sigma", 1.0))
        # gaussian over time axis only
        y = gaussian_filter1d(x, sigma=sigma, axis=0, mode="nearest")
        return y

    if method == "median":
        window = int(params.get("temporal_window", 5))
        window = max(window, 3)
        if window % 2 == 0:
            window += 1
        # median filter over time only
        # size=(time, angle, chan)
        y = median_filter(x, size=(window, 1, 1), mode="nearest")
        return y

    if method == "savgol":
        window = int(params.get("temporal_window", 7))
        window = max(window, 5)
        if window % 2 == 0:
            window += 1
        poly = int(params.get("temporal_savgol_poly", 2))
        poly = min(poly, window - 1)

        # apply per (K,C)
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

    if method == "ewma":
        alpha = float(params.get("temporal_ewma_alpha", 0.25))
        alpha = float(np.clip(alpha, 0.01, 0.99))

        y = np.empty_like(x)
        y[0] = x[0]
        for t in range(1, T):
            y[t] = alpha * x[t] + (1 - alpha) * y[t - 1]
        return y

    raise ValueError(f"Unknown temporal_method: {method}")
