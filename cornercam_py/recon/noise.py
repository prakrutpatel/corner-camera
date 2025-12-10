from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from ..preprocess.frame import preprocess_frame
from .obs import get_obs_vec

# Optional time-adaptive support
try:
    from ..utils.video import get_video_props, read_frame_at_index
except Exception:
    get_video_props = None
    read_frame_at_index = None


def _flatten(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _mad(x: np.ndarray, eps: float = 1e-12) -> float:
    """
    Median absolute deviation (robust scale).
    Returns MAD (not scaled-to-sigma).
    """
    x = _flatten(x)
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + eps)


def _trimmed_mean(x: np.ndarray, trim_frac: float = 0.1) -> float:
    """
    Compute trimmed mean of values (robust to outliers).
    trim_frac=0.1 removes 10% low and 10% high.
    """
    x = _flatten(x)
    if x.size == 0:
        return 0.0
    trim_frac = float(np.clip(trim_frac, 0.0, 0.49))
    if trim_frac <= 0:
        return float(np.mean(x))
    x_sorted = np.sort(x)
    k = int(np.floor(trim_frac * x_sorted.size))
    if 2 * k >= x_sorted.size:
        return float(np.mean(x_sorted))
    x_cut = x_sorted[k:-k]
    return float(np.mean(x_cut))


def _legacy_estimator(var_img_proc: np.ndarray) -> float:
    """
    Closest to the original MATLAB intent:
        median(median(mean(input_fmt,3)))
    where input_fmt is preprocessed variance.
    """
    if var_img_proc.ndim == 3:
        mean_chan = var_img_proc.mean(axis=2)
    else:
        mean_chan = var_img_proc
    # double median
    return float(np.median(np.median(mean_chan, axis=0)))


def _global_pool(var_img_proc: np.ndarray) -> np.ndarray:
    """
    Collapse channel dimension into a single 2D noise field
    for consistent scalar estimation.
    """
    if var_img_proc.ndim == 3:
        return var_img_proc.mean(axis=2)
    return var_img_proc


def _obs_pool(var_img_proc: np.ndarray, params: dict) -> np.ndarray:
    """
    Sample preprocessed variance at observation locations.
    We reuse get_obs_vec which already interpolates.
    Returns vector of sampled values (M,).
    """
    # get_obs_vec expects obs_xlocs_proc/obs_ylocs_proc to exist
    samples, nanrows = get_obs_vec(var_img_proc, params)
    # samples is (M, C) or (M, 1)
    if samples.ndim == 2:
        vals = samples.mean(axis=1)
    else:
        vals = samples.reshape(-1)
    return vals


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute weighted median of values.
    """
    v = _flatten(values)
    w = _flatten(weights)
    if v.size == 0:
        return 0.0
    # sort by value
    idx = np.argsort(v)
    v = v[idx]
    w = w[idx]
    wsum = np.sum(w)
    if wsum <= 0:
        return float(np.median(v))
    cdf = np.cumsum(w) / wsum
    j = int(np.searchsorted(cdf, 0.5))
    j = min(max(j, 0), v.size - 1)
    return float(v[j])


def estimate_frame_noise(
    params: dict,
    *,
    srcfile: Optional[str] = None,
) -> float:
    """
    Pluggable noise estimator.

    Required:
      params["mean_datafile"] -> .npz with "variance" and "back_img".

    Optional flags (all with safe defaults):
      params["noise_model"] : str
          "legacy" (default)
          "mad"
          "trimmed"
          "obs_legacy"
          "obs_mad"
          "obs_trimmed"
          "weighted_obs"

      params["noise_trim_frac"] : float (default 0.1)

      params["noise_use_obs_points"] : bool (default False)
          If True, automatically maps global models to obs_* equivalents.

      params["noise_weighted_eps"] : float (default 1e-6)

      params["noise_time_adaptive"] : bool (default False)
          Uses sample frames from the actual sequence to estimate noise
          on observation residuals relative to background.

      params["noise_time_samples"] : int (default 8)
    """
    
    noise_model = params.get("noise_model", "legacy")
    trim_frac = float(params.get("noise_trim_frac", 0.1))
    use_obs = bool(params.get("noise_use_obs_points", False))
    w_eps = float(params.get("noise_weighted_eps", 1e-6))
    print(f"Using noise model: {noise_model}")

    time_adapt = bool(params.get("noise_time_adaptive", False))
    time_samples = int(params.get("noise_time_samples", 8))

    # If time-adaptive is requested and we can do it, do it first.
    if time_adapt and srcfile and read_frame_at_index is not None:
        try:
            return estimate_noise_time_adaptive(
                srcfile, params, nsamples=time_samples
            )
        except Exception as e:
            print(f"[WARN] time-adaptive noise failed, falling back to static. Reason: {e}")

    d = np.load(params["mean_datafile"])
    variance = d["variance"].astype(np.float64)

    # Preprocess variance exactly like frames
    var_img_proc = preprocess_frame(variance, params)

    # Auto-map global models to obs variants if requested
    if use_obs and not noise_model.startswith("obs_") and noise_model != "weighted_obs":
        noise_model = "obs_" + noise_model

    # --- Models ---
    if noise_model == "legacy":
        return _legacy_estimator(var_img_proc)

    if noise_model == "mad":
        field = _global_pool(var_img_proc)
        return _mad(field)

    if noise_model == "trimmed":
        field = _global_pool(var_img_proc)
        return _trimmed_mean(field, trim_frac)

    # Observation-limited variants
    if noise_model == "obs_legacy":
        vals = _obs_pool(var_img_proc, params)
        # mimic legacy by median of sampled values
        return float(np.median(vals))

    if noise_model == "obs_mad":
        vals = _obs_pool(var_img_proc, params)
        return _mad(vals)

    if noise_model == "obs_trimmed":
        vals = _obs_pool(var_img_proc, params)
        return _trimmed_mean(vals, trim_frac)

    # Spatially weighted in observation region
    # Idea: downweight high-variance outliers (glare/edges)
    # weights = 1 / (val + eps)
    if noise_model == "weighted_obs":
        vals = _obs_pool(var_img_proc, params)
        weights = 1.0 / (vals + w_eps)
        return _weighted_median(vals, weights)

    raise ValueError(f"Unknown noise_model: {noise_model}")


def estimate_noise_time_adaptive(
    srcfile: str,
    params: dict,
    *,
    nsamples: int = 8,
) -> float:
    """
    Time-adaptive noise estimate using real frames.

    We sample frames between startframe and endframe (if present),
    compute residuals relative to background, and estimate robust
    scale on observation samples.

    This is more expensive but can help when:
      - lighting flickers,
      - phone auto-exposure drifts,
      - the background model isn't stable.

    Requires:
      read_frame_at_index, preprocess_frame
    """
    if read_frame_at_index is None:
        raise RuntimeError("time-adaptive noise requires video helpers")

    d = np.load(params["mean_datafile"])
    back_img = d["back_img"].astype(np.float64)
    back_img = preprocess_frame(back_img, params)

    # Determine frame range
    startf = int(params.get("startframe", 0))
    endf = int(params.get("endframe", startf + 1))
    step = max(int(params.get("step", 1)), 1)

    if endf <= startf:
        endf = startf + step

    # Uniform samples over the interval
    nsamples = max(nsamples, 2)
    frameidx = np.linspace(startf, endf, nsamples).round().astype(int)

    residual_vals = []
    for fidx in frameidx:
        fr = read_frame_at_index(srcfile, int(fidx)).astype(np.float64)
        fr = preprocess_frame(fr, params)

        # residual in observation space
        y, _ = get_obs_vec(fr - back_img, params)

        if y.ndim == 2:
            residual_vals.append(y.mean(axis=1))
        else:
            residual_vals.append(y.reshape(-1))

    if not residual_vals:
        return 0.0

    residual_vals = np.concatenate(residual_vals, axis=0)

    # Robust scale from residuals
    return _mad(residual_vals)
