from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class MotionEstimate:
    direction: str  # "increasing-angle", "decreasing-angle", or "uncertain"
    median_shift_bins_per_step: float
    median_speed_bins_per_sec: float
    median_omega_rad_per_sec: float
    shifts: np.ndarray  # per-time-step estimated shifts


def _to_gray_2d(outframes: np.ndarray) -> np.ndarray:
    x = outframes.astype(np.float64, copy=False)
    if x.ndim == 3:
        # average channels
        x = x.mean(axis=2)
    return x


def estimate_motion_by_correlation(
    outframes: np.ndarray,
    *,
    theta_lim: Optional[Tuple[float, float]] = None,
    fps: Optional[float] = None,
    step: Optional[int] = None,
    max_shift: int = 10,
    use_abs: bool = True,
) -> MotionEstimate:
    """
    Estimate motion direction + speed using 1D shift alignment of successive time slices.

    This operates on the T x K reconstruction matrix.
    We look for the best shift Δk that maximizes correlation between x[t] and x[t+1].

    Args:
        outframes: (T,K) or (T,K,C)
        theta_lim: (theta1, theta2) in radians if you want omega output
        fps: video fps
        step: reconstruction step in video frames
        max_shift: search range in bins
        use_abs: correlate abs signal (often more robust)

    Returns:
        MotionEstimate with bins/step, bins/sec, rad/sec
    """
    x = _to_gray_2d(outframes)
    T, K = x.shape

    if use_abs:
        x = np.abs(x)

    # normalize per row to reduce brightness bias
    x = x - x.mean(axis=1, keepdims=True)
    denom = x.std(axis=1, keepdims=True) + 1e-9
    x = x / denom

    shifts = []

    # brute force small-shift correlation
    for t in range(T - 1):
        a = x[t]
        b = x[t + 1]

        best_s = 0
        best_score = -np.inf

        for s in range(-max_shift, max_shift + 1):
            if s < 0:
                aa = a[-s:]
                bb = b[: K + s]
            elif s > 0:
                aa = a[: K - s]
                bb = b[s:]
            else:
                aa = a
                bb = b

            if aa.size < 8:
                continue

            score = float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-9))
            if score > best_score:
                best_score = score
                best_s = s

        shifts.append(best_s)

    shifts = np.asarray(shifts, dtype=np.float64)

    # robust central tendency
    med_shift = float(np.median(shifts)) if shifts.size else 0.0

    # direction label
    if abs(med_shift) < 0.25:
        direction = "uncertain"
    elif med_shift > 0:
        direction = "increasing-angle"
    else:
        direction = "decreasing-angle"

    # bins/sec
    bins_per_sec = 0.0
    if fps and step:
        dt = float(step) / float(fps)
        bins_per_sec = med_shift / (dt + 1e-9)

    # omega rad/sec
    omega = 0.0
    if theta_lim is not None and fps and step:
        theta1, theta2 = theta_lim
        dtheta = (theta2 - theta1) / max(K, 1)
        dt = float(step) / float(fps)
        omega = bins_per_sec * dtheta  # (bins/sec)*(rad/bin)

    return MotionEstimate(
        direction=direction,
        median_shift_bins_per_step=med_shift,
        median_speed_bins_per_sec=float(bins_per_sec),
        median_omega_rad_per_sec=float(omega),
        shifts=shifts,
    )
