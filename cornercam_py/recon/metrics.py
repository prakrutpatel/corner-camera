from __future__ import annotations

import os
import csv
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class ReconMetrics:
    # Lower is usually better (less temporal noise / jitter)
    temporal_abs_diff: float
    temporal_hf_ratio: float

    # Higher is usually better (stronger dynamic signal)
    motion_energy: float
    peakiness: float


def compute_recon_metrics(outframes: np.ndarray) -> ReconMetrics:
    """
    Compute lightweight, model-agnostic proxy metrics for CornerCam outputs.

    outframes expected shape: (T, K, C) or (T, K)
    We treat metrics as heuristic indicators, not ground-truth quality.
    """
    x = outframes.astype(np.float64, copy=False)
    if x.ndim == 2:
        x = x[:, :, None]

    T, K, C = x.shape
    eps = 1e-9

    # 1) Temporal absolute difference (noise/jitter proxy)
    if T >= 2:
        td = np.abs(x[1:] - x[:-1]).mean()
    else:
        td = 0.0

    # 2) Temporal high-frequency ratio
    if T >= 3:
        smooth = np.copy(x)
        smooth[1:-1] = (x[:-2] + x[1:-1] + x[2:]) / 3.0
        residual = x - smooth
        hf_energy = (residual ** 2).mean()
        total_energy = (x ** 2).mean() + eps
        hf_ratio = hf_energy / total_energy
    else:
        hf_ratio = 0.0

    # 3) Motion energy
    motion = x.std(axis=0).mean()

    # 4) Peakiness
    absx = np.abs(x)
    p95 = float(np.percentile(absx, 95))
    med = float(np.median(absx)) + eps
    peak = p95 / med

    return ReconMetrics(
        temporal_abs_diff=float(td),
        temporal_hf_ratio=float(hf_ratio),
        motion_energy=float(motion),
        peakiness=float(peak),
    )


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) < 1e-3:
        return f"{v:.3e}"
    return f"{v:.4f}"


def append_metrics_csv(
    log_path: str,
    *,
    expname: str,
    stage: str,
    noise_preset: str,
    temporal_preset: str,
    lambda_used: float,
    metrics: ReconMetrics,
):
    """
    Append one row to a compact metrics CSV.
    Only essential columns requested by you.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    cols = [
        "expname",
        "stage",
        "noise_preset",
        "temporal_preset",
        "lambda",
        "temporal_abs_diff",
        "temporal_hf_ratio",
        "motion_energy",
        "peakiness",
    ]

    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not file_exists:
            w.writeheader()

        w.writerow({
            "expname": expname,
            "stage": stage,
            "noise_preset": noise_preset,
            "temporal_preset": temporal_preset,
            "lambda": float(lambda_used),
            "temporal_abs_diff": metrics.temporal_abs_diff,
            "temporal_hf_ratio": metrics.temporal_hf_ratio,
            "motion_energy": metrics.motion_energy,
            "peakiness": metrics.peakiness,
        })


def print_recon_metrics(
    tag: str,
    outframes: np.ndarray,
    *,
    baseline: Optional[ReconMetrics] = None,
    extra: Optional[Dict[str, float]] = None,
) -> ReconMetrics:
    """
    Print metrics with direction hints and optional delta vs baseline.
    """
    m = compute_recon_metrics(outframes)

    lines = []
    lines.append(f"\n[METRICS] {tag}")
    lines.append(f"  temporal_abs_diff (↓ cleaner): { _fmt(m.temporal_abs_diff) }")
    lines.append(f"  temporal_hf_ratio (↓ cleaner): { _fmt(m.temporal_hf_ratio) }")
    lines.append(f"  motion_energy     (↑ stronger): { _fmt(m.motion_energy) }")
    lines.append(f"  peakiness         (↑ clearer):  { _fmt(m.peakiness) }")

    if extra:
        for k, v in extra.items():
            lines.append(f"  {k}: { _fmt(float(v)) }")

    if baseline is not None:
        def d(cur, base):
            return cur - base

        lines.append("  Δ vs baseline:")
        lines.append(f"    temporal_abs_diff: { _fmt(d(m.temporal_abs_diff, baseline.temporal_abs_diff)) } (↓ good)")
        lines.append(f"    temporal_hf_ratio: { _fmt(d(m.temporal_hf_ratio, baseline.temporal_hf_ratio)) } (↓ good)")
        lines.append(f"    motion_energy:     { _fmt(d(m.motion_energy, baseline.motion_energy)) } (↑ good)")
        lines.append(f"    peakiness:         { _fmt(d(m.peakiness, baseline.peakiness)) } (↑ good)")

    print("\n".join(lines))
    return m