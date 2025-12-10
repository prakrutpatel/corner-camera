from __future__ import annotations

import os
import shutil
import copy
from typing import List, Tuple
import pandas as pd
from .test_corner import test_corner


NOISE_PRESETS = [
    "baseline",
    "robust_global",
    "robust_obs",
    "weighted_obs",
    "time_adapt_obs",
]

TEMPORAL_PRESETS = [
    "none",
    "gaussian",
    "median",
    "savgol",
    "ewma",
]


def _get_run_name(exp_module) -> str:
    # Prefer explicit run_name if you add it to example params
    rn = getattr(exp_module, "run_name", None)
    if rn:
        return rn

    # fallback to module filename like "outdoor_bricks"
    mod = getattr(exp_module, "__name__", "experiment")
    return mod.split(".")[-1]


def prepare_out_folder(out_root: str, run_name: str, clear: bool = True) -> str:
    out_dir = os.path.join(out_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    if clear:
        # wipe only contents of this experiment folder
        for item in os.listdir(out_dir):
            p = os.path.join(out_dir, item)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    os.remove(p)
                except OSError:
                    pass
    return out_dir


def sweep_corner(
    *,
    datafolder: str,
    exp_module,
    out_root: str,
    debug: bool = False,
    sampling: str = "rays",
    start_time: float = 2.0,
    end_time: float = 22.0,
    step: int = 6,
    clear_output: bool = True,
    noise_presets: List[str] = None,
    temporal_presets: List[str] = None,
    mode: str = "full",  # "full" or "separate"
):
    """
    Runs a systematic sweep and saves all outputs in one clean folder.

    mode:
      - "full": baseline + noise-only + temporal-only + all combinations
      - "separate": only baseline + each knob separately (no cross-product)
    """

    noise_presets = noise_presets or list(NOISE_PRESETS)
    temporal_presets = temporal_presets or list(TEMPORAL_PRESETS)

    run_name = _get_run_name(exp_module)
    out_dir = prepare_out_folder(out_root, run_name, clear=clear_output)
    metrics_log = os.path.join(out_dir, "metrics.csv")
    # build run list
    runs: List[Tuple[str, str]] = []

    # Baseline (noise baseline + temporal none)
    runs.append(("baseline", "none"))

    # Noise-only
    for n in noise_presets:
        if n != "baseline":
            runs.append((n, "none"))

    # Temporal-only
    for t in temporal_presets:
        if t not in ("none", "off"):
            runs.append(("baseline", t))

    if mode == "full":
        # Full cross product
        for n in noise_presets:
            for t in temporal_presets:
                if (n, t) == ("baseline", "none"):
                    continue
                # avoid duplicates already added above
                if t in ("none", "off") and n != "baseline":
                    continue
                if n == "baseline" and t not in ("none", "off"):
                    continue
                runs.append((n, t))

    # de-duplicate while preserving order
    seen = set()
    runs_unique = []
    for r in runs:
        if r not in seen:
            runs_unique.append(r)
            seen.add(r)

    print(f"[SWEEP] Saving all results to: {out_dir}")
    print(f"[SWEEP] Total configs: {len(runs_unique)}")

    for noise_model, temporal_model in runs_unique:
        print(f"\n[SWEEP] noise={noise_model} | temporal={temporal_model}")

        # Each call creates a fresh params dict inside test_corner,
        # so you won't get param contamination.
        test_corner(
            datafolder=datafolder,
            exp_module=exp_module,
            debug=debug,
            sampling=sampling,
            start_time=start_time,
            end_time=end_time,
            step=step,
            noise_model=noise_model,
            temporal_model=temporal_model,
            resfolder=out_dir,
            metrics_log_path=metrics_log,
        )

    print("\n[SWEEP] Done.")
    df = pd.read_csv(metrics_log)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)
    return out_dir
