from __future__ import annotations

import os
import shutil
from typing import List, Tuple, Optional

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
    # If you want to include adaptive later, just uncomment:
    "adaptive_gaussian",
    "adaptive_median",
    "adaptive_savgol",
    "adaptive_ewma",
]


def _get_run_name(exp_module) -> str:
    rn = getattr(exp_module, "run_name", None)
    if rn:
        return rn
    mod = getattr(exp_module, "__name__", "experiment")
    return mod.split(".")[-1]


def prepare_out_folder(out_root: str, run_name: str, clear: bool = True) -> str:
    out_dir = os.path.join(out_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    if clear:
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


def _build_runs(
    noise_presets: List[str],
    temporal_presets: List[str],
    mode: str,
) -> List[Tuple[str, str]]:
    """
    Returns ordered list of (noise_preset, temporal_preset).
    """

    # Ensure baseline + none exist even if caller overrides lists
    if "baseline" not in noise_presets:
        noise_presets = ["baseline"] + noise_presets
    if "none" not in temporal_presets and "off" not in temporal_presets:
        temporal_presets = ["none"] + temporal_presets

    runs: List[Tuple[str, str]] = []

    if mode == "full":
        # True cross-product
        for n in noise_presets:
            for t in temporal_presets:
                # normalize "off" to "none" if someone includes it
                tt = "none" if t == "off" else t
                runs.append((n, tt))

        # de-dup while preserving order
        seen = set()
        runs_unique = []
        for r in runs:
            if r not in seen:
                runs_unique.append(r)
                seen.add(r)
        return runs_unique

    if mode == "separate":
        # Baseline
        runs.append(("baseline", "none"))

        # Noise-only
        for n in noise_presets:
            if n != "baseline":
                runs.append((n, "none"))

        # Temporal-only
        for t in temporal_presets:
            tt = "none" if t == "off" else t
            if tt != "none":
                runs.append(("baseline", tt))

        seen = set()
        runs_unique = []
        for r in runs:
            if r not in seen:
                runs_unique.append(r)
                seen.add(r)
        return runs_unique

    raise ValueError(f"Unknown mode: {mode}. Use 'full' or 'separate'.")


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
    noise_presets: Optional[List[str]] = None,
    temporal_presets: Optional[List[str]] = None,
    mode: str = "full",
):
    """
    Runs a systematic sweep and saves all outputs in one clean folder.

    mode:
      - "full": every combination of noise_presets x temporal_presets
      - "separate": baseline + noise-only + temporal-only
    """

    noise_presets = noise_presets or list(NOISE_PRESETS)
    temporal_presets = temporal_presets or list(TEMPORAL_PRESETS)

    run_name = _get_run_name(exp_module)
    out_dir = prepare_out_folder(out_root, run_name, clear=clear_output)

    metrics_log = os.path.join(out_dir, "metrics.csv")

    runs_unique = _build_runs(noise_presets, temporal_presets, mode)

    print(f"[SWEEP] Saving all results to: {out_dir}")
    print(f"[SWEEP] Mode: {mode}")
    print(f"[SWEEP] Total configs: {len(runs_unique)}")

    for noise_preset, temporal_preset in runs_unique:
        print(f"\n[SWEEP] noise={noise_preset} | temporal={temporal_preset}")

        test_corner(
            datafolder=datafolder,
            exp_module=exp_module,
            debug=debug,
            sampling=sampling,
            start_time=start_time,
            end_time=end_time,
            step=step,
            noise_model=noise_preset,
            temporal_preset=temporal_preset,   # <-- important
            resfolder=out_dir,
            metrics_log_path=metrics_log,
        )

    print("\n[SWEEP] Done.")

    # Convenience print
    if os.path.exists(metrics_log):
        df = pd.read_csv(metrics_log)
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(df)

    return out_dir
