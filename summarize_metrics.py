from __future__ import annotations

import argparse
import os
import glob
import sys
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd


ESSENTIAL_COLS = [
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


def find_metric_files(inputs: List[str]) -> List[str]:
    files = []
    for p in inputs:
        if os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, "**", "metrics.csv"), recursive=True))
        elif "*" in p or "?" in p or "[" in p:
            files.extend(glob.glob(p, recursive=True))
        else:
            files.append(p)

    # de-dup + keep existing
    out = []
    seen = set()
    for f in files:
        f = os.path.abspath(f)
        if f in seen:
            continue
        if os.path.exists(f) and os.path.isfile(f):
            out.append(f)
            seen.add(f)
    return out


def load_metrics(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for path in paths:
        try:
            df = pd.read_csv(path)
            df["__source__"] = path
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Normalize column names just in case
    df.columns = [c.strip() for c in df.columns]

    # Keep only essential columns if present
    missing = [c for c in ESSENTIAL_COLS if c not in df.columns]
    if missing:
        print(f"[WARN] Missing expected columns: {missing}")
        # still proceed with what's available

    # Coerce numeric columns
    for c in ["lambda", "temporal_abs_diff", "temporal_hf_ratio", "motion_energy", "peakiness"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _safe_z(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True)
    if sd is None or sd == 0 or np.isnan(sd):
        return pd.Series([0.0] * len(x), index=x.index)
    return (x - mu) / sd


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create three scores:
      - cleanliness_score: higher is better (derived from lower temporal metrics)
      - signal_score: higher is better
      - balanced_score: combines both

    We use z-scores within the provided dataframe.
    """
    out = df.copy()

    # Handle missing cols gracefully
    for col in ["temporal_abs_diff", "temporal_hf_ratio", "motion_energy", "peakiness"]:
        if col not in out.columns:
            out[col] = np.nan

    # z for each metric
    z_td = _safe_z(out["temporal_abs_diff"])
    z_hf = _safe_z(out["temporal_hf_ratio"])
    z_mo = _safe_z(out["motion_energy"])
    z_pk = _safe_z(out["peakiness"])

    # Lower is better for temporal metrics -> invert sign
    out["cleanliness_score"] = (-z_td + -z_hf) / 2.0
    out["signal_score"] = (z_mo + z_pk) / 2.0

    # Balanced: equal weight
    out["balanced_score"] = (out["cleanliness_score"] + out["signal_score"]) / 2.0

    return out


def group_key(row: pd.Series) -> Tuple:
    """
    Grouping key for 'config-level' summary.
    """
    return (
        row.get("expname", ""),
        row.get("noise_preset", ""),
        row.get("temporal_preset", ""),
        row.get("stage", ""),
    )


def summarize_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate repeated rows (e.g., multiple videos)
    into mean scores per config.
    """
    needed = ["expname", "noise_preset", "temporal_preset", "stage",
              "lambda", "cleanliness_score", "signal_score", "balanced_score",
              "temporal_abs_diff", "temporal_hf_ratio", "motion_energy", "peakiness"]

    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    gcols = ["expname", "noise_preset", "temporal_preset", "stage"]

    agg = df.groupby(gcols, dropna=False).agg(
        n=("balanced_score", "count"),
        lambda_mean=("lambda", "mean"),
        temporal_abs_diff=("temporal_abs_diff", "mean"),
        temporal_hf_ratio=("temporal_hf_ratio", "mean"),
        motion_energy=("motion_energy", "mean"),
        peakiness=("peakiness", "mean"),
        cleanliness_score=("cleanliness_score", "mean"),
        signal_score=("signal_score", "mean"),
        balanced_score=("balanced_score", "mean"),
    ).reset_index()

    # nicer formatting
    agg["lambda_mean"] = agg["lambda_mean"].round(6)
    return agg


def print_top(df_cfg: pd.DataFrame, col: str, n: int, title: str):
    if df_cfg.empty or col not in df_cfg.columns:
        print(f"\n{title}\n  [No data]")
        return

    show_cols = [
        "expname", "stage", "noise_preset", "temporal_preset", "n",
        "lambda_mean", "temporal_abs_diff", "temporal_hf_ratio",
        "motion_energy", "peakiness",
        "cleanliness_score", "signal_score", "balanced_score"
    ]
    show_cols = [c for c in show_cols if c in df_cfg.columns]

    top = df_cfg.sort_values(col, ascending=False).head(n)

    print(f"\n{title}")
    print(top[show_cols].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(
        description="Summarize CornerCam ablation metrics from compact metrics.csv logs."
    )
    ap.add_argument(
        "inputs",
        nargs="*",
        default=[],
        help="Paths to metrics.csv, folders containing them, or glob patterns."
    )
    ap.add_argument(
        "--root",
        default=None,
        help="Optional root folder to search recursively for metrics.csv."
    )
    ap.add_argument(
        "--top",
        type=int,
        default=8,
        help="How many top configs to show per category."
    )
    ap.add_argument(
        "--stage",
        default=None,
        help="Filter to a specific stage (e.g., 'pre-temporal' or 'post-temporal(gaussian)')."
    )
    ap.add_argument(
        "--exp",
        default=None,
        help="Filter to a specific expname (e.g., 'outdoors')."
    )
    ap.add_argument(
        "--save-merged",
        default=None,
        help="If set, saves merged per-row dataframe to this path."
    )
    ap.add_argument(
        "--save-config",
        default=None,
        help="If set, saves aggregated per-config summary to this path."
    )

    args = ap.parse_args()

    inputs = list(args.inputs)

    if args.root:
        inputs.append(args.root)

    if not inputs:
        # default: try current dir
        inputs = [os.getcwd()]

    files = find_metric_files(inputs)

    if not files:
        print("[ERROR] No metrics.csv found.")
        print("Tip: pass --root Y:\\results or a specific metrics.csv path.")
        sys.exit(1)

    print(f"[OK] Found {len(files)} metrics file(s).")

    df = load_metrics(files)
    if df.empty:
        print("[ERROR] Could not load any valid metrics.")
        sys.exit(1)

    # Basic filtering
    if args.stage and "stage" in df.columns:
        df = df[df["stage"].astype(str) == str(args.stage)]

    if args.exp and "expname" in df.columns:
        df = df[df["expname"].astype(str) == str(args.exp)]

    # Add scores
    df_scored = add_scores(df)

    # Aggregate per config
    df_cfg = summarize_by_config(df_scored)

    # Print overviews
    print_top(
        df_cfg, "cleanliness_score", args.top,
        title=f"Top {args.top} configs by CLEANLINESS (higher score = lower temporal noise)"
    )
    print_top(
        df_cfg, "signal_score", args.top,
        title=f"Top {args.top} configs by SIGNAL (higher score = stronger dynamics/contrast)"
    )
    print_top(
        df_cfg, "balanced_score", args.top,
        title=f"Top {args.top} configs by BALANCED score"
    )

    # Save outputs if requested
    if args.save_merged:
        outp = os.path.abspath(args.save_merged)
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        df_scored.to_csv(outp, index=False)
        print(f"\n[OK] Saved merged rows to: {outp}")

    if args.save_config:
        outp = os.path.abspath(args.save_config)
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        df_cfg.to_csv(outp, index=False)
        print(f"[OK] Saved per-config summary to: {outp}")


if __name__ == "__main__":
    main()

# 1) Summarize everything under your results drive
# python summarize_metrics.py --root Y:\results

# 2) Only one experiment folder
# python summarize_metrics.py Y:\results\outdoor_bricks

# 3) Only post-temporal gaussian stage
# python summarize_metrics.py --root Y:\results --stage "post-temporal(gaussian)"