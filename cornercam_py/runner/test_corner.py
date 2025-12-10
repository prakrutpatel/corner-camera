from __future__ import annotations

import os
from .test_run import test_run


def apply_noise_preset(params, preset: str):
    # store name for filenames
    params["noise_preset"] = preset

    if preset == "baseline":
        params["noise_model"] = "legacy"
        params["noise_use_obs_points"] = False
        params["noise_time_adaptive"] = False

    elif preset == "robust_global":
        params["noise_model"] = "mad"
        params["noise_use_obs_points"] = False
        params["noise_time_adaptive"] = False

    elif preset == "robust_obs":
        params["noise_model"] = "mad"
        params["noise_use_obs_points"] = True
        params["noise_time_adaptive"] = False

    elif preset == "weighted_obs":
        params["noise_model"] = "weighted_obs"
        params["noise_use_obs_points"] = True
        params["noise_time_adaptive"] = False
        params["noise_weighted_eps"] = 1e-6

    elif preset == "time_adapt_obs":
        params["noise_model"] = "mad"
        params["noise_use_obs_points"] = True
        params["noise_time_adaptive"] = True
        params["noise_time_samples"] = 8

    else:
        raise ValueError(preset)


def apply_temporal_preset(params, preset: str):
    # store name for filenames
    params["temporal_preset"] = preset

    if preset in (None, "none", "off"):
        params["temporal_denoise"] = False
        params["temporal_method"] = "none"
        return

    params["temporal_denoise"] = True

    if preset == "gaussian":
        params["temporal_method"] = "gaussian"
        params["temporal_sigma"] = 1.0
        params["temporal_window"] = 7

    elif preset == "median":
        params["temporal_method"] = "median"
        params["temporal_window"] = 7

    elif preset == "savgol":
        params["temporal_method"] = "savgol"
        params["temporal_window"] = 7
        params["temporal_savgol_poly"] = 2

    elif preset == "ewma":
        params["temporal_method"] = "ewma"
        params["temporal_ewma_alpha"] = 0.25

    else:
        raise ValueError(preset)


def test_corner(
    datafolder: str,
    exp_module,
    *,
    debug: bool = True,
    sampling: str = "rays",
    start_time: float = 2.0,
    end_time: float = 22.0,
    step: int = 6,
    noise_model: str = "baseline",
    temporal_model: str = "gaussian",
    resfolder: str | None = None,
    metrics_log_path: str | None = None,
):
    """
    Python analogue of test_corner.m with switchable noise + temporal presets.
    """

    expname = exp_module.expname
    calname = exp_module.calname
    vidtype = exp_module.vidtype
    input_type = exp_module.input_type
    names = list(exp_module.names)

    # default params
    params = {}
    params["ncorners"] = 1
    params["corner_idx"] = 1
    params["rectify"] = True

    params["sampling"] = sampling
    params["online"] = 1
    params["sub_mean"] = 1

    params["beta"] = 1 / (0.085 ** 2)

    params["filter_width"] = 5
    params["downlevs"] = 2
    params["corner_r"] = 0

    params["nsamples"] = 200

    params["metrics_debug"] = True
    
    if metrics_log_path:
        params["metrics_log_path"] = metrics_log_path

    # rays
    rstep = 8
    ncircles = 50
    params["rs"] = list(range(rstep, rstep * ncircles + 1, rstep))

    # even_arc
    params["arc_res"] = 1

    # grid
    params["grid_r"] = 60

    # Apply presets
    apply_noise_preset(params, noise_model)
    apply_temporal_preset(params, temporal_model)

    # Run
    return test_run(
        datafolder=datafolder,
        expname=expname,
        calname=calname,
        names=names,
        vidtype=vidtype,
        params=params,
        input_type=input_type,
        debug=debug,
        start_time=start_time,
        end_time=end_time,
        step=step,
        resfolder=resfolder,
    )
