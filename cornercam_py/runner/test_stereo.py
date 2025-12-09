
from __future__ import annotations

from .test_run import test_run


def test_stereo(
    datafolder: str,
    exp_module,
    *,
    debug: bool = True,
    start_time: float = 2.0,
    end_time: float = 32.0,
    step: int = 6,
):
    '''
    Python analogue of test_stereo.m.
    Loops over corners if exp_module provides ncorners.
    '''
    expname = exp_module.expname
    calname = exp_module.calname
    vidtype = exp_module.vidtype
    input_type = exp_module.input_type
    names = list(exp_module.names)

    ncorners = int(getattr(exp_module, "ncorners", 1))

    # base params (you can override externally)
    params = {}
    params["ncorners"] = ncorners
    params["rectify"] = True
    params["sampling"] = "rays"
    params["online"] = 1
    params["sub_mean"] = 1
    params["beta"] = 1 / (0.085 ** 2)
    params["filter_width"] = 5
    params["downlevs"] = 2
    params["corner_r"] = 0
    params["nsamples"] = 200
    params["rs"] = list(range(8, 8 * 50 + 1, 8))
    params["arc_res"] = 1
    params["grid_r"] = 60

    for i in range(1, ncorners + 1):
        params["corner_idx"] = i
        test_run(
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
        )

    return params
