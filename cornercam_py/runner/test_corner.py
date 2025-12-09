
from __future__ import annotations

import os

from .test_run import test_run


def test_corner(
    datafolder: str,
    exp_module,
    *,
    debug: bool = True,
    sampling: str = "rays",
    start_time: float = 2.0,
    end_time: float = 22.0,
    step: int = 6,
):
    '''
    Python analogue of test_corner.m.
    exp_module is a Python module like example_params.indoor_loc1
    holding:
        expname, calname, vidtype, input_type, names
    '''

    expname = exp_module.expname
    calname = exp_module.calname
    vidtype = exp_module.vidtype
    input_type = exp_module.input_type
    names = list(exp_module.names)

    # default params matching MATLAB test_corner.m
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

    # rays
    rstep = 8
    ncircles = 50
    params["rs"] = list(range(rstep, rstep * ncircles + 1, rstep))

    # even_arc
    params["arc_res"] = 1

    # grid
    params["grid_r"] = 60

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
    )
