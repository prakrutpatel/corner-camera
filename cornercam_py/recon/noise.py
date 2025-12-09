
from __future__ import annotations

import numpy as np
from ..preprocess.frame import preprocess_frame

def estimate_frame_noise(params: dict) -> float:
    '''
    Python equivalent of estimateFrameNoise.m.
    Uses precomputed variance from mean_datafile.
    '''
    d = np.load(params["mean_datafile"])
    variance = d["variance"]
    input_fmt = preprocess_frame(variance.astype(np.float64), params)
    # MATLAB: median(median(mean(input_fmt,3)))
    mean_chan = input_fmt.mean(axis=2) if input_fmt.ndim == 3 else input_fmt
    return float(np.median(mean_chan))
