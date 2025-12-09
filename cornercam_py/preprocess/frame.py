
from __future__ import annotations
import numpy as np
from ..pyr.blur import blur_dn_color
from ..pyr.filters import binomial_filter

def preprocess_frame(frame: np.ndarray, params: dict) -> np.ndarray:
    '''
    Python equivalent of preprocessFrame.m:
    crop then blur and downsample.
    Expects frame in RGB float or uint8.
    params must contain xrange, yrange, downlevs, filter_width.
    '''
    xmin, xmax = params["xrange"]
    ymin, ymax = params["yrange"]
    # MATLAB indexing is inclusive and 1-based.
    # Convert to python slices (0-based, end-exclusive)
    x0 = max(int(xmin) - 1, 0)
    x1 = min(int(xmax), frame.shape[1])
    y0 = max(int(ymin) - 1, 0)
    y1 = min(int(ymax), frame.shape[0])

    cropped = frame[y0:y1, x0:x1, ...]
    filt = binomial_filter(int(params.get("filter_width", 5)))
    out = blur_dn_color(cropped.astype(np.float64), int(params.get("downlevs", 2)), filt)
    return out

