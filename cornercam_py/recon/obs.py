
from __future__ import annotations

import numpy as np
from typing import Tuple
from scipy.ndimage import map_coordinates

def get_obs_vec(frame: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Python equivalent of getObsVec.m.
    Interpolates pixel values at observation locations.

    We use obs_xlocs_proc / obs_ylocs_proc (preprocessed coordinates).
    '''
    x = np.asarray(params["obs_xlocs_proc"], dtype=np.float64)
    y = np.asarray(params["obs_ylocs_proc"], dtype=np.float64)

    # map_coordinates expects coords in order (row, col)
    samples = []
    nanmask_total = None

    if frame.ndim == 2:
        coords = np.vstack([y, x])
        vals = map_coordinates(frame, coords, order=1, mode="nearest")
        samples = vals[:, None]
        nanmask_total = np.isnan(vals)
    else:
        for c in range(frame.shape[2]):
            coords = np.vstack([y, x])
            vals = map_coordinates(frame[:, :, c], coords, order=1, mode="nearest")
            samples.append(vals)
            if nanmask_total is None:
                nanmask_total = np.isnan(vals)
            else:
                nanmask_total |= np.isnan(vals)
        samples = np.stack(samples, axis=1)

    nanrows = nanmask_total
    # MATLAB deletes any rows that have any NaN across channels.
    samples = samples[~nanrows, :]
    return samples, nanrows
