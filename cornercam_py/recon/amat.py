
from __future__ import annotations

import numpy as np

def get_amat(params: dict) -> np.ndarray:
    '''
    Python equivalent of getAmat.m.
    Returns a matrix mapping hidden 1D scene coefficients to observed samples.
    This includes a leading constant-light column.

    Output shape: (M, N+1) where N=params['nsamples'].
    '''
    angles = np.asarray(params["obs_angles"], dtype=np.float64)
    theta1, theta2 = params["theta_lim"]
    thetas = np.linspace(theta1, theta2, int(params["nsamples"]))
    if len(thetas) < 2:
        raise ValueError("nsamples too small")
    tdelta = thetas[1] - thetas[0]
    tdir = np.sign(tdelta) if tdelta != 0 else 1.0

    amat = np.zeros((angles.size, thetas.size + 1), dtype=np.float64)

    for i, ang in enumerate(angles):
        idx = int(np.sum((ang - thetas) * tdir >= 0))
        if idx <= 0:
            amat[i, 0] = 1.0
        else:
            d = (ang - thetas[idx-1]) / tdelta  # idx in MATLAB is count; convert to 0-based
            amat[i, 0:idx] = 1.0
            # idx column in MATLAB is idx+1 (since constant in col 1)
            # We'll match the same formula positions.
            amat[i, idx] = 0.5 * (2 - d) * d + 0.5
            if idx < amat.shape[1] - 1:
                amat[i, idx + 1] = 0.5 * d**2
    return amat
