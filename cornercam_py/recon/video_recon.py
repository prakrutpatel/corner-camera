
from __future__ import annotations

import numpy as np
from typing import Tuple

from ..utils.video import get_video_props, read_frame_at_time, read_frame_at_index
from ..preprocess.frame import preprocess_frame
from .amat import get_amat
from .obs import get_obs_vec

def video_recon(srcfile: str, params: dict) -> Tuple[np.ndarray, dict]:
    '''
    Python equivalent of videoRecon.m.
    Returns:
        outframes: (T, nsamples, C) float
        params: updated params

    Patch notes:
      - Uses frame-index seeking (read_frame_at_index) instead of time-based seeking
        for tighter parity with MATLAB VideoReader indexing.
    '''
    props = get_video_props(srcfile)

    # Background image
    if params.get("online", False):
        # take the frame before the start
        back_idx = int(params["startframe"] - params["step"])
        back_idx = max(back_idx, 0)
        back_img = read_frame_at_index(srcfile, back_idx).astype(np.float64)
    else:
        d = np.load(params["mean_datafile"])
        back_img = d["back_img"].astype(np.float64)

    back_img = preprocess_frame(back_img, params)
    mean_pixel = back_img.mean(axis=(0, 1), keepdims=False)
    nchans = back_img.shape[2] if back_img.ndim == 3 else 1

    frameidx = np.arange(params["startframe"], params["endframe"] + 1, params["step"], dtype=int)

    amat = get_amat(params)
    _, nanrows = get_obs_vec(back_img, params)
    amat = amat[~nanrows, :]

    # Spatial prior
    # MATLAB:
    # bmat = eye(K) - diag(ones(K-1,1), 1); bmat = bmat(1:end-1,:)
    K = amat.shape[1]
    bmat = np.eye(K) - np.diag(np.ones(K-1), 1)
    bmat = bmat[:-1, :]
    bmat[0, :] = 0  # don't use constant light to smooth

    cmat = np.eye(bmat.shape[1])
    cmat[0, :] = 0

    reg = params["beta"] * (bmat.T @ bmat + cmat)
    lam = float(params["lambda"])

    # Gain matrix: (K x M)
    # gain = inv(amat'*amat/lam + reg)*(amat'/lam)
    AtA = (amat.T @ amat) / lam
    rhs = amat.T / lam
    gain = np.linalg.solve(AtA + reg, rhs)

    outframes_full = np.zeros((len(frameidx), K, nchans), dtype=np.float64)

    for i, fidx in enumerate(frameidx):
        print(f"Frame {int(fidx)}")

        framen = read_frame_at_index(srcfile, int(fidx)).astype(np.float64)
        framen = preprocess_frame(framen, params)

        if params.get("sub_mean", False):
            y_raw, _ = get_obs_vec(framen - back_img, params)
            y = y_raw + mean_pixel[None, :]
        else:
            y, _ = get_obs_vec(framen - back_img, params)

        if nchans == 1:
            outframes_full[i, :, 0] = gain @ y[:, 0]
        else:
            for c in range(nchans):
                outframes_full[i, :, c] = gain @ y[:, c]

        if params.get("online", False):
            back_img = back_img * (i / (i + 1.0)) + framen / (i + 1.0)
            mean_pixel = back_img.mean(axis=(0, 1), keepdims=False)

    # drop constant-light column
    outframes = outframes_full[:, 1:, :]
    return outframes, params
