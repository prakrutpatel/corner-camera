
'''
Approximate re-implementation of the Simoncelli pyramid routines
used by the MATLAB code (blurDn.m / blurDnClr.m / corrDn.m).
We do NOT attempt to replicate the exact MEX implementation, but
we match the intended operation: blur with a separable kernel and
downsample by 2 per level with reflect-style borders.

This is sufficient for reproducing CornerCam reconstructions.
'''
from __future__ import annotations

import numpy as np
import cv2
from typing import Optional

from .filters import binomial_filter

def _ensure_kernel(filt):
    if isinstance(filt, str):
        # Support only binomN naming used in MATLAB defaults
        # e.g. 'binom5'
        m = None
        try:
            import re
            m = re.match(r"binom(\d+)", filt)
        except Exception:
            m = None
        if not m:
            raise ValueError(f"Unsupported named filter: {filt}")
        sz = int(m.group(1))
        return binomial_filter(sz)
    arr = np.asarray(filt, dtype=np.float64)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    return arr

def blur_dn_gray(im: np.ndarray, nlevs: int = 1, filt="binom5") -> np.ndarray:
    '''
    Roughly matches MATLAB blurDn for 2D single-channel images.
    '''
    if nlevs is None:
        nlevs = 1
    if nlevs <= 0:
        return im
    k = _ensure_kernel(filt)

    # recursion first (MATLAB does recursive call before filtering at this level)
    if nlevs > 1:
        im = blur_dn_gray(im, nlevs - 1, k)

    # Apply separable correlation/conv with reflect1-like borders.
    # MATLAB corrDn uses correlation then picks START:STEP:STOP; in our use-case
    # we only care about STEP=[2 2] downsampling.
    # We can approximate with cv2.sepFilter2D then subsample.
    kx = k.astype(np.float64)
    ky = k.astype(np.float64)

    # OpenCV sepFilter2D expects kernel for convolution (not correlation),
    # but binomial kernel is symmetric so it doesn't matter.
    blurred = cv2.sepFilter2D(im.astype(np.float64), ddepth=-1,
                              kernelX=kx, kernelY=ky,
                              borderType=cv2.BORDER_REFLECT)
    # downsample by 2
    return blurred[::2, ::2]

def blur_dn_color(im: np.ndarray, nlevs: int = 1, filt="binom5") -> np.ndarray:
    '''
    Equivalent of MATLAB blurDnClr.
    Accepts HxWxC or HxW.
    '''
    if im.ndim == 2:
        return blur_dn_gray(im, nlevs, filt)
    chans = []
    for c in range(im.shape[2]):
        chans.append(blur_dn_gray(im[:, :, c], nlevs, filt))
    return np.stack(chans, axis=2)

