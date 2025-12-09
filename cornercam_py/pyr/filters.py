
from __future__ import annotations
import numpy as np

def binomial_filter(sz: int) -> np.ndarray:
    '''
    Python equivalent of MATLAB binomialFilter.m
    Returns 1D kernel of length sz containing binomial coefficients
    of order (sz-1) scaled by 0.5^(sz-1).
    '''
    if sz < 2:
        raise ValueError("size argument must be larger than 1")
    kernel = np.array([0.5, 0.5], dtype=np.float64)
    for _ in range(sz - 2):
        kernel = np.convolve(np.array([0.5, 0.5], dtype=np.float64), kernel)
    # return as 1D
    return kernel

