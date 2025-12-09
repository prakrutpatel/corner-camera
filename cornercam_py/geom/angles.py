
from __future__ import annotations
import numpy as np

def tangent_angle(xq, yq, cx, cy, cr):
    '''
    Python equivalent of tangentAngle.m.
    xq, yq can be arrays.
    '''
    xq = np.asarray(xq, dtype=np.float64)
    yq = np.asarray(yq, dtype=np.float64)

    if cr <= 0:
        return np.arctan2(yq, xq)
    y = yq - cy
    x = xq - cx
    alpha = np.arctan2(y, x)
    d = np.sqrt(y**2 + x**2)
    # avoid invalid values
    d = np.maximum(d, 1e-9)
    theta = np.arcsin(np.clip(cr / d, -1.0, 1.0))
    beta = alpha + theta - np.pi/2.0
    angle = np.arctan2(y - np.sin(beta), x - np.cos(beta))
    return angle
