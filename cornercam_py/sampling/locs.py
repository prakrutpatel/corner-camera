
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict

from ..geom.homography import map_to_rectified, map_from_rectified
from ..geom.angles import tangent_angle

def _rowvec(x):
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(1, -1) if x.ndim == 1 else x

def set_rays_xy_locs(params: dict) -> dict:
    rs = np.asarray(params["rs"], dtype=np.float64).reshape(-1)
    params["rs"] = rs

    if params.get("rectify", False):
        cx, cy = map_to_rectified(params["homography"], params["corner"][0], params["corner"][1])
        center = np.array([float(cx), float(cy)])
    else:
        center = np.array(params["corner"], dtype=np.float64)

    theta1, theta2 = params["theta_lim"]
    angles = np.linspace(theta1, theta2, params["nsamples"])
    # Outer product: angles as column vs rs as row
    xq = center[0] + np.cos(angles)[:, None] * rs[None, :]
    yq = center[1] + np.sin(angles)[:, None] * rs[None, :]
    obs_angles = np.arctan2(yq.ravel() - center[1], xq.ravel() - center[0])

    if params.get("rectify", False):
        xq, yq = map_from_rectified(params["homography"], xq, yq)

    # MATLAB post-fix
    obs_angles = np.where(obs_angles <= 0, theta2, obs_angles)

    params["obs_angles"] = obs_angles
    params["obs_xlocs"] = xq.ravel()
    params["obs_ylocs"] = yq.ravel()
    return params

def set_even_arc_xy_locs(params: dict) -> dict:
    rs = np.asarray(params["rs"], dtype=np.float64).reshape(-1)
    params["rs"] = rs

    tdir = np.sign(params["theta_lim"][1] - params["theta_lim"][0])
    if tdir == 0:
        tdir = 1.0

    if params.get("rectify", False):
        cx, cy = map_to_rectified(params["homography"], params["corner"][0], params["corner"][1])
        center = np.array([float(cx), float(cy)])
    else:
        center = np.array(params["corner"], dtype=np.float64)

    res = float(params.get("arc_res", 1.0))
    all_angles = []
    all_x = []
    all_y = []
    theta1, theta2 = params["theta_lim"]

    for r in rs:
        nangles = int(round((theta2 - theta1) * r * tdir / res))
        nangles = max(nangles, 2)
        ang = np.linspace(theta1, theta2, nangles)
        xq = center[0] + r * np.cos(ang)
        yq = center[1] + r * np.sin(ang)
        all_angles.append(ang)
        all_x.append(xq)
        all_y.append(yq)

    obs_angles = np.concatenate(all_angles)
    xq = np.concatenate(all_x)
    yq = np.concatenate(all_y)

    if params.get("rectify", False):
        xq, yq = map_from_rectified(params["homography"], xq, yq)

    obs_angles = np.where(obs_angles <= 0, theta2, obs_angles)

    params["obs_angles"] = obs_angles
    params["obs_xlocs"] = xq.ravel()
    params["obs_ylocs"] = yq.ravel()
    return params

def set_grid_xy_locs(params: dict) -> dict:
    cr = float(params.get("corner_r", 0))
    grid_step = 2 ** int(params.get("downlevs", 0))

    theta1, theta2 = params["theta_lim"]

    if params.get("rectify", False):
        cx, cy = map_to_rectified(params["homography"], params["corner"][0], params["corner"][1])
        center = np.array([float(cx), float(cy)])

        # MATLAB conditional for step directions
        if np.mod(abs(theta1 - np.pi), np.pi) < 5 * (np.pi/180):
            xstep = np.sign(np.cos(theta1)) * grid_step
            ystep = np.sign(np.sin(theta2)) * grid_step
        else:
            xstep = np.sign(np.cos(theta2)) * grid_step
            ystep = np.sign(np.sin(theta1)) * grid_step

        xmax = params["grid_r"] * xstep
        ymax = params["grid_r"] * ystep

        # ndgrid equivalent
        yq, xq = np.meshgrid(
            np.arange(ystep, ymax + ystep, ystep),
            np.arange(xstep, xmax + xstep, xstep),
            indexing="ij"
        )

        obs_angles = tangent_angle(
            xq.ravel(), yq.ravel(),
            -np.sign(xstep) * cr, np.sign(ystep) * cr, cr
        )

        xq = np.round(xq + center[0])
        yq = np.round(yq + center[1])

        xq, yq = map_from_rectified(params["homography"], xq, yq)
    else:
        center = np.array(params["corner"], dtype=np.float64)
        ii, jj = np.meshgrid(
            np.arange(1, params["grid_r"] + 1),
            np.arange(1, params["grid_r"] + 1),
            indexing="ij"
        )
        step = grid_step * np.sqrt(2)

        xq = center[0] + step * (ii * np.cos(theta1) + jj * np.cos(theta2))
        yq = center[1] + step * (ii * np.sin(theta1) + jj * np.sin(theta2))

        cx = -np.cos(theta1) - np.cos(theta2)
        cy = np.sin(theta1) + np.sin(theta2)

        obs_angles = tangent_angle(
            xq.ravel() - center[0],
            yq.ravel() - center[1],
            cx, cy, cr
        )

    obs_angles = np.where(obs_angles <= 0, theta2, obs_angles)
    params["obs_angles"] = obs_angles
    params["obs_xlocs"] = np.asarray(xq).ravel()
    params["obs_ylocs"] = np.asarray(yq).ravel()
    return params

def set_obs_xy_locs(params: dict) -> dict:
    '''
    Python equivalent of setObsXYLocs.m.
    Requires params['corner_data'] path.
    '''
    d = np.load(params["corner_data"], allow_pickle=True)

    def _ensure_nx2(arr, name):
        arr = np.asarray(arr)

        # Common case: single point stored as shape (2,)
        if arr.ndim == 1 and arr.size == 2:
            return arr.reshape(1, 2)

        # If MATLAB saved as 2xN instead of Nx2
        if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] != 2:
            arr = arr.T

        # If still 1D but longer, try to reshape into Nx2
        if arr.ndim == 1 and arr.size % 2 == 0:
            arr = arr.reshape(-1, 2)

        # Final sanity
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"{name} has unexpected shape {arr.shape}; expected (N,2)")

        return arr.astype(np.float64)

    corner_all = _ensure_nx2(d["corner"], "corner")
    wall_all = _ensure_nx2(d["wall_point"], "wall_point")
    ground_all = _ensure_nx2(d["ground_point"], "ground_point")
    framesize = d["framesize"]
    # framesize stored as [H,W,C]
    H = int(framesize[0]); W = int(framesize[1])

    idx = int(params.get("corner_idx", 1)) - 1  # MATLAB 1-index
    corner = corner_all[idx, :]
    wall_pt = wall_all[idx, :]
    ground_pt = ground_all[idx, :]

    wall_vec = wall_pt - corner
    ground_vec = ground_pt - corner

    if params.get("rectify", False):
        cx, cy = map_to_rectified(params["homography"], corner[0], corner[1])
        wx, wy = map_to_rectified(params["homography"], wall_pt[0], wall_pt[1])
        gx, gy = map_to_rectified(params["homography"], ground_pt[0], ground_pt[1])

        wall_angle = np.mod(np.arctan2(wy - cy, wx - cx), 2*np.pi)
        wall_angle = np.mod(np.round(wall_angle * 2/np.pi) * np.pi/2, 2*np.pi)
        ground_angle = np.mod(np.arctan2(gy - cy, gx - cx), 2*np.pi)
        ground_angle = np.mod(np.round(ground_angle * 2/np.pi) * np.pi/2, 2*np.pi)
    else:
        wall_angle = np.mod(np.arctan2(wall_vec[1], wall_vec[0]), 2*np.pi)
        ground_angle = np.mod(np.arctan2(ground_vec[1], ground_vec[0]), 2*np.pi)

    diff = ground_angle - wall_angle
    if abs(diff) < np.pi:
        end_angle = wall_angle + diff
    else:
        end_angle = wall_angle - np.sign(diff) * (2*np.pi - abs(diff))

    theta_lim = np.array([wall_angle, end_angle], dtype=np.float64)

    print(f"corner located at {corner[0]:.2f}, {corner[1]:.2f}")
    print(f"theta limits set to {theta_lim[0]:.2f}, {theta_lim[1]:.2f}")

    params["corner"] = corner
    params["theta_lim"] = theta_lim
    params["framesize"] = (H, W)

    sampling = params.get("sampling", "rays")
    if sampling == "rays":
        params = set_rays_xy_locs(params)
    elif sampling == "even_arc":
        params = set_even_arc_xy_locs(params)
    elif sampling == "grid":
        params = set_grid_xy_locs(params)
    else:
        raise ValueError("unsupported sampling scheme")

    # Adjusting for preprocessing (crop, blur and downsample)
    filter_width = int(params.get("filter_width", 5))
    downlevs = int(params.get("downlevs", 2))
    pad = filter_width * (2 ** (downlevs + 1))

    xmin = max(1, int(np.floor(params["obs_xlocs"].min())) - pad)
    xmax = min(W, int(np.ceil(params["obs_xlocs"].max())) + pad)
    ymin = max(1, int(np.floor(params["obs_ylocs"].min())) - pad)
    ymax = min(H, int(np.ceil(params["obs_ylocs"].max())) + pad)

    params["xrange"] = (xmin, xmax)
    params["yrange"] = (ymin, ymax)

    # Convert obs locations into coordinates in preprocessed image.
    # MATLAB:
    #   obs_xlocs = (obs_xlocs - xmin)/(2^downlevs)+1
    # We'll store 0-indexed float coordinates for numpy:
    #   x' = (x - xmin)/(2^downlevs)
    scale = 2 ** downlevs
    params["obs_xlocs_proc"] = (params["obs_xlocs"] - xmin) / scale
    params["obs_ylocs_proc"] = (params["obs_ylocs"] - ymin) / scale

    return params
