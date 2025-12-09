
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt

from ..utils.video import read_frame_at_time

@dataclass
class CornerData:
    calimg: np.ndarray
    framesize: tuple
    corner: np.ndarray
    wall_point: np.ndarray
    ground_point: np.ndarray

def _select_points(img: np.ndarray, n: int, title: str) -> np.ndarray:
    plt.figure()
    if img.ndim == 3:
        plt.imshow(img[:, :, 0], cmap="gray")
    else:
        plt.imshow(img, cmap="gray")
    plt.title(title)
    pts = plt.ginput(n, timeout=-1)
    plt.close()
    if len(pts) != n:
        raise RuntimeError(f"Expected {n} points for {title}")
    return np.array(pts, dtype=np.float64)

def save_corner_data(cornerfile: str, ncorners: int,
                     input_type: str = "video",
                     overwrite: bool = False,
                     out_path: Optional[str] = None,
                     corner_pts: Optional[np.ndarray] = None,
                     wall_pts: Optional[np.ndarray] = None,
                     ground_pts: Optional[np.ndarray] = None) -> str:
    '''
    Python equivalent of saveCornerData.m.
    Saves .npz file with corner, wall_point, ground_point and calimg.
    '''
    folder, name = os.path.split(cornerfile)
    stem, _ = os.path.splitext(name)
    if out_path is None:
        out_path = os.path.join(folder, f"{stem}_cornerdata.npz")

    if (not overwrite) and os.path.exists(out_path):
        print(f"Using corner data in {out_path}")
        return out_path
    
    mat_path = os.path.join(folder, f"{stem}_cornerdata.mat")
    if (not overwrite) and os.path.exists(mat_path):
        print(f"Found MATLAB corner data: {mat_path}")
        md = load_cornerdata_mat(mat_path)

        def _norm(arr):
            arr = np.asarray(arr)
            if arr.ndim == 1 and arr.size == 2:
                arr = arr.reshape(1, 2)
            if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] != 2:
                arr = arr.T
            if arr.ndim == 1 and arr.size % 2 == 0:
                arr = arr.reshape(-1, 2)
            return arr

        corner = _norm(md["corner"])
        wall_point = _norm(md["wall_point"])
        ground_point = _norm(md["ground_point"])

        np.savez_compressed(
            out_path,
            calimg=md["calimg"],
            framesize=md["framesize"],
            corner=corner,
            wall_point=wall_point,
            ground_point=ground_point,
            cornerfile=cornerfile,
        )
        print(f"Converted to: {out_path}")
        return out_path

    print(f"Saving corner data in {out_path}")

    if input_type == "video":
        calimg = read_frame_at_time(cornerfile, 1.0)
    elif input_type == "images":
        calimg = cv2.cvtColor(cv2.imread(cornerfile), cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("input_type must be 'video' or 'images'")

    h, w = calimg.shape[0], calimg.shape[1]

    if corner_pts is None:
        corner_pts = _select_points(calimg, ncorners, f"Please select {ncorners} corner(s)")
    if wall_pts is None:
        wall_pts = _select_points(calimg, ncorners, "For every corner, please select a point on the base of the wall")
    if ground_pts is None:
        ground_pts = _select_points(calimg, ncorners, "For every corner, please select an angular endpoint")

    np.savez_compressed(
        out_path,
        calimg=calimg,
        framesize=np.array([h, w, calimg.shape[2] if calimg.ndim == 3 else 1]),
        corner=corner_pts,
        wall_point=wall_pts,
        ground_point=ground_pts,
        cornerfile=cornerfile,
    )
    return out_path


def load_cornerdata_mat(mat_path: str) -> dict:
    """
    Load MATLAB corner data from *_cornerdata.mat.

    Expected variables from CornerCam MATLAB:
      - calimg
      - framesize
      - corner
      - wall_point
      - ground_point
    """
    try:
        from scipy.io import loadmat
        md = loadmat(mat_path)
    except NotImplementedError:
        # MATLAB v7.3 (HDF5)
        import h5py
        md = {}
        with h5py.File(mat_path, "r") as f:
            for k in f.keys():
                md[k] = f[k][()]

    def get(name, default=None):
        if name not in md:
            return default
        x = md[name]
        try:
            x = x.squeeze()
        except Exception:
            pass
        return x

    out = {
        "calimg": get("calimg"),
        "framesize": get("framesize"),
        "corner": get("corner"),
        "wall_point": get("wall_point"),
        "ground_point": get("ground_point"),
    }
    return out


def show_corner_points(corner_data_path: str):
    """
    Visualize all stored corner/wall/ground points on the calibration frame.
    Useful for understanding what to click for new datasets.
    """
    d = np.load(corner_data_path, allow_pickle=True)
    calimg = d["calimg"]
    corners = np.asarray(d["corner"], dtype=np.float64)
    wall = np.asarray(d["wall_point"], dtype=np.float64)
    ground = np.asarray(d["ground_point"], dtype=np.float64)

    import matplotlib.pyplot as plt

    plt.figure()
    if calimg.ndim == 3:
        plt.imshow(calimg[:, :, 0], cmap="gray")
    else:
        plt.imshow(calimg, cmap="gray")

    plt.title("Corner calibration image + stored points")

    # Plot all sets (different markers)
    plt.scatter(corners[:, 0], corners[:, 1], s=80, marker="o", label="corner")
    plt.scatter(wall[:, 0], wall[:, 1], s=80, marker="s", label="wall point")
    plt.scatter(ground[:, 0], ground[:, 1], s=80, marker="^", label="ground point")

    # Annotate each triplet with index
    n = corners.shape[0]
    for i in range(n):
        x, y = corners[i]
        plt.text(x + 5, y + 5, f"C{i+1}", fontsize=11)
        x, y = wall[i]
        plt.text(x + 5, y + 5, f"W{i+1}", fontsize=11)
        x, y = ground[i]
        plt.text(x + 5, y + 5, f"G{i+1}", fontsize=11)

    plt.legend()
    plt.axis("image")
    plt.show()