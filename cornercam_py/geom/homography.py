
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List

import numpy as np
import cv2
import matplotlib.pyplot as plt

from ..utils.video import read_frame_at_time

@dataclass
class HomographyData:
    # Homography that maps rectified -> original (MATLAB variable v)
    v: np.ndarray
    # Inverse homography that maps original -> rectified (MATLAB variable v_inv)
    v_inv: np.ndarray
    minx: float
    maxx: float
    miny: float
    maxy: float
    h_new: int
    w_new: int
    # We store vectors used for grid generation for debugging
    hvec: np.ndarray
    wvec: np.ndarray

def solve_homography_dlt(pin: np.ndarray, pout: np.ndarray) -> np.ndarray:
    '''
    Direct Linear Transform using SVD, matching solveHomography.m logic.
    pin, pout: (N,2) arrays
    returns 3x3 matrix H such that [x_out,y_out,1]^T ~ H [x_in,y_in,1]^T
    '''
    pin = np.asarray(pin, dtype=np.float64)
    pout = np.asarray(pout, dtype=np.float64)
    if pin.shape != pout.shape or pin.shape[1] != 2:
        raise ValueError("pin and pout must be same shape (N,2)")
    n = pin.shape[0]
    if n < 4:
        raise ValueError("Need at least 4 matching points")

    X = pin[:, 0]
    Y = pin[:, 1]
    x = pout[:, 0]
    y = pout[:, 1]

    rows0 = np.zeros((n, 3))
    rowsXY = -np.stack([X, Y, np.ones(n)], axis=1)

    hx = np.concatenate([rowsXY, rows0, (x*X)[:, None], (x*Y)[:, None], x[:, None]], axis=1)
    hy = np.concatenate([rows0, rowsXY, (y*X)[:, None], (y*Y)[:, None], y[:, None]], axis=1)
    h = np.concatenate([hx, hy], axis=0)  # (2n, 9)
    # We want the right singular vector associated with smallest singular value.
    # In MATLAB code, they used U from svd(h) and took U(:,9) then reshaped and transposed.
    # That's slightly unusual; typical DLT uses Vh[-1].
    # We follow the more standard approach but normalize to be safe.
    _, _, vh = np.linalg.svd(h, full_matrices=False)
    v = vh[-1, :].reshape(3, 3)
    return v

def _select_four_points(img: np.ndarray) -> np.ndarray:
    '''
    Uses matplotlib ginput to allow user to click 4 points.
    Returns (4,2) float array in image coordinates.
    '''
    plt.figure()
    if img.ndim == 3:
        plt.imshow(img[:, :, 0], cmap="gray")
    else:
        plt.imshow(img, cmap="gray")
    plt.title("Select 4 points on a square counter-clockwise from top left")
    pts = plt.ginput(4, timeout=-1)
    plt.close()
    if len(pts) != 4:
        raise RuntimeError("Expected 4 points")
    return np.array(pts, dtype=np.float64)

def save_homography_data(gridfile: str, input_type: str = "video",
                         overwrite: bool = False,
                         out_path: Optional[str] = None,
                         pts_img: Optional[np.ndarray] = None) -> str:
    '''
    Python equivalent of saveHomographyData.m.
    Saves an .npz with keys matching MATLAB variable names as closely as possible.
    '''
    folder, name = os.path.split(gridfile)
    stem, _ = os.path.splitext(name)
    if out_path is None:
        out_path = os.path.join(folder, f"{stem}_homography.npz")

    if (not overwrite) and os.path.exists(out_path):
        print(f"Using homography data in {out_path}")
        return out_path
    
    mat_path = os.path.join(folder, f"{stem}_homography.mat")
    if (not overwrite) and os.path.exists(mat_path):
        print(f"Found MATLAB homography: {mat_path}")
        md = load_homography_mat(mat_path)

        pts = md.get("pts", None)
        pts_ref = md.get("pts_ref", None)

        save_dict = {
            "v": md["v"],
            "v_inv": md["v_inv"],
            "minx": md["minx"],
            "maxx": md["maxx"],
            "miny": md["miny"],
            "maxy": md["maxy"],
            "h_new": int(md["h_new"]),
            "w_new": int(md["w_new"]),
            "hvec": md.get("hvec", None),
            "wvec": md.get("wvec", None),
            "gridfile": md.get("gridfile", ""),
        }

        if pts is not None:
            save_dict["pts_img"] = pts  # normalize naming
        if pts_ref is not None:
            save_dict["pts_ref"] = pts_ref

        np.savez_compressed(out_path, **{k:v for k,v in save_dict.items() if v is not None})
        print(f"Converted to: {out_path}")
        return out_path

    print(f"Saving homography data in {out_path}")

    if input_type == "video":
        gridimg = read_frame_at_time(gridfile, 1.0)
    elif input_type == "images":
        gridimg = cv2.cvtColor(cv2.imread(gridfile), cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("input_type must be 'video' or 'images'")

    if pts_img is None:
        pts_img = _select_four_points(gridimg)

    # Reference square points in CCW order from top-left
    pts_ref = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float64)

    # Compute homographies
    # MATLAB: v = solveHomography(pts_ref', pts')
    # So v maps rectified(ref) -> image
    v = solve_homography_dlt(pts_ref, pts_img)
    # MATLAB: v_inv = solveHomography(pts', pts_ref')
    v_inv = solve_homography_dlt(pts_img, pts_ref)

    h, w = gridimg.shape[0], gridimg.shape[1]
    sz = min(h, w) / 3.0

    # Map image corners through v_inv to find bounds in rectified coord space
    def apply(H, x, y):
        p = H @ np.array([x, y, 1.0], dtype=np.float64)
        return p[0]/p[2], p[1]/p[2]

    tl = apply(v_inv, 1, 1)
    bl = apply(v_inv, 1, h)
    tr = apply(v_inv, w, 1)
    br = apply(v_inv, w, h)

    xs = [tl[0], bl[0], tr[0], br[0]]
    ys = [tl[1], bl[1], tr[1], br[1]]
    minx, maxx = float(min(xs)), float(max(xs))
    miny, maxy = float(min(ys)), float(max(ys))

    h_new = int(round((maxy - miny) * sz))
    w_new = int(round((maxx - minx) * sz))
    # MATLAB warning: rounded to be a square - but uses independent hvec/wvec lengths
    # We'll keep same logic (could be rectangular).
    hvec = np.linspace(miny, maxy, max(h_new, 2))
    wvec = np.linspace(minx, maxx, max(w_new, 2))

    # Save
    np.savez_compressed(
        out_path,
        v=v,
        v_inv=v_inv,
        minx=minx, maxx=maxx, miny=miny, maxy=maxy,
        h_new=h_new, w_new=w_new,
        hvec=hvec, wvec=wvec,
        gridfile=gridfile,
        pts_img=pts_img,
        pts_ref=pts_ref,
        framesize=np.array([h, w], dtype=np.int32),
    )
    return out_path

def map_to_rectified(hdata_path: str, xin, yin):
    d = np.load(hdata_path)
    v_inv = d["v_inv"]
    minx = float(d["minx"]); maxx = float(d["maxx"])
    miny = float(d["miny"]); maxy = float(d["maxy"])
    h_new = int(d["h_new"]); w_new = int(d["w_new"])

    xin_arr = np.asarray(xin, dtype=np.float64)
    yin_arr = np.asarray(yin, dtype=np.float64)
    ones = np.ones_like(xin_arr)
    coords = v_inv @ np.stack([xin_arr.ravel(), yin_arr.ravel(), ones.ravel()], axis=0)
    xout = (coords[0] / coords[2]).reshape(xin_arr.shape)
    yout = (coords[1] / coords[2]).reshape(yin_arr.shape)

    xout = (xout - minx) / (maxx - minx) * w_new
    yout = (yout - miny) / (maxy - miny) * h_new
    return xout, yout

def map_from_rectified(hdata_path: str, xin, yin):
    d = np.load(hdata_path)
    v = d["v"]
    minx = float(d["minx"]); maxx = float(d["maxx"])
    miny = float(d["miny"]); maxy = float(d["maxy"])
    h_new = int(d["h_new"]); w_new = int(d["w_new"])

    xin_arr = np.asarray(xin, dtype=np.float64)
    yin_arr = np.asarray(yin, dtype=np.float64)

    xin_norm = xin_arr / w_new * (maxx - minx) + minx
    yin_norm = yin_arr / h_new * (maxy - miny) + miny

    ones = np.ones_like(xin_norm)
    coords = v @ np.stack([xin_norm.ravel(), yin_norm.ravel(), ones.ravel()], axis=0)
    xout = (coords[0] / coords[2]).reshape(xin_arr.shape)
    yout = (coords[1] / coords[2]).reshape(yin_arr.shape)
    return xout, yout

def load_homography_mat(mat_path: str) -> dict:
    """
    Load MATLAB homography data from *_homography.mat.

    Expected variables (from CornerCam MATLAB):
      v, v_inv, minx, maxx, miny, maxy, h_new, w_new, hvec, wvec
    """
    try:
        from scipy.io import loadmat
        md = loadmat(mat_path)
    except NotImplementedError:
        # Likely MATLAB v7.3 (HDF5)
        import h5py
        md = {}
        with h5py.File(mat_path, "r") as f:
            for k in f.keys():
                md[k] = f[k][()]

    def get(name, default=None):
        if name not in md:
            return default
        x = md[name]
        # unwrap MATLAB arrays
        try:
            x = x.squeeze()
        except Exception:
            pass
        return x

    out = {
        "v": get("v"),
        "v_inv": get("v_inv"),
        "minx": float(get("minx")),
        "maxx": float(get("maxx")),
        "miny": float(get("miny")),
        "maxy": float(get("maxy")),
        "h_new": int(get("h_new")),
        "w_new": int(get("w_new")),
        "hvec": get("hvec"),
        "wvec": get("wvec"),
    }
    return out

def _get_pts_from_hdata(d):
    # Python save_homography_data stores pts_img
    if "pts_img" in d.files:
        return np.asarray(d["pts_img"], dtype=np.float64)
    # MATLAB file often stores pts
    if "pts" in d.files:
        return np.asarray(d["pts"], dtype=np.float64)
    return None

def show_homography_points(hdata_path: str):
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from ..utils.video import read_frame_at_time

    d = np.load(hdata_path, allow_pickle=True)

    # Try to locate the grid image source if stored
    gridfile = None
    if "gridfile" in d.files:
        gridfile = str(d["gridfile"])
    elif "gridfile" in d and isinstance(d["gridfile"], (str, bytes)):
        gridfile = d["gridfile"]

    # Fallback: if you don't store gridfile in converted npz,
    # you can just skip image background and only plot points.
    gridimg = None
    if gridfile and os.path.exists(gridfile):
        if gridfile.lower().endswith((".mov", ".mp4", ".avi", ".mkv")):
            gridimg = read_frame_at_time(gridfile, 1.0)
        else:
            im = cv2.imread(gridfile)
            if im is not None:
                gridimg = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Accept both key conventions
    pts_img = None
    if "pts_img" in d.files:
        pts_img = d["pts_img"]
    elif "pts" in d.files:
        pts_img = d["pts"]

    pts_ref = None
    if "pts_ref" in d.files:
        pts_ref = d["pts_ref"]

    plt.figure()
    if gridimg is not None:
        if gridimg.ndim == 3:
            plt.imshow(gridimg[:, :, 0], cmap="gray")
        else:
            plt.imshow(gridimg, cmap="gray")
    else:
        plt.gca().invert_yaxis()  # just to resemble image coords

    if pts_img is None:
        plt.title("Homography debug: no stored points found (this is OK).")
        plt.show()
        return

    pts_img = np.asarray(pts_img).reshape(-1, 2)
    plt.scatter(pts_img[:, 0], pts_img[:, 1], s=60)

    for i, (x, y) in enumerate(pts_img, start=1):
        plt.text(x + 5, y + 5, str(i), fontsize=12)

    title = "Homography points on calibration grid"
    if pts_ref is not None:
        title += " (pts_ref also present)"
    plt.title(title)
    plt.show()
