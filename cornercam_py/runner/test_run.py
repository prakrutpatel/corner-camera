
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio

from ..geom.homography import save_homography_data, show_homography_points
from ..geom.corner_data import save_corner_data, show_corner_points
from ..sampling.locs import set_obs_xy_locs
from ..recon.input_props import get_input_properties
from ..recon.mean_image import save_mean_image
from ..recon.noise import estimate_frame_noise
from ..recon.video_recon import video_recon
from ..preprocess.frame import preprocess_frame


def _param_string(params: dict, name: str) -> str:
    '''
    Creates a filename-friendly parameter string close to MATLAB behavior.
    The MATLAB code uses a formatted string with corner idx / sampling / rectify etc.
    We'll include the most relevant fields for reproducibility.
    '''
    pieces = [
        name,
        f"corner{int(params.get('corner_idx', 1))}",
        f"{params.get('sampling', 'rays')}",
        f"rect{int(bool(params.get('rectify', False)))}",
        f"down{int(params.get('downlevs', 0))}",
        f"fw{int(params.get('filter_width', 5))}",
        f"ns{int(params.get('nsamples', 200))}",
    ]
    # include rs summary for rays/even_arc
    if "rs" in params:
        rs = np.asarray(params["rs"]).reshape(-1)
        if rs.size:
            pieces.append(f"r{int(rs.min())}-{int(rs.max())}-{int(rs.size)}")
    return "_".join(pieces)


def test_run(
    datafolder: str,
    expname: str,
    calname: str,
    names: list,
    vidtype: str,
    params: dict,
    *,
    input_type: str = "video",
    debug: bool = False,
    start_time: float = 2.0,
    end_time: float = 22.0,
    step: int = 6,
    resfolder: str | None = None,
):
    '''
    Python equivalent of test_run.m.

    Expected layout mirroring the official example:
        datafolder/
           indoors/
              loc1_calibration.MOV
              loc1_one_person_walking.MOV
              ...

    Args:
        step: interpreted as FRAME step (not seconds), matching MATLAB videoRecon.
    '''

    expfolder = os.path.join(datafolder, expname)
    if resfolder is None:
        resfolder = os.path.join(expfolder, "results")
    os.makedirs(resfolder, exist_ok=True)

    gridfile = os.path.join(expfolder, f"{calname}{vidtype}")
    cornerfile = os.path.join(expfolder, f"{names[0]}{vidtype}")

    # Homography data (if rectify enabled)
    if params.get("rectify", False):
        params["homography"] = save_homography_data(gridfile, input_type=input_type, overwrite=False)

    # Corner annotation data
    ncorners = int(params.get("ncorners", 1))
    params["corner_data"] = save_corner_data(cornerfile, ncorners, input_type=input_type, overwrite=False)

    # Visualize stored annotation points (raw)
    if debug:
        if params.get("rectify", False) and "homography" in params:
            show_homography_points(params["homography"])
        if "corner_data" in params:
            show_corner_points(params["corner_data"])
    
    # Set sampling locations + crop ranges
    params = set_obs_xy_locs(params)

    # Debug visualization of calibration frame with transformed corner
    if debug:
        d = np.load(params["corner_data"], allow_pickle=True)
        calimg = d["calimg"].astype(np.float64)
        corners = d["corner"]
        idx = int(params.get("corner_idx", 1)) - 1
        c = corners[idx].copy()

        # Convert corner into preprocessed coords exactly like MATLAB debug block
        xmin, xmax = params["xrange"]
        ymin, ymax = params["yrange"]
        scale = 2 ** int(params.get("downlevs", 2))
        c_proc = np.array([
            (c[0] - xmin) / scale,
            (c[1] - ymin) / scale
        ], dtype=np.float64)

        calimg_p = preprocess_frame(calimg, params)
        plt.figure()
        plt.imshow(calimg_p.astype(np.uint8))
        plt.scatter([c_proc[0]], [c_proc[1]], s=60)
        plt.title("Preprocessed calibration frame + corner (debug)")
        plt.show()

    # Run reconstruction for each video
    for name in names:
        srcfile = os.path.join(expfolder, f"{name}{vidtype}")
        if not os.path.exists(srcfile):
            print(f"[WARN] Missing video: {srcfile}")
            continue

        # Fill frame-rate properties
        params = get_input_properties(srcfile, input_type, params)

        # Convert time bounds to frames
        frame_rate = float(params["frame_rate"])
        params["startframe"] = int(round(start_time * frame_rate))
        params["endframe"] = int(round(end_time * frame_rate))
        params["step"] = int(step)

        # Mean/variance file (needed for offline noise estimate)
        # MATLAB uses saveMeanImage -> saveMeanVideoFrame
        params["mean_datafile"] = save_mean_image(srcfile, input_type=input_type, overwrite=True)

        # Estimate noise and set lambda
        noise = estimate_frame_noise(params)
        params["lambda"] = float(noise)

        paramstr = _param_string(params, name)
        outfile = os.path.join(resfolder, f"out_{paramstr}.npz")
        imfile = os.path.join(resfolder, f"im_{paramstr}.png")

        if os.path.exists(outfile):
            print(f"{outfile} already exists")

        outframes, params = video_recon(srcfile, params)

        # Match MATLAB visualization scaling:
        # scaled = (outframes + 0.05)/0.1;
        scaled = (outframes + 0.05) / 0.1
        scaled = np.clip(scaled, 0.0, 1.0)

        if debug:
            plt.figure()
            # show first channel if multi-chan
            plt.imshow(scaled[:, :, 0] if scaled.ndim == 3 else scaled, aspect="auto")
            plt.title(f"Scaled reconstruction: {name}")
            plt.xlabel("Hidden angle index")
            plt.ylabel("Time index")
            plt.show()

        # Save image
        # imageio expects HxW or HxWxC
        # outframes is (T, nsamples, C)
        if scaled.ndim == 3 and scaled.shape[2] >= 3:
            iio.imwrite(imfile, (scaled * 255).astype(np.uint8))
        else:
            # save grayscale
            gray = scaled[:, :, 0] if scaled.ndim == 3 else scaled
            iio.imwrite(imfile, (gray * 255).astype(np.uint8))

        # Save npz results (params saved in a JSON-friendly way)
        safe_params = {}
        for k, v in params.items():
            try:
                if isinstance(v, (str, int, float, bool, type(None))):
                    safe_params[k] = v
                elif isinstance(v, (list, tuple)):
                    safe_params[k] = v
                else:
                    # fall back to string representation
                    safe_params[k] = str(v)
            except Exception:
                safe_params[k] = str(v)

        np.savez_compressed(
            outfile,
            outframes=outframes,
            scaled=scaled,
            params_json=np.array([safe_params], dtype=object),
            srcfile=srcfile,
        )

        print(f"[OK] Saved: {outfile}")
        print(f"[OK] Saved: {imfile}")

    return params
