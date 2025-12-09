
from __future__ import annotations

import os
import numpy as np

from ..utils.video import get_video_props, read_frame_at_time

def save_mean_video_frame(srcfile: str, overwrite: bool = False) -> str:
    '''
    Python equivalent of saveMeanVideoFrame.m.
    Saves .npz containing back_img and variance.
    '''
    folder, name = os.path.split(srcfile)
    stem, _ = os.path.splitext(name)
    outfile = os.path.join(folder, f"{stem}_mean_vidframe.npz")

    props = get_video_props(srcfile)
    # MATLAB times: 2 : round(FrameRate/30) : Duration-2
    step = max(int(round(props.frame_rate / 30.0)), 1)
    # convert to times in seconds using frame index step
    # We'll sample by frame index for stability.
    start_idx = int(round(2 * props.frame_rate))
    end_idx = max(int(round((props.duration - 2) * props.frame_rate)), start_idx + 1)
    indices = list(range(start_idx, end_idx + 1, step))

    if overwrite or (not os.path.exists(outfile)):
        print(f"Saving mean video frame in {outfile}")
        acc = None
        count = 0
        for idx in indices:
            t = idx / props.frame_rate
            frame = read_frame_at_time(srcfile, t).astype(np.float64)
            if acc is None:
                acc = np.zeros_like(frame, dtype=np.float64)
            acc += frame
            count += 1
        back_img = acc / max(count, 1)
        np.savez_compressed(outfile, back_img=back_img)
    else:
        back_img = np.load(outfile)["back_img"]

    # Ensure variance exists
    data = np.load(outfile)
    if "variance" not in data.files:
        print(f"Computing the variance and putting in {outfile}")
        back_img = data["back_img"].astype(np.float64)
        var_acc = np.zeros_like(back_img, dtype=np.float64)
        count = 0
        props = get_video_props(srcfile)
        for idx in indices:
            t = idx / props.frame_rate
            frame = read_frame_at_time(srcfile, t).astype(np.float64)
            var_acc += (frame - back_img) ** 2
            count += 1
        variance = var_acc / max(count - 1, 1)
        # rewrite file with both arrays
        np.savez_compressed(outfile, back_img=back_img, variance=variance)

    return outfile
