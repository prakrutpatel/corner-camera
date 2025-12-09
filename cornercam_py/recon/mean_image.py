
from __future__ import annotations

from .mean_frame import save_mean_video_frame

def save_mean_image(srcfile: str, input_type: str = "video", overwrite: bool = False) -> str:
    '''
    Python equivalent of saveMeanImage.m.
    '''
    if input_type == "video":
        return save_mean_video_frame(srcfile, overwrite=overwrite)
    elif input_type == "images":
        raise NotImplementedError("Directory-of-images input not supported yet.")
    else:
        raise ValueError("input_type must be 'video' or 'images'")
