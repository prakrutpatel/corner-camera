
from __future__ import annotations

from typing import Dict
from ..utils.video import get_video_props

def get_input_properties(srcfile: str, input_type: str, params: Dict) -> Dict:
    '''
    Python equivalent of getInputProperties.m.
    Fills:
        params['frame_rate']
        params['maxframe']
    '''
    if input_type == "video":
        props = get_video_props(srcfile)
        params["frame_rate"] = props.frame_rate
        params["maxframe"] = int(round(props.duration * props.frame_rate))
        return params
    elif input_type == "images":
        raise NotImplementedError("Directory-of-images input not supported yet (matches MATLAB).")
    else:
        raise ValueError("input_type must be 'video' or 'images'")
