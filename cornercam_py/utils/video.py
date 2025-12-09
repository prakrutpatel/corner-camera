
'''
Video helpers wrapping OpenCV to approximate MATLAB VideoReader usage.
'''
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

@dataclass
class VideoProps:
    frame_rate: float
    frame_count: int
    duration: float
    width: int
    height: int

def get_video_props(path: str) -> VideoProps:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    return VideoProps(frame_rate=fps, frame_count=frame_count,
                      duration=duration, width=width, height=height)

def read_frame_at_time(path: str, time_sec: float) -> np.ndarray:
    '''
    Mimics:
        v = VideoReader(path); v.CurrentTime = t; frame = readFrame(v)
    We seek by time. For some codecs OpenCV may land on nearest keyframe.
    '''
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    # set position in milliseconds
    cap.set(cv2.CAP_PROP_POS_MSEC, max(time_sec, 0.0) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame at t={time_sec:.3f}s from {path}")
    # OpenCV returns BGR uint8
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def read_frame_at_index(path: str, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(index, 0))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame idx={index} from {path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def iter_frames_by_index(path: str, indices: np.ndarray) -> Iterator[Tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield int(idx), frame
    cap.release()
