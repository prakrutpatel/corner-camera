
CornerCam (Turning Corners into Cameras) - Python Reimplementation

This package is a best effort translation of the MIT CornerCam MATLAB code.

Dependencies (suggested):
  - numpy
  - scipy
  - opencv-python
  - matplotlib
  - imageio

Typical usage:
  Change "datafolder" in test_corner.py
  python test_corner.py

Notes:
  - Homography and corner selection use matplotlib ginput.
  - Pyramid functions are approximated with OpenCV separable filtering.
  - Outputs are saved as .npz rather than .mat.
