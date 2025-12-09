
CornerCam (Turning Corners into Cameras) - Python Reimplementation

This package is a best-effort translation of the MIT CornerCam MATLAB code.

Dependencies (suggested):
  - numpy
  - scipy
  - opencv-python
  - matplotlib
  - imageio

Typical usage:

  from cornercam_py.runner.test_corner import test_corner
  from cornercam_py.example_params import indoor_loc1

  datafolder = r"C:\Users\admin\Downloads\example_videos\example_videos"
  test_corner(datafolder, indoor_loc1, debug=True)

Notes:
  - Homography and corner selection use matplotlib ginput.
  - Pyramid functions are approximated with OpenCV separable filtering.
  - Outputs are saved as .npz rather than .mat.
