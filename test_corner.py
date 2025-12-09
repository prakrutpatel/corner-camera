import sys
from pathlib import Path

# Ensure we can import cornercam_py without installing as a package
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))

from cornercam_py.runner.test_corner import test_corner
from cornercam_py.example_params import indoor_loc1, indoor_loc2, outdoor_bricks, outdoor_concrete

# CHANGE THIS to where you extracted the official example videos
datafolder = r"C:\Users\admin\Downloads\example_videos\example_videos"

test_corner(
    datafolder=datafolder,
    exp_module=outdoor_concrete,
    debug=True,
    sampling="rays",
    start_time=2.0,
    end_time=22.0,
    step=6,
)
