import sys
from pathlib import Path

# Ensure we can import cornercam_py without installing as a package
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))

from cornercam_py.runner.test_corner import test_corner
from cornercam_py.example_params import indoor_loc1, indoor_loc2, indoor_loc3, indoor_loc4, indoor_loc5, outdoor_bricks, outdoor_concrete

# CHANGE THIS to where you extracted the official example videos
datafolder = r"C:\Users\admin\Downloads\example_videos\example_videos"

test_corner(
    datafolder=datafolder,
    exp_module=outdoor_bricks,
    debug=True,
    sampling="rays",
    start_time=2.0,  #2  16
    end_time=22.0,   #22
    step=6,          #6   
    noise_model= "time_adapt_obs"
    
)
