import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))

from cornercam_py.runner.sweep import sweep_corner
from cornercam_py.example_params import indoor_loc1, indoor_loc2, indoor_loc3, indoor_loc4, indoor_loc5, outdoor_bricks, outdoor_concrete

datafolder = r"C:\Users\admin\Downloads\example_videos\example_videos"

sweep_corner(
    datafolder=datafolder,
    exp_module=indoor_loc1,
    out_root=r"Y:\results",
    debug=False,          # set True if you want windows popping up
    sampling="rays",
    start_time=2.0,
    end_time=22.0,
    step=6,
    clear_output=True,
    #mode="separate",      # start here for sanity
    mode="full",        # later if you want every combo
)
