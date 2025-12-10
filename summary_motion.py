import argparse
import numpy as np

from cornercam_py.recon.motion import estimate_motion_by_correlation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to out_*.npz saved by test_run")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--theta1", type=float, default=None)
    ap.add_argument("--theta2", type=float, default=None)
    ap.add_argument("--max_shift", type=int, default=12)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    outframes = d["outframes"]
    params = None
    if "params_json" in d:
        try:
            params = d["params_json"].item()
        except Exception:
            params = None

    # Try to pull defaults from params if available
    fps = args.fps
    step = args.step
    theta_lim = None

    if params:
        fps = fps or float(params.get("frame_rate", 0) or 0)
        step = step or int(params.get("step", 0) or 0)
        tl = params.get("theta_lim", None)
        if tl is not None:
            try:
                theta_lim = (float(tl[0]), float(tl[1]))
            except Exception:
                theta_lim = None

    if args.theta1 is not None and args.theta2 is not None:
        theta_lim = (args.theta1, args.theta2)

    m = estimate_motion_by_correlation(
        outframes,
        theta_lim=theta_lim,
        fps=fps if fps else None,
        step=step if step else None,
        max_shift=args.max_shift,
    )

    print("\n[MOTION]")
    print(f"  direction: {m.direction}")
    print(f"  median shift: {m.median_shift_bins_per_step:.3f} bins/step")
    print(f"  speed: {m.median_speed_bins_per_sec:.3f} bins/sec")
    if theta_lim is not None and fps and step:
        print(f"  omega: {m.median_omega_rad_per_sec:.4f} rad/sec")


if __name__ == "__main__":
    main()
