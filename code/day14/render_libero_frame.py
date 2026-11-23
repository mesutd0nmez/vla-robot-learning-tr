"""Lecture 14 — Produces a single agentview frame from LIBERO (in the form OpenVLA will see it: rotated by 180°).

Runs in Lecture 10's `vla` environment (mujoco 2.3.7, robosuite 1.4.1, LIBERO). No GPU needed.
    conda activate vla
    python render_libero_frame.py --suite libero_spatial --task 0 --init 0 --out frame_task0.png
Output: a 256x256 RGB PNG (`frame_task0.png`), next to it the raw/flipped comparison `frame_task0_raw_vs_flipped.png`,
and the command text (task.language) on screen.
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl" if os.uname().sysname == "Linux" else "glfw")

import numpy as np
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--init", type=int, default=0, help="which of the 50 stored initial states")
    ap.add_argument("--settle", type=int, default=10, help="wait 10 no-op steps like the OpenVLA eval")
    ap.add_argument("--out", default="frame_task0.png")
    args = ap.parse_args()

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(args.task)[args.init])
    for _ in range(args.settle):                     # run_libero_eval.py: num_steps_wait = 10
        obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
    raw = obs["agentview_image"]
    flipped = raw[::-1, ::-1]                        # libero_utils.get_libero_image: rotate by 180°
    Image.fromarray(flipped).save(args.out)
    side = np.concatenate([raw, flipped], axis=1)
    Image.fromarray(side).save(args.out.replace(".png", "_raw_vs_flipped.png"))
    print("command (task.language):", task.language)
    print("written:", args.out, "| raw (left) vs flipped (right):", args.out.replace(".png", "_raw_vs_flipped.png"))
    env.close()


if __name__ == "__main__":
    main()
