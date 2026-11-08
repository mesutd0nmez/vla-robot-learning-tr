"""Ders 15 — LIBERO-Spatial görev 0–4 için OpenVLA'nın göreceği kareleri üretir (frames/frame_task{t}.png).

Ders 14'ün `render_libero_frame.py` mantığı, 5 görev için döngüde. `vla` env'inde çalışır (mujoco 2.3.7, robosuite 1.4.1, LIBERO).
    conda activate vla
    python render_frames.py --tasks 0,1,2,3,4 --init 0 --out frames
Her kare: init_states[init] → 10 no-op adım (run_libero_eval.py: num_steps_wait) → agentview 256x256 → 180° döndür.
Yanına `frames/contact_sheet.png` (5 kare yan yana, komutlarıyla) yazar — derste/projede kullanmak için.
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl" if os.uname().sysname == "Linux" else "glfw")

import numpy as np
from PIL import Image, ImageDraw
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def render(suite, task_id, init, settle):
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(task_id)[init])
    for _ in range(settle):
        obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
    env.close()
    return obs["agentview_image"][::-1, ::-1], task.language


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="0,1,2,3,4")
    ap.add_argument("--init", type=int, default=0)
    ap.add_argument("--settle", type=int, default=10)
    ap.add_argument("--out", default="frames")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    tiles = []
    for t in [int(x) for x in args.tasks.split(",")]:
        frame, lang = render(suite, t, args.init, args.settle)
        p = os.path.join(args.out, f"frame_task{t}.png")
        Image.fromarray(frame).save(p)
        print(f"görev {t}: {lang}\n  → {p}")
        tile = Image.fromarray(frame).copy()
        ImageDraw.Draw(tile).text((4, 4), f"task {t}", fill=(255, 255, 255))
        tiles.append(tile)
    sheet = Image.new("RGB", (256 * len(tiles), 256))
    for i, tl in enumerate(tiles):
        sheet.paste(tl, (256 * i, 0))
    sheet.save(os.path.join(args.out, "contact_sheet.png"))
    print("yazıldı:", os.path.join(args.out, "contact_sheet.png"))


if __name__ == "__main__":
    main()
