"""Ders 14 — LIBERO'dan tek bir agentview karesi üretir (OpenVLA'nın göreceği biçimde: 180° döndürülmüş).

Ders 10'un `vla` ortamında çalışır (mujoco 2.3.7, robosuite 1.4.1, LIBERO). GPU gerekmez.
    conda activate vla
    python render_libero_frame.py --suite libero_spatial --task 0 --init 0 --out frame_task0.png
Çıktı: 256x256 RGB PNG (`frame_task0.png`), yanında ham/çevrilmiş karşılaştırma `frame_task0_raw_vs_flipped.png`
ve komut metni (task.language) ekrana.
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
    ap.add_argument("--init", type=int, default=0, help="50 kayıtlı başlangıçtan hangisi")
    ap.add_argument("--settle", type=int, default=10, help="OpenVLA eval gibi 10 no-op adım bekle")
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
    flipped = raw[::-1, ::-1]                        # libero_utils.get_libero_image: 180° döndür
    Image.fromarray(flipped).save(args.out)
    side = np.concatenate([raw, flipped], axis=1)
    Image.fromarray(side).save(args.out.replace(".png", "_raw_vs_flipped.png"))
    print("komut (task.language):", task.language)
    print("yazıldı:", args.out, "| ham (sol) vs çevrilmiş (sağ):", args.out.replace(".png", "_raw_vs_flipped.png"))
    env.close()


if __name__ == "__main__":
    main()
