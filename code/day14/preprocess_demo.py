"""Ders 14 — OpenVLA'nın görüntü ön işleme zincirini gerçek bir robosuite karesinde gösterir (model GEREKMEZ).

ham agentview (256x256, robosuite ters verir) → img[::-1, ::-1] (180°) → 224x224 (lanczos) → center crop %90 → 224x224
Kaynak: experiments/robot/libero/libero_utils.py:50-58 ve experiments/robot/openvla_utils.py:132-156.
LIBERO kuruluysa (Ders 10) `--libero` ile aynı zinciri LIBERO Spatial görev 0'da çalıştırır; yoksa robosuite Lift.
    python preprocess_demo.py --out 05-preprocess-pipeline.png
"""
import argparse
import math
import os

os.environ.setdefault("MUJOCO_GL", "egl" if os.uname().sysname == "Linux" else "glfw")

import numpy as np
from PIL import Image


def get_frame_robosuite():
    import robosuite as suite
    env = suite.make("Lift", robots="Panda", has_renderer=False, has_offscreen_renderer=True,
                     use_camera_obs=True, camera_names=["agentview"], camera_heights=256, camera_widths=256,
                     control_freq=20)
    obs = env.reset()
    for _ in range(10):
        obs, _, _, _ = env.step(np.zeros(env.action_dim))
    env.close()
    return obs["agentview_image"], "robosuite Lift (Panda)"


def get_frame_libero(task_id=0):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    s = benchmark.get_benchmark_dict()["libero_spatial"]()
    t = s.get_task(task_id)
    env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file),
                             camera_heights=256, camera_widths=256)
    env.reset()
    obs = env.set_init_state(s.get_task_init_states(task_id)[0])
    for _ in range(10):
        obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
    env.close()
    return obs["agentview_image"], f"LIBERO Spatial görev {task_id}: {t.language}"


def center_crop(img: Image.Image, crop_scale=0.9):
    w, h = img.size
    s = math.sqrt(crop_scale)
    cw, ch = round(w * s), round(h * s)
    l, t = (w - cw) // 2, (h - ch) // 2
    return img.crop((l, t, l + cw, t + ch)).resize((w, h), Image.BILINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero", action="store_true")
    ap.add_argument("--out", default="05-preprocess-pipeline.png")
    a = ap.parse_args()
    raw, title = get_frame_libero() if a.libero else get_frame_robosuite()
    flipped = raw[::-1, ::-1]
    r224 = Image.fromarray(flipped).resize((224, 224), Image.LANCZOS)
    cropped = center_crop(r224)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, im, cap in zip(axes, [raw, flipped, np.array(r224), np.array(cropped)],
                           ["1) ham obs['agentview_image'] 256x256\n(robosuite ters verir)",
                            "2) img[::-1, ::-1] — 180° döndür\n(libero_utils.py:56)",
                            "3) lanczos → 224x224\n(libero_utils.py:44)",
                            "4) center crop %90 alan → 224x224\n(openvla_utils.py:147)  → modele giden"]):
        ax.imshow(im)
        ax.set_title(cap, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"OpenVLA görüntü ön işleme — {title}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(a.out, dpi=150)
    Image.fromarray(np.array(cropped)).save(a.out.replace(".png", "_model_input.png"))
    print("yazıldı:", a.out)


if __name__ == "__main__":
    main()
