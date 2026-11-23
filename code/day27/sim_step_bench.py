"""Lecture 27 — The measurable half of the budget model: how many ms is one LIBERO simulation step?

The model call (GPU) and the simulation step (CPU: MuJoCo physics + offscreen render) together determine
the duration of an episode. We cannot measure the GPU half here; we measure the CPU half.
For each task: number of init states (LIBERO's ceiling for paired n), reset time, step time
(zero action; render included), 128 and 256 px cameras.

  python sim_step_bench.py --libero <LIBERO clone> [--steps 60] [--out results/sim_step.csv]

On the Mac: `conda activate vla` (the Lecture 9–10 environment). On a headless Linux server use MUJOCO_GL=egl or osmesa.
"""
import argparse, csv, os, platform, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--libero", required=True, help="root of the LIBERO repo (contains libero/libero)")
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--tasks", default="libero_spatial:0,libero_spatial:5,libero_goal:0,libero_goal:8")
ap.add_argument("--res", default="128,256")
ap.add_argument("--out", default="results/sim_step.csv")
a = ap.parse_args()

sys.path.insert(0, a.libero)
cfg_dir = os.path.abspath(os.path.join(os.path.dirname(a.out) or ".", ".libero_cfg"))
os.makedirs(cfg_dir, exist_ok=True)
root = os.path.join(os.path.abspath(a.libero), "libero", "libero")
with open(os.path.join(cfg_dir, "config.yaml"), "w") as f:  # skip LIBERO's interactive question
    f.write(f"benchmark_root: {root}\nbddl_files: {root}/bddl_files\ninit_states: {root}/init_files\n"
            f"datasets: {root}/../datasets\nassets: {root}/assets\n")
os.environ["LIBERO_CONFIG_PATH"] = cfg_dir

import torch  # the init state files are torch pickles
_load = torch.load
torch.load = lambda *x, **k: _load(*x, **{**k, "weights_only": False})  # torch>=2.6 default
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.env_wrapper import ControlEnv

rows = []
bench = benchmark.get_benchmark_dict()
for spec in a.tasks.split(","):
    suite, tid = spec.split(":"); tid = int(tid)
    ts = bench[suite](); task = ts.get_task(tid)
    inits = ts.get_task_init_states(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    for res in [0] + list(map(int, a.res.split(","))):   # 0 = render off: physics only
        if res == 0:
            env = ControlEnv(bddl_file_name=bddl, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        else:
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
        if res:
            env.seed(0)
        t0 = time.perf_counter(); env.reset(); env.set_init_state(inits[0]); t_reset = time.perf_counter() - t0
        dummy = np.zeros(7); dummy[-1] = -1.0  # the "wait" action of the LIBERO evals
        for _ in range(5):
            env.step(dummy)  # warm-up
        ts_ = []
        for _ in range(a.steps):
            t0 = time.perf_counter(); obs, r, done, info = env.step(dummy); ts_.append(time.perf_counter() - t0)
        env.close()
        ms = 1000 * np.array(ts_)
        row = dict(suite=suite, task_id=tid, n_init=len(inits), res=res, reset_s=round(t_reset, 2),
                   step_ms_median=round(float(np.median(ms)), 1), step_ms_p90=round(float(np.percentile(ms, 90)), 1),
                   img_shape="x".join(map(str, obs["agentview_image"].shape)) if res else "no render", language=task.language,
                   host=f"{platform.system()}-{platform.machine()}-{os.cpu_count()}cpu", gl=os.environ.get("MUJOCO_GL", "default"))
        rows.append(row); print(row, flush=True)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("written:", a.out)
