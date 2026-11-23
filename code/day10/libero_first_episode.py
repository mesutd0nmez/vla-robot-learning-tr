"""Day 10 — First LIBERO episode.
Opens one LIBERO-Spatial task, sets it to a fixed initial state, applies 60 steps of simple actions,
and saves the agentview + wrist camera frames as a PNG.
Run (vla environment active):  python libero_first_episode.py --task 0
"""
import os, argparse, numpy as np, imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

p = argparse.ArgumentParser()
p.add_argument("--suite", default="libero_spatial")   # libero_spatial | libero_object | libero_goal | libero_10
p.add_argument("--task", type=int, default=0)         # 0..9
p.add_argument("--steps", type=int, default=60)
args = p.parse_args()

suite = benchmark.get_benchmark_dict()[args.suite]()
task = suite.get_task(args.task)
print("Task      :", task.name)
print("Command   :", task.language)                    # <- the English instruction that goes to the VLA
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0)
env.reset()
init_states = suite.get_task_init_states(args.task)   # 50 fixed initial states (benchmark protocol)
obs = env.set_init_state(init_states[0])
print("obs keys:", [k for k in obs.keys() if "image" in k or "eef" in k or "gripper" in k])

frames = []
for t in range(args.steps):
    a = np.zeros(7)
    a[2] = -0.3 if t < 20 else 0.0      # move down for 20 steps
    a[6] = -1.0 if t < 30 else 1.0      # gripper: -1 open, +1 closed
    obs, reward, done, info = env.step(a)
    if t % 20 == 0 or t == args.steps - 1:
        # LIBERO images come out upside down; OpenVLA rotates them by 180° ([::-1, ::-1])
        top = obs["agentview_image"][::-1, ::-1]
        wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1]
        frames.append(np.concatenate([top, wrist], axis=0))
print("Success?  :", env.check_success(), "| last reward:", reward)
env.close()
imageio.imwrite("libero_first_episode.png", np.concatenate(frames, axis=1))
print("saved: libero_first_episode.png")
