"""Lecture 17 — ACTUALLY renders the camera and robot-initial-state perturbations in robosuite (no LIBERO-Plus assets needed).

Why robosuite? The camera (`_view_`) and robot (`_initstate_`) perturbations need no new assets in LIBERO-Plus: one changes the
pose/quat of the agentview camera, the other the 7 joint angles. This script applies the same transformations (the LIBERO-plus math
from camera_geometry.py: z/y rotation around the pivot (0,0,0.8), scaling from the pivot, endpoint yaw/pitch) to a robosuite
table scene with a Panda and plots the frames. If LIBERO is installed, `--libero` does the same on LIBERO-Spatial task 0
(the Lecture 10 environment; `LIBERO_CONFIG_PATH` must be set).

Usage:  MUJOCO_GL=egl python render_perturbations.py <output.png> [--libero] [--size 256] [--robot-csv robot_init.csv]
  --robot-csv: the --csv output of robot_init_spread.py; if not given, random directions with ||Δ||=0.3 rad are generated (same distribution).
"""
import argparse, csv, os, sys
os.environ.setdefault("MUJOCO_GL", "egl" if os.uname().sysname == "Linux" else "glfw")
import numpy as np
from scipy.spatial.transform import Rotation

POS_AV = np.array([0.6586131746834771, 0.0, 1.6103500240372423])   # LIBERO tabletop agentview (problems/*.py)
QUAT_AV = np.array([0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349])  # (w,x,y,z)
PIVOT = np.array([0, 0, 0.8])
BASE_QPOS = np.array([0, -1.61037389e-01, 0.0, -2.44459747e00, 0.0, 2.22675220e00, np.pi / 4])  # MountedPanda.init_qpos


def q_wxyz_to_R(q):  # MuJoCo (w,x,y,z) → scipy Rotation
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def R_to_q_wxyz(R):
    x, y, z, w = R.as_quat(); return np.array([w, x, y, z])


def camera_pose(h=0, v=0, s=1.0, er=0, ev=0):
    """Same order as LIBERO-plus `_setup_camera`: vertical (y axis through the pivot) → horizontal (z) → scale → endpoint yaw → endpoint pitch."""
    pos, R = POS_AV.copy(), q_wxyz_to_R(QUAT_AV)
    if v:
        Ry = Rotation.from_rotvec(np.radians(-v) * np.array([0, 1, 0])); pos = Ry.apply(pos - PIVOT) + PIVOT; R = Ry * R
    if h:
        Rz = Rotation.from_euler("z", h, degrees=True); pos = Rz.apply(pos); R = Rz * R
    if s != 1.0:
        pos = PIVOT + (pos - PIVOT) * s
    if er:
        R = Rotation.from_euler("z", er, degrees=True) * R
    if ev:
        R = Rotation.from_rotvec(np.radians(-ev) * np.array([0, 1, 0])) * R
    return pos, R_to_q_wxyz(R)


def make_env_robosuite(size):
    import robosuite as suite
    env = suite.make("Stack", robots="Panda", has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
                     camera_names="agentview", camera_heights=size, camera_widths=size, control_freq=20)
    env.reset()
    return env, env.sim, "robosuite Stack (Panda); not a LIBERO scene"


def make_env_libero(size):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_spatial"](); task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=size, camera_widths=size); env.seed(0); env.reset()
    env.set_init_state(suite.get_task_init_states(0)[0])
    return env, env.sim, "LIBERO-Spatial task 0, init 0"


def render(sim, size, cam="agentview"):
    sim.forward()
    img = sim.render(camera_name=cam, width=size, height=size)
    return img[::-1]  # the MuJoCo offscreen image comes out upside down (Lecture 10/14)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--libero", action="store_true")
    ap.add_argument("--size", type=int, default=256); ap.add_argument("--robot-csv"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-frame", help="also write the default camera's frame to this PNG (input of noise_on_frame.py)")
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    env, sim, scene = (make_env_libero if a.libero else make_env_robosuite)(a.size)
    m, d = sim.model, sim.data
    cid = m.camera_name2id("agentview")
    orig_pos, orig_quat = m.cam_pos[cid].copy(), m.cam_quat[cid].copy()
    print("scene:", scene, "| the scene's own agentview pos", np.round(orig_pos, 3), "quat", np.round(orig_quat, 3))
    # robot joint addresses (Panda 7 joints)
    jids = [m.joint_name2id(n) for n in m.joint_names if n.startswith("robot0_joint")][:7]
    qadr = [m.jnt_qposadr[j] for j in jids]
    base_q = d.qpos[qadr].copy(); print("the scene's robot qpos", np.round(base_q, 3))

    cams = [("default (LIBERO agentview)", dict()), ("horizontal +30°", dict(h=30)), ("horizontal −45°", dict(h=315)),
            ("horizontal +75° (endpoint)", dict(h=75)), ("vertical 15° + horizontal 30°", dict(v=15, h=30)), ("distance ×1.5", dict(s=1.5)),
            ("distance ×2.0", dict(s=2.0)), ("endpoint yaw +10°, pitch −10°", dict(er=10, ev=350))]
    frames = []
    for title, kw in cams:
        pos, quat = camera_pose(**kw); m.cam_pos[cid] = pos; m.cam_quat[cid] = quat
        frames.append((title, render(sim, a.size)))
    if a.save_frame:
        from PIL import Image; Image.fromarray(frames[0][1]).save(a.save_frame); print("frame written:", a.save_frame)
    m.cam_pos[cid], m.cam_quat[cid] = camera_pose()  # robot frames with the default camera
    # robot initial poses: ||Δ|| = 0.1 / 0.3 / 0.5 rad (three of LIBERO-Plus's 5 fixed magnitudes)
    deltas = []
    if a.robot_csv:
        rows = list(csv.DictReader(open(a.robot_csv, encoding="utf-8")))
        for target in (0.1, 0.3, 0.5):
            r = next(r for r in rows if abs(float(r["norm_delta"]) - target) < 1e-3)
            deltas.append((f"MountedPanda{r['init_state']} (||Δ||={target})", np.array([float(r[f'q{i+1}']) for i in range(7)]) - BASE_QPOS))
    else:
        for target in (0.1, 0.3, 0.5):
            v = rng.normal(size=7); deltas.append((f"random direction, ||Δ||={target}", v / np.linalg.norm(v) * target))
    rob = []
    q0 = BASE_QPOS if a.libero else base_q  # robosuite: its own base; LIBERO: the MountedPanda base
    for title, dq in [("robot base", np.zeros(7))] + deltas:
        d.qpos[qadr] = q0 + dq; d.qvel[:] = 0
        rob.append((title, render(sim, a.size)))
    d.qpos[qadr] = base_q; m.cam_pos[cid], m.cam_quat[cid] = orig_pos, orig_quat

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4, figsize=(13, 10.2))
    for ax, (t, im) in zip(axes[:2].ravel(), frames): ax.imshow(im); ax.set_title(t, fontsize=9.5)
    for ax, (t, im) in zip(axes[2], rob): ax.imshow(im); ax.set_title(t, fontsize=9.5)
    for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("Camera Viewpoints", fontsize=11); axes[1, 0].set_ylabel("Camera Viewpoints", fontsize=11); axes[2, 0].set_ylabel("Robot Initial States", fontsize=11)
    fig.suptitle(f"LIBERO-Plus camera / robot perturbations, real render — scene: {scene} — {a.size}×{a.size} agentview (render_perturbations.py)", fontsize=10.5)
    fig.text(0.5, 0.005, "Camera: LIBERO-plus `_setup_camera` order (vertical→horizontal→scale→endpoint yaw/pitch), pivot (0,0,0.8). Robot: 7 joint angles base + Δ (in LIBERO-Plus ||Δ|| ∈ {0.1,…,0.5} rad).",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96)); fig.savefig(a.out, dpi=150); plt.close(fig)
    print("written:", a.out)
    try: env.close()
    except Exception: pass


if __name__ == "__main__":
    main()
