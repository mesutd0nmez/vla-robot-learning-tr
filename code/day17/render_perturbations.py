"""Ders 17 — Kamera ve robot-başlangıç bozulmalarını robosuite'te GERÇEKTEN render eder (LIBERO-Plus asset'i gerekmez).

Neden robosuite? Kamera (`_view_`) ve robot (`_initstate_`) bozulmaları LIBERO-Plus'ta yeni asset istemez: biri
agentview kamerasının poz/quat'ını, öteki 7 eklem açısını değiştirir. Bu script aynı dönüşümleri (camera_geometry.py'deki
LIBERO-plus matematiği: pivot (0,0,0.8) etrafında z/y dönüşü, pivot'tan ölçekleme, uç yaw/pitch) robosuite'in Panda'lı
bir masa sahnesine uygular ve kareleri çizer. LIBERO kuruluysa `--libero` ile aynı şeyi LIBERO-Spatial görev 0'da yapar
(Ders 10 ortamı; `LIBERO_CONFIG_PATH` ayarlı olmalı).

Kullanım:  MUJOCO_GL=egl python render_perturbations.py <çıktı.png> [--libero] [--size 256] [--robot-csv robot_init.csv]
  --robot-csv: robot_init_spread.py --csv çıktısı; verilmezse ||Δ||=0.3 rad'lık rastgele yönler üretilir (aynı dağılım).
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
    """LIBERO-plus `_setup_camera` ile aynı sıra: dikey (y ekseni, pivot'tan geçen) → yatay (z) → ölçek → uç yaw → uç pitch."""
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
    return env, env.sim, "robosuite Stack (Panda); LIBERO sahnesi değil"


def make_env_libero(size):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_spatial"](); task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=size, camera_widths=size); env.seed(0); env.reset()
    env.set_init_state(suite.get_task_init_states(0)[0])
    return env, env.sim, "LIBERO-Spatial görev 0, init 0"


def render(sim, size, cam="agentview"):
    sim.forward()
    img = sim.render(camera_name=cam, width=size, height=size)
    return img[::-1]  # MuJoCo offscreen görüntüsü ters gelir (Ders 10/14)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--libero", action="store_true")
    ap.add_argument("--size", type=int, default=256); ap.add_argument("--robot-csv"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-frame", help="varsayılan kameranın karesini bu PNG'ye de yaz (noise_on_frame.py girdisi)")
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    env, sim, scene = (make_env_libero if a.libero else make_env_robosuite)(a.size)
    m, d = sim.model, sim.data
    cid = m.camera_name2id("agentview")
    orig_pos, orig_quat = m.cam_pos[cid].copy(), m.cam_quat[cid].copy()
    print("sahne:", scene, "| sahnenin kendi agentview'ı pos", np.round(orig_pos, 3), "quat", np.round(orig_quat, 3))
    # robot eklem adresleri (Panda 7 eklem)
    jids = [m.joint_name2id(n) for n in m.joint_names if n.startswith("robot0_joint")][:7]
    qadr = [m.jnt_qposadr[j] for j in jids]
    base_q = d.qpos[qadr].copy(); print("sahnenin robot qpos'u", np.round(base_q, 3))

    cams = [("varsayılan (LIBERO agentview)", dict()), ("yatay +30°", dict(h=30)), ("yatay −45°", dict(h=315)),
            ("yatay +75° (uç)", dict(h=75)), ("dikey 15° + yatay 30°", dict(v=15, h=30)), ("mesafe ×1.5", dict(s=1.5)),
            ("mesafe ×2.0", dict(s=2.0)), ("uç yaw +10°, pitch −10°", dict(er=10, ev=350))]
    frames = []
    for title, kw in cams:
        pos, quat = camera_pose(**kw); m.cam_pos[cid] = pos; m.cam_quat[cid] = quat
        frames.append((title, render(sim, a.size)))
    if a.save_frame:
        from PIL import Image; Image.fromarray(frames[0][1]).save(a.save_frame); print("kare yazıldı:", a.save_frame)
    m.cam_pos[cid], m.cam_quat[cid] = camera_pose()  # robot kareleri varsayılan kamerayla
    # robot başlangıç pozları: ||Δ|| = 0.1 / 0.3 / 0.5 rad (LIBERO-Plus'ın 5 sabit büyüklüğünden üçü)
    deltas = []
    if a.robot_csv:
        rows = list(csv.DictReader(open(a.robot_csv, encoding="utf-8")))
        for target in (0.1, 0.3, 0.5):
            r = next(r for r in rows if abs(float(r["norm_delta"]) - target) < 1e-3)
            deltas.append((f"MountedPanda{r['init_state']} (||Δ||={target})", np.array([float(r[f'q{i+1}']) for i in range(7)]) - BASE_QPOS))
    else:
        for target in (0.1, 0.3, 0.5):
            v = rng.normal(size=7); deltas.append((f"rastgele yön, ||Δ||={target}", v / np.linalg.norm(v) * target))
    rob = []
    q0 = BASE_QPOS if a.libero else base_q  # robosuite'te kendi tabanı; LIBERO'da MountedPanda tabanı
    for title, dq in [("robot tabanı", np.zeros(7))] + deltas:
        d.qpos[qadr] = q0 + dq; d.qvel[:] = 0
        rob.append((title, render(sim, a.size)))
    d.qpos[qadr] = base_q; m.cam_pos[cid], m.cam_quat[cid] = orig_pos, orig_quat

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4, figsize=(13, 10.2))
    for ax, (t, im) in zip(axes[:2].ravel(), frames): ax.imshow(im); ax.set_title(t, fontsize=9.5)
    for ax, (t, im) in zip(axes[2], rob): ax.imshow(im); ax.set_title(t, fontsize=9.5)
    for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("Camera Viewpoints", fontsize=11); axes[1, 0].set_ylabel("Camera Viewpoints", fontsize=11); axes[2, 0].set_ylabel("Robot Initial States", fontsize=11)
    fig.suptitle(f"LIBERO-Plus kamera / robot bozulmaları gerçek render — sahne: {scene} — {a.size}×{a.size} agentview (render_perturbations.py)", fontsize=10.5)
    fig.text(0.5, 0.005, "Kamera: LIBERO-plus `_setup_camera` sırası (dikey→yatay→ölçek→uç yaw/pitch), pivot (0,0,0.8). Robot: 7 eklem açısı taban + Δ (LIBERO-Plus'ta ||Δ|| ∈ {0.1,…,0.5} rad).",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96)); fig.savefig(a.out, dpi=150); plt.close(fig)
    print("yazıldı:", a.out)
    try: env.close()
    except Exception: pass


if __name__ == "__main__":
    main()
