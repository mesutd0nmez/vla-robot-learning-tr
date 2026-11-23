# day21 — Webots: first scene, extern Python controller, camera recording, FK check

The files of Lecture 21. Assumes Webots **R2025a** (macOS dmg; Apple Silicon + Intel in a single package) and the conda `vla` environment (Python 3.10).
Webots itself could not be run in this session (cloud restriction); `check_webots_env.py` (against the Linux tarball), `ur10e_fk.py`,
`plot_day21.py` and the controller's logic (a dry run with a fake `controller` module) were actually run.

```bash
# 1) Install Webots (downloading from the Terminal avoids the Gatekeeper quarantine flag)
curl -L -O https://github.com/cyberbotics/webots/releases/download/R2025a/webots-R2025a.dmg
open webots-R2025a.dmg && cp -r /Volumes/Webots/Webots.app /Applications/
# 2) environment
conda activate vla
export WEBOTS_HOME=/Applications/Webots.app
python check_webots_env.py                      # you should see import controller : OK
# 3) In Webots: File > Open World... > code/day21/worlds/ders21_masa.wbt, press ▶ to play (the robot waits for the controller)
# 4) in a separate terminal (vla active):
$WEBOTS_HOME/Contents/MacOS/webots-controller controllers/ders21_kol/ders21_kol.py --out out
#    if the launcher fails on an arm Mac (known R2025a bug, PR #6806):
PYTHONPATH=$WEBOTS_HOME/Contents/lib/controller/python python controllers/ders21_kol/ders21_kol.py --out out
# 5) Checks without Webots
python ur10e_fk.py                              # tool position of the 4 poses, above the table?
python ur10e_fk.py --solve                      # joint angles for the targets (scipy)
python plot_day21.py --out 05-decision.png      # decision matrix
```

To download: `curl -O https://raw.githubusercontent.com/mesutd0nmez/vla-robot-learning-tr/main/code/day21/<file>`
(for the world and the controller: `worlds/ders21_masa.wbt`, `controllers/ders21_kol/ders21_kol.py`).

## check_webots_env.py

`check_webots_env.py`

```python
"""Lecture 21 — Webots + conda `vla` connection check (runs without opening Webots).

Checks: is WEBOTS_HOME correct, does the Python API folder exist, can libController be loaded,
where is the launcher (webots-controller), which environment python3 comes from.
Run (vla active):  WEBOTS_HOME=/Applications/Webots.app python check_webots_env.py
"""
import os
import platform
import sys

home = os.environ.get("WEBOTS_HOME")
print("python      :", sys.executable, sys.version.split()[0], "| arch:", platform.machine())
print("WEBOTS_HOME :", home)
if not home:
    print("  -> export WEBOTS_HOME=/Applications/Webots.app   (macOS)  |  .../webots (Linux tarball)")
    sys.exit(1)
mac = sys.platform == "darwin"
pyapi = os.path.join(home, "Contents", "lib", "controller", "python") if mac else os.path.join(home, "lib", "controller", "python")
lib = os.path.join(home, "Contents", "lib", "controller", "libController.dylib") if mac else os.path.join(home, "lib", "controller", "libController.so")
launcher = os.path.join(home, "Contents", "MacOS", "webots-controller") if mac else os.path.join(home, "webots-controller")
for label, path in [("Python API", pyapi), ("libController", lib), ("launcher", launcher)]:
    print(f"{label:12s}: {'OK  ' if os.path.exists(path) else 'MISSING '} {path}")
if platform.machine() == "arm64" and mac:
    print("Apple Silicon: in R2025a the launcher has a bug and does not run on arm Macs (PR #6806, fixed in R2025b).")
    print("  If the launcher fails: PYTHONPATH=" + pyapi + " python ders21_kol.py  (with Webots open)")
sys.path.insert(0, pyapi)
try:
    import controller  # noqa: F401
    from controller import Supervisor, Camera, Motor  # noqa: F401
    print("import controller : OK  (loaded via ctypes:", os.path.basename(lib), "; the numpy/torch of your environment can be used as is)")
except Exception as e:  # pragma: no cover
    print("import controller : ERROR ->", repr(e))
    print("  Possible cause: wrong WEBOTS_HOME, Webots opened under Rosetta (x86_64 dylib), or Python < 3.8")
    sys.exit(2)
print("All good. Now open ders21_masa.wbt in Webots, press play, then start ders21_kol.py.")
```

## ders21_kol.py

`controllers/ders21_kol/ders21_kol.py`

```python
"""Lecture 21 — first arm motion in Webots + camera recording + spatial relation table.

Moves the UR10e (supervisor TRUE, controller <extern>) through four intermediate poses, at each pose saves
the wrist camera and the main 3D view (agentview-like) as PNG/JPG,
reads the objects' world coordinates with the Supervisor API and writes the
left/right/front/back relations with respect to the robot to CSV.

Run (with ders21_masa.wbt open and playing in Webots, in a separate terminal, vla env active):
  macOS : export WEBOTS_HOME=/Applications/Webots.app
          $WEBOTS_HOME/Contents/MacOS/webots-controller code/day21/controllers/ders21_kol/ders21_kol.py --out out
  Linux : $WEBOTS_HOME/webots-controller code/day21/controllers/ders21_kol/ders21_kol.py --out out
If the launcher does not work (see README "without the launcher"):
  PYTHONPATH=$WEBOTS_HOME/Contents/lib/controller/python python ders21_kol.py --out out
"""
import argparse
import csv
import math
import os
import sys

from controller import Supervisor  # Webots' ctypes-based Python API (needs WEBOTS_HOME)

p = argparse.ArgumentParser()
p.add_argument("--out", default="out", help="folder the PNG/JPG/CSV files are written to")
p.add_argument("--settle", type=int, default=60, help="number of steps to wait at each pose")
p.add_argument("--quit", action="store_true", help="quit Webots when done (for CI / headless runs)")
args, _ = p.parse_known_args()
os.makedirs(args.out, exist_ok=True)

robot = Supervisor()
dt = int(robot.getBasicTimeStep())          # basicTimeStep in the world (16 ms)

# --- actuators: the 6 joints of the UR10e (RotationalMotor names in the PROTO) ---
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
motors = [robot.getDevice(j) for j in JOINTS]
sensors = [robot.getDevice(j + "_sensor") for j in JOINTS]
for s in sensors:
    s.enable(dt)
for m in motors:
    m.setVelocity(1.0)                       # rad/s; maxVelocity is 3.14 / 6.28 in the PROTO

# --- sensor: the wrist camera we put on toolSlot ---
cam = robot.getDevice("wrist_camera")
cam.enable(dt)

# --- Supervisor: objects in the scene (DEF names in the world) ---
DEFS = ["BOWL", "PLATE", "STOVE", "CEREAL", "APPLE", "CAN"]
nodes = {d: robot.getFromDef(d) for d in DEFS}
robot_node = robot.getSelf()

# Four poses (rad): [pan, lift, elbow, w1, w2, w3]. The arm leans over the table.
POSES = {
    "00_home":   [0.000, -1.800, 1.200, -1.570, 0.000, 0.000],
    "01_bowl":   [0.582, -0.542, -1.637, -2.533, -1.571, 0.000],
    "02_center": [0.194, -0.868, -1.055, -2.789, -1.571, 0.000],
    "03_plate":  [-0.068, -0.568, -1.593, -2.552, -1.571, 0.000],
}


def world_to_robot(pw, R, t):
    """World point -> robot base frame (R: 3x3 row-major, t: base position)."""
    d = [pw[i] - t[i] for i in range(3)]
    # R is world<-robot; robot coordinate = R^T d
    return [sum(R[3 * k + i] * d[k] for k in range(3)) for i in range(3)]


rows = []
for name, q in POSES.items():
    for m, qi in zip(motors, q):
        m.setPosition(qi)
    for _ in range(args.settle):          # wait for the arm to reach the target (synchronization TRUE)
        if robot.step(dt) == -1:
            sys.exit(0)
    # camera frames
    cam.saveImage(os.path.join(args.out, f"wrist_{name}.png"), 100)
    robot.exportImage(os.path.join(args.out, f"view_{name}.jpg"), 95)
    # joint readings
    qs = [s.getValue() for s in sensors]
    err = max(abs(a - b) for a, b in zip(qs, q))
    print(f"[{name}] t={robot.getTime():6.2f}s  max|q-q*|={err:.3f} rad  "
          + " ".join(f"{v:+.2f}" for v in qs))
    rows.append({"pose": name, "t": round(robot.getTime(), 3), "max_err_rad": round(err, 4),
                 **{j: round(v, 4) for j, v in zip(JOINTS, qs)}})

with open(os.path.join(args.out, "joints.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# --- spatial relations: the Supervisor reads the object positions ---
# The UR10e's "front" is −x in the base frame (shoulder_lift = −90° extends the arm toward −x); since the base is
# rotated −90° about the z axis in the world, its front is world +y and its left is world −x. Below, forward = −x_r, left = −y_r.
t = robot_node.getPosition()
R = robot_node.getOrientation()           # 9 elements, row-major
rel_rows = []
for d, n in nodes.items():
    pw = n.getPosition()
    pr = world_to_robot(pw, R, t)          # base frame
    fwd, left = -pr[0], -pr[1]             # forward (toward the table) and left
    rel_rows.append({"def": d, "x_w": round(pw[0], 3), "y_w": round(pw[1], 3), "z_w": round(pw[2], 3),
                     "ileri_r": round(fwd, 3), "sol_r": round(left, 3),
                     "sol_sag": "sol" if left > 0.03 else ("sag" if left < -0.03 else "orta"),
                     "on_arka": "yakin" if fwd < 0.75 else "uzak"})
    print(f"{d:7s} world=({pw[0]:+.2f},{pw[1]:+.2f},{pw[2]:+.2f})  robot(forward,left)=({fwd:+.2f},{left:+.2f})"
          f"  -> {rel_rows[-1]['sol_sag']}, {rel_rows[-1]['on_arka']}")

# pairwise relations (in the robot frame): is A to the left of / right of / in front of / behind B?
pairs = []
for a in rel_rows:
    for b in rel_rows:
        if a["def"] == b["def"]:
            continue
        d_sol = a["sol_r"] - b["sol_r"]        # >0: A is to the left of B (relative to the robot)
        d_ileri = a["ileri_r"] - b["ileri_r"]  # >0: A is behind B (farther from the robot)
        pairs.append({"a": a["def"], "b": b["def"],
                      "d_sol": round(d_sol, 3), "d_ileri": round(d_ileri, 3),
                      "iliski_tr": ("solunda" if d_sol > 0.05 else "sağında" if d_sol < -0.05 else "aynı hizada")
                                   + " / " + ("gerisinde" if d_ileri > 0.05 else "önünde" if d_ileri < -0.05 else "aynı derinlikte")})
for fname, data in [("objects.csv", rel_rows), ("pairs.csv", pairs)]:
    with open(os.path.join(args.out, fname), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)
print(f"Written: {args.out}/wrist_*.png, view_*.jpg, joints.csv, objects.csv, pairs.csv")
print(f"basicTimeStep={dt} ms; simulation time={robot.getTime():.2f} s")
if args.quit:
    robot.simulationQuit(0)
robot.step(dt)
```

## ur10e_fk.py

`ur10e_fk.py`

```python
"""Lecture 21 — UR10e forward kinematics, with the chain read from the Webots PROTO (Webots not required).

The HingeJoint anchor/axis and endPoint translation values in UR10e.proto are used exactly:
  J1 pan   : axis z, (0, 0, 0.181)      J4 wrist_1 : axis y, (0, 0, 0.571)
  J2 lift  : axis y, (0, 0.176, 0)      J5 wrist_2 : axis z, (0, 0.135, 0)
  J3 elbow : axis y, (0, -0.137, 0.613) J6 wrist_3 : axis y, (0, 0, 0.12)
  toolSlot : (0, 0.12, 0) — the camera is here
Robot base as in ders21_masa.wbt: translation (0, -0.65, 0.74), rotation z 90°.

Output: the tool (camera) position for the four poses in ders21_kol.py, a check whether it is above the table,
and with --plot a top/side view (lectures/images/lecture-21/04-scene-fk.png).
Run:  python ur10e_fk.py [--plot out.png]
"""
import argparse
import math

import numpy as np

CHAIN = [  # (axis, link offset) — in proto order
    ("z", (0, 0, 0.181)),
    ("y", (0, 0.176, 0)),
    ("y", (0, -0.137, 0.613)),
    ("y", (0, 0, 0.571)),
    ("z", (0, 0.135, 0)),
    ("y", (0, 0, 0.12)),
]
TOOL = (0, 0.12, 0)
BASE_T = (0.0, -0.65, 0.74)
BASE_YAW = -math.pi / 2

POSES = {
    "00_home":   [0.000, -1.800, 1.200, -1.570, 0.000, 0.000],
    "01_bowl":   [0.582, -0.542, -1.637, -2.533, -1.571, 0.000],
    "02_center": [0.194, -0.868, -1.055, -2.789, -1.571, 0.000],
    "03_plate":  [-0.068, -0.568, -1.593, -2.552, -1.571, 0.000],
}
# objects in ders21_masa.wbt (DEF, world position, drawing radius m, color)
OBJECTS = [
    ("BOWL",   (-0.22, 0.00, 0.74), 0.09, "#222222"),
    ("PLATE",  (0.22, 0.02, 0.74), 0.11, "#dddddd"),
    ("STOVE",  (0.00, 0.50, 0.74), 0.19, "#888888"),
    ("CEREAL", (-0.35, 0.45, 0.74), 0.08, "#e0a060"),
    ("APPLE",  (0.05, 0.27, 0.79), 0.05, "#c0392b"),
    ("CAN",    (0.36, 0.40, 0.80), 0.035, "#2980b9"),
]
TABLE = (1.0, 1.8, 0.74)  # x, y, height


def rot(axis, q):
    c, s = math.cos(q), math.sin(q)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def fk(q):
    """World position of every joint anchor and of the tool (7+1 points)."""
    R = rot("z", BASE_YAW)
    t = np.array(BASE_T, float)
    pts = [t.copy()]
    for (axis, trans), qi in zip(CHAIN, q):
        t = t + R @ np.array(trans, float)   # Webots: endPoint translation == anchor
        R = R @ rot(axis, qi)                 # joint rotation about the anchor
        pts.append(t.copy())
    tool = t + R @ np.array(TOOL, float)
    pts.append(tool)
    return np.array(pts), R


def on_table(p):
    return abs(p[0]) <= TABLE[0] / 2 and abs(p[1]) <= TABLE[1] / 2 and p[2] > TABLE[2]


TARGETS = {  # --solve: the point where the camera should arrive (26 cm above the object), camera looking down
    "01_bowl":   (-0.22, 0.00, 1.00),
    "02_center": (0.00, 0.25, 1.00),
    "03_plate":  (0.22, 0.02, 1.00),
}


def solve(target, q0):
    """Simple numerical IK (scipy): tool at the target + the +y axis of toolSlot (camera view direction) pointing along world −z."""
    from scipy.optimize import minimize

    def cost(q):
        pts, R = fk(q)
        pos_err = np.sum((pts[-1] - np.array(target)) ** 2)
        look = R @ np.array([0, 1, 0])           # toolSlot +y = camera view direction (Camera rotation z 90°)
        ori_err = np.sum((look - np.array([0, 0, -1])) ** 2)
        below = np.sum(np.clip(TABLE[2] + 0.05 - pts[1:, 2], 0, None) ** 2)  # joints must not go below the table
        return 100 * pos_err + 10 * ori_err + 1000 * below + 0.01 * np.sum((q - np.array(q0)) ** 2)

    rng = np.random.default_rng(21)
    best = None
    starts = [np.array(q0)] + [np.array(q0) + rng.uniform(-0.8, 0.8, 6) for _ in range(40)]
    for s in starts:
        r = minimize(cost, s, method="L-BFGS-B", bounds=[(-3.14, 3.14)] * 6)
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", default=None)
    ap.add_argument("--solve", action="store_true", help="find and print the joint angles for TARGETS with numerical IK")
    a = ap.parse_args()
    if a.solve:
        q0 = [0.0, -1.2, 1.4, -1.8, -1.57, 0.0]
        for name, tgt in TARGETS.items():
            q, f = solve(tgt, q0)
            pts, R = fk(q)
            print(f'    "{name}": [' + ", ".join(f"{v:.3f}" for v in q) + f"],  # tool={np.round(pts[-1], 3).tolist()} cost={f:.4f}")
        return
    print(f"{'pose':10s} {'tool x':>8s} {'tool y':>8s} {'tool z':>8s}  on_table  lowest_joint_z")
    results = {}
    for name, q in POSES.items():
        pts, R = fk(q)
        tool = pts[-1]
        results[name] = pts
        print(f"{name:10s} {tool[0]:8.3f} {tool[1]:8.3f} {tool[2]:8.3f}  {str(on_table(tool)):9s}  {pts[1:].min(axis=0)[2]:.3f}")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2))
        colors = {"00_home": "#999999", "01_bowl": "#ffd166", "02_center": "#cfe8ff", "03_plate": "#d4f5d4"}
        # top view
        ax1.add_patch(Rectangle((-TABLE[0] / 2, -TABLE[1] / 2), TABLE[0], TABLE[1], fc="#f3e2c3", ec="#7a5c2e", lw=1.5))
        for d, p, r, c in OBJECTS:
            ax1.add_patch(Circle((p[0], p[1]), r, fc=c, ec="k", lw=0.8, alpha=0.9))
            ax1.text(p[0], p[1] + r + 0.03, d, ha="center", fontsize=9)
        for name, pts in results.items():
            ax1.plot(pts[:-1, 0], pts[:-1, 1], "-o", color=colors[name], ms=4, lw=2.5, mec="k")
            ax1.plot(pts[-1, 0], pts[-1, 1], "*", color=colors[name], ms=16, mec="k", label=f"{name} (tool)")
        ax1.plot(*BASE_T[:2], "ks", ms=9, label="UR10e base")
        ax1.annotate("agentview camera\n(0, 1.75, 1.55)", xy=(0, 0.9), xytext=(0, 1.15), ha="center", fontsize=9,
                     arrowprops=dict(arrowstyle="->"))
        ax1.set_xlim(-0.9, 0.9); ax1.set_ylim(-1.0, 1.3); ax1.set_aspect("equal")
        ax1.set_xlabel("world x (m)  — the robot faces +y; the robot's LEFT = −x, RIGHT = +x")
        ax1.set_ylabel("world y (m)")
        ax1.set_title("Top view: ders21_masa.wbt objects + the arm in 4 poses (FK, proto geometry)")
        ax1.legend(fontsize=8, loc="lower left")
        # side view (y-z)
        ax2.add_patch(Rectangle((-TABLE[1] / 2, 0), TABLE[1], TABLE[2], fc="#f3e2c3", ec="#7a5c2e", lw=1.5))
        for d, p, r, c in OBJECTS:
            ax2.add_patch(Circle((p[1], p[2] + r * 0.5), r * 0.6, fc=c, ec="k", lw=0.8, alpha=0.9))
        for name, pts in results.items():
            ax2.plot(pts[:-1, 1], pts[:-1, 2], "-o", color=colors[name], ms=4, lw=2.5, mec="k")
            ax2.plot(pts[-1, 1], pts[-1, 2], "*", color=colors[name], ms=16, mec="k")
        ax2.axhline(TABLE[2], color="#7a5c2e", ls="--", lw=1)
        ax2.set_xlim(-1.0, 1.0); ax2.set_ylim(0, 2.0); ax2.set_aspect("equal")
        ax2.set_xlabel("world y (m)"); ax2.set_ylabel("world z (m)")
        ax2.set_title("Side view: does the arm stay above the table? (dashed line = table surface 0.74 m)")
        fig.suptitle("Lecture 21 — Geometry of the Webots scene: objects from the .wbt, the arm computed from the UR10e.proto chain (Webots was not run)",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=150)
        print("saved:", a.plot)


if __name__ == "__main__":
    main()
```

## plot_day21.py

`plot_day21.py`

```python
"""Lecture 21 — figure 05: thesis environment decision matrix (LIBERO+MuJoCo vs Webots).

The scores are SUBJECTIVE (Claude's assessment in this lecture, 0–3); not a source, a starting point for discussion.
After talking to your advisor, change the scores and redraw:  python plot_day21.py --out 05-decision.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CRITERIA = [
    ("Ready-made benchmark + leaderboard (Spatial/Goal, Plus, CF, X)", 3, 0),
    ("VLA eval code available (OpenVLA, VLANeXt, OFT…)",             3, 0),
    ("Training data (50 demos/task, HDF5, RLDS)",                   3, 0),
    ("Ease of installation / running on a Mac M2",                  1, 3),
    ("Room scale, mobile base (RoomSpace-style scene)",              0, 3),
    ("Scene/PROTO authoring, camera placement flexibility",         1, 3),
    ("ROS 2 bridge (transfer to a real robot)",                     0, 3),
    ("Time to the Turkish-command experiment: can it start 'today'", 3, 1),
    ("Comparability of results with the literature",                3, 0),
    ("Same robot (Panda) exists in both environments",              3, 2),
]
LABELS = ["LIBERO + robosuite + MuJoCo", "Webots R2025a"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="05-decision.png")
    a = ap.parse_args()
    M = np.array([[c[1], c[2]] for c in CRITERIA], float)
    fig, ax = plt.subplots(figsize=(11, 6.8))
    cmap = matplotlib.colors.ListedColormap(["#ffd6d6", "#fde9c4", "#d4f5d4", "#9be29b"])
    ax.imshow(M, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
    for i, (name, s1, s2) in enumerate(CRITERIA):
        for j, s in enumerate((s1, s2)):
            ax.text(j, i, ["none", "weak", "medium", "strong"][s], ha="center", va="center", fontsize=11, weight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_yticks(range(len(CRITERIA))); ax.set_yticklabels([c[0] for c in CRITERIA], fontsize=9.5)
    ax.xaxis.tick_top()
    tot = M.sum(axis=0)
    ax.set_xlabel(f"total (equal weights): {tot[0]:.0f} / {tot[1]:.0f}   —   the thesis question needs a benchmark, Webots is not a benchmark",
                  fontsize=10, labelpad=10)
    ax.set_title("Lecture 21 — Thesis environment decision matrix (SUBJECTIVE scoring 0–3; to be updated with the advisor)", fontsize=12, pad=38)
    for i in range(len(CRITERIA) + 1):
        ax.axhline(i - 0.5, color="white", lw=2)
    ax.axvline(0.5, color="white", lw=2)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("saved:", a.out)


if __name__ == "__main__":
    main()
```

## ders21_masa.wbt

`worlds/ders21_masa.wbt`

```text
#VRML_SIM R2025a utf8
# Lecture 21 — a table scene in Webots resembling LIBERO-Spatial:
# UR10e arm (with a wrist camera) + table + black bowl, plate, stove, cereal box, apple, soda can.
# To open: Webots > File > Open World... > this file. The arm controller is <extern>: started from the terminal (see README).

EXTERNPROTO "webots://projects/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "webots://projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"
EXTERNPROTO "webots://projects/objects/floors/protos/Floor.proto"
EXTERNPROTO "webots://projects/objects/tables/protos/Table.proto"
EXTERNPROTO "webots://projects/robots/universal_robots/protos/UR10e.proto"
EXTERNPROTO "webots://projects/objects/fruits/protos/FruitBowl.proto"
EXTERNPROTO "webots://projects/objects/fruits/protos/Apple.proto"
EXTERNPROTO "webots://projects/objects/kitchen/utensils/protos/Plate.proto"
EXTERNPROTO "webots://projects/objects/kitchen/components/protos/HotPlate.proto"
EXTERNPROTO "webots://projects/objects/kitchen/breakfast/protos/CerealBox.proto"
EXTERNPROTO "webots://projects/objects/drinks/protos/Can.proto"

WorldInfo {
  info [
    "Lecture 21: UR10e + table + objects. Webots trial scene for Turkish spatial commands."
  ]
  title "Lecture 21 table scene"
  basicTimeStep 16
}
Viewpoint {
  # agentview-like: a fixed viewpoint across from the robot, looking at the table (computed with vp.py)
  orientation 0 0.558702 0.829369 3.141593
  position 0 1.75 1.55
}
TexturedBackground {
}
TexturedBackgroundLight {
}
Floor {
  size 6 6
}
Table {
  translation 0 0 0
  size 1 1.8 0.74
}
DEF ROBOT UR10e {
  translation 0 -0.65 0.74
  rotation 0 0 1 -1.5708
  controller "<extern>"
  supervisor TRUE
  toolSlot [
    DEF WRIST_CAM Camera {
      translation 0 0.02 0
      rotation 0 0 1 1.5708
      name "wrist_camera"
      fieldOfView 1.2
      width 256
      height 256
    }
  ]
}
DEF BOWL FruitBowl {
  translation -0.22 0.0 0.74
  name "black bowl"
  color 0.05 0.05 0.05
}
DEF PLATE Plate {
  translation 0.22 0.02 0.74
}
DEF STOVE HotPlate {
  translation 0 0.5 0.74
}
DEF CEREAL CerealBox {
  translation -0.35 0.45 0.74
  rotation 0 0 1 0.3
}
DEF APPLE Apple {
  translation 0.05 0.27 0.79
}
DEF CAN Can {
  translation 0.36 0.4 0.8
}
```
