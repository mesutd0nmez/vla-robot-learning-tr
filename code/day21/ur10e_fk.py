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
