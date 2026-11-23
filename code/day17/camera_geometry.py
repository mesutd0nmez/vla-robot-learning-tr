"""Lecture 17 — Computes and plots the LIBERO-Plus camera perturbation with the REAL code (no model/assets/MuJoCo needed).

It extracts the three pure functions (rotate_around_y, rotate_around_z, scale_distance_from_pivot) from
`problems/libero_tabletop_manipulation.py` via `ast` and runs them (the rest of the file needs robosuite, which we
do not import). Then it applies the same order as `_setup_camera` (vertical → horizontal → distance → endpoint yaw/pitch)
to the parameters in the names of the 376 "Camera Viewpoints" tasks in Spatial.

Usage:  python camera_geometry.py <LIBERO-plus root> <output.png> [--suite libero_spatial] [--csv out.csv]
"""
import argparse, ast, csv, json, pathlib
import numpy as np
from scipy.spatial.transform import Rotation  # the functions use this

POS_AV = [0.6586131746834771, 0.0, 1.6103500240372423]          # _setup_camera: default agentview
QUAT_AV = [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]  # (w,x,y,z)
PIVOT = np.array([0, 0, 0.8])


def load_funcs(root):
    src = (pathlib.Path(root) / "libero/libero/envs/problems/libero_tabletop_manipulation.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    ns = {"np": np, "Rotation": Rotation}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("scale_distance_from_pivot", "rotate_around_y", "rotate_around_z"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<libero-plus>", "exec"), ns)
    return ns["rotate_around_y"], ns["rotate_around_z"], ns["scale_distance_from_pivot"]


def setup_camera(fns, h, v, s, er, ev):
    """Exactly the same order as problems/*.py `_setup_camera` lines 319-345."""
    rot_y, rot_z, scale = fns
    pos, quat = POS_AV, QUAT_AV
    if int(v) != 0:
        r = rot_y(original_quat=quat, original_pos=pos, degrees=int(v)); pos, quat = r["new_pos"], r["new_quat"]
    r = rot_z(original_quat=quat, original_pos=pos, degrees=int(h)); pos, quat = r["new_pos"], r["new_quat"]
    if float(s) != 1.0:
        r = scale(original_quat=quat, original_pos=pos, scale_factor=float(s)); pos, quat = r["new_pos"], r["new_quat"]
    if int(er) != 0:
        quat = rot_z(original_quat=quat, degrees=int(er))["new_quat"]
    if int(ev) != 0:
        quat = rot_y(original_quat=quat, degrees=int(ev))["new_quat"]
    return np.array(pos), np.array(quat)


def forward_axis(quat_wxyz):
    """A MuJoCo camera looks along its own -z axis; the view vector in world coordinates."""
    w, x, y, z = quat_wxyz
    return Rotation.from_quat([x, y, z, w]).apply([0, 0, -1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--suite", default="libero_spatial"); ap.add_argument("--csv")
    a = ap.parse_args()
    fns = load_funcs(a.root)
    tasks = json.load(open(pathlib.Path(a.root) / "libero/libero/benchmark/task_classification.json", encoding="utf-8"))[a.suite]
    rows = []
    for t in tasks:
        if t["category"] != "Camera Viewpoints":
            continue
        view, rest = t["name"].split("_view_")[1].split("_initstate_")
        h, v, s, er, ev = view.split("_"); s = int(s) / 100
        pos, quat = setup_camera(fns, h, v, s, er, ev)
        fwd = forward_axis(quat)
        # the point where the view direction intersects the table plane (z=0.8 pivot height)
        tpar = (0.8 - pos[2]) / fwd[2] if fwd[2] < 0 else np.nan
        hit = pos + tpar * fwd
        kind = "rotation" if (int(h) or int(v)) else ("zoom" if s != 1.0 else "endpoint")
        rows.append(dict(name=t["name"], h=int(h), v=int(v), s=s, er=int(er), ev=int(ev), kind=kind,
                         x=pos[0], y=pos[1], z=pos[2], dist=float(np.linalg.norm(pos - PIVOT)),
                         hit_x=hit[0], hit_y=hit[1], difficulty=t["difficulty_level"]))
    base_pos, base_quat = setup_camera(fns, 0, 0, 1.0, 0, 0)
    assert np.allclose(base_pos, POS_AV) and np.allclose(base_quat, QUAT_AV), "the camera must not change with default parameters"
    base_fwd = forward_axis(base_quat); base_hit = base_pos + (0.8 - base_pos[2]) / base_fwd[2] * base_fwd
    kinds = {k: sum(r["kind"] == k for r in rows) for k in ("rotation", "zoom", "endpoint")}
    print(f"{a.suite}: {len(rows)} camera tasks → {kinds}")
    print(f"default camera pos={np.round(base_pos,3)} distance to pivot={np.linalg.norm(base_pos-PIVOT):.3f} m, view-table intersection={np.round(base_hit,3)}")
    d = np.array([r["dist"] for r in rows]); print(f"distance to pivot: min {d.min():.2f} max {d.max():.2f} m (×1.15..×2.0 in the zoom tasks)")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    col = {"rotation": "#ffd166", "zoom": "#cfe8ff", "endpoint": "#d4f5d4"}; edge = {"rotation": "#b07d00", "zoom": "#3a6ea5", "endpoint": "#2e7d32"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    # (a) bird's-eye view
    ax = axes[0]
    ax.add_patch(plt.Rectangle((-0.5, -0.6), 1.0, 1.2, facecolor="#f3e6d0", edgecolor="#a0875a", label="table 1.0×1.2 m"))
    ax.plot(-0.66, 0, "s", color="#555", ms=10, label="robot base (approximate)")
    for k in ("rotation", "zoom", "endpoint"):
        xs = [r["x"] for r in rows if r["kind"] == k]; ys = [r["y"] for r in rows if r["kind"] == k]
        ax.scatter(xs, ys, s=22, color=col[k], edgecolor=edge[k], linewidth=0.6, label=f"{k} ({kinds[k]})", zorder=3)
    ax.plot(base_pos[0], base_pos[1], "*", color="#d62828", ms=16, zorder=5, label="default agentview")
    ax.plot(0, 0, "+", color="#000", ms=10, mew=2, label="pivot (0,0,0.8)")
    ax.set_aspect("equal"); ax.set_xlim(-1.3, 1.8); ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_title("(a) Camera positions, bird's-eye view")
    ax.legend(fontsize=7.5, loc="upper left")
    # (b) side view: horizontal distance to the pivot vs height
    ax = axes[1]
    for k in ("rotation", "zoom", "endpoint"):
        rr = [np.hypot(r["x"], r["y"]) for r in rows if r["kind"] == k]; zz = [r["z"] for r in rows if r["kind"] == k]
        ax.scatter(rr, zz, s=22, color=col[k], edgecolor=edge[k], linewidth=0.6, zorder=3)
    ax.plot(np.hypot(*base_pos[:2]), base_pos[2], "*", color="#d62828", ms=16, zorder=5)
    ax.axhline(0.9, color="#a0875a", lw=1, ls="--"); ax.text(1.45, 0.92, "tabletop z=0.90", fontsize=8, color="#a0875a", ha="right")
    ax.plot(0, 0.8, "+", color="#000", ms=10, mew=2)
    ax.set_xlabel("horizontal distance to the pivot (m)"); ax.set_ylabel("z (m)"); ax.set_title("(b) Side view: zoom = moving away from the pivot")
    ax.set_xlim(-0.05, 1.5); ax.set_ylim(0.7, 2.5)
    # (c) the point where the view direction intersects the table plane
    ax = axes[2]
    ax.add_patch(plt.Rectangle((-0.5, -0.6), 1.0, 1.2, facecolor="#f3e6d0", edgecolor="#a0875a"))
    for k in ("rotation", "zoom", "endpoint"):
        xs = [r["hit_x"] for r in rows if r["kind"] == k]; ys = [r["hit_y"] for r in rows if r["kind"] == k]
        ax.scatter(xs, ys, s=22, color=col[k], edgecolor=edge[k], linewidth=0.6, zorder=3)
    ax.plot(base_hit[0], base_hit[1], "*", color="#d62828", ms=16, zorder=5)
    ax.set_aspect("equal"); ax.set_xlim(-0.7, 0.7); ax.set_ylim(-0.8, 0.8)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_title("(c) Where the view axis intersects the table plane")
    fig.suptitle(f"LIBERO-Plus 'Camera Viewpoints': camera pose of the {len(rows)} {a.suite} tasks — computed with the real "
                 "rotate_around_y/z + scale_distance_from_pivot from the LIBERO-plus code (camera_geometry.py)", fontsize=10.5)
    fig.text(0.5, 0.01, "rotation: horizontal ±75° (+ vertical 15°) around the pivot (0,0,0.8) · zoom: moving ×1.15–2.0 away from the pivot · endpoint: position fixed, only the view changes by ±10° yaw/pitch (the intersection point shifts)",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95)); fig.savefig(a.out, dpi=170); plt.close(fig)
    print("written:", a.out)


if __name__ == "__main__":
    main()
