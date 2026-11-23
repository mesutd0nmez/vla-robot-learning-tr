# day17

Lecture 17 — LIBERO-Plus's seven perturbations in code: carriers (task name / BDDL / problem class), camera math, 500 robot initial poses, sensor noise, object layout, the R1/R2 type of the 50 rewrites, and generating a Turkish `_language_trK.bddl` variant. No script needs a model, a GPU, or LIBERO-Plus assets; they only read text files in the LIBERO-plus clone (robosuite is enough for rendering).

| File | What | Status |
|---|---|---|
| `perturbation_table.py` | Table of 7 perturbations × (carrier, real parameter range, code location, BDDL/assets) — counted from the JSON and the files | ran (Spatial, 2402 tasks) |
| `camera_geometry.py` | Extracts LIBERO-plus's `rotate_around_y/z` + `scale_distance_from_pivot` functions from the file and computes the pose of the 376 camera tasks → figure 02 | ran (real code) |
| `robot_init_spread.py` | Reads the 500 `init_qpos` values in `mounted_panda.py`; joint deviations, ‖Δ‖ ∈ {0.1,…,0.5}, relation to difficulty → figure 03 | ran |
| `render_perturbations.py` | Actually renders the camera (8 poses) and robot (3 inits) perturbations in a robosuite Stack scene; with `--libero`, LIBERO task 0 → figure 04 | ran (robosuite 1.4.1 + mujoco 2.3.7, EGL) |
| `layout_shift.py` | Shift of region centers (cm) in the `_levelL_sampleM` files and the `_add_K` distractor objects → figure 05 | ran (200 + 300 files) |
| `classify_rewrites.py` | Splits the 50 `_language_K` texts of the base task into R1/R2/R0, counts the preserved words → CSV + figure 06 | ran (`datasets/02`) |
| `noise_on_frame.py` | 5 noise families × severity (with the env_wrapper.py parameter tables) applied to a real frame → figure 07 | ran (with wand/ImageMagick) |
| `make_tr_variant.py` + `tr_commands.csv` | Produce a Turkish `_language_trK.bddl`, read it back with the `bddl` parser, write the task map / JSON lines | ran (10/10 exact; expected errors on bad examples) |
| `plot_day17.py` | Figure 09: VLANeXt Table III, 7 perturbations × suite | ran |

Usage (on the Mac, `vla` env; no model needed):
```bash
git clone --depth 1 https://github.com/sylvestf/LIBERO-plus ~/vla-work/LIBERO-plus    # skip if you cloned it in Lecture 16
for f in perturbation_table camera_geometry robot_init_spread layout_shift classify_rewrites noise_on_frame make_tr_variant render_perturbations plot_day17; do
  curl -O https://raw.githubusercontent.com/mesutd0nmez/vla-robot-learning-tr/main/code/day17/$f.py; done
curl -O https://raw.githubusercontent.com/mesutd0nmez/vla-robot-learning-tr/main/code/day17/tr_commands.csv
P=~/vla-work/LIBERO-plus; mkdir -p out17
python perturbation_table.py $P --md out17/table.md
python camera_geometry.py $P out17/02.png --csv out17/cam.csv
python robot_init_spread.py $P out17/03.png --csv out17/robot.csv
python layout_shift.py $P out17/05.png
python classify_rewrites.py $P out17/06.png --csv out17/rewrites.csv
MUJOCO_GL=glfw python render_perturbations.py out17/04.png --robot-csv out17/robot.csv --save-frame out17/frame.png   # add --libero if LIBERO is installed
pip install wand && brew install imagemagick            # for motion_blur (optional)
python noise_on_frame.py out17/frame.png out17/07.png
pip install bddl==1.0.1 future && python make_tr_variant.py $P --commands tr_commands.csv --out out17/tr_variants
```


## perturbation_table.py

```python
"""Lecture 17 — Produces a "perturbation table" of LIBERO-Plus's 7 perturbations by READING THE CODE.

NO model / GPU / assets needed: it only reads text files in the LIBERO-plus clone
(task_classification.json, bddl_files/, envs/robots/mounted_panda.py, envs/problems/*.py).

Usage:
  python perturbation_table.py <LIBERO-plus root> [--suite libero_spatial] [--json out.json] [--md out.md]

Output: for each perturbation (a) the carrier (task name / BDDL file / problem class), (b) the real parameter
range (counted from the task names and files), (c) how many tasks, (d) where it is applied in the code.
"""
import argparse, collections, json, pathlib, re, sys

CATS = ["Camera Viewpoints", "Robot Initial States", "Language Instructions", "Light Conditions",
        "Background Textures", "Sensor Noise", "Objects Layout"]
NOISE_FAMILY = {1: "motion_blur (ImageMagick/wand)", 2: "gaussian_blur (skimage)", 3: "zoom_blur (scipy zoom)",
                4: "fog (plasma_fractal)", 5: "glass_blur (skimage + pixel shuffling)"}


def parse_tail(name):
    """'..._view_H_V_S_ER_EV_initstate_I[_noise_N]' → dict or None (not present in BDDL-carried tasks)."""
    if "_view_" not in name:
        return None
    view, rest = name.split("_view_")[1].split("_initstate_")
    h, v, s, er, ev = view.split("_")
    noise = 0
    if "_noise_" in rest:
        rest, noise = rest.split("_noise_")
    return dict(h=int(h), v=int(v), s=int(s) / 100, er=int(er), ev=int(ev), init=int(rest), noise=int(noise))


def signed(deg):
    """0..359 degrees → -180..180 (camera angles 285..359 are actually negative)."""
    return deg - 360 if deg > 180 else deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--json"); ap.add_argument("--md")
    a = ap.parse_args()
    lib = pathlib.Path(a.root) / "libero" / "libero"
    tasks = json.load(open(lib / "benchmark" / "task_classification.json", encoding="utf-8"))[a.suite]
    bddl_dir = lib / "bddl_files" / a.suite

    by_cat = collections.defaultdict(list)
    for t in tasks:
        by_cat[t["category"]].append(t["name"])

    rows = {}
    # --- Camera
    cams = [parse_tail(n) for n in by_cat["Camera Viewpoints"]]
    hs = [signed(c["h"]) for c in cams]; vs = [c["v"] for c in cams]; ss = [c["s"] for c in cams]
    ers = [signed(c["er"]) for c in cams]; evs = [signed(c["ev"]) for c in cams]
    nz = lambda xs: [x for x in xs if x != 0]
    rows["Camera Viewpoints"] = dict(
        carrier="task name (`_view_H_V_S_ER_EV`)", n=len(cams),
        where="env_wrapper.py:207-218 (parse) → problems/*.py `_setup_camera` (rotate_around_z / rotate_around_y / scale_distance_from_pivot, pivot (0,0,0.8))",
        params=f"horizontal {min(nz(hs))}..{max(nz(hs))}° ({len(nz(hs))} tasks); vertical {sorted(set(vs))}° ; "
               f"distance ×{min(nz([s for s in ss if s != 1.0]) or [1]):.2f}..×{max(ss):.2f} ({sum(s != 1.0 for s in ss)} tasks); "
               f"endpoint yaw {min(nz(ers) or [0])}..{max(nz(ers) or [0])}°, endpoint pitch {min(nz(evs) or [0])}..{max(nz(evs) or [0])}° ({sum(1 for c in cams if c['er'] or c['ev'])} tasks)",
        bddl="no (base BDDL)", assets="no")
    # --- Robot init
    inits = [parse_tail(n)["init"] for n in by_cat["Robot Initial States"]]
    panda = (lib / "envs" / "robots" / "mounted_panda.py").read_text(encoding="utf-8")
    n_cls = len(re.findall(r"^class MountedPanda\d+\(", panda, flags=re.M))
    rows["Robot Initial States"] = dict(
        carrier="task name (`_initstate_I`) → robot class name `MountedPanda{I}`", n=len(inits),
        where="env_wrapper.py:219-220 (`robots=[robots[0]+str(init_state)]`) → envs/robots/mounted_panda.py (each class only returns `init_qpos`)",
        params=f"I = {min(inits)}..{max(inits)} ({len(set(inits))} distinct); {n_cls} classes defined in the file; 7 joint angles (rad)",
        bddl="no", assets="no")
    # --- Language
    langs = by_cat["Language Instructions"]
    lang_files = sorted(bddl_dir.glob("*_language_*.bddl"))
    rows["Language Instructions"] = dict(
        carrier="BDDL file (`_language_K.bddl`, only the `:language` line differs) + fixed task-name tail `_view_0_0_100_0_0_initstate_0`", n=len(langs),
        where="benchmark/__init__.py:46-72 `grab_language_from_filename` → bddl_utils.get_problem_info → bddl.parsing.scan_tokens (`.lower()`)",
        params=f"{len(lang_files)} language files in the folder, {len(langs)} tasks in the benchmark; scene/camera/robot/goal do not change",
        bddl="yes (only :language)", assets="no")
    # --- Light
    lights = by_cat["Light Conditions"]
    light_ids = sorted(int(re.search(r"_light_(\d+)", n).group(1)) for n in lights)
    prob = (lib / "envs" / "problems" / "libero_tabletop_manipulation.py").read_text(encoding="utf-8")
    n_light_cls = len(set(re.findall(r"scenes/lights/(tabletop_light_sync_modified_\d+)\.xml", prob)))
    rows["Light Conditions"] = dict(
        carrier="BDDL file (`_light_K.bddl`, only the `(problem ...)` name differs) → registered problem subclass → `scene_xml`", n=len(lights),
        where="problems/libero_tabletop_manipulation.py `Libero_Tabletop_Manipulation_tabletop_light_sync_modified_K` (scene_xml=scenes/lights/..._K.xml)",
        params=f"K = {min(light_ids)}..{max(light_ids)} ({len(set(light_ids))} distinct in Spatial); {n_light_cls} light scenes registered for tabletop; diffuse/direction/specular/castshadow in the XML",
        bddl="yes (only the problem name)", assets="YES (assets/scenes/lights/*.xml)")
    # --- Background
    bgs = by_cat["Background Textures"]
    n_table = sum("_table_" in n and "_tb_" not in n for n in bgs); n_tb = sum("_tb_" in n for n in bgs)
    tex = set(re.findall(r"scenes/(tabletop_table_[A-Za-z0-9_]+)\.xml", prob))
    rows["Background Textures"] = dict(
        carrier="BDDL file (`_table_K` = table texture, `_tb_K` = table + wall) → problem subclass → `scene_xml`", n=len(bgs),
        where="problems/libero_tabletop_manipulation.py `..._tabletop_table_<Texture>[_Wall_<wall>]` classes",
        params=f"`_table_` {n_table} tasks, `_tb_` {n_tb} tasks; {len(tex)} distinct texture scenes registered for tabletop (e.g. {sorted(tex)[0]})",
        bddl="yes (only the problem name)", assets="YES (assets/scenes/*.xml + textures/)")
    # --- Noise
    noises = [parse_tail(n)["noise"] for n in by_cat["Sensor Noise"]]
    fam = collections.Counter((c - 1) // 10 + 1 for c in noises)
    rows["Sensor Noise"] = dict(
        carrier="task name (`_noise_N`, N = 1..50)", n=len(noises),
        where="env_wrapper.py:285-348 `ControlEnv.step/reset`: ONLY `agentview_image` is modified (the wrist camera is untouched)",
        params="; ".join(f"{NOISE_FAMILY[k]} N={10*(k-1)+1}..{10*k}: {fam.get(k,0)} tasks" for k in range(1, 6)),
        bddl="no", assets="no (but `wand` + libmagickwand are required)")
    # --- Layout
    lays = by_cat["Objects Layout"]
    n_add = sum("_add_" in n for n in lays); n_lvl = sum("_level" in n for n in lays)
    lvl = collections.Counter(int(re.search(r"_level(\d+)_", n).group(1)) for n in lays if "_level" in n)
    add_objs = collections.Counter()
    for f in bddl_dir.glob("*_add_*.bddl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            m = re.search(r"\(On (\S+) main_table_add_object_region", line)
            if m: add_objs[re.sub(r"_\d+$", "", m.group(1))] += 1
    rows["Objects Layout"] = dict(
        carrier="BDDL file: `_add_K` (new object + `add_object_region_K`) and `_levelL_sampleM` (region ranges shifted)", n=len(lays),
        where="bddl_base_domain.py (BDDL `:regions` → placement initializer); new objects in assets/new_objects, turbosquid_objects…",
        params=f"`_add_` {n_add} tasks ({len(add_objs)} distinct added object types, Spatial); `_level` {n_lvl} tasks, level distribution {dict(sorted(lvl.items()))}",
        bddl="yes (:regions, :objects, :init)", assets="YES (new object meshes)")

    # --- output
    print(f"# LIBERO-Plus perturbation table — suite: {a.suite} ({len(tasks)} tasks)\n")
    hdr = "| Perturbation | Tasks | Carrier | Real parameter range | Where in the code | BDDL changes? | Assets needed? |"
    md = [hdr, "|---|---|---|---|---|---|---|"]
    for c in CATS:
        r = rows[c]
        md.append(f"| {c} | {r['n']} | {r['carrier']} | {r['params']} | {r['where']} | {r['bddl']} | {r['assets']} |")
    print("\n".join(md))
    if a.md: pathlib.Path(a.md).write_text("\n".join(md) + "\n", encoding="utf-8")
    if a.json: json.dump({"suite": a.suite, "n_tasks": len(tasks), "rows": rows}, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
```

## camera_geometry.py

```python
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
```

## robot_init_spread.py

```python
"""Lecture 17 — LIBERO-Plus 'Robot Initial States': reads the init_qpos of the 500 MountedPanda subclasses from the file and plots it.

NO model/assets/MuJoCo needed. Every `class MountedPandaK(MountedPanda)` in `envs/robots/mounted_panda.py`
overrides only the `init_qpos` property; this script extracts those 7 numbers with `ast` and computes the difference from the base pose.

Usage:  python robot_init_spread.py <LIBERO-plus root> <output.png> [--csv out.csv] [--suite libero_spatial]
"""
import argparse, ast, csv, json, pathlib, re
import numpy as np

JOINTS = ["j1 base yaw", "j2 shoulder", "j3 upper arm", "j4 elbow", "j5 forearm", "j6 wrist", "j7 hand yaw"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--csv"); ap.add_argument("--suite", default="libero_spatial")
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    src = (root / "libero/libero/envs/robots/mounted_panda.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    qpos = {}
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name.startswith("MountedPanda")):
            continue
        k = node.name[len("MountedPanda"):]
        for fn in node.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == "init_qpos":
                ret = [n for n in ast.walk(fn) if isinstance(n, ast.Return)][0]
                lst = ret.value.args[0]  # np.array([...])
                vals = []
                for e in lst.elts:
                    if isinstance(e, ast.BinOp):  # np.pi / 4
                        vals.append(np.pi / 4)
                    elif isinstance(e, ast.UnaryOp):
                        vals.append(-ast.literal_eval(e.operand))
                    else:
                        vals.append(ast.literal_eval(e))
                qpos[0 if k == "" else int(k)] = np.array(vals, dtype=float)
    base = qpos[0]
    ks = sorted(k for k in qpos if k != 0)
    Q = np.stack([qpos[k] for k in ks]); D = Q - base
    print(f"{len(ks)} variant classes; base init_qpos = {np.round(base, 3)}")
    print("per-joint |Δ| (rad): mean", np.round(np.abs(D).mean(0), 3), " max", np.round(np.abs(D).max(0), 3))
    norm = np.linalg.norm(D, axis=1)
    print(f"7-dim ||Δ||: min {norm.min():.3f} mean {norm.mean():.3f} max {norm.max():.3f} rad")

    tasks = json.load(open(root / "libero/libero/benchmark/task_classification.json", encoding="utf-8"))[a.suite]
    used = {}
    for t in tasks:
        if t["category"] == "Robot Initial States":
            i = int(t["name"].split("_initstate_")[1].split("_")[0]); used[i] = t["difficulty_level"]
    print(f"{a.suite} Robot tasks: {len(used)} distinct I used ({min(used)}..{max(used)})")
    # difficulty vs deviation magnitude (hypothesis test: is a large deviation → hard?)
    diff_norm = {}
    for i, lvl in used.items():
        diff_norm.setdefault(lvl, []).append(np.linalg.norm(qpos[i] - base))
    for lvl in sorted(diff_norm):
        v = np.array(diff_norm[lvl]); print(f"  difficulty L{lvl}: n={len(v):3d}  ||Δ|| mean {v.mean():.3f}  median {np.median(v):.3f}")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["init_state"] + [f"q{i+1}" for i in range(7)] + ["norm_delta", "difficulty_spatial"])
            for k, q in zip(ks, Q):
                w.writerow([k] + [f"{x:.6f}" for x in q] + [f"{np.linalg.norm(q-base):.4f}", used.get(k, "")])

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    bp = ax.boxplot([np.degrees(D[:, j]) for j in range(7)], patch_artist=True, widths=0.6, showfliers=True,
                    flierprops=dict(marker=".", ms=3, color="#999"))
    for p in bp["boxes"]: p.set_facecolor("#ffd166"); p.set_edgecolor("#b07d00")
    for m in bp["medians"]: m.set_color("#000")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xticks(range(1, 8)); ax.set_xticklabels(JOINTS, rotation=20, ha="right")
    ax.set_ylabel("init_qpos − base (degrees)")
    ax.set_title(f"(a) {len(ks)} MountedPandaK classes: deviation per joint (mounted_panda.py)")
    ax.text(0.02, 0.97, f"base: {np.round(np.degrees(base),1)}°", transform=ax.transAxes, fontsize=8, va="top", color="#555")
    ax = axes[1]
    cols = {1: "#d4f5d4", 2: "#e8f5c8", 3: "#ffd166", 4: "#ffb3a7", 5: "#ffd6d6"}
    data = [np.array(diff_norm[l]) for l in sorted(diff_norm)]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, flierprops=dict(marker=".", ms=3, color="#999"))
    for p, l in zip(bp["boxes"], sorted(diff_norm)): p.set_facecolor(cols[l]); p.set_edgecolor("#888")
    for m in bp["medians"]: m.set_color("#000")
    ax.set_xticks(range(1, len(data) + 1)); ax.set_xticklabels([f"L{l}\n(n={len(diff_norm[l])})" for l in sorted(diff_norm)])
    ax.set_ylabel("||Δ init_qpos|| (rad, 7 eklem)"); ax.set_xlabel("difficulty level (how many of 4 models solved it)")
    ax.set_title(f"(b) {a.suite}: deviation magnitude vs difficulty")
    nc = sorted(((round(float(n), 2)), int(c)) for n, c in zip(*np.unique(np.round(norm, 2), return_counts=True)))
    ax.text(0.02, 0.03, "across the 500 classes ||Δ|| = " + ", ".join(f"{n} rad ×{c}" for n, c in nc) + "\n(random direction, 5 fixed magnitudes)",
            transform=ax.transAxes, fontsize=7.5, color="#555", va="bottom")
    fig.suptitle("LIBERO-Plus 'Robot Initial States': the perturbation = only the initial values of the 7 joint angles — read from the real file (robot_init_spread.py)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(a.out, dpi=170); plt.close(fig)
    print("written:", a.out)


if __name__ == "__main__":
    main()
```

## render_perturbations.py

```python
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
```

## layout_shift.py

```python
"""Lecture 17 — LIBERO-Plus 'Objects Layout': measures how far the regions shift in the `_levelL_sampleM.bddl` files and
which object is added in the `_add_K.bddl` files, by comparing with the base BDDL.

NO model/assets needed; only text files are read (the BDDL is parsed with a simple regex; the bddl package is not needed).

Usage:  python layout_shift.py <LIBERO-plus root> <output.png> [--suite libero_spatial] [--csv out.csv]
"""
import argparse, collections, csv, pathlib, re
import numpy as np

RE_REGION = re.compile(r"\(\s*(\w+)\s*\(:target\s+(\w+)\)\s*\(:ranges\s*\(\s*\(([^()]+)\)\s*\)\s*\)", re.S)


def regions(text):
    """region name → (cx, cy) (ranges = x0 y0 x1 y1; center)."""
    out = {}
    for name, target, rng in RE_REGION.findall(text):
        if target != "main_table":
            continue
        x0, y0, x1, y1 = map(float, rng.split())
        out[name] = ((x0 + x1) / 2, (y0 + y1) / 2)
    return out


def init_map(text):
    """object → region ((On obj main_table_region) in the :init block)."""
    m = re.search(r"\(:init(.*?)\)\s*\(:goal", text, re.S)
    return dict(re.findall(r"\(On (\S+) main_table_(\w+)\)", m.group(1))) if m else {}


def objects_of_interest(text):
    m = re.search(r"\(:obj_of_interest(.*?)\)", text, re.S)
    return m.group(1).split() if m else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--suite", default="libero_spatial"); ap.add_argument("--csv")
    a = ap.parse_args()
    d = pathlib.Path(a.root) / "libero/libero/bddl_files" / a.suite
    rows = []
    for f in sorted(d.glob("*_level*_sample*.bddl")):
        m = re.match(r"(.+)_level(\d+)_sample(\d+)\.bddl$", f.name)
        base_name, level, sample = m.group(1), int(m.group(2)), int(m.group(3))
        base = (d / f"{base_name}.bddl").read_text(encoding="utf-8"); var = f.read_text(encoding="utf-8")
        rb, rv = regions(base), regions(var); ib, iv = init_map(base), init_map(var)
        ooi = objects_of_interest(base)
        target = ooi[0] if ooi else None  # in Spatial the first obj_of_interest = the bowl to pick up
        for obj, reg in ib.items():
            if obj not in iv or reg not in rb or iv[obj] not in rv:
                continue
            (bx, by), (vx, vy) = rb[reg], rv[iv[obj]]
            rows.append(dict(base=base_name, level=level, sample=sample, obj=obj, is_target=obj == target,
                             dx_cm=100 * (vx - bx), dy_cm=100 * (vy - by), shift_cm=100 * float(np.hypot(vx - bx, vy - by))))
    print(f"{a.suite}: {len(set((r['base'], r['level'], r['sample']) for r in rows))} level files, {len(rows)} object rows")
    by_level_t = collections.defaultdict(list); by_level_o = collections.defaultdict(list)
    for r in rows:
        (by_level_t if r["is_target"] else by_level_o)[r["level"]].append(r["shift_cm"])
    for L in sorted(by_level_t):
        t, o = np.array(by_level_t[L]), np.array(by_level_o[L])
        print(f"  level {L}: target bowl shift mean {t.mean():5.2f} cm (max {t.max():5.2f}); other objects mean {o.mean():5.2f} cm (max {o.max():5.2f})  n_files={len(t)}")
    # --- _add_ files: added object types and positions
    added = collections.Counter(); add_pos = []
    for f in sorted(d.glob("*_add_*.bddl")):
        txt = f.read_text(encoding="utf-8"); rv = regions(txt)
        for obj, reg in init_map(txt).items():
            if reg.startswith("add_object_region"):
                added[re.sub(r"_\d+$", "", obj)] += 1
                if reg in rv: add_pos.append(rv[reg])
    print(f"  _add_ files: {sum(added.values())} added objects, {len(added)} types:")
    for k, v in added.most_common(): print(f"     {v:3d}  {k}")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1.1, 1, 1.2]})
    ax = axes[0]
    Ls = sorted(by_level_t)
    bp = ax.boxplot([by_level_t[L] for L in Ls], positions=np.arange(len(Ls)) - 0.18, widths=0.32, patch_artist=True, flierprops=dict(marker=".", ms=3))
    for p in bp["boxes"]: p.set_facecolor("#ffd166"); p.set_edgecolor("#b07d00")
    bp2 = ax.boxplot([by_level_o[L] for L in Ls], positions=np.arange(len(Ls)) + 0.18, widths=0.32, patch_artist=True, flierprops=dict(marker=".", ms=3))
    for p in bp2["boxes"]: p.set_facecolor("#cfe8ff"); p.set_edgecolor("#3a6ea5")
    for m in bp["medians"] + bp2["medians"]: m.set_color("#000")
    ax.set_xticks(range(len(Ls))); ax.set_xticklabels([f"level {L}" for L in Ls])
    ax.set_ylabel("shift of the region center (cm)"); ax.set_title("(a) `_levelL_sampleM`: how far do the object regions shift?")
    ax.legend([bp["boxes"][0], bp2["boxes"][0]], ["target bowl (obj_of_interest[0])", "other objects"], fontsize=8, loc="upper left")
    ax = axes[1]
    tgt = [r for r in rows if r["is_target"]]
    cols = {1: "#d4f5d4", 2: "#b7e4b7", 3: "#ffd166", 4: "#ffb3a7", 5: "#ff7f7f"}
    for L in Ls:
        xs = [r["dx_cm"] for r in tgt if r["level"] == L]; ys = [r["dy_cm"] for r in tgt if r["level"] == L]
        ax.scatter(xs, ys, s=26, color=cols[L], edgecolor="#555", linewidth=0.5, label=f"level {L}")
    ax.plot(0, 0, "k+", ms=12, mew=2); ax.set_aspect("equal"); ax.legend(fontsize=8)
    ax.set_xlabel("Δx (cm)"); ax.set_ylabel("Δy (cm)"); ax.set_title("(b) Shift of the target bowl (all base tasks)")
    ax = axes[2]
    names = [k for k, _ in added.most_common()][::-1]; vals = [added[k] for k in names]
    ax.barh(names, vals, color="#ffd6d6", edgecolor="#b00020")
    for i, v in enumerate(vals): ax.text(v + 0.3, i, str(v), va="center", fontsize=8)
    n_add_files = len(list(d.glob("*_add_*.bddl")))
    ax.set_xlabel("in how many `_add_` files"); ax.set_title(f"(c) `_add_K`: added distractors ({sum(vals)} objects / {n_add_files} files)")
    ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(f"LIBERO-Plus 'Objects Layout' ({a.suite}): measured from the BDDL files — base BDDL vs variant, region centers (layout_shift.py)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(a.out, dpi=170); plt.close(fig)
    print("written:", a.out)


if __name__ == "__main__":
    main()
```

## classify_rewrites.py

```python
"""Lecture 17 — LIBERO-Plus 'Language Instructions': reads the 50 rewrites of one base task, assigns them to an R1/R2 type,
counts which words are preserved; produces a CSV + a figure.

NO model/assets needed. The type label is NOT in the JSON; the label here is a rule-based reading following the paper's
Appendix A.6 definitions (R1 distraction = chatty/context sentences; R2 commonsense = the object name replaced by a description;
R3 reasoning chain = in multi-step tasks — not expected in Spatial). Where the rules hold, it is a "hypothesis".
The type labels (R1 dikkat dağıtma, R2 sağduyu, R0 düz parafraz) are data values in the CSV and in datasets/02, so they are kept
as they are; the English display names are in TYPE_EN (used in the plot).

Usage:  python classify_rewrites.py <LIBERO-plus root> <output.png> [--suite libero_spatial] [--base 0] [--csv out.csv]
--base: original LIBERO task order (Lecture 10 list_tasks.py; 0 = between the plate and the ramekin)
"""
import argparse, csv, json, pathlib, re, sys

BASES = {  # same order as Lecture 16 select_plus_tasks.py (LIBERO-Spatial 0–9)
    "libero_spatial": ["pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"]}
CONV = re.compile(r"\b(could you|would you|can you|please|for me|lets|let's|before|remember|i noticed|i need|id appreciate|for our|for better|for proper|"
                  r"the table|needs|isnt|when you have|guests|dinner|dining|cooking|organiz|arrange)\b", re.I)
DESCR = re.compile(r"\b(container|vessel|darkhued|darkcolored|dark colored|flat ?dish|baking dish|ceramic dish|small dish|concave|spherical|rounded|"
                   r"main course|main meal|serving|holder|cooking dish|glazed)\b", re.I)
RELATION = re.compile(r"\b(between|amid|midway|separating|positioned between|situated between)\b", re.I)

TYPE_EN = {"R1 dikkat dağıtma": "R1 distraction", "R2 sağduyu (hafif)": "R2 commonsense (light)",
           "R2 sağduyu (tam)": "R2 commonsense (full)", "R0 düz parafraz": "R0 plain paraphrase"}


def classify(t):
    tl = t.lower()
    conv, descr = bool(CONV.search(tl)), bool(DESCR.search(tl))
    keeps_bowl, keeps_plate = "bowl" in tl, "plate" in tl
    descr_light = bool(re.search(r"\bdark bowl\b|\bblack dish\b|\bthe dish\b", tl))  # 'black bowl'→'dark bowl', 'plate'→'dish'
    if descr and not keeps_bowl:
        typ = "R2 sağduyu (tam)"
    elif descr or descr_light:
        typ = "R2 sağduyu (hafif)"; descr = True
    elif conv:
        typ = "R1 dikkat dağıtma"
    else:
        typ = "R0 düz parafraz"
    return typ, conv, descr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--suite", default="libero_spatial"); ap.add_argument("--base", type=int, default=0); ap.add_argument("--csv")
    a = ap.parse_args()
    root = pathlib.Path(a.root); lib = root / "libero/libero"
    base = BASES[a.suite][a.base]; d = lib / "bddl_files" / a.suite
    tasks = json.load(open(lib / "benchmark/task_classification.json", encoding="utf-8"))[a.suite]
    in_bench = {}
    for i, t in enumerate(tasks):
        m = re.match(re.escape(base) + r"_language_(\d+)_view_", t["name"])
        if m: in_bench[int(m.group(1))] = (i, t["difficulty_level"])
    base_text = re.search(r"\(:language (.*?)\)", (d / f"{base}.bddl").read_text(encoding="utf-8")).group(1)
    rows = []
    for f in sorted(d.glob(f"{base}_language_*.bddl"), key=lambda p: int(re.search(r"_language_(\d+)", p.name).group(1))):
        k = int(re.search(r"_language_(\d+)", f.name).group(1))
        text = re.search(r"\(:language (.*?)\)", f.read_text(encoding="utf-8")).group(1).strip()
        typ, conv, descr = classify(text)
        idx, lvl = in_bench.get(k, (None, None))
        rows.append(dict(k=k, in_benchmark=k in in_bench, task_index=idx, difficulty=lvl, type=typ, words=len(text.split()),
                         keeps_bowl="bowl" in text.lower(), keeps_plate="plate" in text.lower(), keeps_between="between" in text.lower(),
                         relation_word=(RELATION.search(text.lower()) or [""])[0], conversational=conv, descriptive=descr,
                         lowercase_changes=sum(ch.isupper() for ch in text), text=text))
    print(f"base: {base}\n  original :language = '{base_text}'  ({len(base_text.split())} words)")
    print(f"  language files: {len(rows)}; in the benchmark: {sum(r['in_benchmark'] for r in rows)}; missing: {[r['k'] for r in rows if not r['in_benchmark']]}")
    from collections import Counter
    ct = Counter(r["type"] for r in rows); print("  type distribution (rule-based, hypothesis):", dict(ct))
    for typ in sorted(ct):
        rr = [r for r in rows if r["type"] == typ]
        lv = Counter(r["difficulty"] for r in rr if r["in_benchmark"])
        print(f"   {typ:22s} n={len(rr):2d}  words mean {sum(r['words'] for r in rr)/len(rr):4.1f}  'between' kept {sum(r['keeps_between'] for r in rr):2d}  difficulty {dict(sorted(lv.items()))}")
    print("  instead of 'between':", Counter(r["relation_word"] for r in rows if not r["keeps_between"]))
    print("  containing uppercase (scan_tokens .lower() will change it):", sum(r["lowercase_changes"] > 0 for r in rows), "/", len(rows))
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    types = ["R1 dikkat dağıtma", "R2 sağduyu (hafif)", "R2 sağduyu (tam)", "R0 düz parafraz"]
    tcol = {"R1 dikkat dağıtma": "#cfe8ff", "R2 sağduyu (hafif)": "#ffe9a8", "R2 sağduyu (tam)": "#ffd166", "R0 düz parafraz": "#d4f5d4"}
    lcol = {1: "#d4f5d4", 2: "#b7e4b7", 3: "#ffd166", 4: "#ffb3a7", 5: "#ff7f7f", None: "#ddd"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 1.1, 1.1]})
    ax = axes[0]
    left = np.zeros(len(types))
    for lvl in [1, 2, 3, 4, 5, None]:
        vals = [sum(1 for r in rows if r["type"] == t and r["difficulty"] == lvl) for t in types]
        ax.barh([TYPE_EN[t] for t in types], vals, left=left, color=lcol[lvl], edgecolor="#777", label=f"L{lvl}" if lvl else "not in benchmark"); left += vals
    for i, t in enumerate(types): ax.text(left[i] + 0.3, i, str(int(left[i])), va="center", fontsize=9)
    ax.invert_yaxis(); ax.legend(fontsize=7.5, title="difficulty", loc="lower right"); ax.set_xlabel("number of variants")
    ax.set_title(f"(a) Task {a.base}: type of the 50 rewrites (rule-based)")
    ax = axes[1]
    for t in types:
        ws = [r["words"] for r in rows if r["type"] == t]
        ax.scatter(ws, [TYPE_EN[t]] * len(ws), s=40, color=tcol[t], edgecolor="#555", alpha=0.85)
    ax.axvline(len(base_text.split()), color="#d62828", ls="--")
    ax.text(len(base_text.split()) + 0.3, -0.45, f"original: {len(base_text.split())} words", color="#d62828", fontsize=8, va="bottom")
    ax.invert_yaxis(); ax.set_xlabel("number of words"); ax.set_title("(b) Length: words per type")
    ax = axes[2]
    feats = [("'bowl'\npreserved", "keeps_bowl"), ("'plate'\npreserved", "keeps_plate"), ("'between'\npreserved", "keeps_between"), ("has uppercase\n(.lower() changes it)", "lowercase_changes")]
    vals = [sum(bool(r[f]) for r in rows) for _, f in feats]
    ax.bar([f for f, _ in feats], vals, color=["#d4f5d4", "#d4f5d4", "#ffd166", "#ffd6d6"], edgecolor="#777")
    for i, v in enumerate(vals): ax.text(i, v + 0.6, f"{v}/{len(rows)}", ha="center", fontsize=9)
    ax.set_ylim(0, len(rows) + 6); ax.set_ylabel("number of variants"); ax.set_title("(c) Are thesis-critical words preserved?")
    ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"LIBERO-Plus Language perturbation from the inside — {a.suite}, base task {a.base} ({len(rows)} `_language_K.bddl`; the type label is not in the JSON, this is a rule-based reading) — classify_rewrites.py", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(a.out, dpi=170); plt.close(fig)
    print("written:", a.out)


if __name__ == "__main__":
    main()
```

## noise_on_frame.py

```python
"""Lecture 17 — LIBERO-Plus 'Sensor Noise': applies the five perturbations in env_wrapper.py to a frame.

The functions were rewritten from LIBERO-plus `libero/libero/envs/env_wrapper.py` (4976dc3) lines 30-174 with the PARAMETER TABLES
TRANSCRIBED EXACTLY (so as not to run code from an external repo; the algorithms are the same, the known implementations of
Hendrycks & Dietterich ImageNet-C). motion_blur needs `wand` (ImageMagick); if it is missing that column is skipped.

Usage:  python noise_on_frame.py <frame.png> <output.png> [--size 128] [--severity 1,5,10] [--seed 0]
Frame: a real agentview frame like in Lecture 10/14 (LIBERO-Plus eval uses 128x128; scaled with --size).
"""
import argparse, pathlib, sys
from io import BytesIO
import numpy as np
from PIL import Image
from scipy.ndimage import zoom as scizoom
from skimage.filters import gaussian

# --- env_wrapper.py:33-44  motion_blur: (radius, sigma), angle U(-45,45)
MOTION = [(5, 2), (8, 3), (10, 4), (12, 5), (15, 6), (18, 8), (20, 10), (25, 12), (30, 15), (35, 20)]
# --- env_wrapper.py:61  gaussian_blur: sigma
GAUSS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# --- env_wrapper.py:79-88  zoom_blur: zoom factors
ZOOM = [np.arange(1, 1.11, 0.01), np.arange(1, 1.16, 0.01), np.arange(1, 1.21, 0.02), np.arange(1, 1.26, 0.02), np.arange(1, 1.31, 0.03),
        np.arange(1, 1.36, 0.01), np.arange(1, 1.41, 0.01), np.arange(1, 1.46, 0.02), np.arange(1, 1.51, 0.02), np.arange(1, 1.56, 0.03)]
# --- env_wrapper.py:148  fog: (intensity alpha, wibbledecay)
FOG = [(0.5, 3), (1.0, 2.8), (1.5, 2.5), (2.0, 2.2), (2.5, 2.0), (3.0, 1.8), (3.5, 1.6), (4.0, 1.5), (4.5, 1.4), (5.0, 1.3)]
# --- env_wrapper.py:159  glass_blur: (sigma, max_delta, iterations)
GLASS = [(0.5, 1, 3), (0.7, 1, 3), (0.9, 2, 3), (1.0, 2, 2), (1.1, 3, 2), (1.3, 3, 2), (1.5, 4, 2), (1.8, 4, 2), (2.2, 5, 1), (2.5, 5, 1)]


def motion_blur(x, severity=1):
    from wand.image import Image as WandImage  # ImageMagick
    r, s = MOTION[severity - 1]
    buf = BytesIO(); x.save(buf, format="PNG")
    with WandImage(blob=buf.getvalue()) as w:
        w.motion_blur(radius=r, sigma=s, angle=float(np.random.uniform(-45, 45)))
        out = np.array(w)[..., :3]
    return np.clip(out, 0, 255).astype(np.uint8)


def gaussian_blur(x, severity=1):
    return (np.clip(gaussian(np.array(x) / 255.0, sigma=GAUSS[severity - 1], channel_axis=-1), 0, 1) * 255).astype(np.uint8)


def clipped_zoom(img, zf):  # env_wrapper.py:66-76
    h = img.shape[0]; ch = int(np.ceil(h / float(zf))); top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zf, zf, 1), order=1)
    trim = (img.shape[0] - h) // 2
    return img[trim:trim + h, trim:trim + h]


def zoom_blur(x, severity=1):
    c = ZOOM[severity - 1]; x = (np.array(x) / 255.0).astype(np.float32); out = np.zeros_like(x)
    for zf in c: out += clipped_zoom(x, zf)
    return (np.clip((x + out) / (len(c) + 1), 0, 1) * 255).astype(np.uint8)


def plasma_fractal(mapsize=256, wibbledecay=3):  # env_wrapper.py:98-144 (diamond-square)
    maparray = np.empty((mapsize, mapsize), dtype=np.float64); maparray[0, 0] = 0; stepsize = mapsize; wibble = 100
    def wibbledmean(a): return a / 4 + wibble * np.random.uniform(-wibble, wibble, a.shape)
    def fillsquares():
        c = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]; s = c + np.roll(c, -1, 0); s += np.roll(s, -1, 1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(s)
    def filldiamonds():
        m = maparray.shape[0]; dr = maparray[stepsize // 2:m:stepsize, stepsize // 2:m:stepsize]; ul = maparray[0:m:stepsize, 0:m:stepsize]
        maparray[0:m:stepsize, stepsize // 2:m:stepsize] = wibbledmean(dr + np.roll(dr, 1, 0) + ul + np.roll(ul, -1, 1))
        maparray[stepsize // 2:m:stepsize, 0:m:stepsize] = wibbledmean(dr + np.roll(dr, 1, 1) + ul + np.roll(ul, -1, 0))
    while stepsize >= 2:
        fillsquares(); filldiamonds(); stepsize //= 2; wibble /= wibbledecay
    maparray -= maparray.min(); return maparray / maparray.max()


def fog(x, severity=1):
    c = FOG[severity - 1]; x = np.array(x) / 255.0; mx = x.max(); h, w = x.shape[:2]
    x = x + c[0] * plasma_fractal(wibbledecay=c[1])[:h, :w][..., None]
    return (np.clip(x * mx / (mx + c[0]), 0, 1) * 255).astype(np.uint8)


def glass_blur(x, severity=1):
    sg, dmax, iters = GLASS[severity - 1]
    x = np.uint8(gaussian(np.array(x) / 255.0, sigma=sg, channel_axis=-1) * 255); h, w = x.shape[:2]
    for _ in range(iters):
        for hh in range(h - dmax, dmax, -1):
            for ww in range(w - dmax, dmax, -1):
                dx, dy = np.random.randint(-dmax, dmax, size=(2,)); hp, wp = hh + dy, ww + dx
                x[hh, ww], x[hp, wp] = x[hp, wp], x[hh, ww]
    return (np.clip(gaussian(x / 255.0, sigma=sg, channel_axis=-1), 0, 1) * 255).astype(np.uint8)


FAMILIES = [("motion_blur (N=1..10)", 1, motion_blur), ("gaussian_blur (N=11..20)", 11, gaussian_blur),
            ("zoom_blur (N=21..30)", 21, zoom_blur), ("fog (N=31..40)", 31, fog), ("glass_blur (N=41..50)", 41, glass_blur)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame"); ap.add_argument("out")
    ap.add_argument("--size", type=int, default=128); ap.add_argument("--severity", default="1,5,10"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); np.random.seed(a.seed)
    try:
        import wand.image  # noqa
        have_wand = True
    except Exception as e:
        print("WARNING: wand/ImageMagick missing, motion_blur will be skipped:", e, file=sys.stderr); have_wand = False
    img = Image.open(a.frame).convert("RGB")
    if img.size != (a.size, a.size):
        side = min(img.size)
        img = img.crop(((img.width - side) // 2, (img.height - side) // 2, (img.width + side) // 2, (img.height + side) // 2)).resize((a.size, a.size), Image.LANCZOS)
    sev = [int(s) for s in a.severity.split(",")]
    fams = [f for f in FAMILIES if have_wand or f[2] is not motion_blur]

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(sev), len(fams) + 1, figsize=(2.3 * (len(fams) + 1), 2.45 * len(sev)))
    axes = np.atleast_2d(axes)
    for r, s in enumerate(sev):
        axes[r, 0].imshow(img); axes[r, 0].set_ylabel(f"severity {s}", fontsize=10)
        if r == 0: axes[r, 0].set_title("original", fontsize=9)
        for c, (label, base_code, fn) in enumerate(fams, start=1):
            axes[r, c].imshow(fn(img, severity=s))
            if r == 0: axes[r, c].set_title(label, fontsize=8.5)
            axes[r, c].text(0.03, 0.03, f"_noise_{base_code + s - 1}", transform=axes[r, c].transAxes, fontsize=7.5, color="#fff",
                            bbox=dict(facecolor="#000", alpha=0.5, pad=1.5, lw=0))
    for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"LIBERO-Plus 'Sensor Noise': the 5 perturbations of env_wrapper.py ({a.size}×{a.size} agentview frame; parameter tables from the code) — noise_on_frame.py"
                 + ("" if have_wand else "  [motion_blur: wand missing, skipped]"), fontsize=9.5)
    fig.text(0.5, 0.005, "only agentview_image changes; the wrist (eye-in-hand) camera and proprio are untouched (env_wrapper.py:288-308). Frame: " + pathlib.Path(a.frame).name,
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95)); fig.savefig(a.out, dpi=170); plt.close(fig)
    print("written:", a.out, "| wand:", have_wand, "| families:", [f[2].__name__ for f in fams])


if __name__ == "__main__":
    main()
```

## make_tr_variant.py

```python
"""Lecture 17 — Produces a Turkish "Language" variant using the LIBERO-Plus template: `<base>_language_trK.bddl` + a task map line
+ a task_classification.json entry; passes the file through the `bddl` parser and verifies that the text that will go to the model is
THE SAME as what you wrote (Lecture 16 Section 4 rules: lowercase, no `;` and no parentheses).

NO model/assets needed. By default it does NOT touch the clone (outputs go to the --out folder); --install copies into the clone.

Usage:
  python make_tr_variant.py <LIBERO-plus root> --commands commands.csv [--suite libero_spatial] [--out tr_variants/] [--install]
  commands.csv columns: base_index,tr_id,text   (base_index: 0–9 original LIBERO order; tr_id: 1,2,…; text: Turkish command)
  Example row: 0,1,tabak ile ramekin arasındaki siyah kâseyi al ve tabağın üstüne koy
"""
import argparse, csv, json, pathlib, re, shutil, sys, unicodedata

BASES = ["pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"]
TAIL = "_view_0_0_100_0_0_initstate_0"  # fixed tail of the Language tasks (clean camera, robot init 0)


def check_text(t):
    """The pitfalls from Lecture 16: ';' deletes as a comment, '()' breaks the structure, .lower() changes I/İ, U+0307 leaks in."""
    problems = []
    if ";" in t: problems.append("';' present → everything after it is treated as a comment, 'Missing close parenthesis'")
    if "(" in t or ")" in t: problems.append("parenthesis present → nested list, TypeError")
    if t != t.lower(): problems.append(f"uppercase letter present → scan_tokens .lower() changes it: {t.lower()!r}")
    if "̇" in t: problems.append("U+0307 combining dot present")
    if unicodedata.normalize("NFC", t) != t: problems.append("not NFC-normalized")
    if "  " in t or t != t.strip(): problems.append("double/edge whitespace (scan_tokens collapses it to a single space)")
    return problems


def parse_language_like_bddl(path):
    """First the real `bddl` package (pip install bddl==1.0.1 future); if missing, an exact copy of the same two steps."""
    raw = pathlib.Path(path).read_text(encoding="utf-8")
    try:
        from bddl.parsing import scan_tokens  # bddl 1.0.1 parsing.py:14-41 → nested list
        tree = scan_tokens(filename=str(path)); src = "bddl.parsing.scan_tokens"
        lang = next(group for group in tree if isinstance(group, list) and group and group[0] == ":language")
        words = lang[1:]
    except ImportError:
        s = re.sub(r";.*$", "", raw, flags=re.MULTILINE).lower()
        toks = re.findall(r"[()]|[^\s()]+", s); src = "local copy (no bddl package)"
        i = toks.index(":language"); j = i + 1; words = []
        while toks[j] != ")": words.append(toks[j]); j += 1
    # bddl_utils.py:83 — the words are joined with " ".join
    return " ".join(words), src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--commands", required=True); ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--out", default="tr_variants"); ap.add_argument("--install", action="store_true")
    a = ap.parse_args()
    lib = pathlib.Path(a.root) / "libero/libero"; bddl_dir = lib / "bddl_files" / a.suite
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tmap_lines, json_entries, report = [], [], []
    for row in csv.DictReader(open(a.commands, encoding="utf-8")):
        base = BASES[int(row["base_index"])]; k = row["tr_id"].strip(); text = row["text"].strip()
        probs = check_text(text)
        src = (bddl_dir / f"{base}.bddl").read_text(encoding="utf-8")
        n = len(re.findall(r"\(:language .*?\)", src)); assert n == 1, f"{base}: :language line appears {n} times"
        new = re.sub(r"\(:language .*?\)", f"(:language {text})", src, count=1)
        fname = f"{base}_language_tr{k}.bddl"; (out / fname).write_text(new, encoding="utf-8")
        try:
            got, parser = parse_language_like_bddl(out / fname)
        except Exception as e:  # ';' → Missing close parenthesis, '(' → nested list (TypeError)
            got, parser = f"PARSER ERROR: {type(e).__name__}: {e}", "bddl.parsing.scan_tokens"
        same = got == text
        task_name = f"{base}_language_tr{k}{TAIL}"
        tmap_lines.append(f'    "{task_name}",')
        json_entries.append({"id": None, "name": task_name, "category": "Language Instructions", "difficulty_level": None, "lang": "tr"})
        report.append((fname, text, got, same, probs, parser))
        if a.install:
            shutil.copy(out / fname, bddl_dir / fname)
    # --- report
    print(f"# {len(report)} Turkish variants → {out}/   (parser: {report[0][5]})\n")
    print("| file | written | goes to the model (task.language) | same? | warning |\n|---|---|---|---|---|")
    for fname, text, got, same, probs, _ in report:
        print(f"| `{fname.replace(BASES[0], '…')}` | {text} | {got} | {'yes' if same else 'NO'} | {'; '.join(probs) or '—'} |")
    print("\n# libero_suite_task_map.py → append to the END of the libero_task_map[\"%s\"] list (so the indices do not shift):" % a.suite)
    print("\n".join(tmap_lines))
    print("\n# task_classification.json → to the end of the same suite list (id = last id + 1; difficulty_level: not measured with the 4 reference models → null):")
    print(json.dumps(json_entries, ensure_ascii=False, indent=1))
    print("\n# Then: as VLANEXT_TASK_IDS use the 0-based indices of the new tasks = old list length, +1, …  (Lecture 16: id-1 == index)")
    (out / "task_map_lines.txt").write_text("\n".join(tmap_lines) + "\n", encoding="utf-8")
    json.dump(json_entries, open(out / "task_classification_entries.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if a.install: print(f"\nCopied into the clone: {bddl_dir}  (update the task map and JSON by hand)")
    bad = [r for r in report if not r[3] or r[4]]
    if bad: print(f"\nWARNING: in {len(bad)} variants the text changed or there is a warning — write the Turkish in lowercase from the start, without ';' and parentheses.", file=sys.stderr)


if __name__ == "__main__":
    main()
```

## plot_day17.py

```python
"""Lecture 17 — figure 09: 7 perturbations × model in LIBERO-plus (VLANeXt paper Table III; numbers transcribed by hand).

Usage:  python plot_day17.py <output_folder/>
Source: Wu et al. 2026, VLANeXt, arXiv:2602.18532v2 Table III — trained on LIBERO and tested directly on LIBERO-plus
(perturbations unseen). Clean LIBERO numbers are from the same paper's LIBERO table (the values in Lecture 12 figure 04).
"""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = (sys.argv[1] if len(sys.argv) > 1 else ".").rstrip("/") + "/"; os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
CATS = ["Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout"]
T3 = {  # model, suite -> 7 categories
    ("OpenVLA-OFT", "Spatial"): [88.3, 40.0, 80.5, 98.3, 97.3, 96.3, 93.9],
    ("OpenVLA-OFT", "Object"): [38.9, 25.4, 99.0, 73.7, 97.6, 72.3, 71.8],
    ("OpenVLA-OFT", "Goal"): [62.0, 25.2, 53.2, 93.9, 92.5, 75.2, 59.1],
    ("OpenVLA-OFT", "Long"): [38.7, 38.2, 87.0, 89.4, 86.8, 63.5, 76.9],
    ("VLANeXt-B", "Spatial"): [95.7, 78.6, 86.9, 99.7, 98.8, 98.0, 96.6],
    ("VLANeXt-B", "Object"): [99.5, 48.5, 98.6, 99.3, 84.7, 99.8, 78.2],
    ("VLANeXt-B", "Goal"): [96.6, 63.6, 51.5, 97.5, 70.8, 96.8, 63.9],
    ("VLANeXt-B", "Long"): [69.7, 72.0, 90.3, 86.9, 75.8, 81.7, 84.3],
}
CLEAN = {("OpenVLA-OFT", "Spatial"): 97.6, ("OpenVLA-OFT", "Goal"): 97.9, ("VLANeXt-B", "Spatial"): 99.0, ("VLANeXt-B", "Goal"): 96.6}

fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
for ax, suite in zip(axes, ["Spatial", "Goal"]):
    x = np.arange(len(CATS)); w = 0.38
    for i, (model, col) in enumerate([("OpenVLA-OFT", "#3a9a5c"), ("VLANeXt-B", "#e09a00")]):
        vals = T3[(model, suite)]
        bars = ax.bar(x + (i - 0.5) * w, vals, w * 0.94, color=col, label=f"{model} (clean LIBERO {suite}: {CLEAN[(model, suite)]})", zorder=3)
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
        ax.axhline(CLEAN[(model, suite)], color=col, lw=1, ls="--", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(CATS); ax.set_ylim(0, 108); ax.yaxis.grid(True, color="#e5e5e5", zorder=0)
    ax.set_title(f"LIBERO-plus {suite}: success rate (%) per perturbation"); ax.legend(fontsize=8, loc="lower left")
    # highlight the Language column
    ax.axvspan(1.5, 2.5, color="#ffd166", alpha=0.18, zorder=0)
axes[0].set_ylabel("Success rate (%)")
fig.suptitle("Which perturbation breaks what? OpenVLA-OFT vs VLANeXt-B, trained on LIBERO and tested on LIBERO-plus — source: VLANeXt Table III (arXiv:2602.18532v2)", fontsize=10.5)
fig.text(0.5, 0.005, "Dashed line: the same model's clean LIBERO score. On Spatial, Robot (40→79) and Camera break the most; on Goal, Language (53 / 52) is the worst for both models — the only suite where the command has to be read.",
         ha="center", fontsize=8.5, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig(out + "09-vlanext-per-perturbation.png", dpi=170); plt.close(fig)
print("written:", out + "09-vlanext-per-perturbation.png")
```

## tr_commands.csv

```csv
base_index,tr_id,text,source_id
0,1,tabak ile ramekinin arasındaki siyah kaseyi al ve tabağın üstüne koy,sp0_tr
0,2,tabakla ramekin arasında duran siyah kâseyi kaldır ve tabağa yerleştir,sp0_tr2
1,1,ramekinin yanındaki siyah kaseyi al ve tabağın üstüne koy,sp1_tr
1,2,ramekinin yanında duran siyah kâseyi kaldır ve tabağa yerleştir,sp1_tr2
2,1,masanın ortasındaki siyah kaseyi al ve tabağın üstüne koy,sp2_tr
2,2,masanın ortasında duran siyah kâseyi kaldır ve tabağa yerleştir,sp2_tr2
3,1,kurabiye kutusunun üstündeki siyah kaseyi al ve tabağın üstüne koy,sp3_tr
3,2,kurabiye kutusunun üzerinde duran siyah kâseyi kaldır ve tabağa yerleştir,sp3_tr2
4,1,ahşap dolabın üst çekmecesindeki siyah kaseyi al ve tabağın üstüne koy,sp4_tr
4,2,ahşap dolabın üst çekmecesinde duran siyah kâseyi kaldır ve tabağa yerleştir,sp4_tr2
```
