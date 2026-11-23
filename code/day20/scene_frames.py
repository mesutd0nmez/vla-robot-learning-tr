"""Lecture 20 — which one is "the bowl on the left"? In the 10 scenes of LIBERO-Spatial, computes the descriptions that
separate the two black bowls (left/right, front/back, near/far from the plate, near/far from the robot, higher) from the
REAL coordinates of the 50 init states, and writes which bowl is meant in THREE frames for each description:

  libero : LIBERO's own BDDL naming (an observer standing where the agentview camera is; no mirror)
           right = +y, left = -y, front = +x (near the observer), back = -x      [verified from the LIBERO-90/10 region names]
  image  : the pixels the model sees (obs[::-1, ::-1] = horizontal MIRROR of the real image): left = small u, front = large v
  robot  : the robot's own body (base x = -0.66, facing +x): left = +y, front/near = near the base

torch / LIBERO / MuJoCo are NOT needed: .pruned_init is opened with a restricted numpy unpickler (Lecture 19), the camera pose
was taken by hand from libero_tabletop_manipulation.py:188-197 (fovy = MuJoCo default 45 degrees).

Usage:
  python scene_frames.py --libero ~/src/LIBERO                    # 10 scenes × description table
  python scene_frames.py --libero ~/src/LIBERO --csv referents.csv
  python scene_frames.py --libero ~/src/LIBERO --conventions      # left/right/front/back regions in LIBERO-90/10
"""
import argparse, codecs, csv, glob, io, pathlib, pickle, re, zipfile
import numpy as np

SUITE = "libero_spatial"
ORDER_FILE = "libero/libero/benchmark/libero_suite_task_map.py"
CAM_POS = np.array([0.6586131746834771, 0.0, 1.6103500240372423])           # agentview (tabletop)
CAM_QUAT = (0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349)  # w x y z
ROBOT_BASE = np.array([-0.66, 0.0])   # mounted_panda.py:45 → -0.16 - table_length/2, table_length = 1.0
MIN_M, MIN_PX = 0.05, 20.0            # "unambiguous" threshold: at least 5 cm / 20 pixels of difference in all 50 scenes


class _NumpyOnly(pickle.Unpickler):
    def find_class(self, mod, name):
        if name == "_reconstruct" and mod in ("numpy.core.multiarray", "numpy._core.multiarray"):
            try:
                import numpy._core.multiarray as m
            except ImportError:
                import numpy.core.multiarray as m
            return m._reconstruct
        if (mod, name) == ("numpy", "ndarray"): return np.ndarray
        if (mod, name) == ("numpy", "dtype"): return np.dtype
        if (mod, name) == ("_codecs", "encode"): return codecs.encode
        raise pickle.UnpicklingError(f"disallowed object: {mod}.{name}")


def load_init(path):
    z = zipfile.ZipFile(path)
    pkl = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    return np.asarray(_NumpyOnly(io.BytesIO(z.read(pkl))).load())


def official_order(root, suite=SUITE):
    t = (root / ORDER_FILE).read_text(encoding="utf-8")
    return re.findall(r'"([^"]+)"', re.search(r'"%s":\s*\[(.*?)\]' % suite, t, re.S).group(1))


def parse_bddl(path):
    t = pathlib.Path(path).read_text(encoding="utf-8")
    regions = {}
    for m in re.finditer(r"\(\s*(\w+)\s*\n?\s*\(:target\s+(\w+)\)\s*(?:\(:ranges\s*\(\s*\(([^)]*)\)\s*\)\s*\))?", t):
        name, target, rng = m.groups()
        regions[f"{target}_{name}"] = tuple(map(float, rng.split())) if rng else None
    objs = []
    for line in re.search(r"\(:objects(.*?)\)\s*\(:obj_of_interest", t, re.S).group(1).strip().splitlines():
        if " - " in line:
            objs += line.strip().rsplit(" - ", 1)[0].split()
    init_txt = re.search(r"\(:init(.*?)\)\s*\(:goal", t, re.S).group(1)
    where = {o: reg for _, o, reg in re.findall(r"\((On|In)\s+(\S+)\s+(\S+)\)", init_txt)}
    goal = re.sub(r"\s+", " ", t.split("(:goal")[1]).strip()
    return dict(objects=objs, regions=regions, where=where, goal=goal)


def cam_rotation(q=CAM_QUAT):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def project(p, W=256, fovy=45.0, mirror=True):
    """World point → pixel (u, v). The MuJoCo camera looks along -z, +x is right, +y is up.
    mirror=True: the image the model sees (the raw image comes vertically flipped; [::-1, ::-1] turns it into a horizontal mirror)."""
    pc = (np.asarray(p) - CAM_POS) @ cam_rotation()
    f = 0.5 * W / np.tan(np.radians(fovy) / 2)
    u = W / 2 + f * pc[..., 0] / (-pc[..., 2])
    v = W / 2 - f * pc[..., 1] / (-pc[..., 2])
    return np.stack([W - u if mirror else u, v], -1)


def positions(states, objects):
    """{object: (50, 3)} — qpos = 1 time + 7 arm + 2 gripper + 7 per object (x y z qw qx qy qz), BDDL :objects order."""
    return {o: states[:, 10 + 7 * k: 13 + 7 * k] for k, o in enumerate(objects)}


def _pick(score1, score2, thr, smaller_wins):
    """From the scores of the two bowls (50,), return the winner and the SMALLEST gap over the 50 scenes; 'ambiguous' if the sign changes."""
    d = score2 - score1 if smaller_wins else score1 - score2     # > 0 → bowl_1 kazanır
    if (d > 0).all():  who = "bowl_1"
    elif (d < 0).all(): who = "bowl_2"
    else: return "ambiguous", 0.0, False
    margin = float(np.abs(d).min())
    return who, margin, margin >= thr


def descriptors(pos):
    b1, b2, plate = pos["akita_black_bowl_1"], pos["akita_black_bowl_2"], pos["plate_1"]
    uv1, uv2 = project(b1), project(b2)
    dp1, dp2 = np.linalg.norm(b1[:, :2] - plate[:, :2], axis=1), np.linalg.norm(b2[:, :2] - plate[:, :2], axis=1)
    dr1, dr2 = np.linalg.norm(b1[:, :2] - ROBOT_BASE, axis=1), np.linalg.norm(b2[:, :2] - ROBOT_BASE, axis=1)
    rows = []
    def add(desc, frame, s1, s2, thr, smaller, unit):
        who, m, ok = _pick(s1, s2, thr, smaller)
        rows.append(dict(descriptor=desc, frame=frame, referent=who, min_margin=round(m * (100 if unit == "cm" else 1), 1), unit=unit, unambiguous=ok))
    # left / right
    add("left", "libero", b1[:, 1], b2[:, 1], MIN_M, True, "cm")        # left = small y
    add("left", "image", uv1[:, 0], uv2[:, 0], MIN_PX, True, "px")     # left = small u (mirrored image)
    add("left", "robot", b1[:, 1], b2[:, 1], MIN_M, False, "cm")       # the robot's left = large y
    # front (back = the other bowl)
    add("front", "libero", b1[:, 0], b2[:, 0], MIN_M, False, "cm")     # front = large x (near the observer)
    add("front", "image", uv1[:, 1], uv2[:, 1], MIN_PX, False, "px")   # front = lower in the image (large v)
    add("front", "robot", dr1, dr2, MIN_M, True, "cm")                 # in front of / near the robot = near the base
    # frame-independent (reference object is explicit)
    add("near_plate", "none", dp1, dp2, MIN_M, True, "cm")
    add("near_robot", "none", dr1, dr2, MIN_M, True, "cm")
    add("higher", "none", b1[:, 2], b2[:, 2], MIN_M, False, "cm")
    return rows


def analyze(root):
    out = []
    for i, name in enumerate(official_order(root)):
        info = parse_bddl(root / "libero/libero/bddl_files" / SUITE / (name + ".bddl"))
        st = load_init(root / "libero/libero/init_files" / SUITE / (name + ".pruned_init"))
        pos = positions(st, info["objects"])
        # order check: are objects with a table region within ±1.5 cm of their own region (the verification from Lecture 19)
        for o, xyz in pos.items():
            r = info["regions"].get(info["where"].get(o))
            if r:
                x0, y0, x1, y1 = r
                ok = ((xyz[:, 0] > x0 - .015) & (xyz[:, 0] < x1 + .015) & (xyz[:, 1] > y0 - .015) & (xyz[:, 1] < y1 + .015)).all()
                assert ok, f"task {i}: {o} is not in its region — the object-order assumption is broken"
        for d in descriptors(pos):
            out.append(dict(task_id=i, bowl_1_at=info["where"]["akita_black_bowl_1"].replace("main_table_", ""),
                            bowl_2_at=info["where"]["akita_black_bowl_2"].replace("main_table_", ""), **d))
    return out


def conventions(root):
    """Centers of the regions whose names contain left/right/front/back in the LIBERO-90/10 BDDLs: evidence for the frame convention."""
    seen = {}
    for p in sorted(glob.glob(str(root / "libero/libero/bddl_files/libero_90/*.bddl")) + glob.glob(str(root / "libero/libero/bddl_files/libero_10/*.bddl"))):
        scene = re.match(r"([A-Z_]+SCENE\d+)", pathlib.Path(p).name).group(1)
        for k, r in parse_bddl(p)["regions"].items():
            m = re.search(r"(left|right|front|back)", k)
            if r and m and "contain" not in k:
                seen[(scene, k)] = (m.group(1), (r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
    # compare the left/right and front/back pairs in the same scene
    pairs = []
    for (scene, k), (word, x, y) in seen.items():
        other = {"left": "right", "front": "back"}.get(word)
        k2 = k.replace(word, other) if other else None
        if k2 and (scene, k2) in seen:
            _, x2, y2 = seen[(scene, k2)]
            pairs.append((scene, k, k2, word, x, y, x2, y2))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero", required=True); ap.add_argument("--csv"); ap.add_argument("--conventions", action="store_true")
    a = ap.parse_args()
    root = pathlib.Path(a.libero).expanduser()
    if a.conventions:
        pr = conventions(root)
        print(f"{'scene':24s} {'pair':46s} Δx (cm)  Δy (cm)   reading")
        ok_lr = ok_fb = n_lr = n_fb = 0
        for scene, k, k2, word, x, y, x2, y2 in pr:
            dx, dy = (x - x2) * 100, (y - y2) * 100
            if word == "left":
                n_lr += 1; ok_lr += dy < 0; note = "left = smaller y" if dy < 0 else "REVERSED"
            else:
                n_fb += 1; ok_fb += dx > 0; note = "front = larger x" if dx > 0 else "REVERSED"
            label = k.split("table_")[-1] + " | " + k2.split("table_")[-1]
            print(f"{scene:24s} {label[:46]:46s} {dx:7.1f}  {dy:7.1f}   {note}")
        print(f"\nleft/right pairs: {ok_lr}/{n_lr} 'left = -y';  front/back pairs: {ok_fb}/{n_fb} 'front = +x'")
        return
    rows = analyze(root)
    cur = None
    for r in rows:
        if r["task_id"] != cur:
            cur = r["task_id"]; print(f"\ntask {cur}: bowl_1 @ {r['bowl_1_at']} | bowl_2 @ {r['bowl_2_at']}")
        flag = "" if r["unambiguous"] else "   (ambiguous: below threshold)"
        print(f"   {r['descriptor']:11s} {r['frame']:7s} → {r['referent']:8s} min gap {r['min_margin']:6.1f} {r['unit']}{flag}")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"\nwritten: {a.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
