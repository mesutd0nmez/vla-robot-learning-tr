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
