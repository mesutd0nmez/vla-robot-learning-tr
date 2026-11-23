"""Lecture 19 — read the question "is a shortcut possible?" off the BDDL files: in the four LIBERO suites, does the scene
(objects + init regions) alone determine the goal?

Logic: a policy can solve a task without reading the command at all only if the scene "tells" it the task. For each task in the suite
  scene signature = (multiset of object types, init region of each object; objects of the same type are indistinguishable)
  goal            = the (:goal) expression
If more than one goal falls on the same scene signature, scene -> task is not a function: the task cannot be chosen without reading the command.

Usage:
  python suite_structure.py --libero /path/LIBERO            # clone root (reads under libero/libero/bddl_files)
  python suite_structure.py --libero /path/LIBERO --out results/suite_structure.csv --md results/suite_structure.md
Only reads files (with regex); does not import the LIBERO package, no model/GPU needed.
"""
import argparse, csv, pathlib, re
from collections import Counter, defaultdict

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def parse_bddl(path):
    t = path.read_text(encoding="utf-8")
    lang = re.search(r"\(:language\s+([^)]*)\)", t).group(1).strip()
    objs = {}
    m = re.search(r"\(:objects(.*?)\)\s*\(:obj_of_interest", t, re.S)
    for line in m.group(1).strip().splitlines():
        line = line.strip()
        if not line or " - " not in line:
            continue
        names, typ = line.rsplit(" - ", 1)
        for n in names.split():
            objs[n] = typ.strip()
    init_txt = re.search(r"\(:init(.*?)\)\s*\(:goal", t, re.S).group(1)
    init = re.findall(r"\((On|In)\s+(\S+)\s+(\S+)\)", init_txt)
    where = {o: reg for _, o, reg in init}
    goal = re.sub(r"\s+", " ", re.search(r"\(:goal\s*(.*?)\)\s*\)\s*$", t, re.S).group(1).strip())
    # scene signature: object TYPE + region (not names: two black bowls are indistinguishable in the image)
    scene = tuple(sorted((objs.get(o, o), reg) for _, o, reg in init if o in objs))
    fixtures = tuple(sorted((o, reg) for _, o, reg in init if o not in objs))
    # relative goal: replace object names with "type@init-region" -> different physical goals in the same scene become distinct
    rel_goal = re.sub(r"\b([a-z_]+_\d)\b", lambda m_: f"{objs.get(m_.group(1), m_.group(1))}@{where.get(m_.group(1), '-')}", goal)
    return dict(task=path.stem, language=lang, objects=objs, where=where, scene=scene, fixtures=fixtures, goal=goal, rel_goal=rel_goal)


def official_order(root, suite):
    """Order in libero_suite_task_map.py (the benchmark's task_id; NOT alphabetical). The file is read, not imported."""
    t = (root / "libero" / "libero" / "benchmark" / "libero_suite_task_map.py").read_text(encoding="utf-8")
    block = re.search(r'"%s":\s*\[(.*?)\]' % suite, t, re.S).group(1)
    return re.findall(r'"([^"]+)"', block)


def analyze(root):
    rows = []
    for suite in SUITES:
        d = root / "libero" / "libero" / "bddl_files" / suite
        tasks = [parse_bddl(d / (name + ".bddl")) for name in official_order(root, suite)]
        scenes = Counter(t["scene"] + t["fixtures"] for t in tasks)
        by_scene = defaultdict(list)
        for t in tasks:
            by_scene[t["scene"] + t["fixtures"]].append(t)
        n_scenes = len(scenes)
        max_share = max(scenes.values())
        # does the goal follow from the scene: yes if exactly one goal falls on each scene
        deterministic = all(len({x["rel_goal"] for x in v}) == 1 for v in by_scene.values())
        # in the best case without reading the command (picking the most frequent goal per scene), how many tasks are solved
        best_blind = sum(max(Counter(x["rel_goal"] for x in v).values()) for v in by_scene.values())
        obj_sets = Counter(tuple(sorted(Counter(t["objects"].values()).items())) for t in tasks)
        rows.append(dict(suite=suite, n_tasks=len(tasks), n_scenes=n_scenes, n_object_sets=len(obj_sets),
                         max_tasks_per_scene=max_share, scene_determines_goal=deterministic,
                         blind_upper_bound=f"{best_blind}/{len(tasks)}", tasks=tasks, by_scene=by_scene))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero", required=True, help="LIBERO clone root")
    ap.add_argument("--out", default=None)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    rows = analyze(pathlib.Path(a.libero))
    hdr = ["suite", "n_tasks", "n_scenes", "n_object_sets", "max_tasks_per_scene", "scene_determines_goal", "blind_upper_bound"]
    print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join(str(r[h]) for h in hdr) + " |")
    print("\nLegend: n_scenes = distinct scene signatures (object type + init region; fixtures included); scene_determines_goal = exactly one goal per scene;"
          "\nblind_upper_bound = number of tasks a policy that never reads the command but recognizes the scene perfectly could solve (upper bound).")
    sp = next(r for r in rows if r["suite"] == "libero_spatial")
    short = lambda reg: reg.replace("main_table_", "").replace("wooden_cabinet_1_", "cab:").replace("flat_stove_1_", "stove:").replace("_region", "")
    print("\nlibero_spatial — task: bowl regions (BDDL :init) -> target bowl (:goal):")
    for i, t in enumerate(sp["tasks"]):
        tgt = re.search(r"\(On\s+(\S+)\s+plate_1\)", t["goal"]).group(1)
        print(f"  {i}: bowl_1@{short(t['where']['akita_black_bowl_1']):<22} bowl_2@{short(t['where']['akita_black_bowl_2']):<22} target={tgt} ({short(t['where'][tgt])})")
    role = defaultdict(lambda: {"target": [], "other": []})
    for i, t in enumerate(sp["tasks"]):
        tgt = re.search(r"\(On\s+(\S+)\s+plate_1\)", t["goal"]).group(1)
        for o in ("akita_black_bowl_1", "akita_black_bowl_2"):
            role[short(t["where"][o])]["target" if o == tgt else "other"].append(i)
    print("\nRegion -> in which tasks it holds the target / other bowl:")
    for reg, d in sorted(role.items()):
        print(f"  {reg:<22} target:{d['target']}  other:{d['other']}")
    print("\nTasks sharing the same scene signature (cannot be told apart without reading the command):")
    for r in rows:
        for v in r["by_scene"].values():
            if len(v) > 1:
                print(f"  {r['suite']}: " + " | ".join(f"{x['task']} -> {x['rel_goal']}" for x in v))
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(hdr)
            for r in rows:
                w.writerow([r[h] for h in hdr])
    if a.md:
        lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] + ["| " + " | ".join(str(r[h]) for h in hdr) + " |" for r in rows]
        pathlib.Path(a.md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
