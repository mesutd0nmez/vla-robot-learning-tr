# day19

Lecture 19 — Result analysis: is there shortcut learning, and in which suite? Three questions, three tools: (1) `read_patterns.py`, which reads Lecture 18's 4-row table against four patterns and produces a paragraph draft; (2) `suite_structure.py`, which answers "is a shortcut structurally possible?" from the BDDL files (Spatial 9/10, Object 10/10, Goal 1/10, Long 9/10); (3) `init_state_geometry.py`, which extracts scene features to answer "which init states are fragile?" (50 init states, without installing torch/LIBERO). None of them needs a model/GPU; they only read the **files** of the LIBERO repository (`git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO`).

| File | What | Status |
|---|---|---|
| `read_patterns.py` | `episodes.csv` -> Wilson, Δ, discordant pairs, McNemar; decides on one of the four patterns (A/B/C/D, C+B, C+D, unclear); with `--features` the geometry of the discordant scenes, with `--labels` `grasped_bowl` -> faithful/biased; a "one-paragraph interpretation" draft | `--demo` (the 4 patterns of figure 07 classified correctly) and `--selftest` (Lecture 18 table -> B) passed; run end to end with features + labels |
| `suite_structure.py` | Scene signature (object type + init region) and relative goal from the BDDLs of the four suites; is scene -> goal a function, what is the no-command upper bound; Spatial 10 tasks × region table | run on the real files (`libero_suite_task_map.py` order) |
| `init_state_geometry.py` | `.pruned_init` (numpy pickle inside a torch zip) -> (50, 92) array via a restricted unpickler; object (x, y), region verification, bowl-bowl / bowl-plate distances, robot joint deviation -> `init_features_task0.csv` | run for the 10 Spatial tasks + Goal task 8; 5 objects × 50 states inside the region |
| `plot_day19.py` | Figure 02 (no-command SR × suite, three papers + structural upper bound), 03 (50 init states, real), 04 (Spatial scene matrix), 05 (LIBERO-CF faithful/biased) | ran |
| `conditions_goal8.json`, `t4_goal_config.yaml` | The same experiment moved to LIBERO-Goal task 8 (`wrong` = task 1 "put the bowl on the stove") | written; not run |

On the Mac (no model needed; the `vla` env is enough, numpy and matplotlib):
```bash
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO ~/src/LIBERO    # we will only read the files
cd code/day19
python read_patterns.py --demo
python read_patterns.py --selftest
python suite_structure.py --libero ~/src/LIBERO
python init_state_geometry.py --libero ~/src/LIBERO --task 0 --csv init_features_task0.csv
python plot_day19.py --libero ~/src/LIBERO --out ../../lectures/images/lecture-19
# When the Colab run finishes (Lecture 18 analysis output):
python read_patterns.py ../../projects/05-spatial0-4kosul/results/episodes.csv \
       --features ../../datasets/03-libero-spatial0-4cond/data/init_features_task0.csv \
       --labels   ../../projects/05-spatial0-4kosul/results/grasped.csv      # init_state,condition,grasped_bowl
```

## read_patterns.py

```python
"""Lecture 19 — read the 4-condition table (Lecture 18 output) against four patterns and produce a "one-paragraph interpretation" draft.

Input: Lecture 18's episodes.csv (run_id, suite, task_id, condition, init_state, success, ...).
Optional: --features init_features_task0.csv (init_state_geometry.py) -> scene features of the discordant pairs;
          --labels grasped.csv (init_state, condition, grasped_bowl ∈ {bowl_1, bowl_2, none}) -> faithful/biased counts.

Usage:
  python read_patterns.py results/episodes.csv [--features init_features_task0.csv] [--labels grasped.csv]
  python read_patterns.py --demo          # classifies the four representative patterns of Lecture 18 figure 07
  python read_patterns.py --selftest      # Lecture 18 selftest table (9/5/7/4) -> expected output

Rules (heuristic thresholds for the pilot; parameters, not dogma):
  near  = gap ≤ NEAR (default 15 points)   -> "near"
  far   = gap ≥ FAR  (default 25 points)   -> "far"
  A  language is ignored       : empty and wrong are near clean
  C  wrong command 'biased'    : wrong is near clean BUT empty is far (the command says "pick a bowl", the scene chooses which)
  B  Turkish is not read       : language is read (empty far) and tr ≈ empty (tr − empty ≤ NEAR/1.5)
  D  multilingual backbone     : language is read and tr − empty ≥ FAR
  anything in between is "unclear" -> more episodes. C is added to the tr axis: "C+B", "C+D" (panel C of figure 07 has tr=empty, so it is C+B).
If the Wilson intervals overlap (almost always at n=10) the result is a "direction", not a "finding"; the script writes this on every row.
"""
import argparse, csv, math, pathlib, sys
from collections import OrderedDict, defaultdict

NEAR, FAR = 15, 25


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load_episodes(path):
    eps = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            eps[r["condition"]][int(r["init_state"])] = int(r["success"]) == 1
    return eps


def classify(sr, near=NEAR, far=FAR):
    """sr: {condition: percent}. Returns (letter, list of reasons)."""
    c, e, w, t = sr["clean"], sr["empty"], sr.get("wrong"), sr.get("tr")
    why = []
    gap_e = c - e
    lang_read = gap_e >= far
    if gap_e <= near and (w is None or c - w <= near):
        why.append(f"empty is near clean (Δ={gap_e:+.0f}); wrong is near too" if w is not None else f"empty is near clean (Δ={gap_e:+.0f})")
        return "A", why
    if w is not None and c - w <= near and lang_read:
        why.append(f"wrong is near clean (Δ={c - w:+.0f}) but empty is far (Δ={gap_e:+.0f}) -> the 'pick a bowl' part of the command is read, the target comes from the scene")
        letter_w = "C"
    else:
        letter_w = None
    if t is not None and lang_read:
        if t - e <= near / 1.5:
            why.append(f"tr ≈ empty (tr−empty={t - e:+.0f}) -> nothing beyond the scene is gained from Turkish")
            return (letter_w or "B") if letter_w != "C" else "C+B", why
        if t - e >= far:
            why.append(f"tr is well above empty (tr−empty={t - e:+.0f}) -> at least the object/action is extracted from Turkish")
            return (letter_w or "D") if letter_w != "C" else "C+D", why
        why.append(f"tr is between empty and clean (tr−empty={t - e:+.0f}): unclear between B and D")
        return (letter_w or "unclear"), why
    if letter_w:
        return letter_w, why
    why.append(f"empty drop {gap_e:+.0f} points: between {near} and {far} — one can say neither 'ignores' nor 'reads'")
    return "unclear", why


def paragraph(letter, rows, n):
    c = rows["clean"]; e = rows["empty"]; w = rows.get("wrong"); t = rows.get("tr")
    head = {"A": "In this pilot the content of the command did not change the result",
            "B": "The model reads the command but the Turkish command does not rise above the scene baseline",
            "C": "The model reads the action part of the command and picks the target object from the scene",
            "D": "The Turkish command is clearly above the scene baseline",
            "C+B": "Wrong command is 'biased', Turkish command is at the baseline", "C+D": "Wrong command is 'biased' but something is gained from Turkish",
            "unclear": "The pilot gives no direction"}[letter]
    s = (f"{head}: on {n} paired init states of LIBERO-Spatial task 0, success rate is clean {c['k']}/{n}, empty {e['k']}/{n}"
         + (f", wrong {w['k']}/{n}" if w else "") + (f", tr {t['k']}/{n}" if t else "") + ". ")
    s += f"Discordant pairs (clean✓x✗ / clean✗x✓): empty {e['b']}/{e['c']}" + (f", wrong {w['b']}/{w['c']}" if w else "") + (f", tr {t['b']}/{t['c']}" if t else "") + ". "
    s += ("Because the Wilson 95% intervals overlap, this is a directional finding; if the 50-init-state run (Lecture 18 §5) gives the same rates, "
          "significance and effect size are reported together. ")
    if letter in ("A",):
        s += "In Spatial task 0 the scene alone determines the task (suite_structure.py: 9 of the 10 tasks are separable from the scene); the Turkish effect cannot be measured on this task — move to the Goal suite."
    elif letter.startswith("C"):
        s += "If the grasped_bowl labels in the `wrong` videos are mostly biased (bowl_1), this is LIBERO-CF's 'vision overrides language' pattern; if mostly faithful (bowl_2), the model reads the command and the success metric penalizes it."
    elif letter in ("B", "C+B"):
        s += "Chen et al.'s 'foreign language ≈ no command' pattern; the rationale for comparing tokenization (Lecture 21) with a translation layer / Turkish LoRA (Lecture 23)."
    elif letter in ("D", "C+D"):
        s += "Qwen3-VL's multilingual pre-training may be extracting the object/action from Turkish (hypothesis); whether the spatial relation (arasındaki, 'which is between') is read must be tested separately with a distractor condition (Lecture 15)."
    return s


def report(eps, features=None, labels=None, near=NEAR, far=FAR, title=""):
    order = [c for c in ["clean", "empty", "wrong", "tr"] if c in eps] + [c for c in eps if c not in ("clean", "empty", "wrong", "tr")]
    base = eps["clean"]; n = len(base)
    rows = OrderedDict()
    for cnd in order:
        d = eps[cnd]; k = sum(d.values()); lo, hi = wilson(k, len(d))
        common = sorted(set(base) & set(d))
        b = sum(base[i] and not d[i] for i in common); c = sum((not base[i]) and d[i] for i in common)
        rows[cnd] = dict(k=k, n=len(d), sr=100 * k / len(d), lo=100 * lo, hi=100 * hi, b=b, c=c, p=mcnemar(b, c),
                         disc=[i for i in common if base[i] != d[i]])
    if title:
        print(f"\n=== {title}")
    print(f"| condition | k/n | SR | Wilson | Δ | discordant b/c | McNemar | interval vs clean |")
    print("|---|---|---|---|---|---|---|---|")
    cl = rows["clean"]
    for cnd, r in rows.items():
        overlap = "—" if cnd == "clean" else ("overlaps -> direction" if r["hi"] >= cl["lo"] and cl["hi"] >= r["lo"] else "separated")
        d = "—" if cnd == "clean" else f"{r['sr'] - cl['sr']:+.0f}"
        pm = "—" if cnd == "clean" else ("—" if r["p"] != r["p"] else f"{r['p']:.3f}")
        print(f"| {cnd} | {r['k']}/{r['n']} | {r['sr']:.0f}% | {r['lo']:.0f}–{r['hi']:.0f} | {d} | {'—' if cnd == 'clean' else f'{r['b']}/{r['c']}'} | {pm} | {overlap} |")
    sr = {c_: r["sr"] for c_, r in rows.items()}
    letter, why = classify(sr, near, far)
    print(f"\nPattern: **{letter}**  (thresholds: near ≤ {near}, far ≥ {far} points)")
    for w_ in why:
        print(f"  - {w_}")
    # discordant pairs per scene
    disc_all = sorted({i for c_, r in rows.items() if c_ != "clean" for i in r["disc"]})
    if disc_all:
        print("\nDiscordant init states (scenes whose outcome differs from clean):")
        for i in disc_all:
            marks = " ".join(f"{c_}={'✓' if eps[c_][i] else '✗'}" for c_ in order)
            extra = ""
            if features and i in features:
                f_ = features[i]
                extra = f"  | bowl-bowl {100 * f_['d_bowl1_bowl2']:.1f} cm, target-plate {100 * f_['d_bowl1_plate']:.1f} cm, robot deviation {f_['robot_q_dev']:.3f} rad"
            print(f"  init {i:2d}: {marks}{extra}")
        if features:
            fd = [features[i]["robot_q_dev"] for i in disc_all if i in features]
            fa = [features[i]["robot_q_dev"] for i in base if i in features]
            if fd and fa:
                print(f"  robot initial deviation: median {sorted(fd)[len(fd) // 2]:.3f} rad in discordant scenes, {sorted(fa)[len(fa) // 2]:.3f} rad in all scenes "
                      f"(if the gap is large, the fragility comes from the robot's initial pose; objects are within ±1.5 cm, Lecture 19 §3)")
    if labels:
        cnt = defaultdict(lambda: defaultdict(int))
        for (cnd, i), g in labels.items():
            cnt[cnd][g] += 1
        print("\ngrasped_bowl labels (from video, by hand):")
        for cnd, d in cnt.items():
            tot = sum(d.values())
            s_ = ", ".join(f"{g}: {v}/{tot}" for g, v in sorted(d.items()))
            if cnd == "wrong":
                s_ += f"  -> faithful (bowl_2) {100 * d.get('bowl_2', 0) / tot:.0f}%, biased (bowl_1) {100 * d.get('bowl_1', 0) / tot:.0f}% (hand-counted version of the LIBERO-CF grounding measure)"
            print(f"  {cnd}: {s_}")
    print("\nParagraph draft:\n" + paragraph(letter, rows, n))
    return letter


def demo():
    pats = {"A": (9, 8, 8, 8), "B": (9, 4, 4, 4), "C": (9, 4, 8, 4), "D": (9, 4, 4, 7)}
    for name, (c, e, w, t) in pats.items():
        eps = {"clean": {i: i < c for i in range(10)}, "empty": {i: i < e for i in range(10)}, "wrong": {i: i < w for i in range(10)}, "tr": {i: i < t for i in range(10)}}
        got = report(eps, title=f"representative pattern {name} (Lecture 18 figure 07)")
        assert got == {"A": "A", "B": "B", "C": "C+B", "D": "D"}[name], (name, got)
    print("\ndemo OK: the four representative patterns were classified correctly")


def selftest():
    pattern = {"clean": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0], "empty": [1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
               "wrong": [1, 1, 1, 0, 1, 1, 1, 0, 1, 0], "tr": [1, 0, 0, 1, 0, 0, 1, 0, 1, 0]}
    eps = {c: {i: bool(v) for i, v in enumerate(vs)} for c, vs in pattern.items()}
    got = report(eps, title="Lecture 18 selftest table (REPRESENTATIVE)")
    assert got == "B", got   # empty −40 (far), wrong −20 (neither near nor far -> not C), tr−empty = −10 ≤ 10 -> tr ≈ empty -> B
    print("\nselftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="?")
    ap.add_argument("--features"); ap.add_argument("--labels")
    ap.add_argument("--near", type=float, default=NEAR); ap.add_argument("--far", type=float, default=FAR)
    ap.add_argument("--demo", action="store_true"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.demo:
        return demo()
    if a.selftest:
        return selftest()
    if not a.episodes:
        ap.error("give episodes.csv (or --demo / --selftest)")
    eps = load_episodes(a.episodes)
    if "clean" not in eps or "empty" not in eps:
        sys.exit("episodes.csv needs the clean and empty conditions")
    feats = None
    if a.features:
        with open(a.features, encoding="utf-8") as f:
            feats = {int(r["init_state"]): {k: float(v) for k, v in r.items() if k != "init_state"} for r in csv.DictReader(f)}
    labels = None
    if a.labels:
        with open(a.labels, encoding="utf-8") as f:
            labels = {(r["condition"], int(r["init_state"])): r["grasped_bowl"] for r in csv.DictReader(f)}
    report(eps, feats, labels, a.near, a.far, title=pathlib.Path(a.episodes).name)


if __name__ == "__main__":
    main()
```

## suite_structure.py

```python
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
```

## init_state_geometry.py

```python
"""Lecture 19 — data for the question "which init states are fragile, and is it when the bowls are close?": read the 50 init
states of LIBERO-Spatial WITHOUT torch/LIBERO/MuJoCo, extract the object positions, verify them against the BDDL regions,
and write a per-scene feature table.

A .pruned_init file is torch.save(numpy_array) -> a numpy pickle inside a zip; here it is opened with a restricted
Unpickler (only numpy ndarray/dtype constructors are allowed). Array (50, 92): MuJoCo flattened state = [time(1) | qpos | qvel].
qpos = 7 arm joints + 2 gripper + 7 per object (x y z | qw qx qy qz) — the object order is the BDDL (:objects) order.
This order is not an ASSUMPTION but a check: is each object's (x, y) within +-1.5 cm of its own BDDL init region (2 cm x 2 cm)?

Usage:
  python init_state_geometry.py --libero /path/LIBERO                      # summary of the 10 tasks
  python init_state_geometry.py --libero /path/LIBERO --task 0 --csv init_features_task0.csv
Output CSV (the init_state key joins with Lecture 18's episodes.csv):
  init_state, bowl1_x, bowl1_y, bowl2_x, bowl2_y, plate_x, plate_y, d_bowl1_bowl2, d_bowl1_plate, d_bowl2_plate,
  robot_q_dev (norm of the deviation of the 7 joints from the task mean, rad), robot_q1..q7
"""
import argparse, codecs, csv, io, pathlib, pickle, re, zipfile
import numpy as np

ORDER_FILE = "libero/libero/benchmark/libero_suite_task_map.py"
N_ARM, N_GRIP = 7, 2


class _NumpyOnly(pickle.Unpickler):
    def find_class(self, mod, name):
        if (mod, name) in {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct")}:
            import numpy._core.multiarray as m  # numpy>=2; for numpy<2 it is numpy.core.multiarray
            return m._reconstruct
        if (mod, name) == ("numpy", "ndarray"):
            return np.ndarray
        if (mod, name) == ("numpy", "dtype"):
            return np.dtype
        if (mod, name) == ("_codecs", "encode"):
            return codecs.encode
        raise pickle.UnpicklingError(f"disallowed object: {mod}.{name}")


def load_init(path):
    z = zipfile.ZipFile(path)
    pkl = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    return np.asarray(_NumpyOnly(io.BytesIO(z.read(pkl))).load())


def official_order(root, suite):
    t = (root / ORDER_FILE).read_text(encoding="utf-8")
    return re.findall(r'"([^"]+)"', re.search(r'"%s":\s*\[(.*?)\]' % suite, t, re.S).group(1))


def parse_bddl(path):
    t = path.read_text(encoding="utf-8")
    regions = {}
    for m in re.finditer(r"\(\s*(\w+)\s*\n?\s*\(:target\s+(\w+)\)\s*(?:\(:ranges\s*\(\s*\(([^)]*)\)\s*\)\s*\))?", t):
        name, target, rng = m.groups()
        regions[f"{target}_{name}"] = (target, tuple(map(float, rng.split())) if rng else None)
    objs = []
    for line in re.search(r"\(:objects(.*?)\)\s*\(:obj_of_interest", t, re.S).group(1).strip().splitlines():
        if " - " in line:
            names, typ = line.strip().rsplit(" - ", 1)
            objs += names.split()
    init_txt = re.search(r"\(:init(.*?)\)\s*\(:goal", t, re.S).group(1)
    where = {o: reg for _, o, reg in re.findall(r"\((On|In)\s+(\S+)\s+(\S+)\)", init_txt)}
    g = re.search(r"\(On\s+(\S+)\s+plate_1\)", t.split("(:goal")[1])
    return dict(objects=objs, regions=regions, where=where, target=g.group(1) if g else None)


def object_xy(states, k):
    """(x, y, z) columns of the k-th object: 1 + 9 + 7k."""
    base = 1 + N_ARM + N_GRIP + 7 * k
    return states[:, base:base + 3], states[:, base + 3:base + 7]


def check_regions(states, info, tol=0.015):
    """For objects with a table region: is (x, y) within tol of the region rectangle? -> (object, region, matching count, n)"""
    out = []
    for k, o in enumerate(info["objects"]):
        reg = info["where"].get(o)
        if reg not in info["regions"] or info["regions"][reg][1] is None:
            continue
        x0, y0, x1, y1 = info["regions"][reg][1]
        xyz, _ = object_xy(states, k)
        ok = ((xyz[:, 0] >= x0 - tol) & (xyz[:, 0] <= x1 + tol) & (xyz[:, 1] >= y0 - tol) & (xyz[:, 1] <= y1 + tol)).sum()
        out.append((o, reg.replace("main_table_", ""), int(ok), len(states)))
    return out


def features(states, info):
    idx = {o: k for k, o in enumerate(info["objects"])}
    b1, _ = object_xy(states, idx["akita_black_bowl_1"]); b2, _ = object_xy(states, idx["akita_black_bowl_2"]); pl, _ = object_xy(states, idx["plate_1"])
    q = states[:, 1:1 + N_ARM]
    qdev = np.linalg.norm(q - q.mean(0), axis=1)
    rows = []
    for i in range(len(states)):
        rows.append(dict(init_state=i, bowl1_x=b1[i, 0], bowl1_y=b1[i, 1], bowl2_x=b2[i, 0], bowl2_y=b2[i, 1], plate_x=pl[i, 0], plate_y=pl[i, 1],
                         d_bowl1_bowl2=float(np.linalg.norm(b1[i, :2] - b2[i, :2])), d_bowl1_plate=float(np.linalg.norm(b1[i, :2] - pl[i, :2])),
                         d_bowl2_plate=float(np.linalg.norm(b2[i, :2] - pl[i, :2])), robot_q_dev=float(qdev[i]),
                         **{f"robot_q{j + 1}": float(q[i, j]) for j in range(N_ARM)}))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero", required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=None, help="single task; if omitted, a summary of all 10 tasks")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--n", type=int, default=None, help="only the first n init states (Lecture 18 pilot: 10)")
    a = ap.parse_args()
    root = pathlib.Path(a.libero)
    names = official_order(root, a.suite)
    tasks = [a.task] if a.task is not None else range(len(names))
    for ti in tasks:
        name = names[ti]
        info = parse_bddl(root / "libero/libero/bddl_files" / a.suite / f"{name}.bddl")
        states = load_init(root / "libero/libero/init_files" / a.suite / f"{name}.pruned_init")
        raw = root / "libero/libero/init_files" / a.suite / f"{name}.init"
        n_all = load_init(raw).shape[0] if raw.exists() else "none"
        if a.n:
            states = states[:a.n]
        qpos_len = 1 + N_ARM + N_GRIP + 7 * len(info["objects"])
        print(f"\n[{ti}] {name}\n  state {states.shape} (.init: {n_all} states; .pruned_init: {len(states)}), objects {info['objects']}, "
              f"expected qpos length {qpos_len} -> remaining {states.shape[1] - qpos_len} values are qvel (all 0: {np.allclose(states[:, qpos_len:], 0)})")
        for o, reg, ok, n in check_regions(states, info):
            print(f"  region check {o:<32} {reg:<26} {ok}/{n} init states within ±1.5 cm of the region")
        q = states[:, 1:1 + N_ARM]
        print(f"  robot joint std (rad): {np.round(q.std(0), 3)}  | deviation norm median {np.median(np.linalg.norm(q - q.mean(0), axis=1)):.3f}, max {np.linalg.norm(q - q.mean(0), axis=1).max():.3f}")
        for k, o in enumerate(info["objects"]):
            xyz, quat = object_xy(states, k)
            print(f"  {o:<32} x {xyz[:, 0].mean():+.3f}±{xyz[:, 0].std():.3f}  y {xyz[:, 1].mean():+.3f}±{xyz[:, 1].std():.3f}  z {xyz[:, 2].mean():.3f}  "
                  f"spread (max-min) x {np.ptp(xyz[:, 0]) * 100:.1f} cm, y {np.ptp(xyz[:, 1]) * 100:.1f} cm  | quat constant: {np.allclose(quat, quat[0], atol=1e-3)}")
        if "akita_black_bowl_1" in info["objects"] and "akita_black_bowl_2" in info["objects"]:
            rows = features(states, info)
            d12 = np.array([r["d_bowl1_bowl2"] for r in rows]); d1p = np.array([r["d_bowl1_plate"] for r in rows])
            print(f"  bowl-bowl distance {d12.mean() * 100:.1f} cm (min {d12.min() * 100:.1f}, max {d12.max() * 100:.1f}); target bowl-plate {d1p.mean() * 100:.1f} cm "
                  f"(min {d1p.min() * 100:.1f}, max {d1p.max() * 100:.1f})")
            if a.csv and a.task is not None:
                with open(a.csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
                    for r in rows:
                        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
                print(f"  -> {a.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
```

## plot_day19.py

```python
"""Lecture 19 figures (matplotlib). Real values: paper tables (source in the header), BDDL and .pruned_init files.
  python plot_day19.py --libero /path/LIBERO --out ../../lectures/images/lecture-19
Figure 02: no-command success rate × suite (Chen Ø / Zhan blank / Fang vision-only) + structural upper bound (suite_structure.py)
Figure 03: the 50 init states of LIBERO-Spatial task 0 (real), BDDL regions, bowl-bowl distance distribution
Figure 04: Spatial 10 tasks × region matrix (target/other bowl) — tasks 7 and 9 are the same scene
Figure 05: LIBERO-CF grounding: faithful vs biased (Fang et al. Table III)
"""
import argparse, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
YE, BE, GE, RE = "#b8860b", "#3b6fa0", "#2e8b57", "#c0392b"
SUITES = ["Spatial", "Object", "Goal", "Long"]

# ---- real values (paper tables) ----
# Chen et al. 2606.15714 Table 6, Ø (no-command) column; the en column in parentheses
CHEN = {"π0.5": {"en": [97.5, 99.0, 94.0, 93.0], "Ø": [66.0, 64.5, 9.0, 70.0]},
        "OpenVLA-OFT": {"en": [93.5, 100.0, 97.5, 94.5], "Ø": [80.0, 91.5, 10.0, 76.5]},
        "Qwen3-VL-π": {"en": [93.0, 99.5, 97.0, 96.5], "Ø": [41.0, 57.5, 6.0, 60.0]},
        "Qwen3-VL-OFT": {"en": [92.5, 100.0, 99.0, 91.0], "Ø": [34.0, 42.5, 8.5, 45.0]}}
# Zhan et al. 2601.04052 Table 8/9: original / blank
ZHAN = {"π0.5": {"orig": [95.4, 98.4, 97.2, 89.6], "blank": [65.0, 51.8, 11.4, 72.0]},
        "π0": {"orig": [96.8, 98.8, 95.8, 85.2], "blank": [29.8, 41.2, 4.6, 25.2]}}
# Fang et al. 2602.17659 Table I: V+L / vision-only
FANG = {"π0.5": {"vl": [98.8, 98.2, 98.0, 92.4], "v": [68.2, 56.6, 10.2, 71.6]},
        "OpenVLA-OFT": {"vl": [97.6, 98.4, 97.9, 94.5], "v": [83.2, 98.2, 10.0, 85.0]},
        "π0": {"vl": [96.8, 98.8, 95.8, 85.2], "v": [28.0, 26.8, 3.4, 15.6]}}
# suite_structure.py output: upper bound of a policy that does not read the command but recognizes the scene perfectly (number of tasks / 10)
BLIND = {"Spatial": 90, "Object": 100, "Goal": 10, "Long": 90}
# Fang et al. Table III, grounding (faithful / biased), no CAG
CF = {"OpenVLA-OFT": {"CF-Spatial": (6.8, 84.7), "CF-Object": (2.0, 99.6), "CF-Long": (4.0, 62.9), "CF-OOD": (6.0, 87.3)},
      "π0": {"CF-Spatial": (70.9, 37.1), "CF-Object": (6.0, 59.4), "CF-Long": (20.4, 83.1), "CF-OOD": (18.0, 72.4)},
      "π0.5": {"CF-Spatial": (39.3, 61.3), "CF-Object": (11.4, 68.8), "CF-Long": (51.6, 58.4), "CF-OOD": (20.7, 74.0)}}


def fig02(out):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
    series = [("π0.5 — Chen Ø", CHEN["π0.5"]["Ø"], Y, YE), ("π0.5 — Zhan blank", ZHAN["π0.5"]["blank"], Y, YE),
              ("π0.5 — Fang V-only", FANG["π0.5"]["v"], Y, YE), ("OpenVLA-OFT — Chen Ø", CHEN["OpenVLA-OFT"]["Ø"], B, BE),
              ("OpenVLA-OFT — Fang V-only", FANG["OpenVLA-OFT"]["v"], B, BE), ("π0 — Zhan blank", ZHAN["π0"]["blank"], G, GE),
              ("Qwen3-VL-π — Chen Ø", CHEN["Qwen3-VL-π"]["Ø"], R, RE), ("Qwen3-VL-OFT — Chen Ø", CHEN["Qwen3-VL-OFT"]["Ø"], R, RE)]
    for ax, suite, si in zip(axes, SUITES, range(4)):
        xs = np.arange(len(series))
        for x, (name, vals, fc, ec) in zip(xs, series):
            ax.bar(x, vals[si], color=fc, edgecolor=ec, linewidth=1.2)
            ax.text(x, vals[si] + 1.5, f"{vals[si]:.0f}", ha="center", fontsize=8)
        ax.axhline(BLIND[suite], color="k", ls="--", lw=1.2)
        ax.text(len(series) - 0.4, BLIND[suite] + (2.5 if BLIND[suite] > 50 else 14), f"structural upper bound {BLIND[suite]}", ha="right", fontsize=8.5, fontweight="bold")
        ax.set_title(f"LIBERO-{suite}", fontsize=12)
        ax.set_xticks(xs); ax.set_xticklabels([s[0].replace(" — ", "\n") for s in series], rotation=90, fontsize=7.5)
        ax.set_ylim(0, 108); ax.grid(axis="y", alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("success rate (%) — NO COMMAND")
    fig.suptitle("No-command (empty / Ø / vision-only) success rate × suite — three papers, real table values; dashed line: upper bound of a policy that does not read\n"
                 "the command but recognizes the scene perfectly (from BDDL, suite_structure.py: Spatial 9/10, Object 10/10, Goal 1/10, Long 9/10). "
                 "Source: Chen et al. 2606.15714 T6; Zhan et al. 2601.04052 T9; Fang et al. 2602.17659 T1", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(out / "02-empty-by-suite.png", dpi=150); plt.close(fig)


def fig03(out, libero):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from init_state_geometry import load_init, parse_bddl, official_order, object_xy
    root = pathlib.Path(libero)
    name = official_order(root, "libero_spatial")[0]
    info = parse_bddl(root / "libero/libero/bddl_files/libero_spatial" / f"{name}.bddl")
    st = load_init(root / "libero/libero/init_files/libero_spatial" / f"{name}.pruned_init")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.6, 1]})
    colors = {"akita_black_bowl_1": (Y, YE, "target bowl (bowl_1)"), "akita_black_bowl_2": (R, RE, "other bowl (bowl_2) — target of the 'wrong' command"),
              "plate_1": (G, GE, "plate"), "cookies_1": (B, BE, "cookie box"), "glazed_rim_porcelain_ramekin_1": ("#eeeeee", "#777777", "ramekin")}
    for k, o in enumerate(info["objects"]):
        xyz, _ = object_xy(st, k); fc, ec, lab = colors[o]
        reg = info["regions"][info["where"][o]][1]
        ax.add_patch(Rectangle((reg[1], reg[0]), reg[3] - reg[1], reg[2] - reg[0], fill=False, ec="k", lw=2.4, ls=(0, (3, 2)), zorder=7))
        ax.scatter(xyz[:, 1], xyz[:, 0], s=28, c=fc, edgecolors=ec, linewidths=0.6, label=lab, zorder=3, alpha=0.6)
        ax.scatter(xyz[:10, 1], xyz[:10, 0], s=70, facecolors="none", edgecolors="#333333", linewidths=0.9, zorder=4)
        ax.text(xyz[:, 1].mean(), xyz[:, 0].mean() + 0.03, lab.split(" — ")[0], ha="center", fontsize=8.5)
    ax.scatter([], [], s=70, facecolors="none", edgecolors="k", label="black ring: first 10 init states (Lecture 18 pilot)")
    ax.set_xlabel("y (m) — along the table"); ax.set_ylabel("x (m) — decreases toward the robot\n(camera faces the robot, Lecture 17)")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left"); ax.set_ylim(-0.24, 0.12)
    ax.set_title("LIBERO-Spatial task 0: object positions of the 50 init states (REAL, .pruned_init)\ndashed boxes = BDDL :init regions (2 cm × 2 cm); objects within ±1.5 cm after physical settling", fontsize=10)
    b1, _ = object_xy(st, 0); b2, _ = object_xy(st, 1); pl, _ = object_xy(st, 4)
    d12 = np.linalg.norm(b1[:, :2] - b2[:, :2], axis=1) * 100; d1p = np.linalg.norm(b1[:, :2] - pl[:, :2], axis=1) * 100
    q = st[:, 1:8]; qdev = np.linalg.norm(q - q.mean(0), axis=1)
    ax2.hist(d12, bins=12, color=R, edgecolor=RE, label=f"bowl-bowl: {d12.mean():.1f} cm (min {d12.min():.1f}, max {d12.max():.1f})")
    ax2.hist(d1p, bins=12, color=G, edgecolor=GE, alpha=0.9, label=f"target bowl-plate: {d1p.mean():.1f} cm (min {d1p.min():.1f}, max {d1p.max():.1f})")
    ax2.set_xlabel("distance (cm)"); ax2.set_ylabel("number of init states (50)")
    ax2.set_title(f"How similar are the scenes to each other?\nobject spread ≤ 3 cm; bowl orientation constant\nrobot joint std ≈ {q.std(0).mean():.3f} rad (~1°), deviation norm median {np.median(qdev):.3f} rad\n"
                  f"(cf. LIBERO-Plus robot perturbation 0.1–0.5 rad, Lecture 17)", fontsize=10)
    ax2.legend(fontsize=8.5); ax2.grid(alpha=0.3); ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "03-init-states-task0.png", dpi=150); plt.close(fig)


def fig04(out, libero):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from init_state_geometry import parse_bddl, official_order
    import re
    root = pathlib.Path(libero)
    names = official_order(root, "libero_spatial")
    short = lambda reg: reg.replace("main_table_", "").replace("wooden_cabinet_1_", "cab:").replace("flat_stove_1_", "stove:").replace("_region", "") \
        .replace("glazed_rim_porcelain_ramekin_1", "on ramekin").replace("cookies_1", "on cookie box")
    infos = [parse_bddl(root / "libero/libero/bddl_files/libero_spatial" / f"{n}.bddl") for n in names]
    regions = sorted({short(i["where"][o]) for i in infos for o in ("akita_black_bowl_1", "akita_black_bowl_2")})
    M = np.zeros((10, len(regions)))
    for ti, i in enumerate(infos):
        for o in ("akita_black_bowl_1", "akita_black_bowl_2"):
            M[ti, regions.index(short(i["where"][o]))] = 2 if o == i["target"] else 1
    fig, ax = plt.subplots(figsize=(13, 6.2))
    cmap = matplotlib.colors.ListedColormap(["white", R, Y])
    ax.imshow(M, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    for ti in range(10):
        for rj in range(len(regions)):
            if M[ti, rj] == 2: ax.text(rj, ti, "TARGET", ha="center", va="center", fontsize=8.5, fontweight="bold", color=YE)
            elif M[ti, rj] == 1: ax.text(rj, ti, "other", ha="center", va="center", fontsize=8.5, color=RE)
    labels = [f"{ti}: …{re.sub(r'pick_up_the_black_bowl_|_and_place_it_on_the_plate', '', n).replace('_', ' ')}" for ti, n in enumerate(names)]
    ax.set_yticks(range(10)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticks(range(len(regions))); ax.set_xticklabels(regions, rotation=35, ha="right", fontsize=9)
    ax.set_xticks(np.arange(-.5, len(regions), 1), minor=True); ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
    ax.grid(which="minor", color="#dddddd", lw=0.8); ax.tick_params(which="minor", length=0)
    for ti in (7, 9):
        ax.add_patch(Rectangle((-0.5, ti - 0.5), len(regions), 1, fill=False, ec="k", lw=2.2))
    ax.text(len(regions) - 0.5, 8.0, "tasks 7 and 9: SAME scene (stove + top of cabinet),\ntarget reversed — only the command tells them apart", ha="right", va="center", fontsize=9.5,
            bbox=dict(boxstyle="round", fc="white", ec="k"))
    ax.set_title("LIBERO-Spatial: where are the two bowls in each task? (BDDL :init, real; row = task, column = region)\n"
                 "9 of the 10 tasks are separable from the scene -> upper bound 9/10 for a policy that does not read the command; task 0's 'wrong' command (= task 1) describes bowl_2's region", fontsize=10.5)
    fig.tight_layout(); fig.savefig(out / "04-spatial-scene-matrix.png", dpi=150); plt.close(fig)


def fig05(out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    suites = list(CF["π0"])
    for ax, (model, d) in zip(axes, CF.items()):
        xs = np.arange(len(suites)); w = 0.38
        f = [d[s][0] for s in suites]; b = [d[s][1] for s in suites]
        ax.bar(xs - w / 2, f, w, color=G, edgecolor=GE, label="faithful: touched the object in the command")
        ax.bar(xs + w / 2, b, w, color=R, edgecolor=RE, label="biased: touched the object of the training task")
        for x, (fv, bv) in enumerate(zip(f, b)):
            ax.text(x - w / 2, fv + 1.5, f"{fv:.0f}", ha="center", fontsize=8.5); ax.text(x + w / 2, bv + 1.5, f"{bv:.0f}", ha="center", fontsize=8.5)
        ax.set_xticks(xs); ax.set_xticklabels(suites, fontsize=9); ax.set_title(model, fontsize=12)
        ax.set_ylim(0, 108); ax.grid(axis="y", alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("grounding (%)")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("LIBERO-CF (Fang et al. 2602.17659, Table III, no CAG): which object did the gripper go to under the counterfactual command?\n"
                 "The grasped_bowl label in our 'wrong' row is a hand-counted version of this: bowl_2 = faithful, bowl_1 = biased", fontsize=10)
    fig.tight_layout(rect=[0, 0.06, 1, 0.9]); fig.savefig(out / "05-cf-faithful-biased.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--libero", required=True); ap.add_argument("--out", default="../../lectures/images/lecture-19")
    a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig02(out); fig03(out, a.libero); fig04(out, a.libero); fig05(out)
    print("ok:", sorted(p.name for p in out.glob("*.png")))
```

## conditions_goal8.json

```json
{
  "clean": null,
  "empty": "",
  "wrong": "put the bowl on the stove",
  "tr": "kaseyi tabağın üstüne koy"
}
```

## t4_goal_config.yaml

```yaml
# Lecture 19 — the same four-condition experiment on LIBERO-Goal task 8 ("put the bowl on the plate"): one scene, 10 goals -> no-command baseline ≈ 10%.
# Differs from Lecture 18's t4_cond_config.yaml in only two lines: checkpoint and task_suite_name. Same patches (Lecture 12 + Lecture 18).
# Checkpoint: hf download DravenALG/VLANeXt VLANeXt_libero_goal.pt --local-dir checkpoints   (16.4 GB; the Spatial checkpoint is NOT used on Goal)
# Usage (in the VLANeXt root):
#   VLANEXT_TASK_IDS=8 VLANEXT_CONDITIONS="$(cat conditions_goal8.json)" VLANEXT_RUN_TAG=_goal8_4cond \
#   python -m scripts.libero_bench_eval --config config/t4_goal_config.yaml
# wrong = the command of Goal task 1 ("put the bowl on the stove"): feasible in the same scene, different goal -> success(wrong) measures only
# the original goal (bowl on plate); if it comes out ≈ 0 the model reads the command, if ≈ clean the scene overrides the command (Lecture 19 §6).
data:
  augmentation:
    center_crop: true
    center_crop_ratio: 0.9

model:
  diffusion_steps: 5
  scheduler_type: "flow_match"

project:
  use_wandb: false

eval:
  finetuned_checkpoint: "checkpoints/VLANeXt_libero_goal.pt"
  task_suite_name: "libero_goal"
  num_steps_wait: 10
  num_steps_execute: 8
  num_parallel_envs: 2
  num_trials_per_task: 10
  seed: 42
  image_size: 256
  resume_episodes: 0
  resume_successes: 0
```
