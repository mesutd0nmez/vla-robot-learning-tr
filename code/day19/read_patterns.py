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
