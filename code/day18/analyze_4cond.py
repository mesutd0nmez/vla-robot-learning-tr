"""Lecture 18 — Read the eval.log of the 4-condition experiment → episodes.csv + runs.csv + result table with Wilson intervals + paired tests.

Usage:
  python analyze_4cond.py <eval.log> --run-id vlanext_sp0_4cond_01 --out results/
  python analyze_4cond.py --logs clean=a/eval.log empty=b/eval.log wrong=c/eval.log tr=d/eval.log --out results/
        (Plan B: a separate run per condition using Lecture 16's VLANEXT_INSTR_OVERRIDE; the logs have no condition= field)
  python analyze_4cond.py --selftest      # tests with a synthetic log produced with the eval script's format string

Log format (Lecture 18 patch 3): Episode result: task=%d/%d trial=%d condition=%s success=%s rollout_steps=%d episode_time=%s ...
trial = init state index (build_episode_specs: initial_states[episode_idx]) → the conditions see the SAME scene in the same trial → paired.

Outputs (--out folder):
  episodes.csv  run_id, suite, task_id, condition, init_state, success, rollout_steps, episode_time_s
  runs.csv      CLAUDE.md convention: run_id, model, checkpoint, suite, task_id, lang, condition, n_episodes, success_rate (1 row per condition; appended if the file exists)
  table.md      condition | k/n | SR | Wilson 95% | Δ vs clean | discordant pairs (clean✓x✗ / clean✗x✓) | McNemar exact p | Fisher p
No model / GPU needed; only scipy (Fisher) — if missing the Fisher column becomes "—".
"""
import argparse, csv, math, pathlib, re, sys, tempfile
from collections import OrderedDict, defaultdict

EP = re.compile(r"Episode result: task=(\d+)/(\d+) trial=(\d+)(?: condition=(\S+))? success=(True|False) "
                r"rollout_steps=(\d+) episode_time=(\S+)")
SUITE = re.compile(r"Task suite: (\S+)")
CKPT = re.compile(r"checkpoint=(\S+)")
CONDS = re.compile(r"Conditions: (.*)")
OVR = re.compile(r"Instruction override: (.*)")
LANG = {"clean": "en", "empty": "none", "wrong": "en", "para": "en", "tr": "tr", "tr2": "tr"}
HEADER_RUNS = ["run_id", "model", "checkpoint", "suite", "task_id", "lang", "condition", "n_episodes", "success_rate"]


def parse_duration(s):
    """The output of format_duration (src/utils/logging_utils.py) is 'HH:MM:SS'; for flexibility '1m23s' / '45.2s' are also accepted."""
    s = s.strip()
    m = re.fullmatch(r"(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:([\d.]+)s)?", s)
    if m and any(m.groups()):
        h, mi, se = m.groups()
        return 3600 * float(h or 0) + 60 * float(mi or 0) + float(se or 0)
    m = re.fullmatch(r"(\d+):(\d\d):(\d\d(?:\.\d+)?)", s)
    if m:
        return 3600 * int(m.group(1)) + 60 * int(m.group(2)) + float(m.group(3))
    try:
        return float(s)
    except ValueError:
        return float("nan")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def binom_two_sided(b, c):
    """McNemar exact: discordant pair counts b (clean✓, x✗) and c (clean✗, x✓); H0: b ~ Binom(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def fisher(k1, n1, k2, n2):
    try:
        from scipy.stats import fisher_exact
    except ImportError:
        return float("nan")
    return fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]


def read_log(path, forced_condition=None):
    text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    s = SUITE.search(text); c = CKPT.search(text); cl = CONDS.search(text); o = OVR.search(text)
    eps = []
    for m in EP.finditer(text):
        task, _n, trial, cond, ok, steps, t = m.groups()
        cond = forced_condition or cond or "clean"
        eps.append(dict(task_id=int(task) - 1, init_state=int(trial), condition=cond, success=ok == "True",
                        rollout_steps=int(steps), episode_time_s=parse_duration(t)))
    return {"suite": s.group(1) if s else "?", "checkpoint": pathlib.Path(c.group(1)).name if c else "?",
            "conditions_line": (cl.group(1).strip() if cl else None), "override": (o.group(1).strip() if o else None),
            "episodes": eps}


def summarize(eps, order=None):
    by = OrderedDict()
    for e in eps:
        by.setdefault(e["condition"], {})[(e["task_id"], e["init_state"])] = e
    conds = order or list(by)
    if "clean" in conds:
        conds = ["clean"] + [c for c in conds if c != "clean"]
    rows = []
    base = by.get("clean", {})
    for cnd in conds:
        d = by[cnd]; n = len(d); k = sum(e["success"] for e in d.values())
        lo, hi = wilson(k, n)
        r = dict(condition=cnd, k=k, n=n, sr=k / n if n else float("nan"), lo=lo, hi=hi,
                 mean_steps=sum(e["rollout_steps"] for e in d.values()) / n if n else float("nan"),
                 mean_time=sum(e["episode_time_s"] for e in d.values()) / n if n else float("nan"))
        if cnd != "clean" and base:
            common = sorted(set(base) & set(d))
            b = sum(base[key]["success"] and not d[key]["success"] for key in common)
            c = sum((not base[key]["success"]) and d[key]["success"] for key in common)
            kb = sum(base[key]["success"] for key in common); nb = len(common)
            r.update(delta=(k / n - kb / nb) if n and nb else float("nan"), b=b, c=c, n_pairs=len(common),
                     p_mcnemar=binom_two_sided(b, c), p_fisher=fisher(kb, nb, k, n))
        rows.append(r)
    return rows


def fmt_p(p):
    return "—" if p != p else ("<0.001" if p < 0.001 else f"{p:.3f}")


def table_md(rows, suite, task_id):
    out = [f"Suite `{suite}`, task {task_id}; n = number of init states; Wilson 95%; paired tests relative to clean (same init state).", "",
           "| condition | k/n | SR | Wilson 95% | Δ vs clean | discordant (clean✓x✗ / clean✗x✓) | McNemar exact p | Fisher p | mean steps | mean time |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["condition"] == "clean":
            d = "—"; u = "—"; pm = "—"; pf = "—"
        else:
            d = f"{100 * r['delta']:+.0f}"; u = f"{r['b']} / {r['c']} (n={r['n_pairs']})"; pm = fmt_p(r["p_mcnemar"]); pf = fmt_p(r["p_fisher"])
        out.append(f"| {r['condition']} | {r['k']}/{r['n']} | {100 * r['sr']:.0f}% | {100 * r['lo']:.0f}–{100 * r['hi']:.0f} | {d} | {u} | {pm} | {pf} | "
                   f"{r['mean_steps']:.0f} | {r['mean_time']:.0f} s |")
    return "\n".join(out)


def write_outputs(info, eps, rows, run_id, out_dir, model="VLANeXt-B"):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    task_ids = sorted({e["task_id"] for e in eps})
    with (out / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "suite", "task_id", "condition", "init_state", "success", "rollout_steps", "episode_time_s"])
        for e in sorted(eps, key=lambda e: (e["task_id"], e["init_state"], e["condition"])):
            w.writerow([run_id, info["suite"], e["task_id"], e["condition"], e["init_state"], int(e["success"]), e["rollout_steps"],
                        f"{e['episode_time_s']:.1f}"])
    runs = out / "runs.csv"; new = not runs.exists()
    with runs.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER_RUNS)
        for r in rows:
            w.writerow([run_id, model, info["checkpoint"], info["suite"], ",".join(map(str, task_ids)), LANG.get(r["condition"], "?"),
                        r["condition"], r["n"], f"{r['sr']:.3f}"])
    md = table_md(rows, info["suite"], ",".join(map(str, task_ids)))
    (out / "table.md").write_text(md + "\n", encoding="utf-8")
    return md


def selftest():
    # fake log produced with the format string of Lecture 18 patch 3 (libero_bench_eval.py): 10 init × 4 conditions; results are ILLUSTRATIVE
    fmt = ("Episode result: task=%d/%d trial=%d condition=%s success=%s rollout_steps=%d episode_time=%s total_episodes=%d/%d "
           "total_successes=%d task_success_rate=%.1f%% total_success_rate=%.1f%% elapsed=%s avg_episode=%.2fs eta=%s video=%s")
    pattern = {"clean": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0], "empty": [1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
               "wrong": [1, 1, 1, 0, 1, 1, 1, 0, 1, 0], "tr": [1, 0, 0, 1, 0, 0, 1, 0, 1, 0]}
    lines = ["2026-11-11 07:00:00 INFO Task suite: libero_spatial",
             '2026-11-11 07:00:00 INFO Conditions: {"clean": null, "empty": "", "wrong": "pick up ...", "tr": "tabak ile ..."}',
             "2026-11-11 07:00:01 INFO Eval setup: checkpoint=checkpoints/VLANeXt_libero_spatial.pt tasks=10 trials_per_task=10 ..."]
    g = 0; succ = 0
    for init in range(10):
        for cnd in ["clean", "empty", "wrong", "tr"]:
            ok = bool(pattern[cnd][init]); g += 1; succ += ok
            lines.append("2026-11-11 07:05:00 INFO " + fmt % (1, 10, init, cnd, ok, 96 if ok else 220, "00:01:31" if ok else "00:03:04",
                                                             g, 40, succ, 0.0, 100 * succ / g, "00:10:00", 91.0, "00:30:00", "x.mp4"))
    lines.append("2026-11-11 08:00:00 INFO Eval finished: total_episodes=40 total_successes=%d final_success_rate=%.2f%% elapsed=01:02:00" % (succ, 2.5 * succ))
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "eval.log"; p.write_text("\n".join(lines), encoding="utf-8")
        info = read_log(p); eps = info["episodes"]
        assert len(eps) == 40, len(eps)
        rows = summarize(eps)
        assert [r["condition"] for r in rows] == ["clean", "empty", "wrong", "tr"]
        assert rows[0]["k"] == 9 and rows[1]["k"] == 5 and rows[3]["k"] == 4
        tr = rows[3]; assert (tr["b"], tr["c"]) == (5, 0), (tr["b"], tr["c"])       # clean✓tr✗ 5, clean✗tr✓ 0
        assert abs(tr["p_mcnemar"] - 2 * 0.5 ** 5) < 1e-9                          # 0.0625: even if all 5 discordant pairs are one-directional, p = 0.0625
        assert abs(wilson(9, 10)[0] - 0.5958) < 1e-3 and abs(wilson(9, 10)[1] - 0.9821) < 1e-3
        assert abs(parse_duration("00:01:31") - 91) < 1e-9 and abs(parse_duration("1m23s") - 83) < 1e-9
        assert abs(rows[0]["mean_time"] - (9 * 91 + 184) / 10) < 1e-9
        md = write_outputs(info, eps, rows, "selftest", pathlib.Path(d) / "results")
        assert (pathlib.Path(d) / "results" / "runs.csv").read_text().count("\n") == 5
        # Plan B: a log without a condition= field + a forced condition name
        p2 = pathlib.Path(d) / "e2.log"; p2.write_text("\n".join(lines).replace(" condition=tr", "").replace(" condition=clean", ""), encoding="utf-8")
        info2 = read_log(p2, forced_condition="empty"); assert {e["condition"] for e in info2["episodes"]} == {"empty"}
        print(md)
        print("\nselftest OK (numbers are ILLUSTRATIVE; run with a real log)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?")
    ap.add_argument("--logs", nargs="*", default=[], help="Plan B: condition=path pairs")
    ap.add_argument("--run-id", default="vlanext_sp0_4cond_01")
    ap.add_argument("--out", default="results")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.log:
        info = read_log(a.log); eps = info["episodes"]
        if eps and all(e["condition"] == "clean" for e in eps) and info["conditions_line"] in (None, "<none: task.language>"):
            print("WARNING: no condition= field in the log — the Lecture 18 patch was not applied or VLANEXT_CONDITIONS is empty; all counted as 'clean'")
    elif a.logs:
        info = None; eps = []
        for item in a.logs:
            cnd, _, path = item.partition("=")
            inf = read_log(path, forced_condition=cnd); eps += inf["episodes"]; info = info or inf
    else:
        ap.error("give an eval.log or --logs (or --selftest)")
    if not eps:
        sys.exit("No 'Episode result' line in the log — the eval may have crashed before starting; look at the end of the log.")
    print(f"suite={info['suite']}  checkpoint={info['checkpoint']}  episodes={len(eps)}  conditions={info['conditions_line'] or info['override'] or '—'}")
    rows = summarize(eps)
    print(write_outputs(info, eps, rows, a.run_id, a.out))
    print(f"\n→ {a.out}/episodes.csv, runs.csv ({len(rows)} rows appended), table.md")
    inits = defaultdict(set)
    for e in eps:
        inits[e["condition"]].add(e["init_state"])
    if len({frozenset(v) for v in inits.values()}) > 1:
        print("WARNING: the conditions have different sets of init states — the paired test was computed only on the common init states")


if __name__ == "__main__":
    main()
