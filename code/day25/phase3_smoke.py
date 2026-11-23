"""Lecture 25 — Phase 3 tool inventory: run the scripts in code/day17–24 as a dry run (no model, no GPU),
and tabulate which one works with which external input.

Usage (from the repo root or from inside code/day25):
    python phase3_smoke.py                              # no external clone: only those with the "none" dependency
    python phase3_smoke.py --libero ~/src/LIBERO --plus ~/src/LIBERO-plus \
                           --llama code/day22/tokenizer.model --qwen path/to/qwen3/tokenizer.json
    python phase3_smoke.py --only 19,20                 # only the Lecture 19 and 20 scripts
    python phase3_smoke.py --out results                # smoke.csv + smoke.md go here

Each row: lecture (column `ders`), script, command, dependency (none / LIBERO clone / LIBERO-plus clone / bddl package /
sentencepiece+tokenizer / Webots / HF+GPU), result (OK / FAILED / SKIPPED), duration, last line of the output.
"SKIPPED" = the dependency was not provided; "FAILED" = it ran but the exit code is not 0. No script is modified;
output files are written to a temporary folder.
"""
import argparse, csv, os, pathlib, shutil, subprocess, sys, tempfile, time

HERE = pathlib.Path(__file__).resolve().parent
CODE = HERE.parent
ROOT = CODE.parent
DS = ROOT / "datasets"
PY = sys.executable


def manifest(libero, plus, tmp, llama=None, qwen=None):
    """(lecture, script, argv, cwd, dependency, short description). argv None -> skip (dependency missing)."""
    L = str(libero) if libero else None
    P = str(plus) if plus else None
    T = (str(llama), str(qwen)) if (llama and qwen) else None
    try:
        import bddl.parsing  # noqa: F401  (pip install "bddl==1.0.1" future)
        B = True
    except Exception:
        B = False
    t = lambda name: str(tmp / name)
    d = lambda n: CODE / f"day{n}"
    M = []
    # Lecture 17 — perturbation counts from the LIBERO-plus clone
    M += [(17, "perturbation_table.py", [P, "--json", t("pt.json")] if P else None, d(17), "LIBERO-plus clone", "7 perturbations × Spatial counts"),
          (17, "camera_geometry.py", [P, t("cam")] if P else None, d(17), "LIBERO-plus clone", "376 camera poses (V/H/S/ER/EV)"),
          (17, "robot_init_spread.py", [P, t("rob")] if P else None, d(17), "LIBERO-plus clone", "500 Panda init_qpos, ‖Δ‖"),
          (17, "layout_shift.py", [P, t("lay")] if P else None, d(17), "LIBERO-plus clone", "_add_ / _level shifts"),
          (17, "classify_rewrites.py", [P, t("cls")] if P else None, d(17), "LIBERO-plus clone", "the 50 rewrites of task 0, R0/R1/R2"),
          (17, "make_tr_variant.py", [P, "--commands", str(d(17) / "tr_commands.csv"), "--out", t("trv")] if (P and B) else None, d(17), "LIBERO-plus clone + bddl", "Turkish _language_trK BDDL generation (needs bddl 1.0.1)")]
    # Lecture 18 — condition file, statistics, log analysis (selftest)
    M += [(18, "make_conditions.py", ["--task", "0", "--out", t("cond.json")], d(18), "none", "clean/empty/wrong/tr condition JSON"),
          (18, "spec_order_demo.py", [str(d(18) / "conditions_task0.json")], d(18), "none", "init × condition order"),
          (18, "analyze_4cond.py", ["--selftest"], d(18), "none", "eval.log -> table (Wilson)"),
          (18, "power_day18.py", [t("power")], d(18), "none", "Fisher / McNemar power curves")]
    # Lecture 19 — reading protocol, BDDL structure, init state geometry
    M += [(19, "read_patterns.py", ["--selftest"], d(19), "none", "A/B/C/D decision tree"),
          (19, "read_patterns.py --demo", ["--demo"], d(19), "none", "paragraph from an example table"),
          (19, "suite_structure.py", ["--libero", L] if L else None, d(19), "LIBERO clone", "no-command upper bound 9/10/1/9"),
          (19, "init_state_geometry.py", ["--libero", L, "--task", "0"] if L else None, d(19), "LIBERO clone", "geometry of the 50 init states")]
    # Lecture 20 — Turkish set
    M += [(20, "tr_morph.py", ["--selftest"], d(20), "none", "suffix generator, 42 checks"),
          (20, "make_tr_set.py", ["--out", t("set"), "--referents", str(DS / "04-tr-spatial-commands/data/referents.csv")], d(20), "none", "generation of pairs.csv + commands.csv"),
          (20, "validate_set.py", [str(DS / "04-tr-spatial-commands/data"), "--d01", str(DS / "01-libero-spatial-tr-mini/data/commands.csv")], d(20), "none", "set consistency"),
          (20, "scene_frames.py", ["--libero", L] if L else None, d(20), "LIBERO clone", "referent table in three frames"),
          (20, "make_cf_bddl.py", ["--libero", L, "--pairs", str(DS / "04-tr-spatial-commands/data/pairs.csv"), "--out", t("cf")] if (L and B) else None, d(20), "LIBERO clone + bddl", "counterfactual BDDL (needs bddl 1.0.1)")]
    # Lecture 21 — Webots
    M += [(21, "check_webots_env.py", None, d(21), "Webots", "WEBOTS_HOME + import controller"),
          (21, "ur10e_fk.py", [], d(21), "none", "UR10e FK (scipy IK --solve separately)")]
    # Lecture 22 — tokenization
    M += [(22, "tokenize_set.py", ["--llama", T[0], "--qwen", T[1], "--out", t("tok")] if T else None, d(22), "sentencepiece+tokenizer", "100 commands × 2 tokenizers (needs tokenizer.model / tokenizer.json)"),
          (22, "tokens.csv == datasets/05?", ["--diff", t("tok/tokens.csv"), str(DS / "05-tokenization-tr-commands/data/tokens.csv")] if T else None, HERE, "sentencepiece+tokenizer", "reproduction: are the first 7 columns identical"),
          (22, "probe_edge_cases.py", ["--llama", T[0], "--qwen", T[1]] if T else None, d(22), "sentencepiece+tokenizer", "İ.lower, false friends")]
    # Lecture 23 — prediction interval
    M += [(23, "prediction_interval.py", ["--selftest"], d(23), "none", "PI formula, 5 checks"),
          (23, "prediction_interval.py --scores", ["--scores", "90", "80", "90"], d(23), "none", "interval from 3 repeats")]
    # Lecture 24 — parser, budget, MT
    M += [(24, "tr_parse.py --selftest", ["--selftest"], d(24), "none", "parser, 4 checks"),
          (24, "tr_parse.py", ["--csv", str(DS / "04-tr-spatial-commands/data/commands.csv"), "--out", t("parse.csv")], d(24), "none", "80 rows + held-out 12"),
          (24, "qlora_budget.py", [], d(24), "none", "4-bit LoRA memory/time estimate"),
          (24, "mt_layer_mac.py", None, d(24), "HF+GPU", "Opus-MT / NLLB round-trip (downloads a model)")]
    return M


def diff_first_cols(a, b, k=7):
    """Compare the first k columns of two CSVs (the Lecture 22 reproduction check)."""
    ra = [row[:k] for row in csv.reader(open(a, encoding="utf-8"))]
    rb = [row[:k] for row in csv.reader(open(b, encoding="utf-8"))]
    bad = sum(1 for x, y in zip(ra, rb) if x != y) + abs(len(ra) - len(rb))
    print(f"{len(ra)} vs {len(rb)} rows, rows differing in the first {k} columns: {bad} -> {'IDENTICAL' if bad == 0 else 'DIFFERENT'}")
    return 0 if bad == 0 else 1


def run_one(script, argv, cwd, timeout):
    name = script.split()[0]
    if argv and argv[0] == "--diff":                   # built-in check, not a separate script
        cmd = [PY, str(HERE / "phase3_smoke.py")] + argv
    else:
        cmd = [PY, str(cwd / name)] + argv
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"})
        rc, out = p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired:
        rc, out = 124, "timeout"
    dt = time.time() - t0
    lines = [l for l in out.strip().splitlines() if l.strip()]
    last = lines[-1][:110] if lines else ""
    return rc, dt, last, cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--libero", help="LIBERO clone root (Lecture 19/20 scripts)")
    ap.add_argument("--plus", help="LIBERO-plus clone root (Lecture 17 scripts)")
    ap.add_argument("--llama", help="Meta LLaMA-2 tokenizer.model (Lecture 22)")
    ap.add_argument("--qwen", help="Qwen3 tokenizer.json (Lecture 22)")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"), help="(internal use) compare the first 7 columns of two CSVs")
    ap.add_argument("--only", help="comma-separated lecture numbers, e.g. 19,20")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    if a.diff:
        return diff_first_cols(*a.diff)
    only = {int(x) for x in a.only.split(",")} if a.only else None
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="faz3smoke_"))
    rows = []
    for day, script, argv, cwd, need, desc in manifest(a.libero and pathlib.Path(a.libero).expanduser(),
                                                      a.plus and pathlib.Path(a.plus).expanduser(), tmp,
                                                      a.llama and pathlib.Path(a.llama).expanduser(), a.qwen and pathlib.Path(a.qwen).expanduser()):
        if only and day not in only:
            continue
        if argv is None:
            rows.append(dict(ders=day, script=script, bagimlilik=need, sonuc="SKIPPED", sure_s="", son_satir="", aciklama=desc))
            print(f"[--] Lecture {day:2d} {script:34s} SKIPPED ({need})")
            continue
        rc, dt, last, cmd = run_one(script, argv, cwd, a.timeout)
        res = "OK" if rc == 0 else "FAILED"
        rows.append(dict(ders=day, script=script, bagimlilik=need, sonuc=res, sure_s=f"{dt:.1f}", son_satir=last, aciklama=desc))
        print(f"[{'OK' if rc == 0 else '!!'}] Lecture {day:2d} {script:34s} {res:7s} {dt:6.1f}s  {last}")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "smoke.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    ok = sum(r["sonuc"] == "OK" for r in rows); err = sum(r["sonuc"] == "FAILED" for r in rows); sk = sum(r["sonuc"] == "SKIPPED" for r in rows)
    with open(out / "smoke.md", "w", encoding="utf-8") as f:
        f.write(f"# Phase 3 dry run — {ok} OK / {err} FAILED / {sk} SKIPPED (total {len(rows)})\n\n")
        f.write("| Lecture | Script | Dependency | Result | Time | Last line |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['ders']} | `{r['script']}` | {r['bagimlilik']} | {r['sonuc']} | {r['sure_s']} | {r['son_satir'].replace('|', '/')} |\n")
    print(f"\n{ok} OK / {err} FAILED / {sk} SKIPPED -> {out / 'smoke.csv'}, {out / 'smoke.md'}")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
