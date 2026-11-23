# day31

Lecture 31 — Final: the "let's do a project" simulation and the Stage 0 balance sheet. No model; all three scripts ran in this session.

| File | What | Status |
|---|---|---|
| `phase0_inventory.py` | Scans the repo: counts of lectures / images / code / data cards / thesis files, the "ran" marks in the log, the evidence in the repo for the handed-over work; `--png` figure 01 | ran → `results/inventory.csv`, `lectures.csv`, `handoff.md` |
| `design_lint.py` | Checks the 8 fields of the project design template; compares the n in the budget with the MDE from the `code/day27/budget.py` power function; names the lecture to go back to for a missing field; accepts the English and the Turkish field names | ran; selftest OK → `results/lint_demo.txt` |
| `plot_day31.py` | Figure 04: competence radar (lecture coverage + evidence from your hands + self-assessment) | ran |
| `templates/` | `design-template.md` (blank), `P0-ornek.md` (the example Claude filled in, COMPLETE), `P1.md`–`P3.md` (for you) | — |
| `self_assessment.csv` | Competence ↔ lecture mapping and evidence path; the `mesut_0_4` column is yours | — |

```bash
cd code/day31
python phase0_inventory.py --png ../../lectures/images/lecture-31
python design_lint.py --selftest
python design_lint.py templates/P1.md        # after you fill it in
python plot_day31.py
```

## phase0_inventory.py

```python
"""Lecture 31 — Stage 0 balance sheet: scans the repo and extracts the real numbers.

  python phase0_inventory.py                # the repo root ../.. is assumed
  python phase0_inventory.py --png ../../lectures/images/lecture-31

Outputs (results/):
  inventory.csv   metric, value (lectures, images, scripts, data cards, thesis files, projects ...)
  lectures.csv    per lecture: phase, words, images, code files, scripts the log counts as "ran"
  handoff.md      open work handed over from Stage 0 + a present/missing check in the repo
A single source: the repo files. The "ran" information is read from the **Code:** lines of plan/daily-log.md
(those lines were written by the sessions that wrote the lectures; Mesut's own runs are separate: projects/).
Works on both the Turkish original (## Ders N, **Kod:**, çalıştırıldı) and the English version (## Lecture N, **Code:**, ran).
"""
import argparse, csv, glob, os, re
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ap.add_argument("--png", default=None)
a = ap.parse_args()
R = os.path.abspath(a.root)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
J = lambda *p: os.path.join(R, *p)

PHASES = [(1, 8, "Phase 1 Concepts"), (9, 16, "Phase 2 Environment"), (17, 25, "Phase 3 Experiment design"), (26, 31, "Phase 4 Project")]
phase = lambda n: next(p for lo, hi, p in PHASES if lo <= n <= hi)
CODE_EXT = (".py", ".ipynb", ".mjs")
RAN = re.compile(r"çalıştırıldı|doğrulandı|koştu|gerçek çıktı|\bran\b|verified|selftest|run in this session|real output", re.I)

# --- daily-log: lecture blocks and the **Code:** lines
log = open(J("plan", "daily-log.md"), encoding="utf-8").read()
blocks = {int(m.group(1)): m.group(0) for m in re.finditer(r"^## (?:Ders|Lecture) (\d+) —.*?(?=^## |\Z)", log, re.M | re.S)}
ran_by_lec, comp_by_lec = {}, {}
for n, b in blocks.items():
    kod = " ".join(l for l in b.splitlines() if l.startswith(("**Kod:**", "**Code:**")))
    ran = comp = 0
    for m in re.finditer(r"`([\w./-]+\.(?:py|mjs|ipynb))`\s*(\(([^)]*)\))?", kod):
        st = m.group(3) or ""
        if RAN.search(st): ran += 1
        elif "py_compile" in st: comp += 1
    ran_by_lec[n], comp_by_lec[n] = ran, comp

rows = []
for f in sorted(glob.glob(J("lectures", "lecture-*.md"))):
    n = int(re.search(r"lecture-(\d+)", f).group(1))
    txt = open(f, encoding="utf-8").read()
    title = txt.splitlines()[0].lstrip("# ").strip()
    imgs = [p for p in glob.glob(J("lectures", "images", f"lecture-{n:02d}", "*.png"))]
    mmd = glob.glob(J("lectures", "images", f"lecture-{n:02d}", "*.mmd"))
    code = [p for p in glob.glob(J("code", f"day{n:02d}", "**", "*"), recursive=True) if p.endswith(CODE_EXT)]
    rows.append(dict(lecture=n, phase=phase(n), title=title[:90], words=len(txt.split()), png=len(imgs), mmd=len(mmd),
                     code_files=len(code), log_ran=ran_by_lec.get(n, 0), log_py_compile=comp_by_lec.get(n, 0),
                     hypothesis=len(re.findall(r"hipotez|hypothes", txt, re.I)), arxiv=len(set(re.findall(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", txt)))))
with open(os.path.join(OUT, "lectures.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

arxiv_all = set()
for f in glob.glob(J("lectures", "lecture-*.md")):
    arxiv_all |= set(re.findall(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", open(f, encoding="utf-8").read()))
cards = [d for d in glob.glob(J("datasets", "[0-9]*")) if os.path.exists(os.path.join(d, "dataset-card.md"))]
projects = [d for d in glob.glob(J("projects", "*")) if os.path.isdir(d) and not os.path.basename(d).startswith("_")]
thesis = sorted(os.path.basename(p) for p in glob.glob(J("thesis", "*")))
inv = [
    ("lectures", len(rows)), ("lecture_range", f"{min(r['lecture'] for r in rows)}–{max(r['lecture'] for r in rows)}"),
    ("total_words", sum(r["words"] for r in rows)), ("png_images", sum(r["png"] for r in rows)),
    ("mermaid_sources", sum(r["mmd"] for r in rows)), ("code_files", sum(r["code_files"] for r in rows)),
    ("code_folders", len(glob.glob(J("code", "day*")))),
    ("log_ran", sum(r["log_ran"] for r in rows)), ("log_py_compile_only", sum(r["log_py_compile"] for r in rows)),
    ("hypothesis_mentions", sum(r["hypothesis"] for r in rows)), ("distinct_arxiv_ids", len(arxiv_all)),
    ("data_cards", len(cards)), ("thesis_files", len(thesis)), ("thesis_file_names", ";".join(thesis)),
    ("project_folders_mesut", len(projects)), ("projects", ";".join(os.path.basename(p) for p in sorted(projects))),
    ("log_lecture_entries", len(blocks)),
    ("log_task_lines", len(re.findall(r"^\*\*(?:Görev|Task) \(Mesut\):\*\*", log, re.M))),
]
by_phase = Counter()
for r in rows: by_phase[r["phase"]] += 1
for p, c in by_phase.items(): inv.append((f"lectures_{p}", c))
with open(os.path.join(OUT, "inventory.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["metric", "value"]); w.writerows(inv)

# --- handoff: work handed over from Stage 0 — is there evidence in the repo?
def has(path_glob): return bool(glob.glob(J(path_glob)))
def contains(path, pat):
    p = J(path); return os.path.exists(p) and re.search(pat, open(p, encoding="utf-8").read(), re.M) is not None
def s_open():
    p = J("plan", "faz3-open-questions.md")
    if not os.path.exists(p): return 10
    st = [l.rstrip().rstrip("|").split("|")[-1].strip() for l in open(p, encoding="utf-8") if re.match(r"^\| S\d+ ", l)]
    return sum(1 for x in st if x in ("bekliyor", "sorulmadı", "", "waiting", "not asked"))
items = [
    ("S1 — Lecture 18 four-condition run (T4)", "projects/05-spatial0-4kosul/ exists", has("projects/05-*")),
    ("Lecture 12 first T4 eval (VLANeXt)", "projects/01-vlanext-t4-ilk-eval/ exists", has("projects/01-*")),
    ("Lecture 14 OpenVLA Mac single frame", "projects/02-openvla-mac-tek-kare/ exists", has("projects/02-openvla*")),
    ("Lecture 15 Tatar mini-reproduction", "projects/03-tatar-mini/ exists", has("projects/03-*")),
    ("Lecture 16 LIBERO-Plus setup", "projects/04-libero-plus-kurulum/ exists", has("projects/04-*")),
    ("results/runs.csv (experiment-run log)", "results/runs.csv exists", has("results/runs.csv")),
    ("Lecture 25 S1–S10 Status column", f"faz3-open-questions.md: {s_open()} / 10 questions still 'waiting' or 'not asked'", s_open() == 0),
    ("Lecture 26 preregistered predictions", "no '___' left in the predictions block of rq-candidates.md", not contains("thesis/rq-candidates.md", r"___")),
    ("Lecture 27 claim unit + quota", "no '___' left in experiment-matrix.md", not contains("thesis/experiment-matrix.md", r"___")),
    ("K1/K2 calibration (real t_call)", "the 'measured' scenario in code/day27/results/budget_stdout.txt", contains("code/day27/results/budget_stdout.txt", r"ölçüm|\bmeasured\b")),
    ("Lecture 28 ablation plan", "thesis/ablation-plan.md exists", has("thesis/ablation-plan.md")),
    ("Lecture 29 thesis proposal draft", "thesis/proposal-v0.1.md exists", has("thesis/proposal-v0.1.md")),
    ("Lecture 30 advisor meeting package", "slides/advisor-meeting-01.md exists", has("slides/advisor-meeting-01.md")),
]
with open(os.path.join(OUT, "handoff.md"), "w", encoding="utf-8") as fh:
    fh.write("| item | evidence (in the repo) | status |\n|---|---|---|\n")
    for name, ev, ok in items:
        fh.write(f"| {name} | {ev} | {'✔ present' if ok else '✘ missing'} |\n")

for k, v in inv: print(f"{k:28s} {v}")
print()
print(open(os.path.join(OUT, "handoff.md"), encoding="utf-8").read())

if a.png:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    pc = {"Phase 1 Concepts": "#cfe8ff", "Phase 2 Environment": "#d4f5d4", "Phase 3 Experiment design": "#ffd166", "Phase 4 Project": "#ffd6d6"}
    ec = {"Phase 1 Concepts": "#3b6ea8", "Phase 2 Environment": "#2e8b57", "Phase 3 Experiment design": "#c79a00", "Phase 4 Project": "#c0392b"}
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    x = np.array([r["lecture"] for r in rows])
    ax = axes[0]
    ax.bar(x, [r["words"] for r in rows], color=[pc[r["phase"]] for r in rows], edgecolor=[ec[r["phase"]] for r in rows])
    ax.set_ylabel("words / lecture")
    ax.set_title(f"Lecture 31 — Stage 0 inventory (phase0_inventory.py, real scan): {len(rows)} lectures, "
                 f"{sum(r['words'] for r in rows):,} words, {sum(r['png'] for r in rows)} images, {sum(r['code_files'] for r in rows)} code files, "
                 f"{len(arxiv_all)} distinct arXiv ids", fontsize=10.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=pc[p], edgecolor=ec[p], label=p) for p in pc], fontsize=8.5, ncol=4, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax = axes[1]
    w = 0.27
    ax.bar(x - w, [r["png"] for r in rows], w, color="#cfe8ff", edgecolor="#3b6ea8", label="image (PNG)")
    ax.bar(x, [r["code_files"] for r in rows], w, color="#ffd166", edgecolor="#c79a00", label="code file (.py/.ipynb/.mjs)")
    ax.bar(x + w, [r["log_ran"] for r in rows], w, color="#d4f5d4", edgecolor="#2e8b57", label="scripts marked 'ran/verified' in the log (lower bound)")
    ax.set_xticks(x); ax.set_xlabel("lecture"); ax.set_ylabel("count"); ax.legend(fontsize=8.5, loc="upper left"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(a.png, "01-inventory.png"), dpi=160); plt.close(fig)
    print("written:", os.path.join(a.png, "01-inventory.png"))
```

## design_lint.py

```python
"""Lecture 31 — Project design template checker (the automatic half of the rubric).

  python design_lint.py templates/P0-ornek.md          # COMPLETE expected
  python design_lint.py templates/design-template.md   # INCOMPLETE expected (blank template)
  python design_lint.py --selftest

A structural check for each of the eight fields + budget consistency: the MDE for the pooled n you wrote must agree within
±2 points with the power function of code/day27/budget.py (McNemar exact, r = 0.05).
It does not check the quality of the content (is the question any good?) — that is the human half of the rubric.
Both the English field names (Question, Null hypothesis, ...) and the Turkish ones (Soru, Null hipotez, ...) are accepted.
Exit code: 0 COMPLETE, 1 INCOMPLETE, 2 selftest failed.
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "day27"))

FIELDS = ["Question", "Null hypothesis", "Variables", "Cells", "Budget", "Success criterion", "Risks", "If yes / if no"]
ALIAS = {"Soru": "Question", "Null hipotez": "Null hypothesis", "Değişkenler": "Variables", "Hücreler": "Cells", "Bütçe": "Budget",
         "Başarı ölçütü": "Success criterion", "Riskler": "Risks", "Evet / hayır çıkarsa": "If yes / if no"}   # Turkish -> canonical
BACK = {"Question": "Lecture 26 §1", "Null hypothesis": "Lecture 26 §2", "Variables": "Lecture 26 §3", "Cells": "Lecture 27 §1",
        "Budget": "Lecture 27 §4–5", "Success criterion": "Lecture 26 §2, Lecture 19", "Risks": "Lecture 27 §6, Lecture 28",
        "If yes / if no": "Lecture 26 §1 (sixth field)"}


def sections(text):
    out, pat = {}, re.compile(r"^\*\*(.+?):\*\*", re.M)
    ms = list(pat.finditer(text))
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        key = m.group(1).strip()
        out[ALIAS.get(key, key)] = text[m.end():end].strip()
    return out


def num(pattern, s):
    m = re.search(pattern, s)
    return float(m.group(1).replace(",", ".")) if m else None


def lint(text, check_mde=True):
    S, problems = sections(text), []
    for f in FIELDS:
        body = S.get(f)
        if body is None:
            problems.append((f, "field missing")); continue
        if not body or "___" in body:
            problems.append((f, "not filled in ('___' still there)")); continue
        if f == "Null hypothesis" and not (re.search(r"H0", body) and re.search(r"δ\s*=\s*\d", body)):
            problems.append((f, "no H0 label or no δ = number"))
        if f == "Variables" and not (re.search(r"Independent|Bağımsız", body) and re.search(r"Dependent|Bağımlı", body)):
            problems.append((f, "no Independent / Dependent distinction"))
        if f == "Cells" and not (re.search(r"S-(VN|OV)", body) and re.search(r"libero_\w+", body) and re.search(r"n_init\s*\d+", body)):
            problems.append((f, "system (S-VN/S-OV), suite (libero_*) or the n_init number is missing"))
        if f == "Budget":
            h, n, m = num(r"([\d,\.]+)\s*T4 (?:sa|h)", body), num(r"n\s*=\s*(\d+)", body), num(r"MDE\s*=\s*(\d+)", body)
            if None in (h, n, m):
                problems.append((f, "T4 hours, pooled n or the MDE number is missing"))
            elif check_mde:
                from budget import mde
                true = round(100 * mde(int(n)))
                if abs(true - m) > 2:
                    problems.append((f, f"for n = {int(n)} the MDE is {true} points (r = 0.05), you wrote {int(m)}"))
        if f == "Success criterion" and not re.search(r"(p\s*<\s*0[,.]\d+|[Δκδ][^\n]{0,12}?[≥=]\s*[\d,\.]+|\d+\s*(puan|points?))", body):
            problems.append((f, "no numeric threshold (p < 0.05, |Δ| ≥ …, κ ≥ …, δ = …)"))
        if f == "Risks" and len(re.findall(r"^\s*-\s+\S", body, re.M)) < 2:
            problems.append((f, "at least two bullets (- …) are needed"))
        if f == "If yes / if no" and not (re.search(r"Yes|Evet", body) and re.search(r"No\b|Hayır", body)):
            problems.append((f, "both the Yes and the No branch must be written"))
    return problems


def report(path):
    probs = lint(open(path, encoding="utf-8").read())
    ok = not probs
    print(f"{os.path.basename(path)}: {'COMPLETE' if ok else 'INCOMPLETE'} ({len(FIELDS) - len({p[0] for p in probs})}/{len(FIELDS)} fields)")
    for f, why in probs:
        print(f"  - {f}: {why}  → go back to: {BACK[f]}")
    return ok


def selftest():
    good = open(os.path.join(HERE, "templates", "P0-ornek.md"), encoding="utf-8").read()
    assert not lint(good), lint(good)
    blank = open(os.path.join(HERE, "templates", "design-template.md"), encoding="utf-8").read()
    assert len(lint(blank, check_mde=False)) == len(FIELDS)
    bad = good.replace("MDE = 23", "MDE = 10")
    assert any("MDE is 23" in w for _, w in lint(bad)), lint(bad)
    print("selftest OK: the example is COMPLETE, the blank template is 8/8 incomplete, a wrong MDE (10 instead of 23) was caught")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        try:
            selftest(); sys.exit(0)
        except AssertionError as e:
            print("selftest FAILED:", e); sys.exit(2)
    sys.exit(0 if all(report(p) for p in sys.argv[1:]) else 1)
```

## plot_day31.py

```python
"""Lecture 31 — figure 04: competence radar.
Three lines: (1) lecture coverage = number of lectures covering the competence / the maximum (REAL, the mapping in self_assessment.csv);
(2) is there evidence produced by you (a project folder or a block you filled in; 'filled:' = no '___' left) — REAL scan;
(3) your self-assessment 0–4 (the mesut_0_4 column; not drawn if empty).
  python plot_day31.py [output folder]
"""
import csv, glob, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "lectures", "images", "lecture-31")
rows = list(csv.DictReader(open(os.path.join(HERE, "self_assessment.csv"), encoding="utf-8")))
lab = [r["yetkinlik"] for r in rows]
cov = np.array([len(r["dersler"].split(";")) for r in rows], float); cov = cov / cov.max()
def kanit(spec):  # kanit = evidence
    # "filled:path" (or the Turkish "dolu:path") = the file exists AND has no unfilled '___' left (a block Mesut filled in); otherwise a glob
    if spec.startswith(("filled:", "dolu:")):
        p = os.path.join(ROOT, spec.split(":", 1)[1])
        return float(os.path.exists(p) and "___" not in open(p, encoding="utf-8").read())
    return float(bool(glob.glob(os.path.join(ROOT, spec))))
ev = np.array([kanit(r["kanit"]) for r in rows])
me = [r["mesut_0_4"].strip() for r in rows]
ang = np.linspace(0, 2 * np.pi, len(rows), endpoint=False); close = lambda v: np.r_[v, v[:1]]; A = close(ang)
plt.rcParams.update({"font.size": 9.5})
fig = plt.figure(figsize=(10, 8.5)); ax = fig.add_subplot(111, polar=True)
ax.plot(A, close(cov), color="#c79a00", lw=2); ax.fill(A, close(cov), color="#ffd166", alpha=0.45, label="lecture coverage (number of covering lectures, normalized) — real")
ax.plot(A, close(ev), color="#2e8b57", lw=2, marker="o"); ax.fill(A, close(ev), color="#d4f5d4", alpha=0.5, label="evidence from your hands: project folder / filled-in block present (1) / absent (0) — real scan")
if all(m != "" for m in me):
    v = np.array([float(m) / 4 for m in me]); ax.plot(A, close(v), color="#3b6ea8", lw=2, ls="--", label="self-assessment (0–4) — Mesut")
else:
    ax.text(0.5, -0.12, "Self-assessment line is empty: fill in the 'mesut_0_4' column of self_assessment.csv and run again",
            transform=ax.transAxes, ha="center", fontsize=9, color="#c0392b")
ax.set_xticks(ang); ax.set_xticklabels([l.replace(" (", "\n(") for l in lab], fontsize=8.5); ax.tick_params(axis="x", pad=18)
ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["", "", "", ""]); ax.set_ylim(0, 1.05)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), fontsize=8.5)
fig.suptitle("Lecture 31 — Competence map: the gap between what was taught (yellow) and what you have proven by hand (green) is the work Stage 0 hands over", fontsize=10.5, y=0.99)
fig.tight_layout(); os.makedirs(out, exist_ok=True); fig.savefig(os.path.join(out, "04-competence-radar.png"), dpi=160)
print("evidence:", dict(zip(lab, ev.astype(int)))); print("written:", os.path.join(out, "04-competence-radar.png"))
```
