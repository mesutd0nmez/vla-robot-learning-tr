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
