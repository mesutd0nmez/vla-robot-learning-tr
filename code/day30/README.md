# day30

Lecture 30 — preparing the advisor meeting. No model; two scripts and one data file. Output `slides/advisor-meeting-01.md`.

| File | What | Status |
|---|---|---|
| `collect_questions.py` | Writes the lines in the repo's `.md` files that mention "advisor", "S7–S10", "decision", "claim unit", "4-bit" (and the Turkish equivalents) to a CSV with file:line | ran: at the end of Lecture 27 the repo had 184 lines / 37 files; once Lectures 28–31 were added 290 / 45 → `results/questions_raw.csv`, `collect_stdout.txt` (re-run on the English version of the repo: 318 lines / 47 files) |
| `questions.csv` | 11 hand-filtered questions: context, options, our recommendation, what it changes, source, impact, urgency, slot (meeting / quick / email) | written in this lecture (scores are subjective) |
| `plot_day30.py` | Priority order (`results/priority.csv`), figure 02 (impact × urgency) and 03 (30-min agenda) | ran |

On the Mac (`vla` env; only `matplotlib`):
```bash
cd code/day30
python collect_questions.py
python plot_day30.py        # redraw after changing questions.csv
```

## collect_questions.py

```python
"""Lecture 30 — Collect everything that goes to the advisor from the repo: a CSV with file:line sources.

Scans the .md files in the repo with keywords for lines that await a decision
("advisor", "S7", "S9", "S10", "decision" …), writes every match with its source file and line number,
and merges the line into one row if several keys catch it.
Purpose: every question on the meeting list should have a trace in the repo (a question with no trace = an invented question).
Both the Turkish keywords (danışman, karar, iddia birimi) and the English ones (advisor, decision, claim unit) are matched,
so the scan works on either language version of the repo. The key names (CSV values such as `danisman`, `iddia_birimi`) are kept.

  python collect_questions.py                    # repo root ../.. ; results/questions_raw.csv
  python collect_questions.py --root . --out x.csv
"""
import argparse, csv, os, re
from collections import Counter

KEYS = {
    "danisman": re.compile(r"danışman|advis[eo]r", re.I),
    "S7": re.compile(r"\bS7\b"),
    "S8": re.compile(r"\bS8\b"),
    "S9": re.compile(r"\bS9\b"),
    "S10": re.compile(r"\bS10\b"),
    "karar": re.compile(r"\bkarar|\bdecision|\bdecide", re.I),
    "iddia_birimi": re.compile(r"iddia birimi|claim unit", re.I),
    "4bit": re.compile(r"4-bit", re.I),
}
SKIP_DIRS = {".git", "node_modules", "images", "slides"}  # slides: the output of this lecture, so it must not count itself


def scan(root):
    rows = []
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d not in SKIP_DIRS)
        for fn in sorted(fns):
            if not fn.endswith(".md") or fn == "_sidebar.md":
                continue
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, root)
            with open(p, encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    hit = [k for k, rx in KEYS.items() if rx.search(line)]
                    if hit:
                        rows.append(dict(file=rel, line=i, keys="|".join(hit),
                                         text=re.sub(r"\s+", " ", line.strip())[:240]))
    return rows


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(here, "..", ".."))
    ap.add_argument("--out", default=os.path.join(here, "results", "questions_raw.csv"))
    a = ap.parse_args()
    rows = scan(os.path.abspath(a.root))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "line", "keys", "text"]); w.writeheader(); w.writerows(rows)
    kc = Counter(k for r in rows for k in r["keys"].split("|"))
    top = Counter(r["file"].split(os.sep)[0] if os.sep in r["file"] else r["file"] for r in rows)
    print(f"{len(rows)} lines, {len({r['file'] for r in rows})} files → {a.out}")
    print("per key:", dict(sorted(kc.items(), key=lambda x: -x[1])))
    print("per folder:", dict(top.most_common()))
    # Lines addressed explicitly to the advisor (advisor + decision/question) — raw candidates for the meeting list
    cand = [r for r in rows if "danisman" in r["keys"] and re.search(r"karar|sor|konuş|decision|decid|question|ask|discuss|\?", r["text"], re.I)]
    print(f"candidate lines containing 'advisor' + decision/question: {len(cand)}")
    for r in cand[:12]:
        print(f"  {r['file']}:{r['line']}  {r['text'][:110]}")


if __name__ == "__main__":
    main()
```

## plot_day30.py

```python
"""Lecture 30 — figures: (02) impact × urgency priority matrix, (03) timeline of the 30-minute agenda.
Input: questions.csv (11 questions; impact/urgency 1–5 are this lecture's SUBJECTIVE scores — updated with the advisor).
  python plot_day30.py [output folder]
Also results/priority.csv: priority = impact × urgency, sorted; which question goes to the meeting / quick approval / email.
"""
import csv, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

here = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "..", "lectures", "images", "lecture-30")
os.makedirs(out, exist_ok=True)
Q = list(csv.DictReader(open(os.path.join(here, "questions.csv"), encoding="utf-8")))
for q in Q:
    q["impact"], q["urgency"] = int(q["impact"]), int(q["urgency"]); q["prio"] = q["impact"] * q["urgency"]
Q.sort(key=lambda q: (-q["prio"], q["qid"]))
os.makedirs(os.path.join(here, "results"), exist_ok=True)
with open(os.path.join(here, "results", "priority.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["rank", "qid", "short", "impact", "urgency", "priority", "slot"])
    for i, q in enumerate(Q, 1):
        w.writerow([i, q["qid"], q["short"], q["impact"], q["urgency"], q["prio"], q["slot"]])
for i, q in enumerate(Q, 1):
    print(f"{i:2d}. {q['qid']:4s} {q['short']:22s} impact {q['impact']} × urgency {q['urgency']} = {q['prio']:2d}  → {q['slot']}")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
SLOT = {"meeting": (Y, "#c79a00", "discussed in the meeting (decision)"), "quick": (B, "#3b6ea8", "quick approval at the end of the meeting"),
        "email": (G, "#2e8b57", "asked by email")}

# 02 — priority matrix
fig, ax = plt.subplots(figsize=(11, 8))
ax.add_patch(Rectangle((3.5, 3.5), 2, 2, color=R, alpha=0.45, zorder=0))
ax.text(5.45, 5.42, "these first", ha="right", va="top", fontsize=9, color="#c0392b")
from collections import defaultdict
cell = defaultdict(list)
for q in Q:
    cell[(q["urgency"], q["impact"])].append(q)
for (u, i), qs in cell.items():
    for k, q in enumerate(qs):
        dx = (k - (len(qs) - 1) / 2) * 0.4
        fc, ec, _ = SLOT[q["slot"]]
        ax.scatter(u + dx, i, s=1500, color=fc, edgecolor=ec, lw=1.5, zorder=3)
        ax.text(u + dx, i + 0.02, q["qid"], ha="center", va="center", fontsize=10, weight="bold", zorder=4)
        ax.text(u + dx, i - 0.3, q["short"].replace(" (", "\n("), ha="center", va="top", fontsize=7.5, zorder=4)
for s, (fc, ec, lab) in SLOT.items():
    ax.scatter([], [], s=150, color=fc, edgecolor=ec, label=lab)
ax.set_xlim(1.3, 5.6); ax.set_ylim(1.3, 5.6)
ax.set_xticks(range(2, 6)); ax.set_yticks(range(2, 6))
ax.set_xlabel("urgency (1–5): which cell cannot start before the answer arrives?")
ax.set_ylabel("impact (1–5): how much does the answer change the budget / RQ order / thesis sentence?")
ax.grid(alpha=0.25); ax.legend(loc="lower right", fontsize=9)
ax.set_title("Lecture 30 — The 11 questions for the advisor: impact × urgency (SUBJECTIVE scores, this lecture's proposal)\n"
             "the four upper-right questions (Q1 suite, Q2 claim unit, Q3 contribution criterion, Q11 schedule) are the main body of the meeting", fontsize=10.5)
fig.tight_layout(); fig.savefig(os.path.join(out, "02-priority-matrix.png"), dpi=160); plt.close(fig)

# 03 — agenda
agenda = [  # (start min, duration, title, kind)
    (0, 3, "Status\nS1 pending\n(slide 1)", "status"),
    (3, 4, "3 RQs + matrix\n+ budget\n(slides 2–3)", "status"),
    (7, 5, "Q1 suite (S7)\n+ Q2 claim unit", "decision"),
    (12, 5, "Q3 contribution criterion (S10)", "decision"),
    (17, 4, "Q4 LoRA / GPU (S9)", "decision"),
    (21, 3, "Q11 schedule\n+ proposal form", "decision"),
    (24, 3, "quick approval:\nQ5, Q6, Q10", "quick"),
    (27, 3, "summary: decisions,\nowners, date", "closing"),
]
col = {"status": (B, "#3b6ea8"), "decision": (Y, "#c79a00"), "quick": ("#eee", "#888"), "closing": (G, "#2e8b57")}
fig, ax = plt.subplots(figsize=(15, 3.8))
for s, d, t, k in agenda:
    fc, ec = col[k]
    ax.barh(0, d, left=s, color=fc, edgecolor=ec, height=0.6)
    ax.text(s + d / 2, 0, t, ha="center", va="center", fontsize=8.2)
ax.set_xlim(0, 30); ax.set_ylim(-0.6, 0.6); ax.set_yticks([])
ax.set_xticks(range(0, 31, 3)); ax.set_xlabel("minutes")
ax.text(30, 0.48, "Q7 Webots, Q8 Tatar et al., Q9 Cohn → email before the meeting (written answer)", ha="right", fontsize=8.5, color="#2e8b57")
ax.set_title("Lecture 30 — 30-minute agenda: 17 minutes for five decisions (Q1+Q2, Q3, Q4, Q11), 7 minutes of context; the status report is kept short", fontsize=10.5)
fig.tight_layout(); fig.savefig(os.path.join(out, "03-agenda.png"), dpi=160); plt.close(fig)
print("written: 02-priority-matrix.png, 03-agenda.png →", out)
```
