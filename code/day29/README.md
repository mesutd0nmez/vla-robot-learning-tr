# day29

Lecture 29 — a one-page thesis proposal draft. No model. Output file: `thesis/proposal-v0.1.md`.

| File | What | Status |
|---|---|---|
| `proposal_lint.py` | "Definition of done" check of the proposal: body ≤ 700 words (References excluded; the Turkish original was checked against 650, the English text is longer), 8 required sections, a `[Lecture NN]` / `[arXiv:…]` / `[proposal]` / `[calc…]` tag on every sentence/table row that contains a number; accepts both the Turkish and the English headings/tags; `--selftest` | ran: 680/700 words, 12/12 numeric claims tagged, **COMPLETE**; selftest OK → `results/lint.txt`, `results/claims.csv` |
| `plot_day29.py` | Figure 02 (Gantt; T4 hours from `code/day27/results/budget*.csv`) and 03 (risk matrix; scores are an assessment) | ran → `results/plot_stdout.txt` |

On the Mac (`vla` env; only `matplotlib`):
```bash
cd code/day29
python proposal_lint.py --selftest
python proposal_lint.py ../../thesis/proposal-v0.1.md --csv results/claims.csv
python plot_day29.py
```
Exit code of `proposal_lint.py`: 0 = COMPLETE, 1 = INCOMPLETE, 2 = selftest failed.

## proposal_lint.py

```python
"""Lecture 29 — "definition of done" check of the thesis proposal draft.

  python proposal_lint.py ../../thesis/proposal-v0.1.md            # report, exit code 0 = COMPLETE
  python proposal_lint.py ../../thesis/proposal-v0.1.md --max-words 700 --csv results/claims.csv
  python proposal_lint.py --selftest

Three checks:
  1. Body word count ≤ limit (the References section, HTML comments and the "Version:" line excluded) — "one page".
  2. Required sections exist: Problem, Aim, Research questions, Method, Contribution, Schedule, Risk, References.
  3. Every sentence/table row that contains a numeric claim carries a source tag:
     [Lecture NN] · [arXiv:XXXX.XXXXX] · [proposal] · [calc…]. Digits in model/label names
     (RQ-a, S-OV, 4-bit, LLaMA-2, Qwen3-VL, T4, OpenVLA-7B, WP1; no week names) do not count.
Both the Turkish original (Ders NN / öneri / hesap, Problem / Amaç / ...) and the English form (Lecture NN / proposal / calc,
Problem / Aim / ...) are accepted, so the linter works on either language version of thesis/proposal-v0.1.md.
Exit code: 0 COMPLETE, 1 INCOMPLETE, 2 selftest failed.
"""
import argparse, csv, re, sys

# canonical key (used in messages) -> heading regex (Turkish original | English)
REQUIRED = {"Problem": r"^##\s+\d*\.?\s*Problem", "Aim": r"^##\s+\d*\.?\s*(Amaç|Aim)",
            "Research questions": r"^##\s+\d*\.?\s*(Araştırma soru|Research question)", "Method": r"^##\s+\d*\.?\s*(Yöntem|Method)",
            "Contribution": r"^##\s+\d*\.?\s*(Beklenen katkı|Expected contribution)", "Schedule": r"^##\s+\d*\.?\s*(Takvim|Schedule)",
            "Risk": r"^##\s+\d*\.?\s*Risk", "References": r"^##\s+(Kaynaklar|References)"}
TAG = re.compile(r"\[(Ders \d+(?:[–-]\d+)?|Lecture \d+(?:[–-]\d+)?|arXiv:\d{4}\.\d{4,5}|öneri|proposal|hesap[^\]]*|calc[^\]]*)\]")
# Names that contain digits but are not numeric claims (removed first)
NAMES = re.compile(r"(RQ-[a-z]|S-OV|S-VN|\d-bit|bf16|LLaMA-2|Qwen3(-VL)?|OpenVLA-7B|OpenVLA-OFT|T4|M2|İP\d|WP\d|H0-\w+|"
                   r"v\d+\.\d+|lecture-\d+|Ders \d+|Lecture \d+(?:[–-]\d+)?|arXiv:\d{4}\.\d{4,5}|EN-B|tr_\w+|\bK\d\b|\bS\d+\b)")
NUM = re.compile(r"\d")


def body_and_refs(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    lines = text.splitlines()
    out, refs, in_refs = [], [], False
    for ln in lines:
        if re.match(r"^##\s+(Kaynaklar|References)", ln):
            in_refs = True
        if ln.startswith(("Sürüm:", "Version:")):
            continue
        (refs if in_refs else out).append(ln)
    return "\n".join(out), "\n".join(refs)


def words(s):
    s = re.sub(r"[`*_#|>\[\]]", " ", s)
    return [w for w in re.split(r"\s+", s) if re.search(r"\w", w)]


def claim_units(body):
    """Units at the level of a sentence (paragraph/bullet) or a table row."""
    units = []
    for ln in body.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("<!--") or re.match(r"^\|[-\s|]+\|$", s) or s.startswith("**Başlık") \
                or s.startswith("**Title") or s.startswith("**Öğrenci") or s.startswith("**Student"):
            continue
        if s.startswith("|"):
            units.append(s); continue
        s = re.sub(r"^(\d+\.|[-*])\s+", "", s)   # drop the list marker
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ\"“*(])", s):
            units.append(sent)
    return units


def lint(path, max_words):
    text = open(path, encoding="utf-8").read()
    body, refs = body_and_refs(text)
    problems, rows = [], []
    n = len(words(body))
    if n > max_words:
        problems.append(f"body {n} words > {max_words} (one-page limit)")
    for name, rx in REQUIRED.items():
        if not re.search(rx, text, flags=re.M):
            problems.append(f"'{name}' section is missing")
    untagged = 0
    for u in claim_units(body):
        stripped = NAMES.sub(" ", TAG.sub(" ", u))
        has_num = bool(NUM.search(stripped))
        tagged = bool(TAG.search(u))
        if has_num:
            rows.append({"tagged": int(tagged), "tags": ";".join(TAG.findall(u)), "text": u[:160]})
            if not tagged:
                untagged += 1
                problems.append(f"untagged numeric claim: {u[:90]}…")
    return n, rows, problems


def selftest():
    bad_en = ("# T\n\n## 1. Problem\n\nSuccess drops from 97 to 49.\n\n## 2. Aim\n\nx\n\n## 4. Method\n\nn = 50 [Lecture 27].\n\n"
              "## References\n\nA 2024\n")
    bad_tr = ("# T\n\n## 1. Problem\n\nBaşarı 97'den 49'a iniyor.\n\n## 2. Amaç\n\nx\n\n## 4. Yöntem\n\nn = 50 [Ders 27].\n\n"
              "## Kaynaklar\n\nA 2024\n")
    import tempfile, os
    want = {"'Research questions' section is missing", "'Contribution' section is missing", "'Schedule' section is missing",
            "'Risk' section is missing"}
    ok = True
    for label, bad, num in (("English", bad_en, "97"), ("Turkish", bad_tr, "97")):
        p = os.path.join(tempfile.mkdtemp(), "bad.md"); open(p, "w", encoding="utf-8").write(bad)
        n, rows, probs = lint(p, 650)
        good = want <= set(probs) and any("untagged" in x and num in x for x in probs) and not any("n = 50" in x for x in probs)
        ok = ok and good
        print(f"  {label} headings/tags: {len(probs)} problems caught", "OK" if good else "FAILED")
    print("selftest", "OK" if ok else "FAILED", "(expected per file: 4 missing sections + 1 untagged claim; English and Turkish forms)")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?"); ap.add_argument("--max-words", type=int, default=700)  # 650 for the Turkish original; the English text runs longer (680)
    ap.add_argument("--csv"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 2)
    n, rows, probs = lint(a.path, a.max_words)
    print(f"body words: {n} / {a.max_words}")
    print(f"numeric-claim units: {len(rows)}, tagged: {sum(r['tagged'] for r in rows)}")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["tagged", "tags", "text"]); w.writeheader(); w.writerows(rows)
    for p in probs:
        print("  -", p)
    print("RESULT:", "COMPLETE" if not probs else f"INCOMPLETE ({len(probs)})")
    sys.exit(0 if not probs else 1)
```

## plot_day29.py

```python
"""Lecture 29 — figures: 02 Gantt (T4 hours from Lecture 27 budget_lean.csv / budget.csv), 03 risk matrix.

  python plot_day29.py [output folder]
The week ranges in the Gantt are a PROPOSAL (thesis/proposal-v0.1.md §6); hours are the mid scenario of the Lecture 27 budget model.
The probability/impact scores in the risk matrix are this lecture's ASSESSMENT (not a measurement).
"""
import csv, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = sys.argv[1] if len(sys.argv) > 1 else "../../lectures/images/lecture-29"
D27 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "day27", "results")
Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
YD, BD, GD, RD = "#c79a00", "#3b6ea8", "#2e8b57", "#c0392b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def hours(fn):
    rows = list(csv.DictReader(open(os.path.join(D27, fn), encoding="utf-8")))
    h = lambda cells: sum(float(r["h_orta"]) for r in rows if r["cell"] in cells)
    return {"WP1": h({"P1", "P2", "K1", "K2"}), "WP2": h({"A1", "A2", "A3", "A4"}),
            "WP3": h({"B1", "B2", "B3", "B4", "B5"}), "WP4": h({"C1", "C2"})}


lean, full = hours("budget_lean.csv"), hours("budget.csv")
print("T4 hours (mid) lean:", {k: round(v, 1) for k, v in lean.items()}, "total", round(sum(lean.values()), 1))
print("T4 hours (mid) full:", {k: round(v, 1) for k, v in full.items()}, "total", round(sum(full.values()), 1))

# --- 02 Gantt
wp = [("WP1 pilot + calibration (P1, P2, K1, K2; K3 Mac)", 1, 2, B, BD, "WP1"),
      ("WP2 RQ-a core (A1–A4)", 2, 5, Y, YD, "WP2"),
      ("WP3 RQ-b (B1–B5)", 5, 7, Y, YD, "WP3"),
      ("WP4 RQ-c (C1, C2)", 7, 8, Y, YD, "WP4"),
      ("WP5 analysis (McNemar / TOST / κ)", 8, 10, G, GD, None),
      ("WP6 thesis writing", 9, 16, G, GD, None)]
fig, ax = plt.subplots(figsize=(13, 4.8))
for i, (lab, a, b, fc, ec, key) in enumerate(wp):
    yy = len(wp) - 1 - i
    ax.barh(yy, b - a + 1, left=a - 0.5, color=fc, edgecolor=ec, height=0.6)
    txt = lab if key is None else f"{lab} — {lean[key]:.0f} h lean / {full[key]:.0f} h full"
    ax.text(a - 0.4, yy + 0.38, txt, fontsize=8.5, va="bottom")
for wk, lab in [(5, "midterm: RQ-a table"), (16, "final: thesis draft")]:
    ax.axvline(wk + 0.5, color=RD, ls="--", lw=1); ax.text(wk + 0.55, len(wp) - 0.3, lab, color=RD, fontsize=8.5)
ax.set_yticks([]); ax.set_xticks(range(1, 17)); ax.set_xlim(0.4, 17.6); ax.set_ylim(-0.6, len(wp) + 0.2)
ax.set_xlabel("week (week 1 = 30 November 2026 — proposal)")
tl, tf = sum(lean.values()), sum(full.values())
ax.set_title(f"Lecture 29 — Schedule of the proposal: the experiment WPs take ≈ {tl:.0f} T4 h in the lean matrix (full core ≈ {tf:.0f}); "
             f"at Kaggle ~30 h/week that is ≈ {tl/30:.1f} weeks (full: {tf/30:.1f}) — hours from the Lecture 27 budget model (mid)", fontsize=9.5)
ax.grid(axis="x", alpha=0.25)
fig.tight_layout(); os.makedirs(OUT, exist_ok=True); fig.savefig(os.path.join(OUT, "02-gantt.png"), dpi=160); plt.close(fig)

# --- 03 risk matrix (probability × impact, 1–5, assessment)
risks = [("R1 call time slow\n(pessimistic 497 h)", 3, 4), ("R2 4-bit S-OV\ndoes not reach 84.7", 3, 3),
         ("R3 empty near the\nceiling on Spatial", 4, 3), ("R4 bddl_cf\ndoes not load", 2, 3),
         ("R5 no LoRA\napproval (S9)", 3, 2), ("R6 VLANeXt does not\nrun on the T4", 2, 5), ("R7 MT output cannot\nbe produced (HF)", 2, 2)]
fig, ax = plt.subplots(figsize=(9.5, 7))
for x in range(1, 6):
    for y in range(1, 6):
        s = x * y
        c = G if s <= 4 else (Y if s <= 9 else R)
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=c, alpha=0.55, lw=0))
off = {}
for lab, p, imp in risks:
    k = (p, imp); n = off.get(k, 0); off[k] = n + 1
    ax.plot(p, imp + 0.18 - 0.36 * n, "o", color="#333", ms=5)
    ax.text(p + 0.06, imp + 0.18 - 0.36 * n, lab, fontsize=7.8, va="center")
ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5); ax.set_xticks(range(1, 6)); ax.set_yticks(range(1, 6))
ax.set_xlabel("probability (1 low – 5 high)"); ax.set_ylabel("impact (1 small – 5 stops the thesis)")
ax.legend(handles=[Patch(color=G, alpha=0.55, label="monitor"), Patch(color=Y, alpha=0.55, label="have plan B ready"),
                   Patch(color=R, alpha=0.55, label="solve this first")], loc="lower right", fontsize=8.5)
ax.set_title("Lecture 29 — Risk matrix (scores are this lecture's assessment, not a measurement);\nthe plan B for each risk is in thesis/proposal-v0.1.md §7 and in §5 of the lecture", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "03-risk-matrix.png"), dpi=160); plt.close(fig)
print("written: 02-gantt.png, 03-risk-matrix.png →", OUT)
```
