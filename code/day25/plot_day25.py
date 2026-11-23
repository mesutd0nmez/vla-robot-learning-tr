"""Lecture 25 — figures 02, 03, 05, 06. Numbers: the Phase 3 lectures' own outputs (lecture number in the title),
paper tables (arXiv in the title) and results/smoke.csv in this folder (a real run). No representative data.

python plot_day25.py            # writes under lectures/images/lecture-25/
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as _beta, norm

HERE = Path(__file__).resolve().parent
OUT = HERE / ".." / ".." / "lectures" / "images" / "lecture-25"
OUT.mkdir(parents=True, exist_ok=True)
Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"       # yellow model/key, blue input, green output, red problem
YE, BE, GE, RE = "#b8860b", "#3b6fa0", "#2e8b57", "#c0392b"
GRAY = "#9a9a9a"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ---------------------------------------------------------------- 02: measurement status matrix (Phase 3 balance sheet)
# States: 2 = measured with real output in this environment, 1 = script ready, run is Mesut's, 0 = hypothesis / not measured
ROWS = [
    # (row, lecture, state)  — from the Phase 3 daily-log entries
    ("LIBERO-Plus perturbation counts (376 cameras, 500 Pandas, 300 _add_)", 17, 2),
    ("Language 50 rewrites: R1 26 / R2 22 / R0 2, 'between' 48/50", 17, 2),
    ("Turkish _language_trK BDDL generation (bddl 1.0.1, 10/10)", 17, 2),
    ("Four conditions × 10 init states success table (T4)", 18, 1),
    ("Wilson / exact power calculations (n=10, 20, 50)", 18, 2),
    ("No-command upper bound (BDDL): 9/10 · 10/10 · 1/10 · 9/10", 19, 2),
    ("Geometry of the 50 init states (.pruned_init): spread ≤ 3 cm", 19, 2),
    ("Pattern A/B/C/D reading — with the real table", 19, 1),
    ("datasets/04: 20 pairs × 5 surface forms, 90 referent cells, 5 CF BDDL", 20, 2),
    ("Native-speaker check: which bowl is 'soldaki', tr_a/tr_b naturalness", 20, 1),
    ("That the training images are also rotated by 180°", 20, 0),
    ("Webots: setup document, FK/IK, dry-run controller", 21, 2),
    ("Webots real render (does the launcher work on the M2)", 21, 1),
    ("Tokenization: 100 commands × 2 tokenizers (inflation, suffix, action range)", 22, 2),
    ("Qwen3-VL tokenizer = npm Qwen3 tokenizer.json (byte identity)", 22, 1),
    ("PI formula and width simulation", 23, 2),
    ("VLANeXt's run-to-run variability (seed/flow-matching)", 23, 0),
    ("Parser: in-set 79/80, held-out 5/12", 24, 2),
    ("MT round-trip (Opus-MT): exact match, latency", 24, 1),
    ("QLoRA T4 memory 7.7 GB / 65–98 hours", 24, 0),
    ("Where Turkish sits in Qwen3-VL (on the Chinese side or the Arabic side)", 24, 0),
]


def fig02():
    fig, ax = plt.subplots(figsize=(13, 8.2))
    cols = ["Real output in this environment", "Script ready, run is Mesut's", "Hypothesis / not measured"]
    colmap = {2: 0, 1: 1, 0: 2}
    colors = {0: G, 1: Y, 2: R}
    for i, (label, day, st) in enumerate(ROWS):
        y = len(ROWS) - 1 - i
        for j in range(3):
            ax.add_patch(plt.Rectangle((j, y), 1, 1, facecolor=colors[j] if colmap[st] == j else "#f7f7f7",
                                       edgecolor="white", linewidth=2))
        if colmap[st] == 0:
            ax.text(0.5, y + 0.5, "✓", ha="center", va="center", fontsize=12, color=GE, fontweight="bold")
        elif colmap[st] == 1:
            ax.text(1.5, y + 0.5, "→", ha="center", va="center", fontsize=12, color=YE, fontweight="bold")
        else:
            ax.text(2.5, y + 0.5, "?", ha="center", va="center", fontsize=12, color=RE, fontweight="bold")
        ax.text(-0.08, y + 0.5, f"D{day}  {label}", ha="right", va="center", fontsize=8.6)
    ax.set_xlim(0, 3); ax.set_ylim(0, len(ROWS))
    ax.set_xticks([0.5, 1.5, 2.5]); ax.set_xticklabels(cols, fontsize=9.5)
    ax.xaxis.tick_top(); ax.set_yticks([])
    for s in ("left", "bottom"): ax.spines[s].set_visible(False)
    n2 = sum(1 for r in ROWS if r[2] == 2); n1 = sum(1 for r in ROWS if r[2] == 1); n0 = sum(1 for r in ROWS if r[2] == 0)
    ax.set_title(f"Phase 3 balance sheet: {len(ROWS)} measurements — {n2} real, {n1} awaiting Mesut's run, {n0} hypothesis\n"
                 "Source: plan/daily-log.md Lecture 17–24 entries; 'real' = reproduced in this session with phase3_smoke.py (28/30 scripts)",
                 fontsize=10.5, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "02-measured-vs-pending.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 03: the numbers of Phase 3 (four small panels)
def fig03():
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    # (a) no-command upper bound vs the literature empty-command (Lecture 19 §2–3)
    a = axes[0, 0]
    suites = ["Spatial", "Object", "Goal", "Long"]
    bound = [90, 100, 10, 90]
    a.bar(range(4), bound, color=B, edgecolor=BE, width=0.55, label="BDDL upper bound (suite_structure.py)")
    lit = {  # Lecture 19 figure 02: Chen Ø (T6), Zhan blank (T9), Fang vision-only (T1)
        "OpenVLA-OFT": ([80.0, 91.5, 10.0, 76.5], [83.2, 98.2, 10.0, 85.0]),
        "π0.5": ([66.0, 64.5, 9.0, 70.0], [65.0, 51.8, 11.4, 72.0], [68.2, 56.6, 10.2, 71.6]),
        "Qwen3-VL + OFT": ([34.0, 42.5, 8.5, 45.0],),
    }
    mk = {"OpenVLA-OFT": ("o", YE), "π0.5": ("s", GE), "Qwen3-VL + OFT": ("^", RE)}
    for name, series in lit.items():
        arr = np.array(series)
        lo, hi, md = arr.min(0), arr.max(0), arr.mean(0)
        x = np.arange(4) + {"OpenVLA-OFT": -0.12, "π0.5": 0.0, "Qwen3-VL + OFT": 0.12}[name]
        a.errorbar(x, md, yerr=[md - lo, hi - md], fmt=mk[name][0], color=mk[name][1], ms=6, capsize=3, lw=1.2, label=f"{name} empty command")
    a.set_xticks(range(4)); a.set_xticklabels(suites); a.set_ylim(0, 105); a.set_ylabel("success rate (%)")
    a.legend(fontsize=7.5, loc="center", bbox_to_anchor=(0.66, 0.50), framealpha=0.95)
    a.set_title("(a) The no-command ceiling is fixed, its use depends on the model — Lecture 19\nChen 2606.15714 T6 · Zhan 2601.04052 T9 · Fang 2602.17659 T1", fontsize=9.5)
    # (b) tokenization (Lecture 22, real)
    b = axes[0, 1]
    surf = ["en", "tr_a", "tr_ascii", "tr_bare"]
    L = [13.6, 22.4, 17.6, 14.9]; Q = [12.3, 16.9, 16.0, 12.2]
    x = np.arange(4); w = 0.36
    b.bar(x - w / 2, L, w, color=Y, edgecolor=YE, label="LLaMA-2 (OpenVLA)")
    b.bar(x + w / 2, Q, w, color=B, edgecolor=BE, label="Qwen3 (VLANeXt)")
    for i in range(4):
        b.text(x[i] - w / 2, L[i] + 0.3, f"{L[i]:.1f}", ha="center", fontsize=8)
        b.text(x[i] + w / 2, Q[i] + 0.3, f"{Q[i]:.1f}", ha="center", fontsize=8)
    b.set_xticks(x); b.set_xticklabels(surf); b.set_ylabel("tokens / command"); b.set_ylim(0, 26)
    b.legend(fontsize=8, loc="upper right")
    b.set_title("(b) The same 100 commands, two tokenizers — Lecture 22 (tokens.csv, reproduced exactly in this session)\n"
                "tr_a/en inflation 1.63× vs 1.41×; cost of Turkish letters (tr_a − tr_ascii) 4.9 vs 0.9 tokens", fontsize=9.5)
    # (c) Chen Table 1: LLaMA-2 OFT vs Qwen3-VL OFT vs Qwen3-VL GR00T
    c = axes[1, 0]
    langs = ["zh", "fr", "ru", "ar", "∅"]
    rowsC = {"OpenVLA-OFT (LLaMA-2)": ([-28.6, -28.6, -27.8, -29.8, -27.5], Y, YE),
             "Qwen3-VL + OFT head": ([-7.3, -39.1, -48.8, -47.1, -63.1], R, RE),
             "Qwen3-VL + GR00T head": ([-6.1, -33.0, -38.8, -36.0, -48.0], B, BE)}
    x = np.arange(5); w = 0.26
    for k, (name, (v, col, ed)) in enumerate(rowsC.items()):
        c.bar(x + (k - 1) * w, v, w, color=col, edgecolor=ed, label=name)
    c.axhline(0, color="#333", lw=0.8); c.set_xticks(x); c.set_xticklabels(langs)
    c.set_ylabel("difference vs EN (points)"); c.legend(fontsize=8, loc="lower left")
    c.set_title("(c) A 'multilingual backbone' is good only in Chinese — Lecture 24\nChen et al. 2606.15714 Table 1, LIBERO avg.; Turkish not measured (S1 measures it)", fontsize=9.5)
    # (d) Wilson intervals n=10 vs n=50 (Lecture 18)
    d = axes[1, 1]
    cases = [("9/10", 9, 10), ("5/10", 5, 10), ("45/50", 45, 50), ("25/50", 25, 50)]
    for i, (lab, k, n) in enumerate(cases):
        lo, hi = wilson(k, n)
        col = Y if n == 10 else G; ed = YE if n == 10 else GE
        d.barh(i, (hi - lo) * 100, left=lo * 100, color=col, edgecolor=ed, height=0.5)
        d.plot([k / n * 100], [i], "o", color="#333", ms=5)
        d.text(hi * 100 + 1, i, f"{lo*100:.0f}–{hi*100:.0f}", va="center", fontsize=8.5)
    d.set_yticks(range(4)); d.set_yticklabels([c[0] for c in cases]); d.set_xlim(0, 115)
    d.set_xlabel("success rate (%) — Wilson 95%")
    d.set_title("(d) 10 episodes give a direction, 50 episodes give a difference — Lecture 18 §5\nthe 9/10 and 5/10 intervals overlap; 45/50 and 25/50 separate", fontsize=9.5)
    fig.suptitle("The numbers of Phase 3 — all four real (lecture number and source in the panel title)", fontsize=11.5)
    fig.tight_layout(); fig.savefig(OUT / "03-phase3-numbers.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 05: dry-run results (results/smoke.csv, real)
def fig05():
    rows = list(csv.DictReader(open(HERE / "results" / "smoke.csv", encoding="utf-8")))
    days = sorted({int(r["ders"]) for r in rows})
    ok = [sum(1 for r in rows if int(r["ders"]) == d and r["sonuc"] == "OK") for d in days]
    # results may come from an older Turkish run (HATA / ATLANDI) or a new one (FAILED / SKIPPED)
    er = [sum(1 for r in rows if int(r["ders"]) == d and r["sonuc"] in ("HATA", "FAILED")) for d in days]
    sk = [sum(1 for r in rows if int(r["ders"]) == d and r["sonuc"] in ("ATLANDI", "SKIPPED")) for d in days]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(days))
    ax.bar(x, ok, color=G, edgecolor=GE, label="OK (ran, exit 0)")
    ax.bar(x, er, bottom=ok, color=R, edgecolor=RE, label="FAILED")
    ax.bar(x, sk, bottom=np.array(ok) + np.array(er), color="#eeeeee", edgecolor=GRAY, label="SKIPPED (no binary/model)")
    for i, d in enumerate(days):
        names = [r["script"].split()[0] for r in rows if int(r["ders"]) == d and r["sonuc"] in ("ATLANDI", "SKIPPED")]
        if names:
            ax.text(x[i], ok[i] + er[i] + sk[i] + 0.15, "\n".join(names), ha="center", va="bottom", fontsize=7.5, color="#555")
        ax.text(x[i], 0.2, f"{ok[i]}/{ok[i]+er[i]+sk[i]}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"Lecture {d}" for d in days]); ax.set_ylabel("number of scripts / checks")
    ax.set_ylim(0, max(np.array(ok) + np.array(er) + np.array(sk)) + 2.2)
    ax.legend(fontsize=8.5, loc="upper right")
    t_ok, t_all = sum(ok), len(rows)
    ax.set_title(f"Phase 3 tool inventory in a dry run: {t_ok}/{t_all} OK — phase3_smoke.py, this session (no model, no GPU)\n"
                 "Inputs: LIBERO + LIBERO-plus clones (--depth 1), Meta tokenizer.model (sha256 OK), npm Qwen3 tokenizer.json, bddl 1.0.1",
                 fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "05-smoke.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 06: LangGap (Hou & Zhao 2026, arXiv:2603.00592) — π0.5, same scene, only the command changes
def fig06():
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
    suites = ["Spatial", "Goal", "Object"]
    orig = [97.0, 97.0, 100.0]; ext = [5.9, 30.0, 37.7]           # Table VII
    x = np.arange(3); w = 0.36
    a.bar(x - w / 2, orig, w, color=G, edgecolor=GE, label="Original 40 tasks")
    a.bar(x + w / 2, ext, w, color=R, edgecolor=RE, label="Extended 59 tasks (same scene, command changed)")
    for i in range(3):
        a.text(x[i] - w / 2, orig[i] + 1, f"{orig[i]:.1f}", ha="center", fontsize=8.5)
        a.text(x[i] + w / 2, ext[i] + 1, f"{ext[i]:.1f}", ha="center", fontsize=8.5)
    a.set_xticks(x); a.set_xticklabels(suites); a.set_ylim(0, 125); a.set_ylabel("success rate (%) — π0.5")
    a.legend(fontsize=8, loc="upper center", ncol=2, framealpha=0.95)
    a.set_title("Table VII — per suite: Spatial 97.0 → 5.9", fontsize=10)
    dims = ["Object\nchanged", "Goal\nchanged", "Spatial\ndescription changed", "Drawer\naction"]
    vals = [29.3, 0.0, 11.0, 31.7]; ns = ["38 tasks / 760 ep", "13 / 260", "5 / 100", "3 / 60"]
    cols = [B, R, Y, B]; eds = [BE, RE, YE, BE]
    b.bar(range(4), vals, color=cols, edgecolor=eds, width=0.6)
    for i in range(4):
        b.text(i, vals[i] + 1, f"{vals[i]:.1f}", ha="center", fontsize=9)
        b.text(i, -5.5, ns[i], ha="center", fontsize=7.5, color="#555")
    b.set_xticks(range(4)); b.set_xticklabels(dims, fontsize=8.5); b.set_ylim(-8, 45); b.axhline(0, color="#333", lw=0.8)
    b.set_ylabel("success rate (%) — π0.5, extended tasks")
    b.set_title("Table VIII — which component change makes it drop; 'Spatial description' is 5 tasks × 20 episodes", fontsize=10)
    fig.suptitle("LangGap (Hou & Zhao, Feb 2026, arXiv:2603.00592): when one component of the command changes in the same scene, π0.5 drops 93.8 → 21.4\n"
                 "English only, no Turkish; the 'Spatial Description' cell is the English counterpart of block B in datasets/04", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "06-langgap.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig02(); fig03(); fig05(); fig06()
    print("written:", OUT.resolve())
