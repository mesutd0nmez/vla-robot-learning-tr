"""Lecture 24 — figures 02–06. All numbers come from the tables of the papers (source in the titles) or from real outputs in this folder;
the only estimate is the QLoRA memory bar in figure 06 (explicitly labeled 'estimate').

  python plot_day24.py            → ../../lectures/images/lecture-24/0X-*.png
"""
import csv, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / ".." / ".." / "lectures" / "images" / "lecture-24"
OUT.mkdir(parents=True, exist_ok=True)
Y, B, G, R, GR = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6", "#bdbdbd"
YD, BD, GD, RD = "#c9961a", "#4a86c8", "#3f9d5a", "#d45c5c"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------- 02: Dong et al. 2606.11906 Table 2 — inference-time interventions
LANGS = ["ZH", "JA", "KO", "FR", "ES", "AR", "TH", "PT", "VI"]
DONG = {
    "OpenVLA-OFT": {"EN": 97.1,
                    "baseline": [51.7, 59.4, 52.3, 65.3, 64.1, 50.8, 53.2, 63.6, 59.6],
                    "EN-CoT prompt": [56.8, 54.8, 52.0, 62.4, 63.4, 51.1, 52.8, 62.3, 55.1],
                    "average shift": [60.8, 58.9, 54.7, 65.5, 64.5, 53.8, 54.5, 64.2, 57.1],
                    "step-wise shift": [68.8, 68.8, 66.4, 70.9, 69.8, 64.5, 62.6, 70.7, 68.0]},
    "π0.5": {"EN": 96.9,
             "baseline": [56.7, 56.5, 55.4, 61.2, 59.1, 55.7, 56.0, 59.4, 57.3],
             "EN-CoT prompt": [55.2, 55.7, 55.7, 59.9, 57.9, 56.4, 56.5, 57.2, 56.3],
             "average shift": [54.1, 53.9, 52.9, 57.2, 56.2, 53.0, 52.6, 55.6, 54.2],
             "step-wise shift": [80.3, 80.9, 79.8, 82.0, 81.1, 81.8, 79.9, 82.1, 81.9]},
}


def fig02():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    cols = {"baseline": GR, "EN-CoT prompt": B, "average shift": Y, "step-wise shift": G}
    x = np.arange(len(LANGS)); w = 0.2
    for ax, (model, d) in zip(axes, DONG.items()):
        for k, (name, c) in enumerate(cols.items()):
            ax.bar(x + (k - 1.5) * w, d[name], w, color=c, edgecolor="#555", linewidth=0.5, label=name)
        ax.axhline(d["EN"], color=RD, ls="--", lw=1.2)
        ax.text(len(LANGS) - 0.5, d["EN"] + 1, f"EN {d['EN']}", color=RD, ha="right", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(LANGS); ax.set_ylim(0, 105)
        ax.set_title(f"{model} — LIBERO 4-suite average (%)", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("success rate (%)")
    axes[1].legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.95)
    fig.suptitle("Training-free interventions: the 'think in English' prompt does not help, the hidden-state shift does\n"
                 "Dong et al. 2026 (arXiv:2606.11906) Table 2 — 9 languages via Google Translate, no Turkish", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "02-inference-time.png", dpi=170); plt.close(fig)


# ---------------------------------------------------------------- 03: training side — Chen Table 5 + Dong Table 4
CHEN5 = {  # Qwen3-VL + GR00T head, LIBERO 4-suite average, en/zh/fr/ru/ar
    "GR00T (no extra training)": [95.3, 89.1, 62.3, 56.5, 59.3],
    "E-FT (EN-only VLM FT)": [96.3, 84.3, 55.0, 48.5, 52.8],
    "M-FT (multilingual VLM FT)": [95.4, 87.6, 65.1, 57.1, 62.8],
    "E-CT (EN co-training)": [94.0, 86.9, 68.3, 54.5, 57.4],
    "M-CT (multilingual co-training)": [92.6, 88.3, 87.4, 83.1, 89.4],
    "MPCA (M-CT + alignment)": [95.3, 90.6, 89.5, 85.5, 90.8],
}
DONG4 = {"baseline (ZH)": [55.8, 76.0, 9.8, 65.0], "full fine-tune": [94.0, 97.8, 76.4, 74.0],
         "random 50% of steps": [88.8, 90.8, 0.0, 0.0], "critical 50% of steps": [92.6, 96.0, 72.0, 81.2]}


def fig03():
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5), gridspec_kw={"width_ratios": [1.25, 1]})
    M = np.array(list(CHEN5.values()))
    im = a.imshow(M, cmap="YlGn", vmin=40, vmax=100, aspect="auto")
    a.set_xticks(range(5)); a.set_xticklabels(["en", "zh", "fr", "ru", "ar"])
    a.set_yticks(range(len(CHEN5))); a.set_yticklabels(list(CHEN5.keys()), fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            a.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=9, color="black" if M[i, j] > 60 else "#444")
    a.set_title("Training with multilingual VLM data (50k translated VQA, 4 languages)\nChen et al. 2026 (arXiv:2606.15714) Table 5 — Qwen3-VL + GR00T head, LIBERO avg. (%)", fontsize=10)
    a.spines[:].set_visible(False)
    suites = ["Spatial", "Object", "Goal", "Long"]; x = np.arange(4); w = 0.2
    for k, (name, v) in enumerate(DONG4.items()):
        b.bar(x + (k - 1.5) * w, v, w, color=[GR, Y, R, G][k], edgecolor="#555", linewidth=0.5, label=name)
    b.set_xticks(x); b.set_xticklabels(suites); b.set_ylim(0, 125); b.grid(axis="y", alpha=0.3)
    b.set_title("Fine-tuning with commands in the target language (ZH) — OpenVLA-OFT\nDong et al. 2026 Table 4; 'critical step' = a step with a high language gradient", fontsize=10)
    b.legend(fontsize=8, loc="upper center", ncol=2)
    b.set_ylabel("success rate (%)")
    fig.suptitle("Trained solutions: the gap closes when the data is in the target language; English-only extra training (E-FT) WIDENS the gap", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "03-training-side.png", dpi=170); plt.close(fig)


# ---------------------------------------------------------------- 04: backbone comparison — Chen Table 1 (difference from EN, points)
CHEN1 = {
    "OpenVLA-OFT (LLaMA-2 / Prismatic)": [-28.6, -28.6, -27.8, -29.8, -27.5],
    "π0.5 (PaliGemma)": [-39.6, -38.0, -41.8, -41.4, -43.5],
    "Qwen2.5-VL + OFT head": [-17.8, -32.1, -45.3, -59.1, -43.3],
    "Qwen3-VL + OFT head": [-7.3, -39.1, -48.8, -47.1, -63.1],
    "Qwen3-VL + FAST head": [-29.6, -70.8, -76.4, -73.6, -70.5],
    "Qwen3-VL + GR00T head": [-6.1, -33.0, -38.8, -36.0, -48.0],
    "Qwen3-VL + π head": [-12.1, -40.9, -44.6, -46.8, -55.4],
}


def fig04():
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    langs = ["zh", "fr", "ru", "ar", "∅ (empty command)"]
    x = np.arange(len(CHEN1)); w = 0.15
    cols = [YD, BD, GD, RD, "#777"]
    for j, (lg, c) in enumerate(zip(langs, cols)):
        vals = [v[j] for v in CHEN1.values()]
        ax.bar(x + (j - 2) * w, vals, w, color=c, label=lg, edgecolor="none")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(list(CHEN1.keys()), rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("success rate difference vs EN (points)")
    ax.set_ylim(-85, 5); ax.grid(axis="y", alpha=0.3)
    ax.legend(title="command language", fontsize=9, loc="lower left", ncol=5)
    ax.set_title("A 'multilingual backbone' alone is not a solution: Qwen3-VL is good only in Chinese; in French/Russian/Arabic it is WORSE than the LLaMA-2-based OFT\n"
                 "Chen et al. 2026 (arXiv:2606.15714) Table 1 — LIBERO 4-suite average; the backbone and the action head change together (not a clean comparison)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "04-backbones.png", dpi=170); plt.close(fig)


# ---------------------------------------------------------------- 05: rule-based parser — real output
def fig05():
    here = Path(__file__).resolve().parent
    rows = list(csv.DictReader(open(here / "parse_roundtrip.csv", encoding="utf-8")))
    surf = ["tr_a", "tr_ascii", "tr_bare", "tr_b"]
    ok = [sum(1 for r in rows if r["variant"] == s and r["result"] == "ok") for s in surf]
    n = [sum(1 for r in rows if r["variant"] == s) for s in surf]
    # held-out: not from the tr_parse.run_heldout output (parse_out.txt), recomputed
    sys.path.insert(0, str(here))
    import tr_parse
    hrows = list(csv.DictReader(open(here / "heldout_tr.csv", encoding="utf-8")))
    hok = sum(1 for r in hrows if tr_parse.render(tr_parse.parse(r["tr"])) == r["en_gold"])
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    labels = surf + ["held-out\n(12 new sentences)"]
    vals = [100 * o / m for o, m in zip(ok, n)] + [100 * hok / len(hrows)]
    cols = [G, G, G, Y, R]
    bars = ax.bar(labels, vals, color=cols, edgecolor="#555", linewidth=0.6)
    for b_, o, m in zip(bars, ok + [hok], n + [len(hrows)]):
        ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height() + 2, f"{o}/{m}", ha="center", fontsize=10)
    ax.set_ylim(0, 122); ax.set_ylabel("exact match with the EN template (%)")
    ax.axvline(3.5, color="#999", ls=":", lw=1)
    ax.text(1.5, 114, "the parser was written LOOKING AT these 80 rows (upper bound)", ha="center", fontsize=9, color="#555")
    ax.text(4, 114, "written afterwards", ha="center", fontsize=9, color="#555")
    ax.set_title("The cheapest form of solution 4: a 25-word lexicon + suffix stripping + 3 templates (tr_parse.py, real output)\n"
                 "99% inside the set, 42% outside it — the real face of a rule-based translation layer", fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "05-parser.png", dpi=170); plt.close(fig)


# ---------------------------------------------------------------- 06: memory budget — real values + one estimate
def fig06():
    items = [
        ("OpenVLA inference\nbf16", 16.8, B, "OpenVLA Tablo 2"),
        ("OpenVLA inference\nint8", 10.2, B, "OpenVLA Tablo 2"),
        ("OpenVLA inference\nint4", 7.0, B, "OpenVLA Tablo 2"),
        ("LoRA r=32 training\nbatch 16", 72, Y, "finetune.py / README"),
        ("LoRA r=32 training\nsmallest batch", 27, Y, "README: '≥ ~27 GB'"),
        ("LoRA + 4-bit\nbatch 1 (ESTIMATE)", 7.7, R, "qlora_budget.py"),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(items))
    ax.bar(x, [v for _, v, _, _ in items], color=[c for _, _, c, _ in items], edgecolor="#555", linewidth=0.6)
    for i, (_, v, _, src) in enumerate(items):
        ax.text(i, v + 1.5, f"{v:g} GB", ha="center", fontsize=10)
        ax.text(i, -9, src, ha="center", fontsize=7.5, color="#666")
    for yv, lbl, c in [(16, "Colab/Kaggle T4 — 16 GB", RD), (32, "MacBook Pro M2 — 32 GB unified (MPS; no bitsandbytes)", BD), (80, "A100 — 80 GB", GD)]:
        ax.axhline(yv, color=c, ls="--", lw=1.1); ax.text(len(items) - 0.5, yv + 1, lbl, ha="right", color=c, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([n for n, _, _, _ in items], fontsize=9)
    ax.set_ylim(-14, 88); ax.set_ylabel("GPU memory (GB)")
    ax.set_title("Does a Turkish LoRA fine-tune fit Mesut's hardware? — OpenVLA-7B memory map\n"
                 "Blue: inference (OpenVLA paper Table 2); yellow: LoRA training (openvla repo c8f03f4); red: 4-bit QLoRA estimate (not measured)", fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "06-memory.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    fig02(); fig03(); fig04(); fig05(); fig06()
    print("ok →", OUT.resolve())
