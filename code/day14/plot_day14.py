"""Lecture 14 figures: 02 token→action mapping (real libero_spatial q01/q99), 03 sequence layout + EN/TR token
counts (LLaMA-2 tokenizer, counted with llama-tokenizer-js), 04 M2 memory budget (parameter counts approximate).
    python plot_day14.py --out ../../lectures/images/lecture-14
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
DIMS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]
Q01 = np.array([-0.7454732114076613, -0.6616071462631226, -0.9375,
                -0.1071428582072258, -0.20678570866584778, -0.1842857152223587])
Q99 = np.array([0.9375, 0.8758928775787354, 0.9321428537368774,
                0.1039285734295845, 0.17678570747375488, 0.14571428298950195])
BINS = np.linspace(-1, 1, 256)
CENTERS = (BINS[:-1] + BINS[1:]) / 2


def fig02(out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    b = np.arange(255)
    for i, c in zip([0, 2, 3], ["#e07a00", "#1f77b4", "#2ca02c"]):
        act = 0.5 * (CENTERS + 1) * (Q99[i] - Q01[i]) + Q01[i]
        ax1.plot(b, act, lw=2.2, color=c, label=f"{DIMS[i]}  (q01={Q01[i]:.3f}, q99={Q99[i]:.3f})")
    ax1.axvline(127, color="grey", ls="--", lw=1)
    ax1.text(129, -0.85, "bin 127 → 0.0 in [-1,1]\n(token 31872)", fontsize=9, color="grey")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_xlabel("bin index (= 32000 − token id − 1)")
    ax1.set_ylabel("action (LIBERO unit, normalized delta)")
    ax1.set_title("Token → bin → action: linear but NOT symmetric", fontsize=11)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(alpha=0.3)

    x = np.arange(6)
    ax2.bar(x, Q99, color=G, edgecolor="k", label="q99 (value of bin 254)")
    ax2.bar(x, Q01, color=B, edgecolor="k", label="q01 (value of bin 0)")
    zero_act = 0.5 * (0 + 1) * (Q99 - Q01) + Q01
    ax2.scatter(x, zero_act, color="red", zorder=5, s=45, label="\"neutral\" token 31872 produces this action")
    for i in range(6):
        ax2.text(i, zero_act[i] + 0.04, f"{zero_act[i]:+.3f}", ha="center", fontsize=8, color="red")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xticks(x, DIMS)
    ax2.set_title("libero_spatial q01/q99 (dataset_statistics.json, real)", fontsize=11)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(alpha=0.3, axis="y")
    fig.suptitle("OpenVLA action decode — openvla/openvla-7b-finetuned-libero-spatial", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "02-token-to-action.png"), dpi=160)
    plt.close(fig)


def fig03(out):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.2), gridspec_kw={"height_ratios": [1, 1.6]})
    # top: sequence layout (real numbers: BOS 1, patch 256, EN prompt 33, '▁' 1, action 7)
    segs = [("BOS", 1, B), ("256 image patch tokens (DINOv2+SigLIP → projector)", 256, Y),
            ("prompt 33", 33, B), ("▁", 1, R), ("action", 7, G)]
    pos = 0
    for name, n, c in segs:
        w = max(n, 12)
        ax1.barh(0, w, left=pos, color=c, edgecolor="k", height=0.6)
        ax1.text(pos + w / 2, 0, name if n > 12 else f"{name}\n{n}", ha="center", va="center", fontsize=9)
        pos += w
    ax1.set_xlim(0, pos)
    ax1.set_yticks([])
    ax1.set_xlabel("position (LLaMA-2 input sequence: 1 + 256 + 33 + 1 = 291 tokens, then 7 tokens are generated)")
    ax1.set_title("OpenVLA sequence for Spatial task 0 — 'pick up the black bowl between the plate and the ramekin "
                  "and place it on the plate'", fontsize=10)
    # bottom: EN vs TR token counts (llama-tokenizer-js, LLaMA-2 BPE; excluding BOS)
    labels = ["EN command", "TR command", "EN full prompt", "TR full prompt"]
    counts = [20, 32, 33, 45]
    colors = [B, R, B, R]
    ax2.barh(labels[::-1], counts[::-1], color=colors[::-1], edgecolor="k")
    for i, v in enumerate(counts[::-1]):
        ax2.text(v + 0.5, i, str(v), va="center", fontsize=10)
    ax2.set_xlim(0, 60)
    ax2.set_xlabel("token count (LLaMA-2 tokenizer, excluding BOS)")
    ax2.set_title("The same command in Turkish: 20 → 32 tokens (×1.6). "
                  "'arasındaki' = ▁ar|as|ı|nd|aki (5), 'üstüne' = ▁ü|st|ü|ne (4), 'tabağın' = ▁tab|a|ğ|ı|n (5)",
                  fontsize=10)
    ax2.text(35.5, 2.0, "TR: tabak ile ramekinin arasındaki siyah kaseyi al ve tabağın üstüne koy", fontsize=8.5,
             va="center")
    ax2.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "03-sequence-tokens.png"), dpi=160)
    plt.close(fig)


def fig04(out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    names = ["LLaMA-2 7B", "SigLIP So400m/14", "DINOv2 ViT-L/14", "Projector MLP"]
    vals = [6.74, 0.43, 0.30, 0.07]
    ax1.barh(names[::-1], vals[::-1], color=Y, edgecolor="k")
    for i, v in enumerate(vals[::-1]):
        ax1.text(v + 0.08, i, f"{v:.2f} B", va="center", fontsize=10)
    ax1.set_xlim(0, 8)
    ax1.set_xlabel("parameters (billions) — total ≈ 7.5 B (HF card: \"8B params\"); component values approximate")
    ax1.set_title("OpenVLA-7B: where are the parameters? (90% in the language model)", fontsize=11)
    ax1.grid(alpha=0.3, axis="x")

    cats = ["bf16 weights (7.5 B × 2 bytes)", "KV cache + activations (298 tokens, b=1)", "macOS + Python + other"]
    vals2 = [15.1, 0.5, 4.0]
    cols = [Y, G, B]
    bottom = 0
    for c_, v, col in zip(cats, vals2, cols):
        ax2.bar(0, v, bottom=bottom, color=col, edgecolor="k", width=0.5)
        ax2.annotate(f"{c_}: ≈ {v} GB", xy=(0.25, bottom + v / 2), xytext=(0.45, bottom + v / 2),
                     fontsize=8.5, va="center", arrowprops=dict(arrowstyle="-", color="grey"))
        bottom += v
    ax2.axhline(32, color="k", lw=1.2)
    ax2.text(-0.2, 32.6, "M2 32 GB unified memory", fontsize=9)
    ax2.axhline(21.8, color="red", ls="--", lw=1.2)
    ax2.text(-0.2, 23.0, "Metal recommendedMaxWorkingSetSize ≈ 21.8 GB (estimate);\n"
             "PyTorch MPS ceiling = 1.7 × this (PYTORCH_MPS_HIGH_WATERMARK_RATIO)", fontsize=8, color="red")
    ax2.set_ylim(0, 36)
    ax2.set_xlim(-0.3, 2.6)
    ax2.set_xticks([])
    ax2.set_ylabel("GB")
    ax2.set_title("Memory budget on the Mac (estimate; measure with Activity Monitor)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "04-memory-budget.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig02(a.out); fig03(a.out); fig04(a.out)
    print("written:", a.out)
