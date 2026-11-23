"""Lecture 13 figures (matplotlib).

03-denormalize.png   : rescaling VLANeXt's [-1,1] output back to the LIBERO action range.
                       Numbers are from src/datasets/libero_act.py LIBERO_ACTION_STATS["legacy"] (real).
04-connection-language.png : VLANeXt paper Table 1, "VLM-Policy Connection" rows (real).
05-flow-matching-toy.png   : ILLUSTRATIVE 1-D view of the 5-step flow-matching Euler solver
                       (the sigma schedule is real: diffusers FlowMatchEulerDiscreteScheduler, shift=1).
Usage: python plot_day13.py [output_folder]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT.mkdir(parents=True, exist_ok=True)
YEL, BLU, GRN, RED = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
YEL_D, BLU_D, GRN_D, RED_D = "#b8860b", "#2f6fb3", "#2e8b57", "#c0392b"

# ---- libero_act.py LIBERO_ACTION_STATS["legacy"] (first 6 dims; gripper is separate) ----
STATS = {
    "mixed":   ([-0.9375, -0.9375, -0.9375, -0.24214286, -0.375, -0.36428571],
                [0.9375, 0.9375, 0.9375, 0.35571429, 0.375, 0.375]),
    "spatial": ([-0.9375, -0.9375, -0.9375, -0.18857142, -0.3675, -0.36],
                [0.9375, 0.9375, 0.9375, 0.19714285, 0.33642858, 0.375]),
}
DIMS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]


def denorm(raw, lo, hi):
    raw = np.asarray(raw, dtype=np.float32)
    out = raw.copy()
    out[:6] = (raw[:6] + 1) / 2 * (np.array(hi) - np.array(lo)) + np.array(lo)
    out[6] = 1.0 if raw[6] > 0 else -1.0
    return out


def fig03():
    lo, hi = STATS["mixed"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    x = np.arange(6)
    ax.bar(x - 0.18, [h - l for l, h in zip(lo, hi)], 0.36, bottom=lo, color=BLU, edgecolor="#333",
           label="legacy / mixed (the one the checkpoint uses)")
    lo_s, hi_s = STATS["spatial"]
    ax.bar(x + 0.18, [h - l for l, h in zip(lo_s, hi_s)], 0.36, bottom=lo_s, color=YEL, edgecolor="#333",
           label="legacy / spatial (for comparison)")
    for i in range(6):
        ax.text(i - 0.18, hi[i] + 0.02, f"{hi[i]:+.3f}", ha="center", fontsize=7.5)
        ax.text(i - 0.18, lo[i] - 0.07, f"{lo[i]:+.3f}", ha="center", fontsize=7.5)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xticks(x, DIMS)
    ax.set_ylabel("LIBERO action unit (OSC_POSE delta, [-1,1] scale)")
    ax.set_title("Model output [-1,1] → expanded to these ranges\n(action_min/max: libero_act.py, legacy)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(-1.15, 1.15)

    ax = axes[1]
    raw = np.array([0.5, -0.2, 0.0, 0.1, -1.0, 1.0, 0.03])
    real = denorm(raw, lo, hi)
    names = DIMS + ["gripper"]
    y = np.arange(7)[::-1]
    ax.barh(y + 0.18, raw, 0.36, color=YEL, edgecolor="#333", label="raw output")
    ax.barh(y - 0.18, real, 0.36, color=GRN, edgecolor="#333", label="denormalize")
    for yi, r, v in zip(y, raw, real):
        ax.text(1.12, yi, f"{r:+.2f} → {v:+.3f}", va="center", ha="left", fontsize=7.5)
    ax.set_yticks(y, names)
    ax.set_xlim(-1.25, 2.3)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_title("Example: 7-D raw output → real action\n(gripper 0.03 > 0 → +1 = close)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("_denormalize_action: first 6 dims linear, 7th dim thresholded", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "03-denormalize.png", dpi=170)
    plt.close(fig)


def fig04():
    # VLANeXt (Wu et al., ICML 2026) Table 1, "VLM-Policy Connection" rows
    cols = ["LIBERO Spatial\n(original)", "LIBERO-plus Spatial\nLanguage", "LIBERO-plus Spatial\nRobot", "LIBERO-plus Spatial\nOverall"]
    data = {
        "Loose (MetaQuery, no connector)": [90.0, 74.6, 42.0, 53.7],
        "Tight (layer by layer, π0 style)": [90.0, 81.0, 51.7, 55.4],
        "Soft (layer by layer + 16 meta queries)": [91.8, 89.2, 58.3, 56.2],
    }
    colors = [BLU, RED, YEL]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(cols))
    w = 0.26
    for i, (name, vals) in enumerate(data.items()):
        bars = ax.bar(x + (i - 1) * w, vals, w, color=colors[i], edgecolor="#333", label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(x, cols)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("VLM–policy connection type and robustness to language perturbation\n"
                 "VLANeXt Table 1 (Wu et al., 2026; ablation models, Qwen3-VL-2B, Spatial suite)", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "04-connection-language.png", dpi=170)
    plt.close(fig)


def fig05():
    # Real sigma schedule: linspace(1.0, 1/1000, 5) + [0]  (FlowMatchEulerDiscreteScheduler, shift=1)
    sigmas = np.append(np.linspace(1.0, 1 / 1000, 5), 0.0)
    rng = np.random.default_rng(0)
    x0 = 0.35                     # target action (illustrative)
    noise = rng.normal(0, 1, 4)   # 4 different initial noise samples
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for k, n in enumerate(noise):
        x = n
        xs = [x]
        for i in range(5):
            # Perfect-model assumption: velocity = noise - x0 (linear flow); in a real model it is predicted
            v = (x - x0) / max(sigmas[i], 1e-6)   # x_t = (1-s) x0 + s*noise ⇒ v = noise - x0 = (x_t - x0)/s
            x = x + (sigmas[i + 1] - sigmas[i]) * v
            xs.append(x)
        ax.plot(sigmas, xs, marker="o", ms=5, lw=1.8, color=YEL_D if k == 0 else "#999",
                label="Euler trajectory (illustrative)" if k == 0 else None)
    ax.axhline(x0, color=GRN_D, ls="--", lw=1.5, label="target action x₀ (illustrative)")
    ax.invert_xaxis()
    ax.set_xticks([1.0, 0.75025, 0.5005, 0.25075, 0.0], ["1.0", "0.750", "0.500", "0.251", "0.001 → 0"])
    ax.set_xlabel("sigma (1 = pure noise → 0 = clean action); 5 scheduler.step calls")
    ax.set_ylabel("a single action dimension, [-1,1] scale")
    ax.set_title("Flow-matching solver: the 5-step loop inside predict_action\n"
                 "(sigma schedule is real — diffusers FlowMatchEuler, shift=1; trajectory is ILLUSTRATIVE, perfect-model assumption)",
                 fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "05-flow-matching-toy.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig03(); fig04(); fig05()
    print("written:", *(sorted(p.name for p in OUT.glob("0[345]-*.png"))))
