"""Phase 1 (Lectures 1-8) matplotlib figures, English versions.

Recreates the figures that have no .mmd source. One function per figure;
`python tools/figures_phase1.py` writes all of them (dpi 160) into
lectures/images/lecture-0N/. Data values are copied from the lectures / sources
cited there; figures marked "illustrative" use synthetic data.
Colors: yellow #ffd166 (key idea), blue #cfe8ff (input), green #d4f5d4 (output), red #ffd6d6 (problem).
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle

ROOT = Path(__file__).resolve().parent.parent / "lectures" / "images"
DPI = 160
GREEN, AMBER, RED, BLUE, GRAY = "#6cc16c", "#f4b942", "#e06666", "#5b8fc9", "#bbbbbb"

plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"],
    "axes.grid": False, "figure.dpi": DPI, "savefig.dpi": DPI,
    "axes.unicode_minus": True,
})


def _save(fig, lecture, name):
    out = ROOT / f"lecture-{lecture:02d}" / name
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print("wrote", out)


def _grid(ax, axis="y"):
    ax.grid(axis=axis, alpha=0.3)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- Lecture 1
def fig_action_trajectory():
    rng = np.random.default_rng(0)
    T = 120
    t = np.arange(T)
    dx = np.where(t < 40, 0.02, np.where(t < 70, 0.0, -0.015))
    dy = np.where(t < 40, -0.01, np.where(t < 70, 0.0, 0.012))
    dz = np.select([t < 30, t < 40, t < 54, t < 70, t < 95],
                   [-0.01, 0.0, 0.019, 0.0, 0.0], -0.01)
    dz = np.where((t >= 40) & (t < 54), 0.019, dz)
    dz = np.where(t == 40, 0.021, dz)
    dxn, dyn, dzn = (a + rng.normal(0, 0.002, T) for a in (dx, dy, dz))
    grip = ((t >= 38) & (t < 101)).astype(int)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1]}, constrained_layout=True)
    a1.plot(t, dxn, label="Δx"); a1.plot(t, dyn, label="Δy"); a1.plot(t, dzn, label="Δz")
    for x, lab in [(0, "approach object"), (40, "grasp"), (70, "carry"), (100, "release")]:
        a1.axvline(x, color="gray", ls="--", alpha=0.5)
        a1.text(x + 1, 0.0235, lab, color="gray", fontsize=9, va="center")
    a1.set_ylabel("position change (m)")
    a1.legend(loc="upper right"); _grid(a1)
    a1.set_title('Example episode: "kırmızı bardağı tabağa koy" ("put the red mug on the plate")'
                 ' — 7-DoF action sequence (synthetic, illustrative)', fontsize=10.5)
    a2.step(t, grip, where="post", color="#d62728")
    a2.set_yticks([0, 1]); a2.set_yticklabels(["open", "closed"])
    a2.set_ylabel("gripper"); a2.set_xlabel("time step t"); _grid(a2, "both")
    _save(fig, 1, "06-action-trajectory.png")


# ---------------------------------------------------------------- Lecture 2
def fig_action_binning():
    lo, hi = -0.0655, 0.0635
    edges = np.linspace(-0.08, 0.08, 113)
    x = (edges[:-1] + edges[1:]) / 2
    counts = 10 + 420 * np.exp(-(x - 0.001) ** 2 / (2 * 0.0095 ** 2)) \
        + 25 * np.exp(-(x + 0.03) ** 2 / (2 * 0.004 ** 2))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    a1.bar(x, counts, width=edges[1] - edges[0], color="#cfe8ff", edgecolor=BLUE, linewidth=0.8)
    for v, lab, ha in [(lo, "1st percentile", "left"), (hi, "99th percentile", "right")]:
        a1.axvline(v, color="red", ls="--")
        a1.text(v + (0.0005 if ha == "left" else -0.0005), 395, lab, color="red", ha=ha, fontsize=10)
    a1.set_title("Δx distribution in the training data (illustrative)")
    a1.set_xlabel("Δx (m)"); _grid(a1, "both")
    xs = np.linspace(lo, hi, 256)
    a2.plot(xs, np.arange(256), color="#333333", lw=1.8)
    a2.plot([0.0137], [157], "o", color="#ffd166", mec="black", ms=9, zorder=5)
    a2.annotate("Δx = 0.0137 m  →  bin 157\n→ LLaMA token id 31901", xy=(0.0137, 157),
                xytext=(-0.06, 200), arrowprops=dict(arrowstyle="->", color="black"), fontsize=10)
    a2.set_title("256 equal-width bins: continuous value → token")
    a2.set_xlabel("Δx (m)"); a2.set_ylabel("bin index (0–255)"); _grid(a2, "both")
    _save(fig, 2, "04-action-binning.png")


# ---------------------------------------------------------------- Lecture 3
def fig_oft_under_perturbation():
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    labels = ["No perturbation", "Camera viewpoint", "Robot initial\nstate"]
    vals = [97.1, 59.7, 37.2]
    ax.bar(labels, vals, color=[GREEN, AMBER, RED])
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v}%", ha="center", fontsize=11)
    ax.set_ylim(0, 110); ax.set_ylabel("LIBERO success rate (%)")
    ax.set_title("OpenVLA-OFT under LIBERO-Plus (Fei et al. 2025)"); _grid(ax)
    _save(fig, 3, "03-oft-under-perturbation.png")


def fig_libero_leaderboard():
    names = ["RIPT-VLA", "OpenVLA-OFT", "UniVLA", "π0", "Nora", "π0-fast", "WorldVLA", "OpenVLA"]
    vals = [97.5, 97.1, 95.2, 94.2, 87.9, 85.5, 79.1, 76.5]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    y = np.arange(len(names))
    ax.barh(y, vals, color=BLUE)
    ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 0.8, i, f"{v}%", va="center", fontsize=11)
    ax.set_xlim(0, 110); _grid(ax, "x")
    ax.set_xlabel("LIBERO 4-suite mean success (%)")
    ax.set_title("Standard LIBERO: everyone at 95%+ — is the benchmark saturated?")
    _save(fig, 3, "04-libero-leaderboard.png")


# ---------------------------------------------------------------- Lecture 4
def fig_patterns():
    labels = ["correct", "empty", "wrong", "paraphrase", "Turkish"]
    cols = [GREEN, GRAY, RED, BLUE, AMBER]
    A = [95, 92, 10, 94, 90]
    B = [95, 20, 8, 90, 25]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True, constrained_layout=True)
    a1.bar(labels, A, color=cols)
    a1.set_title("Pattern A: scene-memorizer model\n(empty ≈ correct, Turkish ≈ correct → language is not read)", fontsize=10)
    a2.bar(labels, B, color=cols)
    a2.set_title("Pattern B: language-dependent model\n(empty collapses, Turkish ≈ empty → Turkish not understood)", fontsize=10)
    a1.set_ylabel("success rate (%) — hypothetical"); a1.set_ylim(0, 100)
    for a in (a1, a2):
        _grid(a)
    _save(fig, 4, "03-patterns.png")


# ---------------------------------------------------------------- Lecture 5
def fig_category_results():
    cats = ["Steering\n(n=15)", "Pick-place\n(n=20)", "Pushing\n(n=15)", "Overall\n(n=50)"]
    l2 = [0.404, 0.488, 0.614, 0.501]
    cos = [0.622, 0.244, 0.294, 0.372]
    x = np.arange(4); w = 0.4
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.bar(x - w / 2, l2, w, color=RED, label="L2 error")
    ax.bar(x + w / 2, cos, w, color=BLUE, label="cosine similarity")
    ax.set_xticks(x); ax.set_xticklabels(cats); _grid(ax)
    ax.legend(loc="upper right")
    ax.set_title("Tatar et al. 2026, Table I — EN vs TR action vectors\n"
                 "(ANOVA p=0.097: difference between categories not significant)", fontsize=11)
    _save(fig, 5, "02-category-results.png")


def fig_complexity():
    cats = ["Simple (18)", "Medium (16)", "Complex (16)"]
    fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
    ax.bar(cats, [0.551, 0.462, 0.482], color=AMBER)
    ax.set_ylim(0, 0.8); ax.set_ylabel("mean L2"); _grid(ax)
    ax.set_title("Command complexity ↔ error: ρ = 0.002 (no relationship)")
    _save(fig, 5, "03-complexity.png")


def fig_gripper_effect():
    dims = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]
    en = [0.02, -0.01, 0.002, 0.0, 0.0, 0.05, 1.0]
    tr = [0.014, -0.012, 0.001, 0.0, 0.008, 0.04, 0.0]
    x = np.arange(7); w = 0.4
    fig, ax = plt.subplots(figsize=(8, 3.4), constrained_layout=True)
    ax.bar(x - w / 2, en, w, color=GREEN, label="A_EN")
    ax.bar(x + w / 2, tr, w, color=AMBER, label="A_TR")
    ax.set_xticks(x); ax.set_xticklabels(dims); _grid(ax)
    ax.legend(loc="upper left")
    ax.set_title("Illustrative example: 6 dimensions almost identical, gripper differs → L2 = 1.00")
    _save(fig, 5, "04-gripper-effect.png")


# ---------------------------------------------------------------- Lecture 6
def fig_cdc():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    a1.set_xlim(0, 650); a1.set_ylim(0, 650); a1.set_aspect("equal")
    for v in (261, 391):
        a1.axvline(v, color="gray", ls="--", lw=1.8); a1.axhline(v, color="gray", ls="--", lw=1.8)
    cx = [130, 326, 520]; cy = [520, 325, 130]
    names = [["NW", "N", "NE"], ["W", "B/EQ", "E"], ["SW", "S", "SE"]]
    for r in range(3):
        for c in range(3):
            if names[r][c] == "B/EQ":
                a1.text(cx[c], cy[r] + 40, "B/EQ", ha="center", va="center")
            else:
                a1.text(cx[c], cy[r], names[r][c], ha="center", va="center")
    a1.plot(326, 325, "ko", ms=7)
    a1.text(300, 285, "reference (b)", fontsize=9, va="center")
    a1.plot(488, 520, "rs", ms=7); a1.text(498, 560, "a NE b", color="red", fontsize=11, ha="left")
    a1.set_xticks([]); a1.set_yticks([])
    a1.set_title("Projection-based CDC (9 regions)")
    a2.set_aspect("equal"); a2.set_xlim(-1.05, 1.05); a2.set_ylim(-1.05, 1.05)
    cols = {"N": "#dcefd8", "NE": "#c9daea", "E": "#f9c9c9", "SE": "#fbe3f0",
            "S": "#eae2d0", "SW": "#ffffd9", "W": "#fee4c4", "NW": "#e6dcec"}
    ang = {"N": 90, "NE": 45, "E": 0, "SE": -45, "S": -90, "SW": -135, "W": 180, "NW": 135}
    for k, a in ang.items():
        a2.add_patch(Wedge((0, 0), 1, a - 22.5, a + 22.5, fc=cols[k], ec="#999999", lw=1.2))
        r = 0.58 if k in ("N", "S") else 0.62
        if k in ("N", "S"):
            r = 0.77 if k == "N" else 0.77
        a2.text(0.62 * np.cos(np.radians(a)) + (0.18 if k == "E" else 0),
                0.62 * np.sin(np.radians(a)) if k not in ("N", "S") else 0.77 * np.sin(np.radians(a)),
                k, ha="center", va="center")
    a2.plot(0, 0, "ko", ms=6)
    a2.text(0.02, -0.1, "reference (b)", fontsize=9, va="center")
    a2.plot(0.42, 0.62, "rs", ms=7); a2.text(0.5, 0.67, "a NE b", color="red", fontsize=11)
    a2.set_xticks([]); a2.set_yticks([])
    a2.set_title("Cone-based CDC (8 cones)")
    _save(fig, 6, "01-cdc.png")


def fig_rcc8():
    BFC, BEC = "#d6eaff", "#4a82c0"
    YFC, YEC = "#fff0d4", "#d4aa5a"
    MIX, MEC = "#f0e9d4", "#aab8b0"
    fig = plt.figure(figsize=(11, 5.2))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1760); ax.set_ylim(832, 0)
    ax.set_aspect("equal"); ax.axis("off")

    def c(x, y, r, fc, ec, z=1, lw=1.4):
        ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=z))

    ax.text(880, 33, "RCC-8: the 8 topological relations between two regions (blue a, yellow b)",
            ha="center", va="center", fontsize=13)
    heads = [(229, 99, "DC\ndisconnected"), (663, 99, "EC\nexternally connected"),
             (1097, 99, "PO\npartially overlapping"), (1531, 99, "EQ\nequal"),
             (229, 453, "TPP\ntangential proper part"), (663, 453, "NTPP\nnon-tangential proper part"),
             (1097, 453, "TPPi\n(inverse)"), (1531, 453, "NTPPi\n(inverse)")]
    for x, y, s in heads:
        ax.text(x, y, s, ha="center", va="center", fontsize=11, linespacing=1.3)
    c(122, 276, 72, BFC, BEC); c(336, 276, 72, YFC, YEC)
    c(592, 276, 72, BFC, BEC); c(734, 276, 72, YFC, YEC)
    c(1052, 276, 80, BFC, BEC); c(1141, 276, 80, YFC, YEC, z=2)
    ax.add_patch(Circle((1141, 276), 80, fc=YFC, ec=YEC, lw=1.4, alpha=0.6, zorder=3))
    ax.add_patch(Circle((1052, 276), 80, fc="none", ec=BEC, lw=1.4, zorder=4))
    c(1531, 276, 80, MIX, "#9a8858", lw=1.8)
    c(229, 631, 107, YFC, YEC); c(175, 631, 54, MIX, MEC, z=2)
    c(663, 631, 107, YFC, YEC); c(663, 631, 45, MIX, MEC, z=2)
    c(1097, 631, 107, BFC, BEC); c(1150, 631, 54, MIX, YEC, z=2)
    c(1531, 631, 107, BFC, BEC); c(1531, 631, 45, MIX, YEC, z=2)
    _save(fig, 6, "02-rcc8.png")


def fig_relative():
    fig = plt.figure(figsize=(6, 5.2))
    ax = fig.add_axes([34 / 960, 26 / 832, 708 / 960, 744 / 832])
    ax.set_xlim(0, 708); ax.set_ylim(744, 0)
    ax.axvline(354, color="gray", ls="--", lw=1.8); ax.axhline(392, color="gray", ls="--", lw=1.8)
    for x, y, s in [(147, 175, "front-left\nön-sol"), (547, 175, "front-right\nön-sağ"),
                    (147, 540, "back-left\narka-sol"), (547, 540, "back-right\narka-sağ")]:
        ax.text(x, y, s, ha="center", va="center", linespacing=1.3)
    ax.plot(354, 392, "ks", ms=10)
    ax.text(371, 359, "B: chair (reference)", fontsize=9.5, va="center")
    ax.plot(543, 285, "ro", ms=7)
    ax.text(558, 277, "C: bed → (bed, front-right, chair)", color="red", fontsize=9, va="center", ha="left")
    ax.plot(354, 717, "k^", ms=13)
    ax.text(386, 710, "A: observer (at the door, facing north)", fontsize=9, va="center")
    ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.01, 0.975, 'Relative direction (1-cross): "right/left/front/back" depends on the viewpoint',
             fontsize=10.5, va="center")
    _save(fig, 6, "03-relative.png")


# ---------------------------------------------------------------- Lecture 7
def fig_roomspace_results():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    a1.plot([3, 4, 5, 6], [50, 22, 12.5, 4.3], "o-", color=RED, lw=1.8)
    a1.set_ylim(0, 60); _grid(a1, "both")
    a1.set_title("Llama-3-8B, RoomSpace-100: collapse as hops increase")
    a1.set_xlabel("n (number of objects), m = n−1 hops"); a1.set_ylabel("accuracy (%)")
    a2.plot([4, 5, 6, 7, 8, 9], [12.5, 22, 19.7, 44.7, 53.1, 63.5], "o-", color=BLUE, lw=1.8)
    a2.set_ylim(0, 70); _grid(a2, "both")
    a2.set_title("More constraints = fewer hops = easier")
    a2.set_xlabel("m (number of constraints), n = 5 fixed")
    _save(fig, 7, "02-roomspace-results.png")


# ---------------------------------------------------------------- Lecture 8
def fig_vlanext_family():
    labels = ["OpenVLA\n(7B, LLaMA-2)", "VLANeXt-S\n(0.8B)", "VLANeXt-B\n(2B, Qwen3-VL)",
              "VLANeXt-LAM", "VLANeXt-JEPA", "VLANeXt-WAM\n(5B, Wan2.2)", "VLANeXt-L\n(4B)"]
    vals = [76.5, 96.7, 97.4, 97.6, 97.7, 98.2, 98.4]
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.bar(range(7), vals, color=[RED] + [BLUE] * 6)
    ax.set_xticks(range(7)); ax.set_xticklabels(labels, fontsize=8.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.8, f"{v}", ha="center", fontsize=10)
    ax.set_ylim(60, 102); ax.set_ylabel("LIBERO mean (%)"); _grid(ax)
    ax.set_title("VLANeXt family vs OpenVLA (LIBERO 4 suite); VLANeXt-B LIBERO-Plus: 83.9%")
    _save(fig, 8, "03-vlanext-family.png")


FIGURES = [fig_action_trajectory, fig_action_binning, fig_oft_under_perturbation,
           fig_libero_leaderboard, fig_patterns, fig_category_results, fig_complexity,
           fig_gripper_effect, fig_cdc, fig_rcc8, fig_relative, fig_roomspace_results,
           fig_vlanext_family]


def main():
    for f in FIGURES:
        f()


if __name__ == "__main__":
    main()
