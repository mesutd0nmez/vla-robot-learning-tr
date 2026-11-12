"""Ders 19 görselleri (matplotlib). Gerçek değerler: makale tabloları (kaynak başlıkta), BDDL ve .pruned_init dosyaları.
  python plot_day19.py --libero /yol/LIBERO --out ../../lectures/images/lecture-19
Görsel 02: komutsuz success rate × suite (Chen Ø / Zhan blank / Fang vision-only) + yapısal üst sınır (suite_structure.py)
Görsel 03: LIBERO-Spatial görev 0'ın 50 init state'i (gerçek), BDDL bölgeleri, kâse–kâse mesafe dağılımı
Görsel 04: Spatial 10 görev × bölge matrisi (hedef/öteki kâse) — görev 7 ve 9 aynı sahne
Görsel 05: LIBERO-CF grounding: faithful vs biased (Fang vd. Tablo III)
"""
import argparse, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
YE, BE, GE, RE = "#b8860b", "#3b6fa0", "#2e8b57", "#c0392b"
SUITES = ["Spatial", "Object", "Goal", "Long"]

# ---- gerçek değerler (makale tabloları) ----
# Chen vd. 2606.15714 Tablo 6, Ø (komutsuz) sütunu; en sütunu parantezde
CHEN = {"π0.5": {"en": [97.5, 99.0, 94.0, 93.0], "Ø": [66.0, 64.5, 9.0, 70.0]},
        "OpenVLA-OFT": {"en": [93.5, 100.0, 97.5, 94.5], "Ø": [80.0, 91.5, 10.0, 76.5]},
        "Qwen3-VL-π": {"en": [93.0, 99.5, 97.0, 96.5], "Ø": [41.0, 57.5, 6.0, 60.0]},
        "Qwen3-VL-OFT": {"en": [92.5, 100.0, 99.0, 91.0], "Ø": [34.0, 42.5, 8.5, 45.0]}}
# Zhan vd. 2601.04052 Tablo 8/9: original / blank
ZHAN = {"π0.5": {"orig": [95.4, 98.4, 97.2, 89.6], "blank": [65.0, 51.8, 11.4, 72.0]},
        "π0": {"orig": [96.8, 98.8, 95.8, 85.2], "blank": [29.8, 41.2, 4.6, 25.2]}}
# Fang vd. 2602.17659 Tablo I: V+L / vision-only
FANG = {"π0.5": {"vl": [98.8, 98.2, 98.0, 92.4], "v": [68.2, 56.6, 10.2, 71.6]},
        "OpenVLA-OFT": {"vl": [97.6, 98.4, 97.9, 94.5], "v": [83.2, 98.2, 10.0, 85.0]},
        "π0": {"vl": [96.8, 98.8, 95.8, 85.2], "v": [28.0, 26.8, 3.4, 15.6]}}
# suite_structure.py çıktısı: komutu okumayan ama sahneyi mükemmel tanıyan policy'nin üst sınırı (görev sayısı / 10)
BLIND = {"Spatial": 90, "Object": 100, "Goal": 10, "Long": 90}
# Fang vd. Tablo III, grounding (faithful / biased), CAG yok
CF = {"OpenVLA-OFT": {"CF-Spatial": (6.8, 84.7), "CF-Object": (2.0, 99.6), "CF-Long": (4.0, 62.9), "CF-OOD": (6.0, 87.3)},
      "π0": {"CF-Spatial": (70.9, 37.1), "CF-Object": (6.0, 59.4), "CF-Long": (20.4, 83.1), "CF-OOD": (18.0, 72.4)},
      "π0.5": {"CF-Spatial": (39.3, 61.3), "CF-Object": (11.4, 68.8), "CF-Long": (51.6, 58.4), "CF-OOD": (20.7, 74.0)}}


def fig02(out):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
    series = [("π0.5 — Chen Ø", CHEN["π0.5"]["Ø"], Y, YE), ("π0.5 — Zhan blank", ZHAN["π0.5"]["blank"], Y, YE),
              ("π0.5 — Fang V-only", FANG["π0.5"]["v"], Y, YE), ("OpenVLA-OFT — Chen Ø", CHEN["OpenVLA-OFT"]["Ø"], B, BE),
              ("OpenVLA-OFT — Fang V-only", FANG["OpenVLA-OFT"]["v"], B, BE), ("π0 — Zhan blank", ZHAN["π0"]["blank"], G, GE),
              ("Qwen3-VL-π — Chen Ø", CHEN["Qwen3-VL-π"]["Ø"], R, RE), ("Qwen3-VL-OFT — Chen Ø", CHEN["Qwen3-VL-OFT"]["Ø"], R, RE)]
    for ax, suite, si in zip(axes, SUITES, range(4)):
        xs = np.arange(len(series))
        for x, (name, vals, fc, ec) in zip(xs, series):
            ax.bar(x, vals[si], color=fc, edgecolor=ec, linewidth=1.2)
            ax.text(x, vals[si] + 1.5, f"{vals[si]:.0f}", ha="center", fontsize=8)
        ax.axhline(BLIND[suite], color="k", ls="--", lw=1.2)
        ax.text(len(series) - 0.4, BLIND[suite] + (2.5 if BLIND[suite] > 50 else 14), f"yapısal üst sınır {BLIND[suite]}", ha="right", fontsize=8.5, fontweight="bold")
        ax.set_title(f"LIBERO-{suite}", fontsize=12)
        ax.set_xticks(xs); ax.set_xticklabels([s[0].replace(" — ", "\n") for s in series], rotation=90, fontsize=7.5)
        ax.set_ylim(0, 108); ax.grid(axis="y", alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("success rate (%) — KOMUTSUZ")
    fig.suptitle("Komutsuz (boş / Ø / vision-only) success rate × suite — üç makale, gerçek tablo değerleri; kesikli çizgi: komutu okumayan ama sahneyi\n"
                 "mükemmel tanıyan policy'nin üst sınırı (BDDL'den, suite_structure.py: Spatial 9/10, Object 10/10, Goal 1/10, Long 9/10). "
                 "Kaynak: Chen vd. 2606.15714 T6; Zhan vd. 2601.04052 T9; Fang vd. 2602.17659 T1", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(out / "02-empty-by-suite.png", dpi=150); plt.close(fig)


def fig03(out, libero):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from init_state_geometry import load_init, parse_bddl, official_order, object_xy
    root = pathlib.Path(libero)
    name = official_order(root, "libero_spatial")[0]
    info = parse_bddl(root / "libero/libero/bddl_files/libero_spatial" / f"{name}.bddl")
    st = load_init(root / "libero/libero/init_files/libero_spatial" / f"{name}.pruned_init")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.6, 1]})
    colors = {"akita_black_bowl_1": (Y, YE, "hedef kâse (bowl_1)"), "akita_black_bowl_2": (R, RE, "öteki kâse (bowl_2) — 'wrong' komutun hedefi"),
              "plate_1": (G, GE, "tabak"), "cookies_1": (B, BE, "kurabiye kutusu"), "glazed_rim_porcelain_ramekin_1": ("#eeeeee", "#777777", "ramekin")}
    for k, o in enumerate(info["objects"]):
        xyz, _ = object_xy(st, k); fc, ec, lab = colors[o]
        reg = info["regions"][info["where"][o]][1]
        ax.add_patch(Rectangle((reg[1], reg[0]), reg[3] - reg[1], reg[2] - reg[0], fill=False, ec="k", lw=2.4, ls=(0, (3, 2)), zorder=7))
        ax.scatter(xyz[:, 1], xyz[:, 0], s=28, c=fc, edgecolors=ec, linewidths=0.6, label=lab, zorder=3, alpha=0.6)
        ax.scatter(xyz[:10, 1], xyz[:10, 0], s=70, facecolors="none", edgecolors="#333333", linewidths=0.9, zorder=4)
        ax.text(xyz[:, 1].mean(), xyz[:, 0].mean() + 0.03, lab.split(" — ")[0], ha="center", fontsize=8.5)
    ax.scatter([], [], s=70, facecolors="none", edgecolors="k", label="siyah halka: ilk 10 init state (Ders 18 pilotu)")
    ax.set_xlabel("y (m) — masa boyunca"); ax.set_ylabel("x (m) — robota doğru azalır (kamera robotun karşısında, Ders 17)")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left"); ax.set_ylim(-0.24, 0.12)
    ax.set_title("LIBERO-Spatial görev 0: 50 init state'in nesne konumları (GERÇEK, .pruned_init)\nkesikli kutular = BDDL :init bölgeleri (2 cm × 2 cm); nesneler fiziksel oturmadan sonra ±1.5 cm içinde", fontsize=10)
    b1, _ = object_xy(st, 0); b2, _ = object_xy(st, 1); pl, _ = object_xy(st, 4)
    d12 = np.linalg.norm(b1[:, :2] - b2[:, :2], axis=1) * 100; d1p = np.linalg.norm(b1[:, :2] - pl[:, :2], axis=1) * 100
    q = st[:, 1:8]; qdev = np.linalg.norm(q - q.mean(0), axis=1)
    ax2.hist(d12, bins=12, color=R, edgecolor=RE, label=f"kâse–kâse: {d12.mean():.1f} cm (min {d12.min():.1f}, maks {d12.max():.1f})")
    ax2.hist(d1p, bins=12, color=G, edgecolor=GE, alpha=0.9, label=f"hedef kâse–tabak: {d1p.mean():.1f} cm (min {d1p.min():.1f}, maks {d1p.max():.1f})")
    ax2.set_xlabel("mesafe (cm)"); ax2.set_ylabel("init state sayısı (50)")
    ax2.set_title(f"Sahneler birbirine ne kadar benziyor?\nnesne yayılımı ≤ 3 cm; kâse yönelimi sabit\nrobot eklem std ≈ {q.std(0).mean():.3f} rad (~1°), sapma normu medyan {np.median(qdev):.3f} rad\n"
                  f"(karş. LIBERO-Plus robot bozulması 0.1–0.5 rad, Ders 17)", fontsize=10)
    ax2.legend(fontsize=8.5); ax2.grid(alpha=0.3); ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "03-init-states-task0.png", dpi=150); plt.close(fig)


def fig04(out, libero):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from init_state_geometry import parse_bddl, official_order
    import re
    root = pathlib.Path(libero)
    names = official_order(root, "libero_spatial")
    short = lambda reg: reg.replace("main_table_", "").replace("wooden_cabinet_1_", "cab:").replace("flat_stove_1_", "stove:").replace("_region", "") \
        .replace("glazed_rim_porcelain_ramekin_1", "on ramekin").replace("cookies_1", "on cookie box")
    infos = [parse_bddl(root / "libero/libero/bddl_files/libero_spatial" / f"{n}.bddl") for n in names]
    regions = sorted({short(i["where"][o]) for i in infos for o in ("akita_black_bowl_1", "akita_black_bowl_2")})
    M = np.zeros((10, len(regions)))
    for ti, i in enumerate(infos):
        for o in ("akita_black_bowl_1", "akita_black_bowl_2"):
            M[ti, regions.index(short(i["where"][o]))] = 2 if o == i["target"] else 1
    fig, ax = plt.subplots(figsize=(13, 6.2))
    cmap = matplotlib.colors.ListedColormap(["white", R, Y])
    ax.imshow(M, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    for ti in range(10):
        for rj in range(len(regions)):
            if M[ti, rj] == 2: ax.text(rj, ti, "HEDEF", ha="center", va="center", fontsize=8.5, fontweight="bold", color=YE)
            elif M[ti, rj] == 1: ax.text(rj, ti, "öteki", ha="center", va="center", fontsize=8.5, color=RE)
    labels = [f"{ti}: …{re.sub(r'pick_up_the_black_bowl_|_and_place_it_on_the_plate', '', n).replace('_', ' ')}" for ti, n in enumerate(names)]
    ax.set_yticks(range(10)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticks(range(len(regions))); ax.set_xticklabels(regions, rotation=35, ha="right", fontsize=9)
    ax.set_xticks(np.arange(-.5, len(regions), 1), minor=True); ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
    ax.grid(which="minor", color="#dddddd", lw=0.8); ax.tick_params(which="minor", length=0)
    for ti in (7, 9):
        ax.add_patch(Rectangle((-0.5, ti - 0.5), len(regions), 1, fill=False, ec="k", lw=2.2))
    ax.text(len(regions) - 0.5, 8.0, "görev 7 ve 9: AYNI sahne (ocak + dolap üstü),\nhedef ters — yalnızca komut ayırır", ha="right", va="center", fontsize=9.5,
            bbox=dict(boxstyle="round", fc="white", ec="k"))
    ax.set_title("LIBERO-Spatial: hangi görevde iki kâse nerede? (BDDL :init, gerçek; satır = görev, sütun = bölge)\n"
                 "10 görevin 9'u sahneden ayrışır → komutu okumayan policy'nin üst sınırı 9/10; görev 0'ın 'wrong' komutu (= görev 1) bowl_2'nin bölgesini tarif eder", fontsize=10.5)
    fig.tight_layout(); fig.savefig(out / "04-spatial-scene-matrix.png", dpi=150); plt.close(fig)


def fig05(out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    suites = list(CF["π0"])
    for ax, (model, d) in zip(axes, CF.items()):
        xs = np.arange(len(suites)); w = 0.38
        f = [d[s][0] for s in suites]; b = [d[s][1] for s in suites]
        ax.bar(xs - w / 2, f, w, color=G, edgecolor=GE, label="faithful: komuttaki nesneye dokundu")
        ax.bar(xs + w / 2, b, w, color=R, edgecolor=RE, label="biased: eğitimdeki görevin nesnesine dokundu")
        for x, (fv, bv) in enumerate(zip(f, b)):
            ax.text(x - w / 2, fv + 1.5, f"{fv:.0f}", ha="center", fontsize=8.5); ax.text(x + w / 2, bv + 1.5, f"{bv:.0f}", ha="center", fontsize=8.5)
        ax.set_xticks(xs); ax.set_xticklabels(suites, fontsize=9); ax.set_title(model, fontsize=12)
        ax.set_ylim(0, 108); ax.grid(axis="y", alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("grounding (%)")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("LIBERO-CF (Fang vd. 2602.17659, Tablo III, CAG yok): karşı-olgusal komutta gripper hangi nesneye gitti?\n"
                 "Bizim 'wrong' satırımızdaki grasped_bowl etiketi bunun elle sayılmış hâli: bowl_2 = faithful, bowl_1 = biased", fontsize=10)
    fig.tight_layout(rect=[0, 0.06, 1, 0.9]); fig.savefig(out / "05-cf-faithful-biased.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--libero", required=True); ap.add_argument("--out", default="../../lectures/images/lecture-19")
    a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig02(out); fig03(out, a.libero); fig04(out, a.libero); fig05(out)
    print("ok:", sorted(p.name for p in out.glob("*.png")))
