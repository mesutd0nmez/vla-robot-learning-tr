"""Ders 20 görselleri (02–05). Gerçek veri: LIBERO BDDL + .pruned_init (scene_frames.py), pairs.csv, Mesut'un project-2 render'ı.
  python plot_day20.py --libero ~/src/LIBERO --data ../../datasets/04-tr-spatial-commands/data \
      --render ../../projects/project-2/libero_first_episode-task-1.png --out ../../lectures/images/lecture-20
"""
import argparse, csv, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import scene_frames as SF

Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"


def scene(root, task):
    name = SF.official_order(root)[task]
    info = SF.parse_bddl(root / "libero/libero/bddl_files" / SF.SUITE / (name + ".bddl"))
    st = SF.load_init(root / "libero/libero/init_files" / SF.SUITE / (name + ".pruned_init"))
    return SF.positions(st, info["objects"])


LBL = {"akita_black_bowl_1": "bowl_1", "akita_black_bowl_2": "bowl_2", "plate_1": "tabak", "cookies_1": "kurabiye\nkutusu",
       "glazed_rim_porcelain_ramekin_1": "ramekin"}
COL = {"akita_black_bowl_1": "#1f77b4", "akita_black_bowl_2": "#d98c00", "plate_1": "#c0392b", "cookies_1": "#7f6000",
       "glazed_rim_porcelain_ramekin_1": "#666666"}


def topdown(ax, pos, title, mode):
    """Kuş bakışı; ekranda yukarı = -x (robot üstte), sağ = -y → modelin gördüğü görüntüyle aynı yerleşim."""
    ax.add_patch(Rectangle((-0.6, -0.5), 1.2, 1.0, fc="#f7f1e3", ec="#999", lw=1))            # masa (y yatay, x dikey)
    for o, xyz in pos.items():
        ax.scatter(-xyz[:, 1], -xyz[:, 0], s=14, c=COL[o], alpha=0.5, lw=0)
        m = xyz.mean(0)
        ax.annotate(LBL[o], (-m[1], -m[0]), xytext=(7, 6), textcoords="offset points", fontsize=9, color=COL[o], weight="bold")
    ax.scatter([0], [0.66], marker="s", s=260, c=Y, ec="k", zorder=3); ax.text(0, 0.735, "robot tabanı (x = −0.66)", ha="center", fontsize=9)
    ax.scatter([0], [-0.66], marker="v", s=220, c=B, ec="k", zorder=3); ax.text(0, -0.76, "agentview kamerası (x = +0.66)", ha="center", fontsize=9)
    ax.annotate("", xy=(0, 0.48), xytext=(0, 0.62), arrowprops=dict(arrowstyle="<-", color="k"))
    if mode == "lr":
        ax.annotate("", xy=(-0.55, -0.44), xytext=(0.55, -0.44), arrowprops=dict(arrowstyle="<->", color="#2e7d32", lw=2))
        ax.text(-0.57, -0.40, "+y", fontsize=9, color="#2e7d32"); ax.text(0.50, -0.40, "−y", fontsize=9, color="#2e7d32")
        ax.text(-0.58, 0.60, "görüntü: SOL\nrobot: SOL\nLIBERO: right", fontsize=9, va="top", bbox=dict(fc=G, ec="none"))
        ax.text(0.58, 0.60, "görüntü: SAĞ\nrobot: SAĞ\nLIBERO: left", fontsize=9, va="top", ha="right", bbox=dict(fc=R, ec="none"))
    else:
        ax.annotate("", xy=(0.52, 0.42), xytext=(0.52, -0.42), arrowprops=dict(arrowstyle="<->", color="#2e7d32", lw=2))
        ax.text(0.55, 0.36, "−x", fontsize=9, color="#2e7d32"); ax.text(0.55, -0.40, "+x", fontsize=9, color="#2e7d32")
        ax.text(-0.58, 0.44, "robot: ÖN (yakın)\ngörüntü: ARKA (üstte)\nLIBERO: back", fontsize=9, va="top", bbox=dict(fc=R, ec="none"))
        ax.text(-0.58, -0.44, "robot: uzak\ngörüntü: ÖN (altta)\nLIBERO: front", fontsize=9, va="bottom", bbox=dict(fc=G, ec="none"))
    ax.set_xlim(-0.68, 0.68); ax.set_ylim(-0.82, 0.80); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10.5)


def fig_frames(root, render, out):
    fig, axs = plt.subplots(1, 3, figsize=(17, 6.8))
    p1, p8 = scene(root, 1), scene(root, 8)
    ax = axs[0]
    if render and pathlib.Path(render).exists():
        img = plt.imread(render)[:256, :256]
        ax.imshow(img, extent=(0, 256, 256, 0))
        src = "Mesut'un project-2 render'ı (görev 1, init 0, [::-1, ::-1] uygulanmış)"
    else:
        ax.set_facecolor("#eee"); src = "render bulunamadı — yalnızca projeksiyon"
    for o, xyz in p1.items():
        uv = SF.project(xyz[0])
        ax.scatter(*uv, s=70, fc="none", ec=COL[o], lw=2)
        ax.annotate(LBL[o].replace("\n", " "), uv, xytext=(8, -8), textcoords="offset points", fontsize=9, color="white",
                    bbox=dict(fc=COL[o], ec="none", pad=1.5))
    ax.set_xlim(0, 256); ax.set_ylim(256, 0); ax.set_xticks([0, 128, 256]); ax.set_yticks([0, 128, 256])
    ax.set_xlabel("u (piksel) — küçük u = görüntünün solu = dünya +y")
    ax.set_title("(a) Modelin gördüğü kare + init state'ten HESAPLANAN konumlar\n" + src, fontsize=10.5)
    topdown(axs[1], p1, "(b) Görev 1 sahnesi, 50 init state (kuş bakışı, görüntüyle aynı yön)\n«soldaki siyah kase»: görüntü/robot → bowl_1, LIBERO sözleşmesi → bowl_2", "lr")
    topdown(axs[2], p8, "(c) Görev 8 sahnesi, 50 init state\n«öndeki siyah kase»: görüntü/LIBERO → bowl_1, robot → bowl_2", "fb")
    fig.suptitle("Üç çerçeve, bir sahne: sol/sağ'da LIBERO adlandırması, ön/arka'da robot çerçevesi ayrışıyor "
                 "(kaynak: LIBERO .pruned_init + kamera pozu libero_tabletop_manipulation.py; fovy 45°)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig(out / "02-three-frames.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_matrix(pairs, out):
    cols = [("pair_id", "çift", 0.6), ("tier", "tier", 0.5), ("scene", "sahne", 1.1), ("en_short", "EN (ayırt edici kısım)", 3.6), ("rel_seg", "TR ilişki sözcüğü (ayrım)", 2.4),
            ("case", "hâl", 1.9), ("frame_dependent", "çerçeve?", 0.9), ("en_seen_suite", "EN eğitimde?", 1.1), ("refs", "kastedilen kâse\nlibero | görüntü | robot", 2.6)]
    fig, ax = plt.subplots(figsize=(17, 9.2)); ax.axis("off")
    xs = np.cumsum([0] + [c[2] for c in cols]); H = 1.0
    for j, (_, name, w) in enumerate(cols):
        ax.add_patch(Rectangle((xs[j], 0), w, H * 1.3, fc="#444", ec="white")); ax.text(xs[j] + w / 2, H * 0.65, name, color="white", ha="center", va="center", fontsize=9.5, weight="bold")
    blockc = {"A": B, "B": Y, "C": G, "D": R}
    for i, r in enumerate(pairs):
        y = -(i + 1) * H + 0.0
        en = r["en"].replace("pick up the black bowl ", "… ").replace(" and place it on the plate", " …")
        r = dict(r, scene=("Spatial " if r["suite"] == "libero_spatial" else "Goal ") + r["scene_task_id"], en_short=en,
                 refs="—" if r["ref_libero"] == "n/a" else f"{r['ref_libero'][-1]} | {r['ref_image'][-1]} | {r['ref_robot'][-1]}")
        for j, (k, _, w) in enumerate(cols):
            fc = blockc[r["block"]] if j < 3 else "white"
            if k == "refs" and r["ref_libero"] != "n/a":
                s = {r["ref_libero"], r["ref_image"], r["ref_robot"]}
                fc = R if len(s) > 1 else (Y if s == {"bowl_2"} else B)
            if k == "frame_dependent" and r[k] == "evet": fc = R
            if k == "en_seen_suite": fc = G if r[k] == "evet" else Y
            ax.add_patch(Rectangle((xs[j], y), w, H, fc=fc, ec="#bbb", lw=0.6))
            ax.text(xs[j] + (0.08 if k in ("en_short", "rel_seg", "case") else w / 2), y + H / 2, str(r[k]), ha="left" if k in ("en_short", "rel_seg", "case") else "center", va="center", fontsize=9)
    ax.set_xlim(0, xs[-1]); ax.set_ylim(-len(pairs) * H - 1.6, H * 1.4)
    ax.text(0, -len(pairs) * H - 0.55, "Bloklar: A (mavi) Spatial eğitim komutları · B (sarı) aynı sahnelerde eğitim cümlesi OLMAYAN tarifler · C (yeşil) Goal eğitim komutları · D (kırmızı) eğitilmemiş yerleştirme (predicate yok).\n"
            "Tier 1: mevcut checkpoint + mevcut predicate · Tier 2: mevcut checkpoint + karşı-olgusal BDDL (bowl_2) · Tier 3: yeni predicate + fine-tune.  Kâse sütunu: 1 = bowl_1 (eğitimdeki hedef), 2 = bowl_2; kırmızı = çerçeveler ayrışıyor.",
            fontsize=9.5, va="top")
    ax.set_title("datasets/04-tr-spatial-commands — 20 EN–TR çiftinin tasarım matrisi (pairs.csv; kâse sütunu 50 init state'ten hesaplandı)", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "03-design-matrix.png", dpi=130); plt.close(fig)


def fig_referents(data, out):
    rows = list(csv.DictReader(open(data / "referents.csv", encoding="utf-8")))
    keys = [("left", "libero"), ("left", "image"), ("left", "robot"), ("front", "libero"), ("front", "image"), ("front", "robot"),
            ("near_plate", "none"), ("near_robot", "none"), ("higher", "none")]
    names = ["sol\nLIBERO", "sol\ngörüntü", "sol\nrobot", "ön\nLIBERO", "ön\ngörüntü", "ön\nrobot", "tabağa\nyakın", "robota\nyakın", "daha\nyüksekte"]
    d = {(int(r["task_id"]), r["descriptor"], r["frame"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    for t in range(10):
        for j, k in enumerate(keys):
            r = d[(t,) + k]
            ok = r["unambiguous"] == "True"
            fc = R if not ok else (B if r["referent"] == "bowl_1" else Y)
            ax.add_patch(Rectangle((j, 9 - t), 1, 1, fc=fc, ec="white", lw=2))
            txt = "belirsiz" if r["referent"] == "belirsiz" else f"{r['referent']}\n{float(r['min_margin']):.0f} {r['unit']}"
            ax.text(j + .5, 9 - t + .5, txt, ha="center", va="center", fontsize=8.5, color="#333" if ok else "#a00")
        ax.text(-0.1, 9 - t + .5, f"görev {t}\n1: {d[(t,'left','libero')]['bowl_1_at'].replace('_region','').replace('wooden_cabinet_1_','dolap ').replace('flat_stove_1_','ocak ')}\n2: {d[(t,'left','libero')]['bowl_2_at'].replace('_region','').replace('wooden_cabinet_1_','dolap ').replace('flat_stove_1_','ocak ')}",
                ha="right", va="center", fontsize=7.6)
    for j, n in enumerate(names): ax.text(j + .5, 10.15, n, ha="center", va="bottom", fontsize=9.5)
    for x in (3, 6): ax.plot([x, x], [0, 10], color="k", lw=1.5)
    ax.set_xlim(-2.6, 9); ax.set_ylim(0, 11.2); ax.axis("off")
    ax.set_title("«Hangi kâse?» — 10 Spatial sahnesi × 9 tarif; hücre: kastedilen kâse + 50 init state'teki EN KÜÇÜK fark\n"
                 "mavi = bowl_1 (eğitimdeki hedef), sarı = bowl_2, kırmızı = belirsiz (< 5 cm / 20 px ya da işaret değişiyor) — kaynak: LIBERO .pruned_init", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "04-referents.png", dpi=130); plt.close(fig)


def fig_allomorph(pairs, out):
    import validate_set as V
    from collections import Counter, defaultdict
    allo = defaultdict(Counter)
    for r in pairs:
        for tok in r["tr_a_seg"].split():
            for m, s in V.morpheme_of(tok): allo[m][s] += 1
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]; x = 0; ticks = []; shades = ["#e09f00", "#ffd166", "#ffe6a8", "#fff3d6", "#f1c40f", "#fbe9a7"]
    for m in ("GEN", "POSS3", "ACC", "DAT", "LOC", "ki"):
        bottom = 0
        for i, (s, n) in enumerate(allo[m].most_common()):
            ax.bar(x, n, bottom=bottom, color=shades[i % 6], ec="#555", width=0.7)
            if n >= 2: ax.text(x, bottom + n / 2, f"-{s} ×{n}", ha="center", va="center", fontsize=9)
            bottom += n
        small = [f"-{s}" for s, n in allo[m].most_common() if n < 2]
        if small: ax.text(x, bottom + 0.6, " ".join(small) + "\n(birer kez)", ha="center", va="bottom", fontsize=8.5)
        ticks.append(f"{m}\n{len(allo[m])} biçim"); x += 1
    ax.set_xticks(range(len(ticks))); ax.set_xticklabels(ticks); ax.set_ylabel("tr_a'da geçiş sayısı (20 komut)"); ax.set_ylim(0, 31)
    ax.set_title("(a) Bir biçimbirim, birden çok yüzey: 20 komutta ek allomorfları\n(tr_morph ayrımlarından sayıldı — gerçek)", fontsize=10.5)
    ax = axs[1]
    sp = [r for r in pairs if r["suite"] == "libero_spatial"]
    NH, NT = 5, 7
    for row, (key, lab) in enumerate([("en", "EN"), ("tr_a", "TR (tr_a)")]):
        toks = [r[key].split() for r in sp]
        cells = [(k, {t[k] for t in toks}) for k in range(NH)] + [(NH + 1 + j, {t[-(NT - j)] for t in toks}) for j in range(NT)]
        for x, vals in cells:
            ax.add_patch(Rectangle((x, 1 - row), 1, 0.8, fc=G if len(vals) == 1 else Y, ec="white", lw=1.5))
            ax.text(x + .5, 1 - row + .52, str(len(vals)), ha="center", va="center", fontsize=10, weight="bold")
            ax.text(x + .5, 1 - row + .2, next(iter(vals)) if len(vals) == 1 else "", ha="center", va="center", fontsize=6.5)
        ax.text(NH + .5, 1 - row + .4, "…", ha="center", va="center", fontsize=14)
        ax.text(-0.3, 1 - row + .4, lab, ha="right", va="center", fontsize=10)
    ax.text(NH / 2, 2.0, "ilk 5 sözcük (baştan hizalı)", ha="center", fontsize=9); ax.text(NH + 1 + NT / 2, 2.0, "son 7 sözcük (sondan hizalı)", ha="center", fontsize=9)
    ax.set_xlim(-2.2, NH + NT + 1.2); ax.set_ylim(-1.3, 2.4); ax.axis("off")
    ax.text(0, -0.25, "hücre = o konumda 12 Spatial komutunda kaç FARKLI sözcük var\nyeşil = hep aynı (ayırt edici bilgi yok) · sarı = ayırt edici\n"
            "EN: tarif ORTADA — «pick up the black bowl» + tarif + «and place it on the plate»\nTR: tarif EN BAŞTA — tarif + «siyah kaseyi al ve tabağın üstüne koy»", fontsize=9, va="top")
    ax.set_title("(b) Ayırt edici bilgi cümlenin neresinde? (sözcük konumu, 12 Spatial komutu)", fontsize=10.5)
    fig.tight_layout(); fig.savefig(out / "05-allomorphs-position.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero", required=True); ap.add_argument("--data", required=True); ap.add_argument("--render"); ap.add_argument("--out", default=".")
    a = ap.parse_args()
    root, data, out = pathlib.Path(a.libero).expanduser(), pathlib.Path(a.data), pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pairs = list(csv.DictReader(open(data / "pairs.csv", encoding="utf-8")))
    fig_frames(root, a.render, out); fig_matrix(pairs, out); fig_referents(data, out); fig_allomorph(pairs, out)
    print("yazıldı:", *sorted(p.name for p in out.glob("0[2-5]-*.png")))


if __name__ == "__main__":
    main()
