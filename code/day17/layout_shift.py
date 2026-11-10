"""Ders 17 — LIBERO-Plus 'Objects Layout': `_levelL_sampleM.bddl` dosyalarında bölgelerin ne kadar kaydığını ve
`_add_K.bddl` dosyalarında hangi nesnenin eklendiğini temel BDDL ile karşılaştırarak ölçer.

Model/asset GEREKMEZ; yalnızca metin dosyaları okunur (BDDL'i basit bir regex ile ayrıştırır; bddl paketi gerekmez).

Kullanım:  python layout_shift.py <LIBERO-plus kökü> <çıktı.png> [--suite libero_spatial] [--csv out.csv]
"""
import argparse, collections, csv, pathlib, re
import numpy as np

RE_REGION = re.compile(r"\(\s*(\w+)\s*\(:target\s+(\w+)\)\s*\(:ranges\s*\(\s*\(([^()]+)\)\s*\)\s*\)", re.S)


def regions(text):
    """bölge adı → (cx, cy) (ranges = x0 y0 x1 y1; merkez)."""
    out = {}
    for name, target, rng in RE_REGION.findall(text):
        if target != "main_table":
            continue
        x0, y0, x1, y1 = map(float, rng.split())
        out[name] = ((x0 + x1) / 2, (y0 + y1) / 2)
    return out


def init_map(text):
    """nesne → bölge (:init bloğundaki (On obj main_table_region))."""
    m = re.search(r"\(:init(.*?)\)\s*\(:goal", text, re.S)
    return dict(re.findall(r"\(On (\S+) main_table_(\w+)\)", m.group(1))) if m else {}


def objects_of_interest(text):
    m = re.search(r"\(:obj_of_interest(.*?)\)", text, re.S)
    return m.group(1).split() if m else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--suite", default="libero_spatial"); ap.add_argument("--csv")
    a = ap.parse_args()
    d = pathlib.Path(a.root) / "libero/libero/bddl_files" / a.suite
    rows = []
    for f in sorted(d.glob("*_level*_sample*.bddl")):
        m = re.match(r"(.+)_level(\d+)_sample(\d+)\.bddl$", f.name)
        base_name, level, sample = m.group(1), int(m.group(2)), int(m.group(3))
        base = (d / f"{base_name}.bddl").read_text(encoding="utf-8"); var = f.read_text(encoding="utf-8")
        rb, rv = regions(base), regions(var); ib, iv = init_map(base), init_map(var)
        ooi = objects_of_interest(base)
        target = ooi[0] if ooi else None  # Spatial'da ilk obj_of_interest = alınacak kâse
        for obj, reg in ib.items():
            if obj not in iv or reg not in rb or iv[obj] not in rv:
                continue
            (bx, by), (vx, vy) = rb[reg], rv[iv[obj]]
            rows.append(dict(base=base_name, level=level, sample=sample, obj=obj, is_target=obj == target,
                             dx_cm=100 * (vx - bx), dy_cm=100 * (vy - by), shift_cm=100 * float(np.hypot(vx - bx, vy - by))))
    print(f"{a.suite}: {len(set((r['base'], r['level'], r['sample']) for r in rows))} level dosyası, {len(rows)} nesne-satırı")
    by_level_t = collections.defaultdict(list); by_level_o = collections.defaultdict(list)
    for r in rows:
        (by_level_t if r["is_target"] else by_level_o)[r["level"]].append(r["shift_cm"])
    for L in sorted(by_level_t):
        t, o = np.array(by_level_t[L]), np.array(by_level_o[L])
        print(f"  level {L}: hedef kâse kayması ort {t.mean():5.2f} cm (max {t.max():5.2f}); diğer nesneler ort {o.mean():5.2f} cm (max {o.max():5.2f})  n_dosya={len(t)}")
    # --- _add_ dosyaları: eklenen nesne türleri ve konumları
    added = collections.Counter(); add_pos = []
    for f in sorted(d.glob("*_add_*.bddl")):
        txt = f.read_text(encoding="utf-8"); rv = regions(txt)
        for obj, reg in init_map(txt).items():
            if reg.startswith("add_object_region"):
                added[re.sub(r"_\d+$", "", obj)] += 1
                if reg in rv: add_pos.append(rv[reg])
    print(f"  _add_ dosyaları: {sum(added.values())} eklenen nesne, {len(added)} tür:")
    for k, v in added.most_common(): print(f"     {v:3d}  {k}")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1.1, 1, 1.2]})
    ax = axes[0]
    Ls = sorted(by_level_t)
    bp = ax.boxplot([by_level_t[L] for L in Ls], positions=np.arange(len(Ls)) - 0.18, widths=0.32, patch_artist=True, flierprops=dict(marker=".", ms=3))
    for p in bp["boxes"]: p.set_facecolor("#ffd166"); p.set_edgecolor("#b07d00")
    bp2 = ax.boxplot([by_level_o[L] for L in Ls], positions=np.arange(len(Ls)) + 0.18, widths=0.32, patch_artist=True, flierprops=dict(marker=".", ms=3))
    for p in bp2["boxes"]: p.set_facecolor("#cfe8ff"); p.set_edgecolor("#3a6ea5")
    for m in bp["medians"] + bp2["medians"]: m.set_color("#000")
    ax.set_xticks(range(len(Ls))); ax.set_xticklabels([f"level {L}" for L in Ls])
    ax.set_ylabel("bölge merkezinin kayması (cm)"); ax.set_title("(a) `_levelL_sampleM`: nesne bölgeleri ne kadar kayıyor?")
    ax.legend([bp["boxes"][0], bp2["boxes"][0]], ["hedef kâse (obj_of_interest[0])", "diğer nesneler"], fontsize=8, loc="upper left")
    ax = axes[1]
    tgt = [r for r in rows if r["is_target"]]
    cols = {1: "#d4f5d4", 2: "#b7e4b7", 3: "#ffd166", 4: "#ffb3a7", 5: "#ff7f7f"}
    for L in Ls:
        xs = [r["dx_cm"] for r in tgt if r["level"] == L]; ys = [r["dy_cm"] for r in tgt if r["level"] == L]
        ax.scatter(xs, ys, s=26, color=cols[L], edgecolor="#555", linewidth=0.5, label=f"level {L}")
    ax.plot(0, 0, "k+", ms=12, mew=2); ax.set_aspect("equal"); ax.legend(fontsize=8)
    ax.set_xlabel("Δx (cm)"); ax.set_ylabel("Δy (cm)"); ax.set_title("(b) Hedef kâsenin kayması (tüm temel görevler)")
    ax = axes[2]
    names = [k for k, _ in added.most_common()][::-1]; vals = [added[k] for k in names]
    ax.barh(names, vals, color="#ffd6d6", edgecolor="#b00020")
    for i, v in enumerate(vals): ax.text(v + 0.3, i, str(v), va="center", fontsize=8)
    n_add_files = len(list(d.glob("*_add_*.bddl")))
    ax.set_xlabel("kaç `_add_` dosyasında"); ax.set_title(f"(c) `_add_K`: eklenen çeldiriciler ({sum(vals)} nesne / {n_add_files} dosya)")
    ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(f"LIBERO-Plus 'Objects Layout' ({a.suite}): BDDL dosyalarından ölçüldü — temel BDDL vs varyant, bölge merkezleri (layout_shift.py)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(a.out, dpi=170); plt.close(fig)
    print("yazıldı:", a.out)


if __name__ == "__main__":
    main()
