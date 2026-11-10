"""Ders 17 — LIBERO-Plus 'Robot Initial States': 500 MountedPanda alt sınıfının init_qpos'unu dosyadan okur ve çizer.

Model/asset/MuJoCo GEREKMEZ. `envs/robots/mounted_panda.py` içindeki her `class MountedPandaK(MountedPanda)`
yalnızca `init_qpos` property'sini ezer; bu script o 7 sayıyı `ast` ile çeker, taban pozdan farkını hesaplar.

Kullanım:  python robot_init_spread.py <LIBERO-plus kökü> <çıktı.png> [--csv out.csv] [--suite libero_spatial]
"""
import argparse, ast, csv, json, pathlib, re
import numpy as np

JOINTS = ["j1 taban yaw", "j2 omuz", "j3 üst kol", "j4 dirsek", "j5 ön kol", "j6 bilek", "j7 el yaw"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--csv"); ap.add_argument("--suite", default="libero_spatial")
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    src = (root / "libero/libero/envs/robots/mounted_panda.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    qpos = {}
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name.startswith("MountedPanda")):
            continue
        k = node.name[len("MountedPanda"):]
        for fn in node.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == "init_qpos":
                ret = [n for n in ast.walk(fn) if isinstance(n, ast.Return)][0]
                lst = ret.value.args[0]  # np.array([...])
                vals = []
                for e in lst.elts:
                    if isinstance(e, ast.BinOp):  # np.pi / 4
                        vals.append(np.pi / 4)
                    elif isinstance(e, ast.UnaryOp):
                        vals.append(-ast.literal_eval(e.operand))
                    else:
                        vals.append(ast.literal_eval(e))
                qpos[0 if k == "" else int(k)] = np.array(vals, dtype=float)
    base = qpos[0]
    ks = sorted(k for k in qpos if k != 0)
    Q = np.stack([qpos[k] for k in ks]); D = Q - base
    print(f"{len(ks)} varyant sınıfı; taban init_qpos = {np.round(base, 3)}")
    print("eklem başına |Δ| (rad): ort", np.round(np.abs(D).mean(0), 3), " max", np.round(np.abs(D).max(0), 3))
    norm = np.linalg.norm(D, axis=1)
    print(f"7-boyutlu ||Δ||: min {norm.min():.3f} ort {norm.mean():.3f} max {norm.max():.3f} rad")

    tasks = json.load(open(root / "libero/libero/benchmark/task_classification.json", encoding="utf-8"))[a.suite]
    used = {}
    for t in tasks:
        if t["category"] == "Robot Initial States":
            i = int(t["name"].split("_initstate_")[1].split("_")[0]); used[i] = t["difficulty_level"]
    print(f"{a.suite} Robot görevlerinde kullanılan I: {len(used)} farklı ({min(used)}..{max(used)})")
    # zorluk vs sapma büyüklüğü (hipotez testi: büyük sapma → zor mu?)
    diff_norm = {}
    for i, lvl in used.items():
        diff_norm.setdefault(lvl, []).append(np.linalg.norm(qpos[i] - base))
    for lvl in sorted(diff_norm):
        v = np.array(diff_norm[lvl]); print(f"  zorluk L{lvl}: n={len(v):3d}  ||Δ|| ort {v.mean():.3f}  medyan {np.median(v):.3f}")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["init_state"] + [f"q{i+1}" for i in range(7)] + ["norm_delta", "difficulty_spatial"])
            for k, q in zip(ks, Q):
                w.writerow([k] + [f"{x:.6f}" for x in q] + [f"{np.linalg.norm(q-base):.4f}", used.get(k, "")])

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    bp = ax.boxplot([np.degrees(D[:, j]) for j in range(7)], patch_artist=True, widths=0.6, showfliers=True,
                    flierprops=dict(marker=".", ms=3, color="#999"))
    for p in bp["boxes"]: p.set_facecolor("#ffd166"); p.set_edgecolor("#b07d00")
    for m in bp["medians"]: m.set_color("#000")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xticks(range(1, 8)); ax.set_xticklabels(JOINTS, rotation=20, ha="right")
    ax.set_ylabel("init_qpos − taban (derece)")
    ax.set_title(f"(a) {len(ks)} MountedPandaK sınıfı: eklem başına sapma (mounted_panda.py)")
    ax.text(0.02, 0.97, f"taban: {np.round(np.degrees(base),1)}°", transform=ax.transAxes, fontsize=8, va="top", color="#555")
    ax = axes[1]
    cols = {1: "#d4f5d4", 2: "#e8f5c8", 3: "#ffd166", 4: "#ffb3a7", 5: "#ffd6d6"}
    data = [np.array(diff_norm[l]) for l in sorted(diff_norm)]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, flierprops=dict(marker=".", ms=3, color="#999"))
    for p, l in zip(bp["boxes"], sorted(diff_norm)): p.set_facecolor(cols[l]); p.set_edgecolor("#888")
    for m in bp["medians"]: m.set_color("#000")
    ax.set_xticks(range(1, len(data) + 1)); ax.set_xticklabels([f"L{l}\n(n={len(diff_norm[l])})" for l in sorted(diff_norm)])
    ax.set_ylabel("||Δ init_qpos|| (rad, 7 eklem)"); ax.set_xlabel("zorluk seviyesi (4 modelden kaçı çözdü)")
    ax.set_title(f"(b) {a.suite}: sapma büyüklüğü vs zorluk")
    nc = sorted(((round(float(n), 2)), int(c)) for n, c in zip(*np.unique(np.round(norm, 2), return_counts=True)))
    ax.text(0.02, 0.03, "500 sınıfta ||Δ|| = " + ", ".join(f"{n} rad ×{c}" for n, c in nc) + "\n(rastgele yön, 5 sabit büyüklük)",
            transform=ax.transAxes, fontsize=7.5, color="#555", va="bottom")
    fig.suptitle("LIBERO-Plus 'Robot Initial States': bozulma = yalnızca 7 eklem açısının başlangıç değeri — gerçek dosyadan okundu (robot_init_spread.py)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(a.out, dpi=170); plt.close(fig)
    print("yazıldı:", a.out)


if __name__ == "__main__":
    main()
