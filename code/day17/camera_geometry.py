"""Ders 17 — LIBERO-Plus kamera bozulmasını GERÇEK kodla hesaplar ve çizer (model/asset/MuJoCo gerekmez).

`problems/libero_tabletop_manipulation.py` içindeki üç saf fonksiyonu (rotate_around_y, rotate_around_z,
scale_distance_from_pivot) dosyadan `ast` ile çekip çalıştırır (dosyanın geri kalanı robosuite ister, onu
import etmeyiz). Sonra `_setup_camera` ile aynı sırayı (dikey → yatay → mesafe → uç yaw/pitch) Spatial'daki
376 "Camera Viewpoints" görevinin adlarındaki parametrelere uygular.

Kullanım:  python camera_geometry.py <LIBERO-plus kökü> <çıktı.png> [--suite libero_spatial] [--csv out.csv]
"""
import argparse, ast, csv, json, pathlib
import numpy as np
from scipy.spatial.transform import Rotation  # fonksiyonlar bunu kullanıyor

POS_AV = [0.6586131746834771, 0.0, 1.6103500240372423]          # _setup_camera: varsayılan agentview
QUAT_AV = [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]  # (w,x,y,z)
PIVOT = np.array([0, 0, 0.8])


def load_funcs(root):
    src = (pathlib.Path(root) / "libero/libero/envs/problems/libero_tabletop_manipulation.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    ns = {"np": np, "Rotation": Rotation}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("scale_distance_from_pivot", "rotate_around_y", "rotate_around_z"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<libero-plus>", "exec"), ns)
    return ns["rotate_around_y"], ns["rotate_around_z"], ns["scale_distance_from_pivot"]


def setup_camera(fns, h, v, s, er, ev):
    """problems/*.py `_setup_camera` satır 319-345 ile birebir aynı sıra."""
    rot_y, rot_z, scale = fns
    pos, quat = POS_AV, QUAT_AV
    if int(v) != 0:
        r = rot_y(original_quat=quat, original_pos=pos, degrees=int(v)); pos, quat = r["new_pos"], r["new_quat"]
    r = rot_z(original_quat=quat, original_pos=pos, degrees=int(h)); pos, quat = r["new_pos"], r["new_quat"]
    if float(s) != 1.0:
        r = scale(original_quat=quat, original_pos=pos, scale_factor=float(s)); pos, quat = r["new_pos"], r["new_quat"]
    if int(er) != 0:
        quat = rot_z(original_quat=quat, degrees=int(er))["new_quat"]
    if int(ev) != 0:
        quat = rot_y(original_quat=quat, degrees=int(ev))["new_quat"]
    return np.array(pos), np.array(quat)


def forward_axis(quat_wxyz):
    """MuJoCo kamerası kendi -z eksenine bakar; dünya koordinatında bakış vektörü."""
    w, x, y, z = quat_wxyz
    return Rotation.from_quat([x, y, z, w]).apply([0, 0, -1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--suite", default="libero_spatial"); ap.add_argument("--csv")
    a = ap.parse_args()
    fns = load_funcs(a.root)
    tasks = json.load(open(pathlib.Path(a.root) / "libero/libero/benchmark/task_classification.json", encoding="utf-8"))[a.suite]
    rows = []
    for t in tasks:
        if t["category"] != "Camera Viewpoints":
            continue
        view, rest = t["name"].split("_view_")[1].split("_initstate_")
        h, v, s, er, ev = view.split("_"); s = int(s) / 100
        pos, quat = setup_camera(fns, h, v, s, er, ev)
        fwd = forward_axis(quat)
        # bakış doğrultusunun masa düzlemini (z=0.8 pivot yüksekliği) kestiği nokta
        tpar = (0.8 - pos[2]) / fwd[2] if fwd[2] < 0 else np.nan
        hit = pos + tpar * fwd
        kind = "rotation" if (int(h) or int(v)) else ("zoom" if s != 1.0 else "endpoint")
        rows.append(dict(name=t["name"], h=int(h), v=int(v), s=s, er=int(er), ev=int(ev), kind=kind,
                         x=pos[0], y=pos[1], z=pos[2], dist=float(np.linalg.norm(pos - PIVOT)),
                         hit_x=hit[0], hit_y=hit[1], difficulty=t["difficulty_level"]))
    base_pos, base_quat = setup_camera(fns, 0, 0, 1.0, 0, 0)
    assert np.allclose(base_pos, POS_AV) and np.allclose(base_quat, QUAT_AV), "varsayılan parametrelerle kamera değişmemeli"
    base_fwd = forward_axis(base_quat); base_hit = base_pos + (0.8 - base_pos[2]) / base_fwd[2] * base_fwd
    kinds = {k: sum(r["kind"] == k for r in rows) for k in ("rotation", "zoom", "endpoint")}
    print(f"{a.suite}: {len(rows)} kamera görevi → {kinds}")
    print(f"varsayılan kamera pos={np.round(base_pos,3)} pivot'a uzaklık={np.linalg.norm(base_pos-PIVOT):.3f} m, bakış masa kesişimi={np.round(base_hit,3)}")
    d = np.array([r["dist"] for r in rows]); print(f"pivot'a uzaklık: min {d.min():.2f} max {d.max():.2f} m (zoom görevlerinde ×1.15..×2.0)")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    col = {"rotation": "#ffd166", "zoom": "#cfe8ff", "endpoint": "#d4f5d4"}; edge = {"rotation": "#b07d00", "zoom": "#3a6ea5", "endpoint": "#2e7d32"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    # (a) kuşbakışı
    ax = axes[0]
    ax.add_patch(plt.Rectangle((-0.5, -0.6), 1.0, 1.2, facecolor="#f3e6d0", edgecolor="#a0875a", label="masa 1.0×1.2 m"))
    ax.plot(-0.66, 0, "s", color="#555", ms=10, label="robot tabanı (yaklaşık)")
    for k in ("rotation", "zoom", "endpoint"):
        xs = [r["x"] for r in rows if r["kind"] == k]; ys = [r["y"] for r in rows if r["kind"] == k]
        ax.scatter(xs, ys, s=22, color=col[k], edgecolor=edge[k], linewidth=0.6, label=f"{k} ({kinds[k]})", zorder=3)
    ax.plot(base_pos[0], base_pos[1], "*", color="#d62828", ms=16, zorder=5, label="varsayılan agentview")
    ax.plot(0, 0, "+", color="#000", ms=10, mew=2, label="pivot (0,0,0.8)")
    ax.set_aspect("equal"); ax.set_xlim(-1.3, 1.8); ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_title("(a) Kamera konumları, kuşbakışı")
    ax.legend(fontsize=7.5, loc="upper left")
    # (b) yan görünüş: pivot'a yatay uzaklık vs yükseklik
    ax = axes[1]
    for k in ("rotation", "zoom", "endpoint"):
        rr = [np.hypot(r["x"], r["y"]) for r in rows if r["kind"] == k]; zz = [r["z"] for r in rows if r["kind"] == k]
        ax.scatter(rr, zz, s=22, color=col[k], edgecolor=edge[k], linewidth=0.6, zorder=3)
    ax.plot(np.hypot(*base_pos[:2]), base_pos[2], "*", color="#d62828", ms=16, zorder=5)
    ax.axhline(0.9, color="#a0875a", lw=1, ls="--"); ax.text(1.45, 0.92, "masa üstü z=0.90", fontsize=8, color="#a0875a", ha="right")
    ax.plot(0, 0.8, "+", color="#000", ms=10, mew=2)
    ax.set_xlabel("pivot'a yatay uzaklık (m)"); ax.set_ylabel("z (m)"); ax.set_title("(b) Yan görünüş: zoom = pivot'tan uzaklaşma")
    ax.set_xlim(-0.05, 1.5); ax.set_ylim(0.7, 2.5)
    # (c) bakış doğrultusunun masa düzlemini kestiği nokta
    ax = axes[2]
    ax.add_patch(plt.Rectangle((-0.5, -0.6), 1.0, 1.2, facecolor="#f3e6d0", edgecolor="#a0875a"))
    for k in ("rotation", "zoom", "endpoint"):
        xs = [r["hit_x"] for r in rows if r["kind"] == k]; ys = [r["hit_y"] for r in rows if r["kind"] == k]
        ax.scatter(xs, ys, s=22, color=col[k], edgecolor=edge[k], linewidth=0.6, zorder=3)
    ax.plot(base_hit[0], base_hit[1], "*", color="#d62828", ms=16, zorder=5)
    ax.set_aspect("equal"); ax.set_xlim(-0.7, 0.7); ax.set_ylim(-0.8, 0.8)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_title("(c) Bakış ekseninin masa düzlemini kestiği nokta")
    fig.suptitle(f"LIBERO-Plus 'Camera Viewpoints': {a.suite} {len(rows)} görevin kamera pozu — LIBERO-plus kodundaki gerçek "
                 "rotate_around_y/z + scale_distance_from_pivot ile hesaplandı (camera_geometry.py)", fontsize=10.5)
    fig.text(0.5, 0.01, "rotation: yatay ±75° (+ dikey 15°) pivot (0,0,0.8) etrafında · zoom: pivot'tan ×1.15–2.0 uzaklaşma · endpoint: konum sabit, yalnızca bakış ±10° yaw/pitch (kesişim noktası kayar)",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95)); fig.savefig(a.out, dpi=170); plt.close(fig)
    print("yazıldı:", a.out)


if __name__ == "__main__":
    main()
