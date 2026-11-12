"""Ders 19 — "Hangi init state'ler kırılgan, kâseler yakınsa mı?" sorusuna veri: LIBERO-Spatial'ın 50 init state'ini
torch/LIBERO/MuJoCo OLMADAN oku, nesne konumlarını çıkar, BDDL bölgeleriyle doğrula, sahne-başına özellik tablosu yaz.

.pruned_init dosyası torch.save(numpy_array) → zip içinde numpy pickle; burada kısıtlı bir Unpickler ile açılır (yalnızca
numpy ndarray/dtype kurucularına izin verilir). Dizi (50, 92): MuJoCo flatten state = [time(1) | qpos | qvel].
qpos = 7 kol eklemi + 2 gripper + nesne başına 7 (x y z | qw qx qy qz) — nesne sırası BDDL (:objects) sırası.
Bu sıra VARSAYIM değil, kontrol: her nesnenin (x, y)'si kendi BDDL init bölgesinin (2 cm × 2 cm) ±1.5 cm komşuluğunda mı?

Kullanım:
  python init_state_geometry.py --libero /yol/LIBERO                      # 10 görevin özeti
  python init_state_geometry.py --libero /yol/LIBERO --task 0 --csv init_features_task0.csv
Çıktı CSV (init_state anahtarı Ders 18'in episodes.csv'siyle birleşir):
  init_state, bowl1_x, bowl1_y, bowl2_x, bowl2_y, plate_x, plate_y, d_bowl1_bowl2, d_bowl1_plate, d_bowl2_plate,
  robot_q_dev (7 eklemin görev-ortalamasından sapma normu, rad), robot_q1..q7
"""
import argparse, codecs, csv, io, pathlib, pickle, re, zipfile
import numpy as np

ORDER_FILE = "libero/libero/benchmark/libero_suite_task_map.py"
N_ARM, N_GRIP = 7, 2


class _NumpyOnly(pickle.Unpickler):
    def find_class(self, mod, name):
        if (mod, name) in {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct")}:
            import numpy._core.multiarray as m  # numpy>=2; numpy<2'de numpy.core.multiarray
            return m._reconstruct
        if (mod, name) == ("numpy", "ndarray"):
            return np.ndarray
        if (mod, name) == ("numpy", "dtype"):
            return np.dtype
        if (mod, name) == ("_codecs", "encode"):
            return codecs.encode
        raise pickle.UnpicklingError(f"izin verilmeyen nesne: {mod}.{name}")


def load_init(path):
    z = zipfile.ZipFile(path)
    pkl = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    return np.asarray(_NumpyOnly(io.BytesIO(z.read(pkl))).load())


def official_order(root, suite):
    t = (root / ORDER_FILE).read_text(encoding="utf-8")
    return re.findall(r'"([^"]+)"', re.search(r'"%s":\s*\[(.*?)\]' % suite, t, re.S).group(1))


def parse_bddl(path):
    t = path.read_text(encoding="utf-8")
    regions = {}
    for m in re.finditer(r"\(\s*(\w+)\s*\n?\s*\(:target\s+(\w+)\)\s*(?:\(:ranges\s*\(\s*\(([^)]*)\)\s*\)\s*\))?", t):
        name, target, rng = m.groups()
        regions[f"{target}_{name}"] = (target, tuple(map(float, rng.split())) if rng else None)
    objs = []
    for line in re.search(r"\(:objects(.*?)\)\s*\(:obj_of_interest", t, re.S).group(1).strip().splitlines():
        if " - " in line:
            names, typ = line.strip().rsplit(" - ", 1)
            objs += names.split()
    init_txt = re.search(r"\(:init(.*?)\)\s*\(:goal", t, re.S).group(1)
    where = {o: reg for _, o, reg in re.findall(r"\((On|In)\s+(\S+)\s+(\S+)\)", init_txt)}
    g = re.search(r"\(On\s+(\S+)\s+plate_1\)", t.split("(:goal")[1])
    return dict(objects=objs, regions=regions, where=where, target=g.group(1) if g else None)


def object_xy(states, k):
    """k. nesnenin (x, y, z) sütunları: 1 + 9 + 7k."""
    base = 1 + N_ARM + N_GRIP + 7 * k
    return states[:, base:base + 3], states[:, base + 3:base + 7]


def check_regions(states, info, tol=0.015):
    """Masa bölgeli nesneler için: (x, y) bölge dikdörtgeninin tol komşuluğunda mı? → (nesne, bölge, uyan sayısı, n)"""
    out = []
    for k, o in enumerate(info["objects"]):
        reg = info["where"].get(o)
        if reg not in info["regions"] or info["regions"][reg][1] is None:
            continue
        x0, y0, x1, y1 = info["regions"][reg][1]
        xyz, _ = object_xy(states, k)
        ok = ((xyz[:, 0] >= x0 - tol) & (xyz[:, 0] <= x1 + tol) & (xyz[:, 1] >= y0 - tol) & (xyz[:, 1] <= y1 + tol)).sum()
        out.append((o, reg.replace("main_table_", ""), int(ok), len(states)))
    return out


def features(states, info):
    idx = {o: k for k, o in enumerate(info["objects"])}
    b1, _ = object_xy(states, idx["akita_black_bowl_1"]); b2, _ = object_xy(states, idx["akita_black_bowl_2"]); pl, _ = object_xy(states, idx["plate_1"])
    q = states[:, 1:1 + N_ARM]
    qdev = np.linalg.norm(q - q.mean(0), axis=1)
    rows = []
    for i in range(len(states)):
        rows.append(dict(init_state=i, bowl1_x=b1[i, 0], bowl1_y=b1[i, 1], bowl2_x=b2[i, 0], bowl2_y=b2[i, 1], plate_x=pl[i, 0], plate_y=pl[i, 1],
                         d_bowl1_bowl2=float(np.linalg.norm(b1[i, :2] - b2[i, :2])), d_bowl1_plate=float(np.linalg.norm(b1[i, :2] - pl[i, :2])),
                         d_bowl2_plate=float(np.linalg.norm(b2[i, :2] - pl[i, :2])), robot_q_dev=float(qdev[i]),
                         **{f"robot_q{j + 1}": float(q[i, j]) for j in range(N_ARM)}))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero", required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=None, help="tek görev; verilmezse 10 görevin özeti")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--n", type=int, default=None, help="yalnızca ilk n init state (Ders 18 pilotu: 10)")
    a = ap.parse_args()
    root = pathlib.Path(a.libero)
    names = official_order(root, a.suite)
    tasks = [a.task] if a.task is not None else range(len(names))
    for ti in tasks:
        name = names[ti]
        info = parse_bddl(root / "libero/libero/bddl_files" / a.suite / f"{name}.bddl")
        states = load_init(root / "libero/libero/init_files" / a.suite / f"{name}.pruned_init")
        raw = root / "libero/libero/init_files" / a.suite / f"{name}.init"
        n_all = load_init(raw).shape[0] if raw.exists() else "yok"
        if a.n:
            states = states[:a.n]
        qpos_len = 1 + N_ARM + N_GRIP + 7 * len(info["objects"])
        print(f"\n[{ti}] {name}\n  state {states.shape} (.init: {n_all} durum; .pruned_init: {len(states)}), nesneler {info['objects']}, "
              f"beklenen qpos uzunluğu {qpos_len} → kalan {states.shape[1] - qpos_len} değer qvel (hepsi 0: {np.allclose(states[:, qpos_len:], 0)})")
        for o, reg, ok, n in check_regions(states, info):
            print(f"  bölge kontrolü {o:<32} {reg:<26} {ok}/{n} init state bölgenin ±1.5 cm içinde")
        q = states[:, 1:1 + N_ARM]
        print(f"  robot eklem std (rad): {np.round(q.std(0), 3)}  | sapma normu medyan {np.median(np.linalg.norm(q - q.mean(0), axis=1)):.3f}, maks {np.linalg.norm(q - q.mean(0), axis=1).max():.3f}")
        for k, o in enumerate(info["objects"]):
            xyz, quat = object_xy(states, k)
            print(f"  {o:<32} x {xyz[:, 0].mean():+.3f}±{xyz[:, 0].std():.3f}  y {xyz[:, 1].mean():+.3f}±{xyz[:, 1].std():.3f}  z {xyz[:, 2].mean():.3f}  "
                  f"yayılım (maks-min) x {np.ptp(xyz[:, 0]) * 100:.1f} cm, y {np.ptp(xyz[:, 1]) * 100:.1f} cm  | quat sabit: {np.allclose(quat, quat[0], atol=1e-3)}")
        if "akita_black_bowl_1" in info["objects"] and "akita_black_bowl_2" in info["objects"]:
            rows = features(states, info)
            d12 = np.array([r["d_bowl1_bowl2"] for r in rows]); d1p = np.array([r["d_bowl1_plate"] for r in rows])
            print(f"  kâse–kâse mesafesi {d12.mean() * 100:.1f} cm (min {d12.min() * 100:.1f}, maks {d12.max() * 100:.1f}); hedef kâse–tabak {d1p.mean() * 100:.1f} cm "
                  f"(min {d1p.min() * 100:.1f}, maks {d1p.max() * 100:.1f})")
            if a.csv and a.task is not None:
                with open(a.csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
                    for r in rows:
                        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
                print(f"  → {a.csv} ({len(rows)} satır)")


if __name__ == "__main__":
    main()
