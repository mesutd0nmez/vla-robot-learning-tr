"""Ders 20 — "soldaki kâse" hangisi? LIBERO-Spatial'ın 10 sahnesinde iki siyah kâseyi ayıran tarifleri
(sol/sağ, ön/arka, tabağa yakın/uzak, robota yakın/uzak, yüksekte) 50 init state'in GERÇEK koordinatlarından hesaplar
ve her tarif için ÜÇ çerçevede hangi kâsenin kastedildiğini yazar:

  libero : LIBERO'nun kendi BDDL adlandırması (agentview kamerasının yerinde duran gözlemci; ayna yok)
           sağ = +y, sol = -y, ön = +x (gözlemciye yakın), arka = -x      [LIBERO-90/10 bölge adlarından doğrulandı]
  image  : modelin gördüğü piksel (obs[::-1, ::-1] = gerçek görüntünün yatay AYNASI): sol = küçük u, ön = büyük v
  robot  : robotun kendi gövdesi (taban x = -0.66, +x'e bakıyor): sol = +y, ön/yakın = tabana yakın

torch / LIBERO / MuJoCo GEREKMEZ: .pruned_init kısıtlı numpy unpickler ile açılır (Ders 19), kamera pozu
libero_tabletop_manipulation.py:188-197'den elle alınmıştır (fovy = MuJoCo varsayılanı 45 derece).

Kullanım:
  python scene_frames.py --libero ~/src/LIBERO                    # 10 sahne × tarif tablosu
  python scene_frames.py --libero ~/src/LIBERO --csv referents.csv
  python scene_frames.py --libero ~/src/LIBERO --conventions      # LIBERO-90/10'daki left/right/front/back bölgeleri
"""
import argparse, codecs, csv, glob, io, pathlib, pickle, re, zipfile
import numpy as np

SUITE = "libero_spatial"
ORDER_FILE = "libero/libero/benchmark/libero_suite_task_map.py"
CAM_POS = np.array([0.6586131746834771, 0.0, 1.6103500240372423])           # agentview (tabletop)
CAM_QUAT = (0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349)  # w x y z
ROBOT_BASE = np.array([-0.66, 0.0])   # mounted_panda.py:45 → -0.16 - table_length/2, table_length = 1.0
MIN_M, MIN_PX = 0.05, 20.0            # "belirsiz değil" eşiği: 50 sahnenin hepsinde en az 5 cm / 20 piksel fark


class _NumpyOnly(pickle.Unpickler):
    def find_class(self, mod, name):
        if name == "_reconstruct" and mod in ("numpy.core.multiarray", "numpy._core.multiarray"):
            try:
                import numpy._core.multiarray as m
            except ImportError:
                import numpy.core.multiarray as m
            return m._reconstruct
        if (mod, name) == ("numpy", "ndarray"): return np.ndarray
        if (mod, name) == ("numpy", "dtype"): return np.dtype
        if (mod, name) == ("_codecs", "encode"): return codecs.encode
        raise pickle.UnpicklingError(f"izin verilmeyen nesne: {mod}.{name}")


def load_init(path):
    z = zipfile.ZipFile(path)
    pkl = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    return np.asarray(_NumpyOnly(io.BytesIO(z.read(pkl))).load())


def official_order(root, suite=SUITE):
    t = (root / ORDER_FILE).read_text(encoding="utf-8")
    return re.findall(r'"([^"]+)"', re.search(r'"%s":\s*\[(.*?)\]' % suite, t, re.S).group(1))


def parse_bddl(path):
    t = pathlib.Path(path).read_text(encoding="utf-8")
    regions = {}
    for m in re.finditer(r"\(\s*(\w+)\s*\n?\s*\(:target\s+(\w+)\)\s*(?:\(:ranges\s*\(\s*\(([^)]*)\)\s*\)\s*\))?", t):
        name, target, rng = m.groups()
        regions[f"{target}_{name}"] = tuple(map(float, rng.split())) if rng else None
    objs = []
    for line in re.search(r"\(:objects(.*?)\)\s*\(:obj_of_interest", t, re.S).group(1).strip().splitlines():
        if " - " in line:
            objs += line.strip().rsplit(" - ", 1)[0].split()
    init_txt = re.search(r"\(:init(.*?)\)\s*\(:goal", t, re.S).group(1)
    where = {o: reg for _, o, reg in re.findall(r"\((On|In)\s+(\S+)\s+(\S+)\)", init_txt)}
    goal = re.sub(r"\s+", " ", t.split("(:goal")[1]).strip()
    return dict(objects=objs, regions=regions, where=where, goal=goal)


def cam_rotation(q=CAM_QUAT):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def project(p, W=256, fovy=45.0, mirror=True):
    """Dünya noktası → piksel (u, v). MuJoCo kamerası -z'ye bakar, +x sağ, +y yukarı.
    mirror=True: modelin gördüğü görüntü (ham görüntü dikey ters gelir; [::-1, ::-1] onu yatay aynaya çevirir)."""
    pc = (np.asarray(p) - CAM_POS) @ cam_rotation()
    f = 0.5 * W / np.tan(np.radians(fovy) / 2)
    u = W / 2 + f * pc[..., 0] / (-pc[..., 2])
    v = W / 2 - f * pc[..., 1] / (-pc[..., 2])
    return np.stack([W - u if mirror else u, v], -1)


def positions(states, objects):
    """{nesne: (50, 3)} — qpos = 1 zaman + 7 kol + 2 gripper + nesne başına 7 (x y z qw qx qy qz), BDDL :objects sırası."""
    return {o: states[:, 10 + 7 * k: 13 + 7 * k] for k, o in enumerate(objects)}


def _pick(score1, score2, thr, smaller_wins):
    """İki kâsenin skorlarından (50,) kazananı ve 50 sahnedeki EN KÜÇÜK farkı döndür; işaret değişiyorsa 'belirsiz'."""
    d = score2 - score1 if smaller_wins else score1 - score2     # > 0 → bowl_1 kazanır
    if (d > 0).all():  who = "bowl_1"
    elif (d < 0).all(): who = "bowl_2"
    else: return "belirsiz", 0.0, False
    margin = float(np.abs(d).min())
    return who, margin, margin >= thr


def descriptors(pos):
    b1, b2, plate = pos["akita_black_bowl_1"], pos["akita_black_bowl_2"], pos["plate_1"]
    uv1, uv2 = project(b1), project(b2)
    dp1, dp2 = np.linalg.norm(b1[:, :2] - plate[:, :2], axis=1), np.linalg.norm(b2[:, :2] - plate[:, :2], axis=1)
    dr1, dr2 = np.linalg.norm(b1[:, :2] - ROBOT_BASE, axis=1), np.linalg.norm(b2[:, :2] - ROBOT_BASE, axis=1)
    rows = []
    def add(desc, frame, s1, s2, thr, smaller, unit):
        who, m, ok = _pick(s1, s2, thr, smaller)
        rows.append(dict(descriptor=desc, frame=frame, referent=who, min_margin=round(m * (100 if unit == "cm" else 1), 1), unit=unit, unambiguous=ok))
    # sol / sağ
    add("left", "libero", b1[:, 1], b2[:, 1], MIN_M, True, "cm")        # sol = küçük y
    add("left", "image", uv1[:, 0], uv2[:, 0], MIN_PX, True, "px")     # sol = küçük u (aynalı görüntü)
    add("left", "robot", b1[:, 1], b2[:, 1], MIN_M, False, "cm")       # robotun solu = büyük y
    # ön (arka = öteki kâse)
    add("front", "libero", b1[:, 0], b2[:, 0], MIN_M, False, "cm")     # ön = büyük x (gözlemciye yakın)
    add("front", "image", uv1[:, 1], uv2[:, 1], MIN_PX, False, "px")   # ön = görüntüde aşağıda (büyük v)
    add("front", "robot", dr1, dr2, MIN_M, True, "cm")                 # robotun önündeki/yakınındaki = tabana yakın
    # çerçeveden bağımsız (referans nesne açık)
    add("near_plate", "none", dp1, dp2, MIN_M, True, "cm")
    add("near_robot", "none", dr1, dr2, MIN_M, True, "cm")
    add("higher", "none", b1[:, 2], b2[:, 2], MIN_M, False, "cm")
    return rows


def analyze(root):
    out = []
    for i, name in enumerate(official_order(root)):
        info = parse_bddl(root / "libero/libero/bddl_files" / SUITE / (name + ".bddl"))
        st = load_init(root / "libero/libero/init_files" / SUITE / (name + ".pruned_init"))
        pos = positions(st, info["objects"])
        # sıra kontrolü: masa bölgeli nesneler kendi bölgesinin ±1.5 cm içinde mi (Ders 19'daki doğrulama)
        for o, xyz in pos.items():
            r = info["regions"].get(info["where"].get(o))
            if r:
                x0, y0, x1, y1 = r
                ok = ((xyz[:, 0] > x0 - .015) & (xyz[:, 0] < x1 + .015) & (xyz[:, 1] > y0 - .015) & (xyz[:, 1] < y1 + .015)).all()
                assert ok, f"görev {i}: {o} bölgesinde değil — nesne sırası varsayımı bozuk"
        for d in descriptors(pos):
            out.append(dict(task_id=i, bowl_1_at=info["where"]["akita_black_bowl_1"].replace("main_table_", ""),
                            bowl_2_at=info["where"]["akita_black_bowl_2"].replace("main_table_", ""), **d))
    return out


def conventions(root):
    """LIBERO-90/10 BDDL'lerinde adı left/right/front/back içeren bölgelerin merkezleri: çerçeve sözleşmesinin kanıtı."""
    seen = {}
    for p in sorted(glob.glob(str(root / "libero/libero/bddl_files/libero_90/*.bddl")) + glob.glob(str(root / "libero/libero/bddl_files/libero_10/*.bddl"))):
        scene = re.match(r"([A-Z_]+SCENE\d+)", pathlib.Path(p).name).group(1)
        for k, r in parse_bddl(p)["regions"].items():
            m = re.search(r"(left|right|front|back)", k)
            if r and m and "contain" not in k:
                seen[(scene, k)] = (m.group(1), (r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
    # aynı sahnede left/right ve front/back çiftlerini karşılaştır
    pairs = []
    for (scene, k), (word, x, y) in seen.items():
        other = {"left": "right", "front": "back"}.get(word)
        k2 = k.replace(word, other) if other else None
        if k2 and (scene, k2) in seen:
            _, x2, y2 = seen[(scene, k2)]
            pairs.append((scene, k, k2, word, x, y, x2, y2))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero", required=True); ap.add_argument("--csv"); ap.add_argument("--conventions", action="store_true")
    a = ap.parse_args()
    root = pathlib.Path(a.libero).expanduser()
    if a.conventions:
        pr = conventions(root)
        print(f"{'sahne':24s} {'çift':46s} Δx (cm)  Δy (cm)   okuma")
        ok_lr = ok_fb = n_lr = n_fb = 0
        for scene, k, k2, word, x, y, x2, y2 in pr:
            dx, dy = (x - x2) * 100, (y - y2) * 100
            if word == "left":
                n_lr += 1; ok_lr += dy < 0; note = "left = daha küçük y" if dy < 0 else "TERS"
            else:
                n_fb += 1; ok_fb += dx > 0; note = "front = daha büyük x" if dx > 0 else "TERS"
            label = k.split("table_")[-1] + " | " + k2.split("table_")[-1]
            print(f"{scene:24s} {label[:46]:46s} {dx:7.1f}  {dy:7.1f}   {note}")
        print(f"\nleft/right çiftleri: {ok_lr}/{n_lr} 'left = -y';  front/back çiftleri: {ok_fb}/{n_fb} 'front = +x'")
        return
    rows = analyze(root)
    cur = None
    for r in rows:
        if r["task_id"] != cur:
            cur = r["task_id"]; print(f"\ngörev {cur}: bowl_1 @ {r['bowl_1_at']} | bowl_2 @ {r['bowl_2_at']}")
        flag = "" if r["unambiguous"] else "   (belirsiz: eşik altı)"
        print(f"   {r['descriptor']:11s} {r['frame']:7s} → {r['referent']:8s} min fark {r['min_margin']:6.1f} {r['unit']}{flag}")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"\nyazıldı: {a.csv} ({len(rows)} satır)")


if __name__ == "__main__":
    main()
