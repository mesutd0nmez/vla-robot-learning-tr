# day21 — Webots: ilk sahne, extern Python controller, kamera kaydı, FK kontrolü

Ders 21'in dosyaları. Webots **R2025a** (macOS dmg; Apple Silicon + Intel tek paket) ve conda `vla` ortamı (Python 3.10) varsayılır.
Bu oturumda Webots'un kendisi çalıştırılamadı (bulut kısıtı); `check_webots_env.py` (Linux tarball'a karşı), `ur10e_fk.py`,
`plot_day21.py` ve controller'ın mantığı (sahte `controller` modülüyle kuru koşu) gerçekten çalıştırıldı.

```bash
# 1) Webots'u kur (Terminal'den indirmek Gatekeeper etiketini önler)
curl -L -O https://github.com/cyberbotics/webots/releases/download/R2025a/webots-R2025a.dmg
open webots-R2025a.dmg && cp -r /Volumes/Webots/Webots.app /Applications/
# 2) ortam
conda activate vla
export WEBOTS_HOME=/Applications/Webots.app
python check_webots_env.py                      # import controller : OK görmelisin
# 3) Webots'ta File > Open World... > code/day21/worlds/ders21_masa.wbt, ▶ ile oynat (robot controller bekler)
# 4) ayrı terminalde (vla aktif):
$WEBOTS_HOME/Contents/MacOS/webots-controller controllers/ders21_kol/ders21_kol.py --out out
#    launcher arm Mac'te hata verirse (R2025a bilinen hata, PR #6806):
PYTHONPATH=$WEBOTS_HOME/Contents/lib/controller/python python controllers/ders21_kol/ders21_kol.py --out out
# 5) Webots'suz kontroller
python ur10e_fk.py                              # 4 duruşun tool konumu, masa üstü mü
python ur10e_fk.py --solve                      # hedefler için eklem açıları (scipy)
python plot_day21.py --out 05-decision.png      # karar matrisi
```

İndirmek için: `curl -O https://raw.githubusercontent.com/mesutd0nmez/vla-robot-learning-tr/main/code/day21/<dosya>`
(world ve controller için `worlds/ders21_masa.wbt`, `controllers/ders21_kol/ders21_kol.py`).


## check_webots_env.py

`check_webots_env.py`

```python
"""Ders 21 — Webots + conda `vla` bağlantı kontrolü (Webots'u açmadan çalışır).

Kontroller: WEBOTS_HOME doğru mu, Python API klasörü var mı, libController yüklenebiliyor mu,
launcher (webots-controller) nerede, python3 hangi ortamdan geliyor.
Çalıştır (vla aktif):  WEBOTS_HOME=/Applications/Webots.app python check_webots_env.py
"""
import os
import platform
import sys

home = os.environ.get("WEBOTS_HOME")
print("python      :", sys.executable, sys.version.split()[0], "| arch:", platform.machine())
print("WEBOTS_HOME :", home)
if not home:
    print("  -> export WEBOTS_HOME=/Applications/Webots.app   (macOS)  |  .../webots (Linux tarball)")
    sys.exit(1)
mac = sys.platform == "darwin"
pyapi = os.path.join(home, "Contents", "lib", "controller", "python") if mac else os.path.join(home, "lib", "controller", "python")
lib = os.path.join(home, "Contents", "lib", "controller", "libController.dylib") if mac else os.path.join(home, "lib", "controller", "libController.so")
launcher = os.path.join(home, "Contents", "MacOS", "webots-controller") if mac else os.path.join(home, "webots-controller")
for label, path in [("Python API", pyapi), ("libController", lib), ("launcher", launcher)]:
    print(f"{label:12s}: {'VAR ' if os.path.exists(path) else 'YOK '} {path}")
if platform.machine() == "arm64" and mac:
    print("Apple Silicon: R2025a'da launcher'ın arm Mac'te çalışmama hatası var (PR #6806, R2025b'de düzeltildi).")
    print("  Launcher hata verirse: PYTHONPATH=" + pyapi + " python ders21_kol.py  (Webots açıkken)")
sys.path.insert(0, pyapi)
try:
    import controller  # noqa: F401
    from controller import Supervisor, Camera, Motor  # noqa: F401
    print("import controller : OK  (ctypes ile", os.path.basename(lib), "yüklendi; numpy/torch ortamın aynen kullanılabilir)")
except Exception as e:  # pragma: no cover
    print("import controller : HATA ->", repr(e))
    print("  Olası sebep: WEBOTS_HOME yanlış, Rosetta altında açılmış Webots (x86_64 dylib), ya da Python < 3.8")
    sys.exit(2)
print("Hepsi tamam. Şimdi Webots'ta ders21_masa.wbt'yi aç, oynat, sonra ders21_kol.py'yi başlat.")
```

## ders21_kol.py

`controllers/ders21_kol/ders21_kol.py`

```python
"""Ders 21 — Webots'ta ilk kol hareketi + kamera kaydı + mekânsal ilişki tablosu.

UR10e'yi (supervisor TRUE, controller <extern>) dört ara duruştan geçirir, her duruşta
bilek kamerasını ve ana 3B görünümü (agentview benzeri) PNG/JPG olarak kaydeder,
nesnelerin dünya koordinatlarını Supervisor API ile okur ve robota göre
sol/sağ/ön/arka ilişkilerini CSV'ye yazar.

Çalıştır (Webots'ta ders21_masa.wbt açık ve oynatılıyorken, ayrı terminalde, vla ortamı aktif):
  macOS : export WEBOTS_HOME=/Applications/Webots.app
          $WEBOTS_HOME/Contents/MacOS/webots-controller code/day21/controllers/ders21_kol/ders21_kol.py --out out
  Linux : $WEBOTS_HOME/webots-controller code/day21/controllers/ders21_kol/ders21_kol.py --out out
Launcher çalışmazsa (bkz. README "launcher olmadan"):
  PYTHONPATH=$WEBOTS_HOME/Contents/lib/controller/python python ders21_kol.py --out out
"""
import argparse
import csv
import math
import os
import sys

from controller import Supervisor  # Webots'un ctypes tabanlı Python API'si (WEBOTS_HOME gerekir)

p = argparse.ArgumentParser()
p.add_argument("--out", default="out", help="PNG/JPG/CSV'lerin yazılacağı klasör")
p.add_argument("--settle", type=int, default=60, help="her duruşta beklenen adım sayısı")
p.add_argument("--quit", action="store_true", help="bitince Webots'u kapat (CI / headless koşular için)")
args, _ = p.parse_known_args()
os.makedirs(args.out, exist_ok=True)

robot = Supervisor()
dt = int(robot.getBasicTimeStep())          # world'deki basicTimeStep (16 ms)

# --- aktüatörler: UR10e'nin 6 eklemi (PROTO'daki RotationalMotor adları) ---
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
motors = [robot.getDevice(j) for j in JOINTS]
sensors = [robot.getDevice(j + "_sensor") for j in JOINTS]
for s in sensors:
    s.enable(dt)
for m in motors:
    m.setVelocity(1.0)                       # rad/s; PROTO'da maxVelocity 3.14 / 6.28

# --- sensör: toolSlot'a koyduğumuz bilek kamerası ---
cam = robot.getDevice("wrist_camera")
cam.enable(dt)

# --- Supervisor: sahnedeki nesneler (world'deki DEF adları) ---
DEFS = ["BOWL", "PLATE", "STOVE", "CEREAL", "APPLE", "CAN"]
nodes = {d: robot.getFromDef(d) for d in DEFS}
robot_node = robot.getSelf()

# Dört duruş (rad): [pan, lift, elbow, w1, w2, w3]. Kol masanın üstüne doğru eğilir.
POSES = {
    "00_home":   [0.000, -1.800, 1.200, -1.570, 0.000, 0.000],
    "01_bowl":   [0.582, -0.542, -1.637, -2.533, -1.571, 0.000],
    "02_center": [0.194, -0.868, -1.055, -2.789, -1.571, 0.000],
    "03_plate":  [-0.068, -0.568, -1.593, -2.552, -1.571, 0.000],
}


def world_to_robot(pw, R, t):
    """Dünya noktası -> robot tabanı çerçevesi (R: 3x3 satır-major, t: taban konumu)."""
    d = [pw[i] - t[i] for i in range(3)]
    # R dünya<-robot; robot koordinatı = R^T d
    return [sum(R[3 * k + i] * d[k] for k in range(3)) for i in range(3)]


rows = []
for name, q in POSES.items():
    for m, qi in zip(motors, q):
        m.setPosition(qi)
    for _ in range(args.settle):          # kolun hedefe gelmesini bekle (synchronization TRUE)
        if robot.step(dt) == -1:
            sys.exit(0)
    # kamera kareleri
    cam.saveImage(os.path.join(args.out, f"wrist_{name}.png"), 100)
    robot.exportImage(os.path.join(args.out, f"view_{name}.jpg"), 95)
    # eklem okumaları
    qs = [s.getValue() for s in sensors]
    err = max(abs(a - b) for a, b in zip(qs, q))
    print(f"[{name}] t={robot.getTime():6.2f}s  max|q-q*|={err:.3f} rad  "
          + " ".join(f"{v:+.2f}" for v in qs))
    rows.append({"pose": name, "t": round(robot.getTime(), 3), "max_err_rad": round(err, 4),
                 **{j: round(v, 4) for j, v in zip(JOINTS, qs)}})

with open(os.path.join(args.out, "joints.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# --- mekânsal ilişkiler: Supervisor nesne konumlarını okur ---
# UR10e'nin "önü" taban çerçevesinde −x'tir (shoulder_lift = −90° kolu −x'e uzatır); world'de taban
# z ekseninde −90° döndürüldüğü için önü dünya +y, solu dünya −x. Aşağıda ileri = −x_r, sol = −y_r.
t = robot_node.getPosition()
R = robot_node.getOrientation()           # 9 eleman, satır-major
rel_rows = []
for d, n in nodes.items():
    pw = n.getPosition()
    pr = world_to_robot(pw, R, t)          # taban çerçevesi
    fwd, left = -pr[0], -pr[1]             # ileri (masaya doğru) ve sol
    rel_rows.append({"def": d, "x_w": round(pw[0], 3), "y_w": round(pw[1], 3), "z_w": round(pw[2], 3),
                     "ileri_r": round(fwd, 3), "sol_r": round(left, 3),
                     "sol_sag": "sol" if left > 0.03 else ("sag" if left < -0.03 else "orta"),
                     "on_arka": "yakin" if fwd < 0.75 else "uzak"})
    print(f"{d:7s} dünya=({pw[0]:+.2f},{pw[1]:+.2f},{pw[2]:+.2f})  robot(ileri,sol)=({fwd:+.2f},{left:+.2f})"
          f"  -> {rel_rows[-1]['sol_sag']}, {rel_rows[-1]['on_arka']}")

# ikili ilişkiler (robot çerçevesinde): A, B'nin solunda / sağında / önünde / arkasında mı?
pairs = []
for a in rel_rows:
    for b in rel_rows:
        if a["def"] == b["def"]:
            continue
        d_sol = a["sol_r"] - b["sol_r"]        # >0: A, B'nin solunda (robota göre)
        d_ileri = a["ileri_r"] - b["ileri_r"]  # >0: A, B'nin gerisinde (robottan daha uzak)
        pairs.append({"a": a["def"], "b": b["def"],
                      "d_sol": round(d_sol, 3), "d_ileri": round(d_ileri, 3),
                      "iliski_tr": ("solunda" if d_sol > 0.05 else "sağında" if d_sol < -0.05 else "aynı hizada")
                                   + " / " + ("gerisinde" if d_ileri > 0.05 else "önünde" if d_ileri < -0.05 else "aynı derinlikte")})
for fname, data in [("objects.csv", rel_rows), ("pairs.csv", pairs)]:
    with open(os.path.join(args.out, fname), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)
print(f"Yazıldı: {args.out}/wrist_*.png, view_*.jpg, joints.csv, objects.csv, pairs.csv")
print(f"basicTimeStep={dt} ms; simülasyon süresi={robot.getTime():.2f} s")
if args.quit:
    robot.simulationQuit(0)
robot.step(dt)
```

## ur10e_fk.py

`ur10e_fk.py`

```python
"""Ders 21 — UR10e ileri kinematiği, Webots PROTO'sundan okunan zincirle (Webots gerekmez).

UR10e.proto'daki HingeJoint anchor/axis ve endPoint translation değerleri birebir kullanılır:
  J1 pan   : axis z, (0, 0, 0.181)      J4 wrist_1 : axis y, (0, 0, 0.571)
  J2 lift  : axis y, (0, 0.176, 0)      J5 wrist_2 : axis z, (0, 0.135, 0)
  J3 elbow : axis y, (0, -0.137, 0.613) J6 wrist_3 : axis y, (0, 0, 0.12)
  toolSlot : (0, 0.12, 0) — kamera burada
Robot tabanı ders21_masa.wbt'deki gibi: translation (0, -0.65, 0.74), rotation z 90°.

Çıktı: ders21_kol.py'deki dört duruş için tool (kamera) konumu, masa üstünde mi kontrolü,
ve --plot ile üstten/yandan görünüm (lectures/images/lecture-21/04-scene-fk.png).
Çalıştır:  python ur10e_fk.py [--plot cikti.png]
"""
import argparse
import math

import numpy as np

CHAIN = [  # (eksen, bağlantı ötelemesi) — proto sırasıyla
    ("z", (0, 0, 0.181)),
    ("y", (0, 0.176, 0)),
    ("y", (0, -0.137, 0.613)),
    ("y", (0, 0, 0.571)),
    ("z", (0, 0.135, 0)),
    ("y", (0, 0, 0.12)),
]
TOOL = (0, 0.12, 0)
BASE_T = (0.0, -0.65, 0.74)
BASE_YAW = -math.pi / 2

POSES = {
    "00_home":   [0.000, -1.800, 1.200, -1.570, 0.000, 0.000],
    "01_bowl":   [0.582, -0.542, -1.637, -2.533, -1.571, 0.000],
    "02_center": [0.194, -0.868, -1.055, -2.789, -1.571, 0.000],
    "03_plate":  [-0.068, -0.568, -1.593, -2.552, -1.571, 0.000],
}
# ders21_masa.wbt'deki nesneler (DEF, dünya konumu, çizim yarıçapı m, renk)
OBJECTS = [
    ("BOWL",   (-0.22, 0.00, 0.74), 0.09, "#222222"),
    ("PLATE",  (0.22, 0.02, 0.74), 0.11, "#dddddd"),
    ("STOVE",  (0.00, 0.50, 0.74), 0.19, "#888888"),
    ("CEREAL", (-0.35, 0.45, 0.74), 0.08, "#e0a060"),
    ("APPLE",  (0.05, 0.27, 0.79), 0.05, "#c0392b"),
    ("CAN",    (0.36, 0.40, 0.80), 0.035, "#2980b9"),
]
TABLE = (1.0, 1.8, 0.74)  # x, y, yükseklik


def rot(axis, q):
    c, s = math.cos(q), math.sin(q)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def fk(q):
    """Her eklem anchor'ının ve tool'un dünya konumu (7+1 nokta)."""
    R = rot("z", BASE_YAW)
    t = np.array(BASE_T, float)
    pts = [t.copy()]
    for (axis, trans), qi in zip(CHAIN, q):
        t = t + R @ np.array(trans, float)   # Webots: endPoint translation == anchor
        R = R @ rot(axis, qi)                 # eklem dönüşü anchor etrafında
        pts.append(t.copy())
    tool = t + R @ np.array(TOOL, float)
    pts.append(tool)
    return np.array(pts), R


def on_table(p):
    return abs(p[0]) <= TABLE[0] / 2 and abs(p[1]) <= TABLE[1] / 2 and p[2] > TABLE[2]


TARGETS = {  # --solve: kameranın gelmesi istenen nokta (nesnenin 26 cm üstü), kamera aşağı bakacak
    "01_bowl":   (-0.22, 0.00, 1.00),
    "02_center": (0.00, 0.25, 1.00),
    "03_plate":  (0.22, 0.02, 1.00),
}


def solve(target, q0):
    """Basit sayısal IK (scipy): tool hedefte + toolSlot'un +y ekseni (kamera bakışı) dünya −z'ye baksın."""
    from scipy.optimize import minimize

    def cost(q):
        pts, R = fk(q)
        pos_err = np.sum((pts[-1] - np.array(target)) ** 2)
        look = R @ np.array([0, 1, 0])           # toolSlot +y = kamera bakış yönü (Camera rotation z 90°)
        ori_err = np.sum((look - np.array([0, 0, -1])) ** 2)
        below = np.sum(np.clip(TABLE[2] + 0.05 - pts[1:, 2], 0, None) ** 2)  # eklemler masanın altına inmesin
        return 100 * pos_err + 10 * ori_err + 1000 * below + 0.01 * np.sum((q - np.array(q0)) ** 2)

    rng = np.random.default_rng(21)
    best = None
    starts = [np.array(q0)] + [np.array(q0) + rng.uniform(-0.8, 0.8, 6) for _ in range(40)]
    for s in starts:
        r = minimize(cost, s, method="L-BFGS-B", bounds=[(-3.14, 3.14)] * 6)
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", default=None)
    ap.add_argument("--solve", action="store_true", help="TARGETS için eklem açılarını sayısal IK ile bul ve yazdır")
    a = ap.parse_args()
    if a.solve:
        q0 = [0.0, -1.2, 1.4, -1.8, -1.57, 0.0]
        for name, tgt in TARGETS.items():
            q, f = solve(tgt, q0)
            pts, R = fk(q)
            print(f'    "{name}": [' + ", ".join(f"{v:.3f}" for v in q) + f"],  # tool={np.round(pts[-1], 3).tolist()} cost={f:.4f}")
        return
    print(f"{'duruş':10s} {'tool x':>8s} {'tool y':>8s} {'tool z':>8s}  masa_üstü  en_düşük_eklem_z")
    results = {}
    for name, q in POSES.items():
        pts, R = fk(q)
        tool = pts[-1]
        results[name] = pts
        print(f"{name:10s} {tool[0]:8.3f} {tool[1]:8.3f} {tool[2]:8.3f}  {str(on_table(tool)):9s}  {pts[1:].min(axis=0)[2]:.3f}")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2))
        colors = {"00_home": "#999999", "01_bowl": "#ffd166", "02_center": "#cfe8ff", "03_plate": "#d4f5d4"}
        # üstten
        ax1.add_patch(Rectangle((-TABLE[0] / 2, -TABLE[1] / 2), TABLE[0], TABLE[1], fc="#f3e2c3", ec="#7a5c2e", lw=1.5))
        for d, p, r, c in OBJECTS:
            ax1.add_patch(Circle((p[0], p[1]), r, fc=c, ec="k", lw=0.8, alpha=0.9))
            ax1.text(p[0], p[1] + r + 0.03, d, ha="center", fontsize=9)
        for name, pts in results.items():
            ax1.plot(pts[:-1, 0], pts[:-1, 1], "-o", color=colors[name], ms=4, lw=2.5, mec="k")
            ax1.plot(pts[-1, 0], pts[-1, 1], "*", color=colors[name], ms=16, mec="k", label=f"{name} (tool)")
        ax1.plot(*BASE_T[:2], "ks", ms=9, label="UR10e tabanı")
        ax1.annotate("agentview kamerası\n(0, 1.75, 1.55)", xy=(0, 0.9), xytext=(0, 1.15), ha="center", fontsize=9,
                     arrowprops=dict(arrowstyle="->"))
        ax1.set_xlim(-0.9, 0.9); ax1.set_ylim(-1.0, 1.3); ax1.set_aspect("equal")
        ax1.set_xlabel("dünya x (m)  — robot +y yönüne bakıyor; robotun SOLU = −x, SAĞI = +x")
        ax1.set_ylabel("dünya y (m)")
        ax1.set_title("Üstten: ders21_masa.wbt nesneleri + 4 duruşta kol (FK, proto geometrisi)")
        ax1.legend(fontsize=8, loc="lower left")
        # yandan (y-z)
        ax2.add_patch(Rectangle((-TABLE[1] / 2, 0), TABLE[1], TABLE[2], fc="#f3e2c3", ec="#7a5c2e", lw=1.5))
        for d, p, r, c in OBJECTS:
            ax2.add_patch(Circle((p[1], p[2] + r * 0.5), r * 0.6, fc=c, ec="k", lw=0.8, alpha=0.9))
        for name, pts in results.items():
            ax2.plot(pts[:-1, 1], pts[:-1, 2], "-o", color=colors[name], ms=4, lw=2.5, mec="k")
            ax2.plot(pts[-1, 1], pts[-1, 2], "*", color=colors[name], ms=16, mec="k")
        ax2.axhline(TABLE[2], color="#7a5c2e", ls="--", lw=1)
        ax2.set_xlim(-1.0, 1.0); ax2.set_ylim(0, 2.0); ax2.set_aspect("equal")
        ax2.set_xlabel("dünya y (m)"); ax2.set_ylabel("dünya z (m)")
        ax2.set_title("Yandan: kol masanın üstünde kalıyor mu? (kesikli çizgi = masa yüzeyi 0.74 m)")
        fig.suptitle("Ders 21 — Webots sahnesinin geometrisi: nesneler .wbt'den, kol UR10e.proto zincirinden hesaplandı (Webots çalıştırılmadı)",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=150)
        print("kaydedildi:", a.plot)


if __name__ == "__main__":
    main()
```

## plot_day21.py

`plot_day21.py`

```python
"""Ders 21 — görsel 05: tez ortamı karar matrisi (LIBERO+MuJoCo vs Webots).

Puanlar ÖZNEL (Claude'un bu dersteki değerlendirmesi, 0–3); kaynak değil, tartışma başlangıcıdır.
Danışmanla konuştuktan sonra puanları değiştirip yeniden çiz:  python plot_day21.py --out 05-decision.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CRITERIA = [
    ("Hazır benchmark + liderlik tablosu (Spatial/Goal, Plus, CF, X)", 3, 0),
    ("VLA eval kodu hazır (OpenVLA, VLANeXt, OFT…)",                   3, 0),
    ("Eğitim verisi (50 demo/görev, HDF5, RLDS)",                      3, 0),
    ("Mac M2'de kurulum / çalışma kolaylığı",                          1, 3),
    ("Oda ölçeği, mobil taban (RoomSpace tarzı sahne)",                 0, 3),
    ("Sahne/PROTO üretimi, kamera yerleşimi esnekliği",                 1, 3),
    ("ROS 2 köprüsü (gerçek robota geçiş)",                             0, 3),
    ("Türkçe komut deneyine süre: 'bugün' başlanabilir mi",            3, 1),
    ("Sonuçların literatürle karşılaştırılabilirliği",                  3, 0),
    ("Aynı robot (Panda) iki ortamda da var",                           3, 2),
]
LABELS = ["LIBERO + robosuite + MuJoCo", "Webots R2025a"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="05-decision.png")
    a = ap.parse_args()
    M = np.array([[c[1], c[2]] for c in CRITERIA], float)
    fig, ax = plt.subplots(figsize=(11, 6.8))
    cmap = matplotlib.colors.ListedColormap(["#ffd6d6", "#fde9c4", "#d4f5d4", "#9be29b"])
    ax.imshow(M, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
    for i, (name, s1, s2) in enumerate(CRITERIA):
        for j, s in enumerate((s1, s2)):
            ax.text(j, i, ["yok", "zayıf", "orta", "güçlü"][s], ha="center", va="center", fontsize=11, weight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_yticks(range(len(CRITERIA))); ax.set_yticklabels([c[0] for c in CRITERIA], fontsize=9.5)
    ax.xaxis.tick_top()
    tot = M.sum(axis=0)
    ax.set_xlabel(f"toplam (eşit ağırlık): {tot[0]:.0f} / {tot[1]:.0f}   —   tez sorusu benchmark ister, Webots bir benchmark değil",
                  fontsize=10, labelpad=10)
    ax.set_title("Ders 21 — Tez ortamı karar matrisi (ÖZNEL puanlama 0–3; danışmanla güncellenecek)", fontsize=12, pad=38)
    for i in range(len(CRITERIA) + 1):
        ax.axhline(i - 0.5, color="white", lw=2)
    ax.axvline(0.5, color="white", lw=2)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("kaydedildi:", a.out)


if __name__ == "__main__":
    main()
```

## ders21_masa.wbt

`worlds/ders21_masa.wbt`

```text
#VRML_SIM R2025a utf8
# Ders 21 — Webots'ta LIBERO-Spatial'a benzeyen bir masa sahnesi:
# UR10e kol (bilek kamerasıyla) + masa + siyah kâse, tabak, ocak, mısır gevreği kutusu, elma, kutu içecek.
# Açmak için: Webots > File > Open World... > bu dosya. Kol controller'ı <extern>: terminalden başlatılır (bkz. README).

EXTERNPROTO "webots://projects/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "webots://projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"
EXTERNPROTO "webots://projects/objects/floors/protos/Floor.proto"
EXTERNPROTO "webots://projects/objects/tables/protos/Table.proto"
EXTERNPROTO "webots://projects/robots/universal_robots/protos/UR10e.proto"
EXTERNPROTO "webots://projects/objects/fruits/protos/FruitBowl.proto"
EXTERNPROTO "webots://projects/objects/fruits/protos/Apple.proto"
EXTERNPROTO "webots://projects/objects/kitchen/utensils/protos/Plate.proto"
EXTERNPROTO "webots://projects/objects/kitchen/components/protos/HotPlate.proto"
EXTERNPROTO "webots://projects/objects/kitchen/breakfast/protos/CerealBox.proto"
EXTERNPROTO "webots://projects/objects/drinks/protos/Can.proto"

WorldInfo {
  info [
    "Ders 21: UR10e + masa + nesneler. Türkçe mekânsal komutlar için Webots deneme sahnesi."
  ]
  title "Ders 21 masa sahnesi"
  basicTimeStep 16
}
Viewpoint {
  # agentview benzeri: robotun karşısından, masaya bakan sabit bakış (vp.py ile hesaplandı)
  orientation 0 0.558702 0.829369 3.141593
  position 0 1.75 1.55
}
TexturedBackground {
}
TexturedBackgroundLight {
}
Floor {
  size 6 6
}
Table {
  translation 0 0 0
  size 1 1.8 0.74
}
DEF ROBOT UR10e {
  translation 0 -0.65 0.74
  rotation 0 0 1 -1.5708
  controller "<extern>"
  supervisor TRUE
  toolSlot [
    DEF WRIST_CAM Camera {
      translation 0 0.02 0
      rotation 0 0 1 1.5708
      name "wrist_camera"
      fieldOfView 1.2
      width 256
      height 256
    }
  ]
}
DEF BOWL FruitBowl {
  translation -0.22 0.0 0.74
  name "black bowl"
  color 0.05 0.05 0.05
}
DEF PLATE Plate {
  translation 0.22 0.02 0.74
}
DEF STOVE HotPlate {
  translation 0 0.5 0.74
}
DEF CEREAL CerealBox {
  translation -0.35 0.45 0.74
  rotation 0 0 1 0.3
}
DEF APPLE Apple {
  translation 0.05 0.27 0.79
}
DEF CAN Can {
  translation 0.36 0.4 0.8
}
```
