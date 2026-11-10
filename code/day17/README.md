# day17

Ders 17 — LIBERO-Plus'ın yedi bozulması kodda: taşıyıcılar (görev adı / BDDL / problem sınıfı), kamera matematiği, 500 robot başlangıç pozu, sensör gürültüsü, nesne yerleşimi, 50 yeniden yazımın R1/R2 türü ve Türkçe `_language_trK.bddl` varyantı üretimi. Hiçbir script model, GPU ya da LIBERO-Plus asset'i istemez; yalnızca LIBERO-plus klonundaki metin dosyaları okunur (render için robosuite yeterlidir).

| Dosya | Ne | Durum |
|---|---|---|
| `perturbation_table.py` | 7 bozulma × (taşıyıcı, gerçek parametre aralığı, kod yeri, BDDL/asset) tablosu — JSON ve dosyalardan sayar | çalıştırıldı (Spatial, 2402 görev) |
| `camera_geometry.py` | LIBERO-plus'ın `rotate_around_y/z` + `scale_distance_from_pivot` fonksiyonlarını dosyadan çekip 376 kamera görevinin pozunu hesaplar → görsel 02 | çalıştırıldı (gerçek kod) |
| `robot_init_spread.py` | `mounted_panda.py`'deki 500 `init_qpos`'u okur; eklem sapmaları, ‖Δ‖ ∈ {0.1,…,0.5}, zorlukla ilişki → görsel 03 | çalıştırıldı |
| `render_perturbations.py` | Kamera (8 poz) ve robot (3 init) bozulmalarını robosuite Stack sahnesinde gerçekten render eder; `--libero` ile LIBERO görev 0 → görsel 04 | çalıştırıldı (robosuite 1.4.1 + mujoco 2.3.7, EGL) |
| `layout_shift.py` | `_levelL_sampleM` dosyalarında bölge merkezlerinin kayması (cm) ve `_add_K` çeldirici nesneleri → görsel 05 | çalıştırıldı (200 + 300 dosya) |
| `classify_rewrites.py` | Temel görevin 50 `_language_K` metnini R1/R2/R0'a ayırır, korunan kelimeleri sayar → CSV + görsel 06 | çalıştırıldı (`datasets/02`) |
| `noise_on_frame.py` | 5 gürültü ailesi × şiddet (env_wrapper.py parametre tablolarıyla) gerçek bir kareye → görsel 07 | çalıştırıldı (wand/ImageMagick ile) |
| `make_tr_variant.py` + `tr_commands.csv` | Türkçe `_language_trK.bddl` üret, `bddl` ayrıştırıcısıyla geri oku, task map / JSON satırlarını yaz | çalıştırıldı (10/10 birebir; kötü örneklerde beklenen hatalar) |
| `plot_day17.py` | Görsel 09: VLANeXt Tablo III, 7 bozulma × suite | çalıştırıldı |

Kullanım (Mac'te, `vla` env; model gerekmez):
```bash
git clone --depth 1 https://github.com/sylvestf/LIBERO-plus ~/vla-work/LIBERO-plus    # Ders 16'da klonladıysan atla
for f in perturbation_table camera_geometry robot_init_spread layout_shift classify_rewrites noise_on_frame make_tr_variant render_perturbations plot_day17; do
  curl -O https://raw.githubusercontent.com/mesutd0nmez/vla-robot-learning-tr/main/code/day17/$f.py; done
curl -O https://raw.githubusercontent.com/mesutd0nmez/vla-robot-learning-tr/main/code/day17/tr_commands.csv
P=~/vla-work/LIBERO-plus; mkdir -p out17
python perturbation_table.py $P --md out17/table.md
python camera_geometry.py $P out17/02.png --csv out17/cam.csv
python robot_init_spread.py $P out17/03.png --csv out17/robot.csv
python layout_shift.py $P out17/05.png
python classify_rewrites.py $P out17/06.png --csv out17/rewrites.csv
MUJOCO_GL=glfw python render_perturbations.py out17/04.png --robot-csv out17/robot.csv --save-frame out17/frame.png   # LIBERO kuruluysa --libero ekle
pip install wand && brew install imagemagick            # motion_blur için (isteğe bağlı)
python noise_on_frame.py out17/frame.png out17/07.png
pip install bddl==1.0.1 future && python make_tr_variant.py $P --commands tr_commands.csv --out out17/tr_variants
```


## perturbation_table.py

```python
"""Ders 17 — LIBERO-Plus'ın 7 bozulmasını KODDAN okuyarak bir "perturbation tablosu" çıkarır.

Model / GPU / asset GEREKMEZ: yalnızca LIBERO-plus klonundaki metin dosyalarını okur
(task_classification.json, bddl_files/, envs/robots/mounted_panda.py, envs/problems/*.py).

Kullanım:
  python perturbation_table.py <LIBERO-plus kökü> [--suite libero_spatial] [--json out.json] [--md out.md]

Çıktı: her bozulma için (a) taşıyıcı (görev adı / BDDL dosyası / problem sınıfı), (b) gerçek parametre
aralığı (görev adlarından ve dosyalardan sayılır), (c) kaç görev, (d) kodda nerede uygulanıyor.
"""
import argparse, collections, json, pathlib, re, sys

CATS = ["Camera Viewpoints", "Robot Initial States", "Language Instructions", "Light Conditions",
        "Background Textures", "Sensor Noise", "Objects Layout"]
NOISE_FAMILY = {1: "motion_blur (ImageMagick/wand)", 2: "gaussian_blur (skimage)", 3: "zoom_blur (scipy zoom)",
                4: "fog (plasma_fractal)", 5: "glass_blur (skimage + piksel karıştırma)"}


def parse_tail(name):
    """'..._view_H_V_S_ER_EV_initstate_I[_noise_N]' → dict ya da None (BDDL-taşımalı görevlerde yok)."""
    if "_view_" not in name:
        return None
    view, rest = name.split("_view_")[1].split("_initstate_")
    h, v, s, er, ev = view.split("_")
    noise = 0
    if "_noise_" in rest:
        rest, noise = rest.split("_noise_")
    return dict(h=int(h), v=int(v), s=int(s) / 100, er=int(er), ev=int(ev), init=int(rest), noise=int(noise))


def signed(deg):
    """0..359 derece → -180..180 (kamera açıları 285..359 aslında negatif)."""
    return deg - 360 if deg > 180 else deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--json"); ap.add_argument("--md")
    a = ap.parse_args()
    lib = pathlib.Path(a.root) / "libero" / "libero"
    tasks = json.load(open(lib / "benchmark" / "task_classification.json", encoding="utf-8"))[a.suite]
    bddl_dir = lib / "bddl_files" / a.suite

    by_cat = collections.defaultdict(list)
    for t in tasks:
        by_cat[t["category"]].append(t["name"])

    rows = {}
    # --- Camera
    cams = [parse_tail(n) for n in by_cat["Camera Viewpoints"]]
    hs = [signed(c["h"]) for c in cams]; vs = [c["v"] for c in cams]; ss = [c["s"] for c in cams]
    ers = [signed(c["er"]) for c in cams]; evs = [signed(c["ev"]) for c in cams]
    nz = lambda xs: [x for x in xs if x != 0]
    rows["Camera Viewpoints"] = dict(
        carrier="görev adı (`_view_H_V_S_ER_EV`)", n=len(cams),
        where="env_wrapper.py:207-218 (parse) → problems/*.py `_setup_camera` (rotate_around_z / rotate_around_y / scale_distance_from_pivot, pivot (0,0,0.8))",
        params=f"yatay {min(nz(hs))}..{max(nz(hs))}° ({len(nz(hs))} görev); dikey {sorted(set(vs))}° ; "
               f"mesafe ×{min(nz([s for s in ss if s != 1.0]) or [1]):.2f}..×{max(ss):.2f} ({sum(s != 1.0 for s in ss)} görev); "
               f"uç yaw {min(nz(ers) or [0])}..{max(nz(ers) or [0])}°, uç pitch {min(nz(evs) or [0])}..{max(nz(evs) or [0])}° ({sum(1 for c in cams if c['er'] or c['ev'])} görev)",
        bddl="hayır (temel BDDL)", assets="hayır")
    # --- Robot init
    inits = [parse_tail(n)["init"] for n in by_cat["Robot Initial States"]]
    panda = (lib / "envs" / "robots" / "mounted_panda.py").read_text(encoding="utf-8")
    n_cls = len(re.findall(r"^class MountedPanda\d+\(", panda, flags=re.M))
    rows["Robot Initial States"] = dict(
        carrier="görev adı (`_initstate_I`) → robot sınıfı adı `MountedPanda{I}`", n=len(inits),
        where="env_wrapper.py:219-220 (`robots=[robots[0]+str(init_state)]`) → envs/robots/mounted_panda.py (her sınıf yalnızca `init_qpos` döndürür)",
        params=f"I = {min(inits)}..{max(inits)} ({len(set(inits))} farklı); dosyada {n_cls} sınıf tanımlı; 7 eklem açısı (rad)",
        bddl="hayır", assets="hayır")
    # --- Language
    langs = by_cat["Language Instructions"]
    lang_files = sorted(bddl_dir.glob("*_language_*.bddl"))
    rows["Language Instructions"] = dict(
        carrier="BDDL dosyası (`_language_K.bddl`, yalnızca `:language` satırı farklı) + görev adı kuyruğu sabit `_view_0_0_100_0_0_initstate_0`", n=len(langs),
        where="benchmark/__init__.py:46-72 `grab_language_from_filename` → bddl_utils.get_problem_info → bddl.parsing.scan_tokens (`.lower()`)",
        params=f"klasörde {len(lang_files)} dil dosyası, benchmark'ta {len(langs)} görev; sahne/kamera/robot/hedef değişmez",
        bddl="evet (yalnızca :language)", assets="hayır")
    # --- Light
    lights = by_cat["Light Conditions"]
    light_ids = sorted(int(re.search(r"_light_(\d+)", n).group(1)) for n in lights)
    prob = (lib / "envs" / "problems" / "libero_tabletop_manipulation.py").read_text(encoding="utf-8")
    n_light_cls = len(set(re.findall(r"scenes/lights/(tabletop_light_sync_modified_\d+)\.xml", prob)))
    rows["Light Conditions"] = dict(
        carrier="BDDL dosyası (`_light_K.bddl`, yalnızca `(problem ...)` adı farklı) → kayıtlı problem alt sınıfı → `scene_xml`", n=len(lights),
        where="problems/libero_tabletop_manipulation.py `Libero_Tabletop_Manipulation_tabletop_light_sync_modified_K` (scene_xml=scenes/lights/..._K.xml)",
        params=f"K = {min(light_ids)}..{max(light_ids)} (Spatial'da {len(set(light_ids))} farklı); tabletop için {n_light_cls} ışık sahnesi kayıtlı; XML'de diffuse/direction/specular/castshadow",
        bddl="evet (yalnızca problem adı)", assets="EVET (assets/scenes/lights/*.xml)")
    # --- Background
    bgs = by_cat["Background Textures"]
    n_table = sum("_table_" in n and "_tb_" not in n for n in bgs); n_tb = sum("_tb_" in n for n in bgs)
    tex = set(re.findall(r"scenes/(tabletop_table_[A-Za-z0-9_]+)\.xml", prob))
    rows["Background Textures"] = dict(
        carrier="BDDL dosyası (`_table_K` = masa dokusu, `_tb_K` = masa + duvar) → problem alt sınıfı → `scene_xml`", n=len(bgs),
        where="problems/libero_tabletop_manipulation.py `..._tabletop_table_<Doku>[_Wall_<duvar>]` sınıfları",
        params=f"`_table_` {n_table} görev, `_tb_` {n_tb} görev; tabletop için {len(tex)} farklı doku sahnesi kayıtlı (ör. {sorted(tex)[0]})",
        bddl="evet (yalnızca problem adı)", assets="EVET (assets/scenes/*.xml + textures/)")
    # --- Noise
    noises = [parse_tail(n)["noise"] for n in by_cat["Sensor Noise"]]
    fam = collections.Counter((c - 1) // 10 + 1 for c in noises)
    rows["Sensor Noise"] = dict(
        carrier="görev adı (`_noise_N`, N = 1..50)", n=len(noises),
        where="env_wrapper.py:285-348 `ControlEnv.step/reset`: YALNIZCA `agentview_image` değiştirilir (bilek kamerası dokunulmaz)",
        params="; ".join(f"{NOISE_FAMILY[k]} N={10*(k-1)+1}..{10*k}: {fam.get(k,0)} görev" for k in range(1, 6)),
        bddl="hayır", assets="hayır (ama `wand` + libmagickwand gerekir)")
    # --- Layout
    lays = by_cat["Objects Layout"]
    n_add = sum("_add_" in n for n in lays); n_lvl = sum("_level" in n for n in lays)
    lvl = collections.Counter(int(re.search(r"_level(\d+)_", n).group(1)) for n in lays if "_level" in n)
    add_objs = collections.Counter()
    for f in bddl_dir.glob("*_add_*.bddl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            m = re.search(r"\(On (\S+) main_table_add_object_region", line)
            if m: add_objs[re.sub(r"_\d+$", "", m.group(1))] += 1
    rows["Objects Layout"] = dict(
        carrier="BDDL dosyası: `_add_K` (yeni nesne + `add_object_region_K`) ve `_levelL_sampleM` (bölge aralıkları kaydırılmış)", n=len(lays),
        where="bddl_base_domain.py (BDDL `:regions` → placement initializer); yeni nesneler assets/new_objects, turbosquid_objects…",
        params=f"`_add_` {n_add} görev ({len(add_objs)} farklı eklenen nesne türü, Spatial); `_level` {n_lvl} görev, seviye dağılımı {dict(sorted(lvl.items()))}",
        bddl="evet (:regions, :objects, :init)", assets="EVET (yeni nesne mesh'leri)")

    # --- çıktı
    print(f"# LIBERO-Plus perturbation tablosu — suite: {a.suite} ({len(tasks)} görev)\n")
    hdr = "| Bozulma | Görev | Taşıyıcı | Gerçek parametre aralığı | Kodda nerede | BDDL değişir? | Asset gerekir? |"
    md = [hdr, "|---|---|---|---|---|---|---|"]
    for c in CATS:
        r = rows[c]
        md.append(f"| {c} | {r['n']} | {r['carrier']} | {r['params']} | {r['where']} | {r['bddl']} | {r['assets']} |")
    print("\n".join(md))
    if a.md: pathlib.Path(a.md).write_text("\n".join(md) + "\n", encoding="utf-8")
    if a.json: json.dump({"suite": a.suite, "n_tasks": len(tasks), "rows": rows}, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
```

## camera_geometry.py

```python
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
```

## robot_init_spread.py

```python
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
```

## render_perturbations.py

```python
"""Ders 17 — Kamera ve robot-başlangıç bozulmalarını robosuite'te GERÇEKTEN render eder (LIBERO-Plus asset'i gerekmez).

Neden robosuite? Kamera (`_view_`) ve robot (`_initstate_`) bozulmaları LIBERO-Plus'ta yeni asset istemez: biri
agentview kamerasının poz/quat'ını, öteki 7 eklem açısını değiştirir. Bu script aynı dönüşümleri (camera_geometry.py'deki
LIBERO-plus matematiği: pivot (0,0,0.8) etrafında z/y dönüşü, pivot'tan ölçekleme, uç yaw/pitch) robosuite'in Panda'lı
bir masa sahnesine uygular ve kareleri çizer. LIBERO kuruluysa `--libero` ile aynı şeyi LIBERO-Spatial görev 0'da yapar
(Ders 10 ortamı; `LIBERO_CONFIG_PATH` ayarlı olmalı).

Kullanım:  MUJOCO_GL=egl python render_perturbations.py <çıktı.png> [--libero] [--size 256] [--robot-csv robot_init.csv]
  --robot-csv: robot_init_spread.py --csv çıktısı; verilmezse ||Δ||=0.3 rad'lık rastgele yönler üretilir (aynı dağılım).
"""
import argparse, csv, os, sys
os.environ.setdefault("MUJOCO_GL", "egl" if os.uname().sysname == "Linux" else "glfw")
import numpy as np
from scipy.spatial.transform import Rotation

POS_AV = np.array([0.6586131746834771, 0.0, 1.6103500240372423])   # LIBERO tabletop agentview (problems/*.py)
QUAT_AV = np.array([0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349])  # (w,x,y,z)
PIVOT = np.array([0, 0, 0.8])
BASE_QPOS = np.array([0, -1.61037389e-01, 0.0, -2.44459747e00, 0.0, 2.22675220e00, np.pi / 4])  # MountedPanda.init_qpos


def q_wxyz_to_R(q):  # MuJoCo (w,x,y,z) → scipy Rotation
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def R_to_q_wxyz(R):
    x, y, z, w = R.as_quat(); return np.array([w, x, y, z])


def camera_pose(h=0, v=0, s=1.0, er=0, ev=0):
    """LIBERO-plus `_setup_camera` ile aynı sıra: dikey (y ekseni, pivot'tan geçen) → yatay (z) → ölçek → uç yaw → uç pitch."""
    pos, R = POS_AV.copy(), q_wxyz_to_R(QUAT_AV)
    if v:
        Ry = Rotation.from_rotvec(np.radians(-v) * np.array([0, 1, 0])); pos = Ry.apply(pos - PIVOT) + PIVOT; R = Ry * R
    if h:
        Rz = Rotation.from_euler("z", h, degrees=True); pos = Rz.apply(pos); R = Rz * R
    if s != 1.0:
        pos = PIVOT + (pos - PIVOT) * s
    if er:
        R = Rotation.from_euler("z", er, degrees=True) * R
    if ev:
        R = Rotation.from_rotvec(np.radians(-ev) * np.array([0, 1, 0])) * R
    return pos, R_to_q_wxyz(R)


def make_env_robosuite(size):
    import robosuite as suite
    env = suite.make("Stack", robots="Panda", has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
                     camera_names="agentview", camera_heights=size, camera_widths=size, control_freq=20)
    env.reset()
    return env, env.sim, "robosuite Stack (Panda); LIBERO sahnesi değil"


def make_env_libero(size):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_spatial"](); task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=size, camera_widths=size); env.seed(0); env.reset()
    env.set_init_state(suite.get_task_init_states(0)[0])
    return env, env.sim, "LIBERO-Spatial görev 0, init 0"


def render(sim, size, cam="agentview"):
    sim.forward()
    img = sim.render(camera_name=cam, width=size, height=size)
    return img[::-1]  # MuJoCo offscreen görüntüsü ters gelir (Ders 10/14)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--libero", action="store_true")
    ap.add_argument("--size", type=int, default=256); ap.add_argument("--robot-csv"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-frame", help="varsayılan kameranın karesini bu PNG'ye de yaz (noise_on_frame.py girdisi)")
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    env, sim, scene = (make_env_libero if a.libero else make_env_robosuite)(a.size)
    m, d = sim.model, sim.data
    cid = m.camera_name2id("agentview")
    orig_pos, orig_quat = m.cam_pos[cid].copy(), m.cam_quat[cid].copy()
    print("sahne:", scene, "| sahnenin kendi agentview'ı pos", np.round(orig_pos, 3), "quat", np.round(orig_quat, 3))
    # robot eklem adresleri (Panda 7 eklem)
    jids = [m.joint_name2id(n) for n in m.joint_names if n.startswith("robot0_joint")][:7]
    qadr = [m.jnt_qposadr[j] for j in jids]
    base_q = d.qpos[qadr].copy(); print("sahnenin robot qpos'u", np.round(base_q, 3))

    cams = [("varsayılan (LIBERO agentview)", dict()), ("yatay +30°", dict(h=30)), ("yatay −45°", dict(h=315)),
            ("yatay +75° (uç)", dict(h=75)), ("dikey 15° + yatay 30°", dict(v=15, h=30)), ("mesafe ×1.5", dict(s=1.5)),
            ("mesafe ×2.0", dict(s=2.0)), ("uç yaw +10°, pitch −10°", dict(er=10, ev=350))]
    frames = []
    for title, kw in cams:
        pos, quat = camera_pose(**kw); m.cam_pos[cid] = pos; m.cam_quat[cid] = quat
        frames.append((title, render(sim, a.size)))
    if a.save_frame:
        from PIL import Image; Image.fromarray(frames[0][1]).save(a.save_frame); print("kare yazıldı:", a.save_frame)
    m.cam_pos[cid], m.cam_quat[cid] = camera_pose()  # robot kareleri varsayılan kamerayla
    # robot başlangıç pozları: ||Δ|| = 0.1 / 0.3 / 0.5 rad (LIBERO-Plus'ın 5 sabit büyüklüğünden üçü)
    deltas = []
    if a.robot_csv:
        rows = list(csv.DictReader(open(a.robot_csv, encoding="utf-8")))
        for target in (0.1, 0.3, 0.5):
            r = next(r for r in rows if abs(float(r["norm_delta"]) - target) < 1e-3)
            deltas.append((f"MountedPanda{r['init_state']} (||Δ||={target})", np.array([float(r[f'q{i+1}']) for i in range(7)]) - BASE_QPOS))
    else:
        for target in (0.1, 0.3, 0.5):
            v = rng.normal(size=7); deltas.append((f"rastgele yön, ||Δ||={target}", v / np.linalg.norm(v) * target))
    rob = []
    q0 = BASE_QPOS if a.libero else base_q  # robosuite'te kendi tabanı; LIBERO'da MountedPanda tabanı
    for title, dq in [("robot tabanı", np.zeros(7))] + deltas:
        d.qpos[qadr] = q0 + dq; d.qvel[:] = 0
        rob.append((title, render(sim, a.size)))
    d.qpos[qadr] = base_q; m.cam_pos[cid], m.cam_quat[cid] = orig_pos, orig_quat

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4, figsize=(13, 10.2))
    for ax, (t, im) in zip(axes[:2].ravel(), frames): ax.imshow(im); ax.set_title(t, fontsize=9.5)
    for ax, (t, im) in zip(axes[2], rob): ax.imshow(im); ax.set_title(t, fontsize=9.5)
    for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("Camera Viewpoints", fontsize=11); axes[1, 0].set_ylabel("Camera Viewpoints", fontsize=11); axes[2, 0].set_ylabel("Robot Initial States", fontsize=11)
    fig.suptitle(f"LIBERO-Plus kamera / robot bozulmaları gerçek render — sahne: {scene} — {a.size}×{a.size} agentview (render_perturbations.py)", fontsize=10.5)
    fig.text(0.5, 0.005, "Kamera: LIBERO-plus `_setup_camera` sırası (dikey→yatay→ölçek→uç yaw/pitch), pivot (0,0,0.8). Robot: 7 eklem açısı taban + Δ (LIBERO-Plus'ta ||Δ|| ∈ {0.1,…,0.5} rad).",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96)); fig.savefig(a.out, dpi=150); plt.close(fig)
    print("yazıldı:", a.out)
    try: env.close()
    except Exception: pass


if __name__ == "__main__":
    main()
```

## layout_shift.py

```python
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
```

## classify_rewrites.py

```python
"""Ders 17 — LIBERO-Plus 'Language Instructions': bir temel görevin 50 yeniden yazımını okur, R1/R2 türüne ayırır,
hangi kelimelerin korunduğunu sayar; CSV + görsel üretir.

Model/asset GEREKMEZ. Tür etiketi JSON'da YOK; buradaki etiket makalenin Ek A.6 tanımlarına göre kural tabanlı
bir okumadır (R1 dikkat dağıtma = sohbet/bağlam cümleleri; R2 sağduyu = nesne adının tarifle değişmesi;
R3 akıl yürütme zinciri = çok adımlı görevlerde — Spatial'da beklenmez). Kuralların tuttuğu yerde "hipotez"dir.

Kullanım:  python classify_rewrites.py <LIBERO-plus kökü> <çıktı.png> [--suite libero_spatial] [--base 0] [--csv out.csv]
--base: orijinal LIBERO görev sırası (Ders 10 list_tasks.py; 0 = between the plate and the ramekin)
"""
import argparse, csv, json, pathlib, re, sys

BASES = {  # Ders 16 select_plus_tasks.py ile aynı sıra (LIBERO-Spatial 0–9)
    "libero_spatial": ["pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"]}
CONV = re.compile(r"\b(could you|would you|can you|please|for me|lets|let's|before|remember|i noticed|i need|id appreciate|for our|for better|for proper|"
                  r"the table|needs|isnt|when you have|guests|dinner|dining|cooking|organiz|arrange)\b", re.I)
DESCR = re.compile(r"\b(container|vessel|darkhued|darkcolored|dark colored|flat ?dish|baking dish|ceramic dish|small dish|concave|spherical|rounded|"
                   r"main course|main meal|serving|holder|cooking dish|glazed)\b", re.I)
RELATION = re.compile(r"\b(between|amid|midway|separating|positioned between|situated between)\b", re.I)


def classify(t):
    tl = t.lower()
    conv, descr = bool(CONV.search(tl)), bool(DESCR.search(tl))
    keeps_bowl, keeps_plate = "bowl" in tl, "plate" in tl
    descr_light = bool(re.search(r"\bdark bowl\b|\bblack dish\b|\bthe dish\b", tl))  # 'black bowl'→'dark bowl', 'plate'→'dish'
    if descr and not keeps_bowl:
        typ = "R2 sağduyu (tam)"
    elif descr or descr_light:
        typ = "R2 sağduyu (hafif)"; descr = True
    elif conv:
        typ = "R1 dikkat dağıtma"
    else:
        typ = "R0 düz parafraz"
    return typ, conv, descr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--suite", default="libero_spatial"); ap.add_argument("--base", type=int, default=0); ap.add_argument("--csv")
    a = ap.parse_args()
    root = pathlib.Path(a.root); lib = root / "libero/libero"
    base = BASES[a.suite][a.base]; d = lib / "bddl_files" / a.suite
    tasks = json.load(open(lib / "benchmark/task_classification.json", encoding="utf-8"))[a.suite]
    in_bench = {}
    for i, t in enumerate(tasks):
        m = re.match(re.escape(base) + r"_language_(\d+)_view_", t["name"])
        if m: in_bench[int(m.group(1))] = (i, t["difficulty_level"])
    base_text = re.search(r"\(:language (.*?)\)", (d / f"{base}.bddl").read_text(encoding="utf-8")).group(1)
    rows = []
    for f in sorted(d.glob(f"{base}_language_*.bddl"), key=lambda p: int(re.search(r"_language_(\d+)", p.name).group(1))):
        k = int(re.search(r"_language_(\d+)", f.name).group(1))
        text = re.search(r"\(:language (.*?)\)", f.read_text(encoding="utf-8")).group(1).strip()
        typ, conv, descr = classify(text)
        idx, lvl = in_bench.get(k, (None, None))
        rows.append(dict(k=k, in_benchmark=k in in_bench, task_index=idx, difficulty=lvl, type=typ, words=len(text.split()),
                         keeps_bowl="bowl" in text.lower(), keeps_plate="plate" in text.lower(), keeps_between="between" in text.lower(),
                         relation_word=(RELATION.search(text.lower()) or [""])[0], conversational=conv, descriptive=descr,
                         lowercase_changes=sum(ch.isupper() for ch in text), text=text))
    print(f"temel: {base}\n  orijinal :language = '{base_text}'  ({len(base_text.split())} kelime)")
    print(f"  dil dosyası: {len(rows)}; benchmark'ta: {sum(r['in_benchmark'] for r in rows)}; eksik: {[r['k'] for r in rows if not r['in_benchmark']]}")
    from collections import Counter
    ct = Counter(r["type"] for r in rows); print("  tür dağılımı (kural tabanlı, hipotez):", dict(ct))
    for typ in sorted(ct):
        rr = [r for r in rows if r["type"] == typ]
        lv = Counter(r["difficulty"] for r in rr if r["in_benchmark"])
        print(f"   {typ:22s} n={len(rr):2d}  kelime ort {sum(r['words'] for r in rr)/len(rr):4.1f}  'between' korunan {sum(r['keeps_between'] for r in rr):2d}  zorluk {dict(sorted(lv.items()))}")
    print("  'between' yerine:", Counter(r["relation_word"] for r in rows if not r["keeps_between"]))
    print("  büyük harf içeren (scan_tokens .lower() değiştirecek):", sum(r["lowercase_changes"] > 0 for r in rows), "/", len(rows))
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    types = ["R1 dikkat dağıtma", "R2 sağduyu (hafif)", "R2 sağduyu (tam)", "R0 düz parafraz"]
    tcol = {"R1 dikkat dağıtma": "#cfe8ff", "R2 sağduyu (hafif)": "#ffe9a8", "R2 sağduyu (tam)": "#ffd166", "R0 düz parafraz": "#d4f5d4"}
    lcol = {1: "#d4f5d4", 2: "#b7e4b7", 3: "#ffd166", 4: "#ffb3a7", 5: "#ff7f7f", None: "#ddd"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 1.1, 1.1]})
    ax = axes[0]
    left = np.zeros(len(types))
    for lvl in [1, 2, 3, 4, 5, None]:
        vals = [sum(1 for r in rows if r["type"] == t and r["difficulty"] == lvl) for t in types]
        ax.barh(types, vals, left=left, color=lcol[lvl], edgecolor="#777", label=f"L{lvl}" if lvl else "benchmark'ta yok"); left += vals
    for i, t in enumerate(types): ax.text(left[i] + 0.3, i, str(int(left[i])), va="center", fontsize=9)
    ax.invert_yaxis(); ax.legend(fontsize=7.5, title="zorluk", loc="lower right"); ax.set_xlabel("varyant sayısı")
    ax.set_title(f"(a) Görev {a.base}: 50 yeniden yazımın türü (kural tabanlı)")
    ax = axes[1]
    for t in types:
        ws = [r["words"] for r in rows if r["type"] == t]
        ax.scatter(ws, [t] * len(ws), s=40, color=tcol[t], edgecolor="#555", alpha=0.85)
    ax.axvline(len(base_text.split()), color="#d62828", ls="--")
    ax.text(len(base_text.split()) + 0.3, -0.45, f"orijinal: {len(base_text.split())} kelime", color="#d62828", fontsize=8, va="bottom")
    ax.invert_yaxis(); ax.set_xlabel("kelime sayısı"); ax.set_title("(b) Uzunluk: tür başına kelime sayısı")
    ax = axes[2]
    feats = [("'bowl'\nkorunmuş", "keeps_bowl"), ("'plate'\nkorunmuş", "keeps_plate"), ("'between'\nkorunmuş", "keeps_between"), ("büyük harf var\n(.lower() değiştirir)", "lowercase_changes")]
    vals = [sum(bool(r[f]) for r in rows) for _, f in feats]
    ax.bar([f for f, _ in feats], vals, color=["#d4f5d4", "#d4f5d4", "#ffd166", "#ffd6d6"], edgecolor="#777")
    for i, v in enumerate(vals): ax.text(i, v + 0.6, f"{v}/{len(rows)}", ha="center", fontsize=9)
    ax.set_ylim(0, len(rows) + 6); ax.set_ylabel("varyant sayısı"); ax.set_title("(c) Tez için kritik kelimeler korunuyor mu?")
    ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"LIBERO-Plus Language bozulması içeriden — {a.suite}, temel görev {a.base} ({len(rows)} `_language_K.bddl`; tür etiketi JSON'da yok, buradaki kural tabanlı okumadır) — classify_rewrites.py", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(a.out, dpi=170); plt.close(fig)
    print("yazıldı:", a.out)


if __name__ == "__main__":
    main()
```

## noise_on_frame.py

```python
"""Ders 17 — LIBERO-Plus 'Sensor Noise': env_wrapper.py'deki beş bozulmayı bir kareye uygular.

Fonksiyonlar LIBERO-plus `libero/libero/envs/env_wrapper.py` (4976dc3) satır 30-174'ten, PARAMETRE TABLOLARI
BİREBİR aktarılarak yeniden yazıldı (dış depodan kod çalıştırmamak için; algoritmalar aynı, Hendrycks & Dietterich
ImageNet-C'nin bilinen uygulamaları). motion_blur için `wand` (ImageMagick) gerekir; yoksa o sütun atlanır.

Kullanım:  python noise_on_frame.py <kare.png> <çıktı.png> [--size 128] [--severity 1,5,10] [--seed 0]
Kare: Ders 10/14'teki gibi gerçek bir agentview karesi (LIBERO-Plus eval'de 128x128 kullanılır; --size ile ölçeklenir).
"""
import argparse, pathlib, sys
from io import BytesIO
import numpy as np
from PIL import Image
from scipy.ndimage import zoom as scizoom
from skimage.filters import gaussian

# --- env_wrapper.py:33-44  motion_blur: (radius, sigma), açı U(-45,45)
MOTION = [(5, 2), (8, 3), (10, 4), (12, 5), (15, 6), (18, 8), (20, 10), (25, 12), (30, 15), (35, 20)]
# --- env_wrapper.py:61  gaussian_blur: sigma
GAUSS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# --- env_wrapper.py:79-88  zoom_blur: zoom faktörleri
ZOOM = [np.arange(1, 1.11, 0.01), np.arange(1, 1.16, 0.01), np.arange(1, 1.21, 0.02), np.arange(1, 1.26, 0.02), np.arange(1, 1.31, 0.03),
        np.arange(1, 1.36, 0.01), np.arange(1, 1.41, 0.01), np.arange(1, 1.46, 0.02), np.arange(1, 1.51, 0.02), np.arange(1, 1.56, 0.03)]
# --- env_wrapper.py:148  fog: (yoğunluk alpha, wibbledecay)
FOG = [(0.5, 3), (1.0, 2.8), (1.5, 2.5), (2.0, 2.2), (2.5, 2.0), (3.0, 1.8), (3.5, 1.6), (4.0, 1.5), (4.5, 1.4), (5.0, 1.3)]
# --- env_wrapper.py:159  glass_blur: (sigma, max_delta, iterations)
GLASS = [(0.5, 1, 3), (0.7, 1, 3), (0.9, 2, 3), (1.0, 2, 2), (1.1, 3, 2), (1.3, 3, 2), (1.5, 4, 2), (1.8, 4, 2), (2.2, 5, 1), (2.5, 5, 1)]


def motion_blur(x, severity=1):
    from wand.image import Image as WandImage  # ImageMagick
    r, s = MOTION[severity - 1]
    buf = BytesIO(); x.save(buf, format="PNG")
    with WandImage(blob=buf.getvalue()) as w:
        w.motion_blur(radius=r, sigma=s, angle=float(np.random.uniform(-45, 45)))
        out = np.array(w)[..., :3]
    return np.clip(out, 0, 255).astype(np.uint8)


def gaussian_blur(x, severity=1):
    return (np.clip(gaussian(np.array(x) / 255.0, sigma=GAUSS[severity - 1], channel_axis=-1), 0, 1) * 255).astype(np.uint8)


def clipped_zoom(img, zf):  # env_wrapper.py:66-76
    h = img.shape[0]; ch = int(np.ceil(h / float(zf))); top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zf, zf, 1), order=1)
    trim = (img.shape[0] - h) // 2
    return img[trim:trim + h, trim:trim + h]


def zoom_blur(x, severity=1):
    c = ZOOM[severity - 1]; x = (np.array(x) / 255.0).astype(np.float32); out = np.zeros_like(x)
    for zf in c: out += clipped_zoom(x, zf)
    return (np.clip((x + out) / (len(c) + 1), 0, 1) * 255).astype(np.uint8)


def plasma_fractal(mapsize=256, wibbledecay=3):  # env_wrapper.py:98-144 (diamond-square)
    maparray = np.empty((mapsize, mapsize), dtype=np.float64); maparray[0, 0] = 0; stepsize = mapsize; wibble = 100
    def wibbledmean(a): return a / 4 + wibble * np.random.uniform(-wibble, wibble, a.shape)
    def fillsquares():
        c = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]; s = c + np.roll(c, -1, 0); s += np.roll(s, -1, 1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(s)
    def filldiamonds():
        m = maparray.shape[0]; dr = maparray[stepsize // 2:m:stepsize, stepsize // 2:m:stepsize]; ul = maparray[0:m:stepsize, 0:m:stepsize]
        maparray[0:m:stepsize, stepsize // 2:m:stepsize] = wibbledmean(dr + np.roll(dr, 1, 0) + ul + np.roll(ul, -1, 1))
        maparray[stepsize // 2:m:stepsize, 0:m:stepsize] = wibbledmean(dr + np.roll(dr, 1, 1) + ul + np.roll(ul, -1, 0))
    while stepsize >= 2:
        fillsquares(); filldiamonds(); stepsize //= 2; wibble /= wibbledecay
    maparray -= maparray.min(); return maparray / maparray.max()


def fog(x, severity=1):
    c = FOG[severity - 1]; x = np.array(x) / 255.0; mx = x.max(); h, w = x.shape[:2]
    x = x + c[0] * plasma_fractal(wibbledecay=c[1])[:h, :w][..., None]
    return (np.clip(x * mx / (mx + c[0]), 0, 1) * 255).astype(np.uint8)


def glass_blur(x, severity=1):
    sg, dmax, iters = GLASS[severity - 1]
    x = np.uint8(gaussian(np.array(x) / 255.0, sigma=sg, channel_axis=-1) * 255); h, w = x.shape[:2]
    for _ in range(iters):
        for hh in range(h - dmax, dmax, -1):
            for ww in range(w - dmax, dmax, -1):
                dx, dy = np.random.randint(-dmax, dmax, size=(2,)); hp, wp = hh + dy, ww + dx
                x[hh, ww], x[hp, wp] = x[hp, wp], x[hh, ww]
    return (np.clip(gaussian(x / 255.0, sigma=sg, channel_axis=-1), 0, 1) * 255).astype(np.uint8)


FAMILIES = [("motion_blur (N=1..10)", 1, motion_blur), ("gaussian_blur (N=11..20)", 11, gaussian_blur),
            ("zoom_blur (N=21..30)", 21, zoom_blur), ("fog (N=31..40)", 31, fog), ("glass_blur (N=41..50)", 41, glass_blur)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame"); ap.add_argument("out")
    ap.add_argument("--size", type=int, default=128); ap.add_argument("--severity", default="1,5,10"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); np.random.seed(a.seed)
    try:
        import wand.image  # noqa
        have_wand = True
    except Exception as e:
        print("UYARI: wand/ImageMagick yok, motion_blur atlanacak:", e, file=sys.stderr); have_wand = False
    img = Image.open(a.frame).convert("RGB")
    if img.size != (a.size, a.size):
        side = min(img.size)
        img = img.crop(((img.width - side) // 2, (img.height - side) // 2, (img.width + side) // 2, (img.height + side) // 2)).resize((a.size, a.size), Image.LANCZOS)
    sev = [int(s) for s in a.severity.split(",")]
    fams = [f for f in FAMILIES if have_wand or f[2] is not motion_blur]

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(sev), len(fams) + 1, figsize=(2.3 * (len(fams) + 1), 2.45 * len(sev)))
    axes = np.atleast_2d(axes)
    for r, s in enumerate(sev):
        axes[r, 0].imshow(img); axes[r, 0].set_ylabel(f"severity {s}", fontsize=10)
        if r == 0: axes[r, 0].set_title("orijinal", fontsize=9)
        for c, (label, base_code, fn) in enumerate(fams, start=1):
            axes[r, c].imshow(fn(img, severity=s))
            if r == 0: axes[r, c].set_title(label, fontsize=8.5)
            axes[r, c].text(0.03, 0.03, f"_noise_{base_code + s - 1}", transform=axes[r, c].transAxes, fontsize=7.5, color="#fff",
                            bbox=dict(facecolor="#000", alpha=0.5, pad=1.5, lw=0))
    for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"LIBERO-Plus 'Sensor Noise': env_wrapper.py'nin 5 bozulması ({a.size}×{a.size} agentview karesi; parametre tabloları koddan) — noise_on_frame.py"
                 + ("" if have_wand else "  [motion_blur: wand yok, atlandı]"), fontsize=9.5)
    fig.text(0.5, 0.005, "yalnızca agentview_image değişir; bilek (eye-in-hand) kamerası ve proprio dokunulmaz (env_wrapper.py:288-308). Kare: " + pathlib.Path(a.frame).name,
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95)); fig.savefig(a.out, dpi=170); plt.close(fig)
    print("yazıldı:", a.out, "| wand:", have_wand, "| aileler:", [f[2].__name__ for f in fams])


if __name__ == "__main__":
    main()
```

## make_tr_variant.py

```python
"""Ders 17 — LIBERO-Plus şablonuyla Türkçe "Language" varyantı üretir: `<temel>_language_trK.bddl` + task map satırı
+ task_classification.json girdisi; dosyayı `bddl` ayrıştırıcısından geçirip modele gidecek metnin yazdığınla
AYNI olduğunu doğrular (Ders 16 Bölüm 4 kuralları: küçük harf, `;` ve parantez yok).

Model/asset GEREKMEZ. Varsayılan olarak klona DOKUNMAZ (çıktılar --out klasörüne yazılır); --install ile klona kopyalar.

Kullanım:
  python make_tr_variant.py <LIBERO-plus kökü> --commands komutlar.csv [--suite libero_spatial] [--out tr_variants/] [--install]
  komutlar.csv sütunları: base_index,tr_id,text   (base_index: 0–9 orijinal LIBERO sırası; tr_id: 1,2,…; text: Türkçe komut)
  Örnek satır: 0,1,tabak ile ramekin arasındaki siyah kâseyi al ve tabağın üstüne koy
"""
import argparse, csv, json, pathlib, re, shutil, sys, unicodedata

BASES = ["pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"]
TAIL = "_view_0_0_100_0_0_initstate_0"  # Language görevlerinin sabit kuyruğu (temiz kamera, robot init 0)


def check_text(t):
    """Ders 16'daki tuzaklar: ';' yorum siler, '()' yapıyı bozar, .lower() I/İ'yi değiştirir, U+0307 sızar."""
    problems = []
    if ";" in t: problems.append("';' var → sonrası yorum sayılır, 'Missing close parenthesis'")
    if "(" in t or ")" in t: problems.append("parantez var → iç içe liste, TypeError")
    if t != t.lower(): problems.append(f"büyük harf var → scan_tokens .lower() değiştirir: {t.lower()!r}")
    if "̇" in t: problems.append("U+0307 birleşik nokta var")
    if unicodedata.normalize("NFC", t) != t: problems.append("NFC normalize değil")
    if "  " in t or t != t.strip(): problems.append("çift/uç boşluk (scan_tokens tek boşluğa indirir)")
    return problems


def parse_language_like_bddl(path):
    """Önce gerçek `bddl` paketi (pip install bddl==1.0.1 future); yoksa aynı iki adımın birebir kopyası."""
    raw = pathlib.Path(path).read_text(encoding="utf-8")
    try:
        from bddl.parsing import scan_tokens  # bddl 1.0.1 parsing.py:14-41 → iç içe liste
        tree = scan_tokens(filename=str(path)); src = "bddl.parsing.scan_tokens"
        lang = next(group for group in tree if isinstance(group, list) and group and group[0] == ":language")
        words = lang[1:]
    except ImportError:
        s = re.sub(r";.*$", "", raw, flags=re.MULTILINE).lower()
        toks = re.findall(r"[()]|[^\s()]+", s); src = "yerel kopya (bddl paketi yok)"
        i = toks.index(":language"); j = i + 1; words = []
        while toks[j] != ")": words.append(toks[j]); j += 1
    # bddl_utils.py:83 — kelimeler " ".join ile birleştirilir
    return " ".join(words), src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--commands", required=True); ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--out", default="tr_variants"); ap.add_argument("--install", action="store_true")
    a = ap.parse_args()
    lib = pathlib.Path(a.root) / "libero/libero"; bddl_dir = lib / "bddl_files" / a.suite
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tmap_lines, json_entries, report = [], [], []
    for row in csv.DictReader(open(a.commands, encoding="utf-8")):
        base = BASES[int(row["base_index"])]; k = row["tr_id"].strip(); text = row["text"].strip()
        probs = check_text(text)
        src = (bddl_dir / f"{base}.bddl").read_text(encoding="utf-8")
        n = len(re.findall(r"\(:language .*?\)", src)); assert n == 1, f"{base}: :language satırı {n} kez"
        new = re.sub(r"\(:language .*?\)", f"(:language {text})", src, count=1)
        fname = f"{base}_language_tr{k}.bddl"; (out / fname).write_text(new, encoding="utf-8")
        try:
            got, parser = parse_language_like_bddl(out / fname)
        except Exception as e:  # ';' → Missing close parenthesis, '(' → iç içe liste (TypeError)
            got, parser = f"AYRIŞTIRICI HATASI: {type(e).__name__}: {e}", "bddl.parsing.scan_tokens"
        same = got == text
        task_name = f"{base}_language_tr{k}{TAIL}"
        tmap_lines.append(f'    "{task_name}",')
        json_entries.append({"id": None, "name": task_name, "category": "Language Instructions", "difficulty_level": None, "lang": "tr"})
        report.append((fname, text, got, same, probs, parser))
        if a.install:
            shutil.copy(out / fname, bddl_dir / fname)
    # --- rapor
    print(f"# {len(report)} Türkçe varyant → {out}/   (ayrıştırıcı: {report[0][5]})\n")
    print("| dosya | yazılan | modele gidecek (task.language) | aynı mı | uyarı |\n|---|---|---|---|---|")
    for fname, text, got, same, probs, _ in report:
        print(f"| `{fname.replace(BASES[0], '…')}` | {text} | {got} | {'evet' if same else 'HAYIR'} | {'; '.join(probs) or '—'} |")
    print("\n# libero_suite_task_map.py → libero_task_map[\"%s\"] listesinin SONUNA ekle (indeksler kaymasın):" % a.suite)
    print("\n".join(tmap_lines))
    print("\n# task_classification.json → aynı suite listesinin sonuna (id = son id + 1; difficulty_level: 4 referans modelle ölçülmedi → null):")
    print(json.dumps(json_entries, ensure_ascii=False, indent=1))
    print("\n# Sonra: VLANEXT_TASK_IDS olarak yeni görevlerin 0 tabanlı indeksleri = eski liste uzunluğu, +1, …  (Ders 16: id-1 == indeks)")
    (out / "task_map_lines.txt").write_text("\n".join(tmap_lines) + "\n", encoding="utf-8")
    json.dump(json_entries, open(out / "task_classification_entries.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if a.install: print(f"\nKlona kopyalandı: {bddl_dir}  (task map ve JSON'u elle güncelle)")
    bad = [r for r in report if not r[3] or r[4]]
    if bad: print(f"\nUYARI: {len(bad)} varyantta metin değişti ya da uyarı var — Türkçe'yi baştan küçük harfle, ';' ve parantezsiz yaz.", file=sys.stderr)


if __name__ == "__main__":
    main()
```

## plot_day17.py

```python
"""Ders 17 — görsel 09: LIBERO-plus'ta 7 bozulma × model (VLANeXt makalesi Tablo III; sayılar elle aktarıldı).

Kullanım:  python plot_day17.py <çıktı_klasörü/>
Kaynak: Wu vd. 2026, VLANeXt, arXiv:2602.18532v2 Tablo III — LIBERO'da eğitilip LIBERO-plus'ta doğrudan test
(bozulma görülmemiş). Temiz LIBERO sayıları aynı makalenin LIBERO tablosundan (Ders 12 görsel 04'teki değerler).
"""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = (sys.argv[1] if len(sys.argv) > 1 else ".").rstrip("/") + "/"; os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
CATS = ["Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout"]
T3 = {  # model, suite -> 7 kategori
    ("OpenVLA-OFT", "Spatial"): [88.3, 40.0, 80.5, 98.3, 97.3, 96.3, 93.9],
    ("OpenVLA-OFT", "Object"): [38.9, 25.4, 99.0, 73.7, 97.6, 72.3, 71.8],
    ("OpenVLA-OFT", "Goal"): [62.0, 25.2, 53.2, 93.9, 92.5, 75.2, 59.1],
    ("OpenVLA-OFT", "Long"): [38.7, 38.2, 87.0, 89.4, 86.8, 63.5, 76.9],
    ("VLANeXt-B", "Spatial"): [95.7, 78.6, 86.9, 99.7, 98.8, 98.0, 96.6],
    ("VLANeXt-B", "Object"): [99.5, 48.5, 98.6, 99.3, 84.7, 99.8, 78.2],
    ("VLANeXt-B", "Goal"): [96.6, 63.6, 51.5, 97.5, 70.8, 96.8, 63.9],
    ("VLANeXt-B", "Long"): [69.7, 72.0, 90.3, 86.9, 75.8, 81.7, 84.3],
}
CLEAN = {("OpenVLA-OFT", "Spatial"): 97.6, ("OpenVLA-OFT", "Goal"): 97.9, ("VLANeXt-B", "Spatial"): 99.0, ("VLANeXt-B", "Goal"): 96.6}

fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
for ax, suite in zip(axes, ["Spatial", "Goal"]):
    x = np.arange(len(CATS)); w = 0.38
    for i, (model, col) in enumerate([("OpenVLA-OFT", "#3a9a5c"), ("VLANeXt-B", "#e09a00")]):
        vals = T3[(model, suite)]
        bars = ax.bar(x + (i - 0.5) * w, vals, w * 0.94, color=col, label=f"{model} (temiz LIBERO {suite}: {CLEAN[(model, suite)]})", zorder=3)
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
        ax.axhline(CLEAN[(model, suite)], color=col, lw=1, ls="--", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(CATS); ax.set_ylim(0, 108); ax.yaxis.grid(True, color="#e5e5e5", zorder=0)
    ax.set_title(f"LIBERO-plus {suite}: bozulma başına success rate (%)"); ax.legend(fontsize=8, loc="lower left")
    # Language sütununu vurgula
    ax.axvspan(1.5, 2.5, color="#ffd166", alpha=0.18, zorder=0)
axes[0].set_ylabel("Success rate (%)")
fig.suptitle("Hangi bozulma neyi kırıyor? OpenVLA-OFT vs VLANeXt-B, LIBERO'da eğitilip LIBERO-plus'ta test — kaynak: VLANeXt Tablo III (arXiv:2602.18532v2)", fontsize=10.5)
fig.text(0.5, 0.005, "Kesikli çizgi: aynı modelin temiz LIBERO skoru. Spatial'da Robot (40→79) ve Camera en çok kıran; Goal'da Language (53 / 52) iki modelde de en kötü — komut okunmak zorunda olan tek suite.",
         ha="center", fontsize=8.5, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig(out + "09-vlanext-per-perturbation.png", dpi=170); plt.close(fig)
print("yazıldı:", out + "09-vlanext-per-perturbation.png")
```

## tr_commands.csv

```csv
base_index,tr_id,text,source_id
0,1,tabak ile ramekinin arasındaki siyah kaseyi al ve tabağın üstüne koy,sp0_tr
0,2,tabakla ramekin arasında duran siyah kâseyi kaldır ve tabağa yerleştir,sp0_tr2
1,1,ramekinin yanındaki siyah kaseyi al ve tabağın üstüne koy,sp1_tr
1,2,ramekinin yanında duran siyah kâseyi kaldır ve tabağa yerleştir,sp1_tr2
2,1,masanın ortasındaki siyah kaseyi al ve tabağın üstüne koy,sp2_tr
2,2,masanın ortasında duran siyah kâseyi kaldır ve tabağa yerleştir,sp2_tr2
3,1,kurabiye kutusunun üstündeki siyah kaseyi al ve tabağın üstüne koy,sp3_tr
3,2,kurabiye kutusunun üzerinde duran siyah kâseyi kaldır ve tabağa yerleştir,sp3_tr2
4,1,ahşap dolabın üst çekmecesindeki siyah kaseyi al ve tabağın üstüne koy,sp4_tr
4,2,ahşap dolabın üst çekmecesinde duran siyah kâseyi kaldır ve tabağa yerleştir,sp4_tr2
```
