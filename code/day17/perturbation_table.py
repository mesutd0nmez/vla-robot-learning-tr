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
