"""Ders 19 — "Shortcut mümkün mü?" sorusunu BDDL'den oku: dört LIBERO suite'inde sahne (nesneler + init bölgeleri)
hedefi (goal) tek başına belirliyor mu?

Mantık: bir policy komutu hiç okumadan görevi ancak sahne görevi "söylüyorsa" çözebilir. Suite'teki her görev için
  sahne imzası = (nesne türleri çoklu kümesi, her nesnenin init bölgesi; aynı türden nesneler ayırt edilemez)
  hedef       = (:goal) ifadesi
Aynı sahne imzasına birden fazla hedef düşüyorsa sahne→görev bir fonksiyon değildir: komut okunmadan görev seçilemez.

Kullanım:
  python suite_structure.py --libero /yol/LIBERO            # klon kökü (libero/libero/bddl_files altını okur)
  python suite_structure.py --libero /yol/LIBERO --out results/suite_structure.csv --md results/suite_structure.md
Yalnızca dosya okur (regex ile); LIBERO paketini import etmez, model/GPU gerekmez.
"""
import argparse, csv, pathlib, re
from collections import Counter, defaultdict

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def parse_bddl(path):
    t = path.read_text(encoding="utf-8")
    lang = re.search(r"\(:language\s+([^)]*)\)", t).group(1).strip()
    objs = {}
    m = re.search(r"\(:objects(.*?)\)\s*\(:obj_of_interest", t, re.S)
    for line in m.group(1).strip().splitlines():
        line = line.strip()
        if not line or " - " not in line:
            continue
        names, typ = line.rsplit(" - ", 1)
        for n in names.split():
            objs[n] = typ.strip()
    init_txt = re.search(r"\(:init(.*?)\)\s*\(:goal", t, re.S).group(1)
    init = re.findall(r"\((On|In)\s+(\S+)\s+(\S+)\)", init_txt)
    where = {o: reg for _, o, reg in init}
    goal = re.sub(r"\s+", " ", re.search(r"\(:goal\s*(.*?)\)\s*\)\s*$", t, re.S).group(1).strip())
    # sahne imzası: nesne TÜRÜ + bölge (adlar değil: iki siyah kâse görüntüde ayırt edilemez)
    scene = tuple(sorted((objs.get(o, o), reg) for _, o, reg in init if o in objs))
    fixtures = tuple(sorted((o, reg) for _, o, reg in init if o not in objs))
    # göreli hedef: nesne adlarını "tür@init-bölgesi" ile değiştir → aynı sahnede farklı fiziksel hedefler ayrışır
    rel_goal = re.sub(r"\b([a-z_]+_\d)\b", lambda m_: f"{objs.get(m_.group(1), m_.group(1))}@{where.get(m_.group(1), '-')}", goal)
    return dict(task=path.stem, language=lang, objects=objs, where=where, scene=scene, fixtures=fixtures, goal=goal, rel_goal=rel_goal)


def official_order(root, suite):
    """libero_suite_task_map.py'deki sıra (benchmark'ın task_id'si; alfabetik DEĞİL). Dosya okunur, import edilmez."""
    t = (root / "libero" / "libero" / "benchmark" / "libero_suite_task_map.py").read_text(encoding="utf-8")
    block = re.search(r'"%s":\s*\[(.*?)\]' % suite, t, re.S).group(1)
    return re.findall(r'"([^"]+)"', block)


def analyze(root):
    rows = []
    for suite in SUITES:
        d = root / "libero" / "libero" / "bddl_files" / suite
        tasks = [parse_bddl(d / (name + ".bddl")) for name in official_order(root, suite)]
        scenes = Counter(t["scene"] + t["fixtures"] for t in tasks)
        by_scene = defaultdict(list)
        for t in tasks:
            by_scene[t["scene"] + t["fixtures"]].append(t)
        n_scenes = len(scenes)
        max_share = max(scenes.values())
        # sahneden hedef çıkar mı: her sahneye tek hedef düşüyorsa evet
        deterministic = all(len({x["rel_goal"] for x in v}) == 1 for v in by_scene.values())
        # komut okunmadan en iyi durumda (sahneye en sık düşen hedefi seçerek) kaç görev çözülür
        best_blind = sum(max(Counter(x["rel_goal"] for x in v).values()) for v in by_scene.values())
        obj_sets = Counter(tuple(sorted(Counter(t["objects"].values()).items())) for t in tasks)
        rows.append(dict(suite=suite, n_tasks=len(tasks), n_scenes=n_scenes, n_object_sets=len(obj_sets),
                         max_tasks_per_scene=max_share, scene_determines_goal=deterministic,
                         blind_upper_bound=f"{best_blind}/{len(tasks)}", tasks=tasks, by_scene=by_scene))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero", required=True, help="LIBERO klon kökü")
    ap.add_argument("--out", default=None)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    rows = analyze(pathlib.Path(a.libero))
    hdr = ["suite", "n_tasks", "n_scenes", "n_object_sets", "max_tasks_per_scene", "scene_determines_goal", "blind_upper_bound"]
    print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join(str(r[h]) for h in hdr) + " |")
    print("\nAçıklama: n_scenes = farklı sahne imzası (nesne türü + init bölgesi; fixture dahil); scene_determines_goal = her sahneye tek hedef;"
          "\nblind_upper_bound = komutu hiç okumayan ama sahneyi mükemmel tanıyan bir policy'nin çözebileceği görev sayısı (üst sınır).")
    sp = next(r for r in rows if r["suite"] == "libero_spatial")
    short = lambda reg: reg.replace("main_table_", "").replace("wooden_cabinet_1_", "cab:").replace("flat_stove_1_", "stove:").replace("_region", "")
    print("\nlibero_spatial — görev: kâse bölgeleri (BDDL :init) → hedef kâse (:goal):")
    for i, t in enumerate(sp["tasks"]):
        tgt = re.search(r"\(On\s+(\S+)\s+plate_1\)", t["goal"]).group(1)
        print(f"  {i}: bowl_1@{short(t['where']['akita_black_bowl_1']):<22} bowl_2@{short(t['where']['akita_black_bowl_2']):<22} hedef={tgt} ({short(t['where'][tgt])})")
    role = defaultdict(lambda: {"hedef": [], "öteki": []})
    for i, t in enumerate(sp["tasks"]):
        tgt = re.search(r"\(On\s+(\S+)\s+plate_1\)", t["goal"]).group(1)
        for o in ("akita_black_bowl_1", "akita_black_bowl_2"):
            role[short(t["where"][o])]["hedef" if o == tgt else "öteki"].append(i)
    print("\nBölge → hangi görevlerde hedef / öteki kâse:")
    for reg, d in sorted(role.items()):
        print(f"  {reg:<22} hedef:{d['hedef']}  öteki:{d['öteki']}")
    print("\nAynı sahne imzasını paylaşan görevler (komut okunmadan ayrılamaz):")
    for r in rows:
        for v in r["by_scene"].values():
            if len(v) > 1:
                print(f"  {r['suite']}: " + " | ".join(f"{x['task']} → {x['rel_goal']}" for x in v))
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(hdr)
            for r in rows:
                w.writerow([r[h] for h in hdr])
    if a.md:
        lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] + ["| " + " | ".join(str(r[h]) for h in hdr) + " |" for r in rows]
        pathlib.Path(a.md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
