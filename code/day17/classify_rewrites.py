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
