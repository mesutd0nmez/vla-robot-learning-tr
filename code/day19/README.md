# day19

Ders 19 — Sonuç analizi: shortcut learning var mı, hangi suite'te? Üç soruya üç araç: (1) Ders 18'in 4 satırlık tablosunu dört örüntüye göre okuyan ve paragraf taslağı üreten `read_patterns.py`; (2) "shortcut yapısal olarak mümkün mü?" sorusunu BDDL'den cevaplayan `suite_structure.py` (Spatial 9/10, Object 10/10, Goal 1/10, Long 9/10); (3) "hangi init state kırılgan?" sorusuna sahne özellikleri çıkaran `init_state_geometry.py` (50 init state, torch/LIBERO kurulmadan). Hiçbiri model/GPU istemez; yalnızca LIBERO deposunun **dosyalarını** okurlar (`git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO`).

| Dosya | Ne | Durum |
|---|---|---|
| `read_patterns.py` | `episodes.csv` → Wilson, Δ, uyumsuz çiftler, McNemar; dört örüntüden (A/B/C/D, C+B, C+D, belirsiz) birine karar; `--features` ile uyumsuz sahnelerin geometrisi, `--labels` ile `grasped_bowl` → faithful/biased; "1 paragraf yorum" taslağı | `--demo` (görsel 07'nin 4 örüntüsü doğru sınıflandı) ve `--selftest` (Ders 18 tablosu → B) geçti; features + labels ile uçtan uca çalıştırıldı |
| `suite_structure.py` | Dört suite'in BDDL'lerinden sahne imzası (nesne türü + init bölgesi) ve göreli hedef; sahne→hedef bir fonksiyon mu, komutsuz üst sınır kaç; Spatial 10 görev × bölge tablosu | gerçek dosyalarla çalıştırıldı (`libero_suite_task_map.py` sırası) |
| `init_state_geometry.py` | `.pruned_init` (torch zip içinde numpy pickle) → kısıtlı unpickler ile (50, 92) dizi; nesne (x, y), bölge doğrulaması, kâse–kâse / kâse–tabak mesafeleri, robot eklem sapması → `init_features_task0.csv` | 10 Spatial görevi + Goal görev 8 için çalıştırıldı; 5 nesne × 50 state bölge içinde |
| `plot_day19.py` | Görsel 02 (komutsuz SR × suite, üç makale + yapısal üst sınır), 03 (50 init state, gerçek), 04 (Spatial sahne matrisi), 05 (LIBERO-CF faithful/biased) | çalıştırıldı |
| `conditions_goal8.json`, `t4_goal_config.yaml` | Aynı deneyin LIBERO-Goal görev 8'e taşınmış hâli (`wrong` = görev 1 "put the bowl on the stove") | yazıldı; koşulmadı |

Mac'te (model gerekmez; `vla` env yeter, numpy ve matplotlib):
```bash
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO ~/src/LIBERO    # yalnızca dosyaları okuyacağız
cd code/day19
python read_patterns.py --demo
python read_patterns.py --selftest
python suite_structure.py --libero ~/src/LIBERO
python init_state_geometry.py --libero ~/src/LIBERO --task 0 --csv init_features_task0.csv
python plot_day19.py --libero ~/src/LIBERO --out ../../lectures/images/lecture-19
# Colab koşun bittiğinde (Ders 18 analiz çıktısı):
python read_patterns.py ../../projects/05-spatial0-4kosul/results/episodes.csv \
       --features ../../datasets/03-libero-spatial0-4cond/data/init_features_task0.csv \
       --labels   ../../projects/05-spatial0-4kosul/results/grasped.csv      # init_state,condition,grasped_bowl
```

## read_patterns.py

```python
"""Ders 19 — 4 koşullu tabloyu (Ders 18 çıktısı) dört örüntüye göre oku ve "1 paragraf yorum" taslağı üret.

Girdi: Ders 18'in episodes.csv'si (run_id, suite, task_id, condition, init_state, success, ...).
İsteğe bağlı: --features init_features_task0.csv (init_state_geometry.py) → uyumsuz çiftlerin sahne özellikleri;
              --labels grasped.csv (init_state, condition, grasped_bowl ∈ {bowl_1, bowl_2, none}) → faithful/biased sayımı.

Kullanım:
  python read_patterns.py results/episodes.csv [--features init_features_task0.csv] [--labels grasped.csv]
  python read_patterns.py --demo          # Ders 18 görsel 07'nin dört temsili örüntüsünü sınıflandırır
  python read_patterns.py --selftest      # Ders 18 selftest tablosu (9/5/7/4) → beklenen çıktı

Kurallar (pilot için sezgisel eşikler; parametre, dogma değil):
  near  = fark ≤ NEAR (varsayılan 15 puan)   → "yakın"
  far   = fark ≥ FAR  (varsayılan 25 puan)   → "uzak"
  A  dil görmezden geliniyor : empty ve wrong clean'e yakın
  C  yanlış komut 'biased'   : wrong clean'e yakın AMA empty uzak (komut 'bir kâse al' diyor, sahne hangisini seçiyor)
  B  Türkçe okunmuyor        : dil okunuyor (empty uzak) ve tr ≈ empty (tr − empty ≤ NEAR/1.5)
  D  çok dilli omurga        : dil okunuyor ve tr − empty ≥ FAR
  aradakiler "belirsiz" → daha çok episode. C, tr eksenine eklenir: "C+B", "C+D" (görsel 07'nin C paneli tr=empty olduğu için C+B'dir).
Wilson aralıkları çakışıyorsa (n=10'da hemen hep çakışır) sonuç "yön", "bulgu" değil; script bunu her satıra yazar.
"""
import argparse, csv, math, pathlib, sys
from collections import OrderedDict, defaultdict

NEAR, FAR = 15, 25


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load_episodes(path):
    eps = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            eps[r["condition"]][int(r["init_state"])] = int(r["success"]) == 1
    return eps


def classify(sr, near=NEAR, far=FAR):
    """sr: {koşul: yüzde}. Dönüş: (harf, gerekçe listesi)."""
    c, e, w, t = sr["clean"], sr["empty"], sr.get("wrong"), sr.get("tr")
    why = []
    gap_e = c - e
    lang_read = gap_e >= far
    if gap_e <= near and (w is None or c - w <= near):
        why.append(f"empty clean'e yakın (Δ={gap_e:+.0f}); wrong da yakın" if w is not None else f"empty clean'e yakın (Δ={gap_e:+.0f})")
        return "A", why
    if w is not None and c - w <= near and lang_read:
        why.append(f"wrong clean'e yakın (Δ={c - w:+.0f}) ama empty uzak (Δ={gap_e:+.0f}) → komutun 'bir kâse al' kısmı okunuyor, hedef sahneden")
        letter_w = "C"
    else:
        letter_w = None
    if t is not None and lang_read:
        if t - e <= near / 1.5:
            why.append(f"tr ≈ empty (tr−empty={t - e:+.0f}) → Türkçe'den sahnenin üstüne bir şey çıkmıyor")
            return (letter_w or "B") if letter_w != "C" else "C+B", why
        if t - e >= far:
            why.append(f"tr empty'nin çok üstünde (tr−empty={t - e:+.0f}) → Türkçe'den en azından nesne/eylem çıkıyor")
            return (letter_w or "D") if letter_w != "C" else "C+D", why
        why.append(f"tr empty ile clean arasında (tr−empty={t - e:+.0f}): B ile D arasında belirsiz")
        return (letter_w or "belirsiz"), why
    if letter_w:
        return letter_w, why
    why.append(f"empty düşüşü {gap_e:+.0f} puan: {near} ile {far} arasında — ne 'görmezden geliyor' ne 'okuyor' denebilir")
    return "belirsiz", why


def paragraph(letter, rows, n):
    c = rows["clean"]; e = rows["empty"]; w = rows.get("wrong"); t = rows.get("tr")
    head = {"A": "Bu pilotta komutun içeriği sonucu değiştirmedi",
            "B": "Model komutu okuyor ama Türkçe komut sahne tabanının üstüne çıkmıyor",
            "C": "Model komutun eylem kısmını okuyor, hedef nesneyi sahneden seçiyor",
            "D": "Türkçe komut sahne tabanının belirgin üstünde",
            "C+B": "Yanlış komut 'biased', Türkçe komut tabanda", "C+D": "Yanlış komut 'biased' ama Türkçe'den bir şey çıkıyor",
            "belirsiz": "Pilot yön vermiyor"}[letter]
    s = (f"{head}: LIBERO-Spatial görev 0'da {n} eşli init state'te success rate clean {c['k']}/{n}, empty {e['k']}/{n}"
         + (f", wrong {w['k']}/{n}" if w else "") + (f", tr {t['k']}/{n}" if t else "") + ". ")
    s += f"Uyumsuz çiftler (clean✓x✗ / clean✗x✓): empty {e['b']}/{e['c']}" + (f", wrong {w['b']}/{w['c']}" if w else "") + (f", tr {t['b']}/{t['c']}" if t else "") + ". "
    s += ("Wilson %95 aralıkları çakıştığı için bu bir yön bulgusudur; 50 init state'lik koşu (Ders 18 §5) aynı oranları verirse "
          "anlamlılık ve etki büyüklüğü birlikte raporlanır. ")
    if letter in ("A",):
        s += "Spatial görev 0'da sahne görevi tek başına belirliyor (suite_structure.py: 10 görevin 9'u sahneden ayrışır); Türkçe etkisi bu görevde ölçülemez — Goal suite'e geç."
    elif letter.startswith("C"):
        s += "`wrong` videolarındaki grasped_bowl etiketi biased (bowl_1) ağırlıklıysa LIBERO-CF'in 'vision overrides language' örüntüsü; faithful (bowl_2) ağırlıklıysa model komutu okuyor ve success metriği onu cezalandırıyor."
    elif letter in ("B", "C+B"):
        s += "Chen vd.'nin 'yabancı dil ≈ komut yok' örüntüsü; tokenizasyon (Ders 21) ve çeviri katmanı / Türkçe LoRA (Ders 23) karşılaştırmasının gerekçesi."
    elif letter in ("D", "C+D"):
        s += "Qwen3-VL'nin çok dilli ön-eğitimi Türkçe'den nesne/eylem çıkarıyor olabilir (hipotez); mekânsal ilişkinin (arasındaki) okunup okunmadığı distractor koşuluyla ayrıca test edilmeli (Ders 15)."
    return s


def report(eps, features=None, labels=None, near=NEAR, far=FAR, title=""):
    order = [c for c in ["clean", "empty", "wrong", "tr"] if c in eps] + [c for c in eps if c not in ("clean", "empty", "wrong", "tr")]
    base = eps["clean"]; n = len(base)
    rows = OrderedDict()
    for cnd in order:
        d = eps[cnd]; k = sum(d.values()); lo, hi = wilson(k, len(d))
        common = sorted(set(base) & set(d))
        b = sum(base[i] and not d[i] for i in common); c = sum((not base[i]) and d[i] for i in common)
        rows[cnd] = dict(k=k, n=len(d), sr=100 * k / len(d), lo=100 * lo, hi=100 * hi, b=b, c=c, p=mcnemar(b, c),
                         disc=[i for i in common if base[i] != d[i]])
    if title:
        print(f"\n=== {title}")
    print(f"| koşul | k/n | SR | Wilson | Δ | uyumsuz b/c | McNemar | clean ile aralık |")
    print("|---|---|---|---|---|---|---|---|")
    cl = rows["clean"]
    for cnd, r in rows.items():
        overlap = "—" if cnd == "clean" else ("çakışıyor → yön" if r["hi"] >= cl["lo"] and cl["hi"] >= r["lo"] else "ayrışıyor")
        d = "—" if cnd == "clean" else f"{r['sr'] - cl['sr']:+.0f}"
        pm = "—" if cnd == "clean" else ("—" if r["p"] != r["p"] else f"{r['p']:.3f}")
        print(f"| {cnd} | {r['k']}/{r['n']} | {r['sr']:.0f}% | {r['lo']:.0f}–{r['hi']:.0f} | {d} | {'—' if cnd == 'clean' else f'{r['b']}/{r['c']}'} | {pm} | {overlap} |")
    sr = {c_: r["sr"] for c_, r in rows.items()}
    letter, why = classify(sr, near, far)
    print(f"\nÖrüntü: **{letter}**  (eşikler: yakın ≤ {near}, uzak ≥ {far} puan)")
    for w_ in why:
        print(f"  - {w_}")
    # uyumsuz çiftler sahne bazında
    disc_all = sorted({i for c_, r in rows.items() if c_ != "clean" for i in r["disc"]})
    if disc_all:
        print("\nUyumsuz init state'ler (clean ile farklı sonuç veren sahneler):")
        for i in disc_all:
            marks = " ".join(f"{c_}={'✓' if eps[c_][i] else '✗'}" for c_ in order)
            extra = ""
            if features and i in features:
                f_ = features[i]
                extra = f"  | kâse–kâse {100 * f_['d_bowl1_bowl2']:.1f} cm, hedef–tabak {100 * f_['d_bowl1_plate']:.1f} cm, robot sapma {f_['robot_q_dev']:.3f} rad"
            print(f"  init {i:2d}: {marks}{extra}")
        if features:
            fd = [features[i]["robot_q_dev"] for i in disc_all if i in features]
            fa = [features[i]["robot_q_dev"] for i in base if i in features]
            if fd and fa:
                print(f"  robot başlangıç sapması: uyumsuz sahnelerde medyan {sorted(fd)[len(fd) // 2]:.3f} rad, tüm sahnelerde {sorted(fa)[len(fa) // 2]:.3f} rad "
                      f"(fark büyükse kırılganlık robotun başlangıç pozundan; nesneler ±1.5 cm içinde, Ders 19 §3)")
    if labels:
        cnt = defaultdict(lambda: defaultdict(int))
        for (cnd, i), g in labels.items():
            cnt[cnd][g] += 1
        print("\ngrasped_bowl etiketleri (videodan, elle):")
        for cnd, d in cnt.items():
            tot = sum(d.values())
            s_ = ", ".join(f"{g}: {v}/{tot}" for g, v in sorted(d.items()))
            if cnd == "wrong":
                s_ += f"  → faithful (bowl_2) {100 * d.get('bowl_2', 0) / tot:.0f}%, biased (bowl_1) {100 * d.get('bowl_1', 0) / tot:.0f}% (LIBERO-CF grounding ölçüsünün elle sayılmış hâli)"
            print(f"  {cnd}: {s_}")
    print("\nParagraf taslağı:\n" + paragraph(letter, rows, n))
    return letter


def demo():
    pats = {"A": (9, 8, 8, 8), "B": (9, 4, 4, 4), "C": (9, 4, 8, 4), "D": (9, 4, 4, 7)}
    for name, (c, e, w, t) in pats.items():
        eps = {"clean": {i: i < c for i in range(10)}, "empty": {i: i < e for i in range(10)}, "wrong": {i: i < w for i in range(10)}, "tr": {i: i < t for i in range(10)}}
        got = report(eps, title=f"temsili örüntü {name} (Ders 18 görsel 07)")
        assert got == {"A": "A", "B": "B", "C": "C+B", "D": "D"}[name], (name, got)
    print("\ndemo OK: dört temsili örüntü doğru sınıflandı")


def selftest():
    pattern = {"clean": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0], "empty": [1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
               "wrong": [1, 1, 1, 0, 1, 1, 1, 0, 1, 0], "tr": [1, 0, 0, 1, 0, 0, 1, 0, 1, 0]}
    eps = {c: {i: bool(v) for i, v in enumerate(vs)} for c, vs in pattern.items()}
    got = report(eps, title="Ders 18 selftest tablosu (TEMSİLİ)")
    assert got == "B", got   # empty −40 (uzak), wrong −20 (ne yakın ne uzak → C değil), tr−empty = −10 ≤ 10 → tr ≈ empty → B
    print("\nselftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="?")
    ap.add_argument("--features"); ap.add_argument("--labels")
    ap.add_argument("--near", type=float, default=NEAR); ap.add_argument("--far", type=float, default=FAR)
    ap.add_argument("--demo", action="store_true"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.demo:
        return demo()
    if a.selftest:
        return selftest()
    if not a.episodes:
        ap.error("episodes.csv ver (ya da --demo / --selftest)")
    eps = load_episodes(a.episodes)
    if "clean" not in eps or "empty" not in eps:
        sys.exit("episodes.csv'de clean ve empty koşulları gerekli")
    feats = None
    if a.features:
        with open(a.features, encoding="utf-8") as f:
            feats = {int(r["init_state"]): {k: float(v) for k, v in r.items() if k != "init_state"} for r in csv.DictReader(f)}
    labels = None
    if a.labels:
        with open(a.labels, encoding="utf-8") as f:
            labels = {(r["condition"], int(r["init_state"])): r["grasped_bowl"] for r in csv.DictReader(f)}
    report(eps, feats, labels, a.near, a.far, title=pathlib.Path(a.episodes).name)


if __name__ == "__main__":
    main()
```

## suite_structure.py

```python
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
```

## init_state_geometry.py

```python
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
```

## plot_day19.py

```python
"""Ders 19 görselleri (matplotlib). Gerçek değerler: makale tabloları (kaynak başlıkta), BDDL ve .pruned_init dosyaları.
  python plot_day19.py --libero /yol/LIBERO --out ../../lectures/images/lecture-19
Görsel 02: komutsuz success rate × suite (Chen Ø / Zhan blank / Fang vision-only) + yapısal üst sınır (suite_structure.py)
Görsel 03: LIBERO-Spatial görev 0'ın 50 init state'i (gerçek), BDDL bölgeleri, kâse–kâse mesafe dağılımı
Görsel 04: Spatial 10 görev × bölge matrisi (hedef/öteki kâse) — görev 7 ve 9 aynı sahne
Görsel 05: LIBERO-CF grounding: faithful vs biased (Fang vd. Tablo III)
"""
import argparse, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
YE, BE, GE, RE = "#b8860b", "#3b6fa0", "#2e8b57", "#c0392b"
SUITES = ["Spatial", "Object", "Goal", "Long"]

# ---- gerçek değerler (makale tabloları) ----
# Chen vd. 2606.15714 Tablo 6, Ø (komutsuz) sütunu; en sütunu parantezde
CHEN = {"π0.5": {"en": [97.5, 99.0, 94.0, 93.0], "Ø": [66.0, 64.5, 9.0, 70.0]},
        "OpenVLA-OFT": {"en": [93.5, 100.0, 97.5, 94.5], "Ø": [80.0, 91.5, 10.0, 76.5]},
        "Qwen3-VL-π": {"en": [93.0, 99.5, 97.0, 96.5], "Ø": [41.0, 57.5, 6.0, 60.0]},
        "Qwen3-VL-OFT": {"en": [92.5, 100.0, 99.0, 91.0], "Ø": [34.0, 42.5, 8.5, 45.0]}}
# Zhan vd. 2601.04052 Tablo 8/9: original / blank
ZHAN = {"π0.5": {"orig": [95.4, 98.4, 97.2, 89.6], "blank": [65.0, 51.8, 11.4, 72.0]},
        "π0": {"orig": [96.8, 98.8, 95.8, 85.2], "blank": [29.8, 41.2, 4.6, 25.2]}}
# Fang vd. 2602.17659 Tablo I: V+L / vision-only
FANG = {"π0.5": {"vl": [98.8, 98.2, 98.0, 92.4], "v": [68.2, 56.6, 10.2, 71.6]},
        "OpenVLA-OFT": {"vl": [97.6, 98.4, 97.9, 94.5], "v": [83.2, 98.2, 10.0, 85.0]},
        "π0": {"vl": [96.8, 98.8, 95.8, 85.2], "v": [28.0, 26.8, 3.4, 15.6]}}
# suite_structure.py çıktısı: komutu okumayan ama sahneyi mükemmel tanıyan policy'nin üst sınırı (görev sayısı / 10)
BLIND = {"Spatial": 90, "Object": 100, "Goal": 10, "Long": 90}
# Fang vd. Tablo III, grounding (faithful / biased), CAG yok
CF = {"OpenVLA-OFT": {"CF-Spatial": (6.8, 84.7), "CF-Object": (2.0, 99.6), "CF-Long": (4.0, 62.9), "CF-OOD": (6.0, 87.3)},
      "π0": {"CF-Spatial": (70.9, 37.1), "CF-Object": (6.0, 59.4), "CF-Long": (20.4, 83.1), "CF-OOD": (18.0, 72.4)},
      "π0.5": {"CF-Spatial": (39.3, 61.3), "CF-Object": (11.4, 68.8), "CF-Long": (51.6, 58.4), "CF-OOD": (20.7, 74.0)}}


def fig02(out):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
    series = [("π0.5 — Chen Ø", CHEN["π0.5"]["Ø"], Y, YE), ("π0.5 — Zhan blank", ZHAN["π0.5"]["blank"], Y, YE),
              ("π0.5 — Fang V-only", FANG["π0.5"]["v"], Y, YE), ("OpenVLA-OFT — Chen Ø", CHEN["OpenVLA-OFT"]["Ø"], B, BE),
              ("OpenVLA-OFT — Fang V-only", FANG["OpenVLA-OFT"]["v"], B, BE), ("π0 — Zhan blank", ZHAN["π0"]["blank"], G, GE),
              ("Qwen3-VL-π — Chen Ø", CHEN["Qwen3-VL-π"]["Ø"], R, RE), ("Qwen3-VL-OFT — Chen Ø", CHEN["Qwen3-VL-OFT"]["Ø"], R, RE)]
    for ax, suite, si in zip(axes, SUITES, range(4)):
        xs = np.arange(len(series))
        for x, (name, vals, fc, ec) in zip(xs, series):
            ax.bar(x, vals[si], color=fc, edgecolor=ec, linewidth=1.2)
            ax.text(x, vals[si] + 1.5, f"{vals[si]:.0f}", ha="center", fontsize=8)
        ax.axhline(BLIND[suite], color="k", ls="--", lw=1.2)
        ax.text(len(series) - 0.4, BLIND[suite] + (2.5 if BLIND[suite] > 50 else 14), f"yapısal üst sınır {BLIND[suite]}", ha="right", fontsize=8.5, fontweight="bold")
        ax.set_title(f"LIBERO-{suite}", fontsize=12)
        ax.set_xticks(xs); ax.set_xticklabels([s[0].replace(" — ", "\n") for s in series], rotation=90, fontsize=7.5)
        ax.set_ylim(0, 108); ax.grid(axis="y", alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("success rate (%) — KOMUTSUZ")
    fig.suptitle("Komutsuz (boş / Ø / vision-only) success rate × suite — üç makale, gerçek tablo değerleri; kesikli çizgi: komutu okumayan ama sahneyi\n"
                 "mükemmel tanıyan policy'nin üst sınırı (BDDL'den, suite_structure.py: Spatial 9/10, Object 10/10, Goal 1/10, Long 9/10). "
                 "Kaynak: Chen vd. 2606.15714 T6; Zhan vd. 2601.04052 T9; Fang vd. 2602.17659 T1", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(out / "02-empty-by-suite.png", dpi=150); plt.close(fig)


def fig03(out, libero):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from init_state_geometry import load_init, parse_bddl, official_order, object_xy
    root = pathlib.Path(libero)
    name = official_order(root, "libero_spatial")[0]
    info = parse_bddl(root / "libero/libero/bddl_files/libero_spatial" / f"{name}.bddl")
    st = load_init(root / "libero/libero/init_files/libero_spatial" / f"{name}.pruned_init")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.6, 1]})
    colors = {"akita_black_bowl_1": (Y, YE, "hedef kâse (bowl_1)"), "akita_black_bowl_2": (R, RE, "öteki kâse (bowl_2) — 'wrong' komutun hedefi"),
              "plate_1": (G, GE, "tabak"), "cookies_1": (B, BE, "kurabiye kutusu"), "glazed_rim_porcelain_ramekin_1": ("#eeeeee", "#777777", "ramekin")}
    for k, o in enumerate(info["objects"]):
        xyz, _ = object_xy(st, k); fc, ec, lab = colors[o]
        reg = info["regions"][info["where"][o]][1]
        ax.add_patch(Rectangle((reg[1], reg[0]), reg[3] - reg[1], reg[2] - reg[0], fill=False, ec="k", lw=2.4, ls=(0, (3, 2)), zorder=7))
        ax.scatter(xyz[:, 1], xyz[:, 0], s=28, c=fc, edgecolors=ec, linewidths=0.6, label=lab, zorder=3, alpha=0.6)
        ax.scatter(xyz[:10, 1], xyz[:10, 0], s=70, facecolors="none", edgecolors="#333333", linewidths=0.9, zorder=4)
        ax.text(xyz[:, 1].mean(), xyz[:, 0].mean() + 0.03, lab.split(" — ")[0], ha="center", fontsize=8.5)
    ax.scatter([], [], s=70, facecolors="none", edgecolors="k", label="siyah halka: ilk 10 init state (Ders 18 pilotu)")
    ax.set_xlabel("y (m) — masa boyunca"); ax.set_ylabel("x (m) — robota doğru azalır (kamera robotun karşısında, Ders 17)")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left"); ax.set_ylim(-0.24, 0.12)
    ax.set_title("LIBERO-Spatial görev 0: 50 init state'in nesne konumları (GERÇEK, .pruned_init)\nkesikli kutular = BDDL :init bölgeleri (2 cm × 2 cm); nesneler fiziksel oturmadan sonra ±1.5 cm içinde", fontsize=10)
    b1, _ = object_xy(st, 0); b2, _ = object_xy(st, 1); pl, _ = object_xy(st, 4)
    d12 = np.linalg.norm(b1[:, :2] - b2[:, :2], axis=1) * 100; d1p = np.linalg.norm(b1[:, :2] - pl[:, :2], axis=1) * 100
    q = st[:, 1:8]; qdev = np.linalg.norm(q - q.mean(0), axis=1)
    ax2.hist(d12, bins=12, color=R, edgecolor=RE, label=f"kâse–kâse: {d12.mean():.1f} cm (min {d12.min():.1f}, maks {d12.max():.1f})")
    ax2.hist(d1p, bins=12, color=G, edgecolor=GE, alpha=0.9, label=f"hedef kâse–tabak: {d1p.mean():.1f} cm (min {d1p.min():.1f}, maks {d1p.max():.1f})")
    ax2.set_xlabel("mesafe (cm)"); ax2.set_ylabel("init state sayısı (50)")
    ax2.set_title(f"Sahneler birbirine ne kadar benziyor?\nnesne yayılımı ≤ 3 cm; kâse yönelimi sabit\nrobot eklem std ≈ {q.std(0).mean():.3f} rad (~1°), sapma normu medyan {np.median(qdev):.3f} rad\n"
                  f"(karş. LIBERO-Plus robot bozulması 0.1–0.5 rad, Ders 17)", fontsize=10)
    ax2.legend(fontsize=8.5); ax2.grid(alpha=0.3); ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "03-init-states-task0.png", dpi=150); plt.close(fig)


def fig04(out, libero):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from init_state_geometry import parse_bddl, official_order
    import re
    root = pathlib.Path(libero)
    names = official_order(root, "libero_spatial")
    short = lambda reg: reg.replace("main_table_", "").replace("wooden_cabinet_1_", "cab:").replace("flat_stove_1_", "stove:").replace("_region", "") \
        .replace("glazed_rim_porcelain_ramekin_1", "on ramekin").replace("cookies_1", "on cookie box")
    infos = [parse_bddl(root / "libero/libero/bddl_files/libero_spatial" / f"{n}.bddl") for n in names]
    regions = sorted({short(i["where"][o]) for i in infos for o in ("akita_black_bowl_1", "akita_black_bowl_2")})
    M = np.zeros((10, len(regions)))
    for ti, i in enumerate(infos):
        for o in ("akita_black_bowl_1", "akita_black_bowl_2"):
            M[ti, regions.index(short(i["where"][o]))] = 2 if o == i["target"] else 1
    fig, ax = plt.subplots(figsize=(13, 6.2))
    cmap = matplotlib.colors.ListedColormap(["white", R, Y])
    ax.imshow(M, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    for ti in range(10):
        for rj in range(len(regions)):
            if M[ti, rj] == 2: ax.text(rj, ti, "HEDEF", ha="center", va="center", fontsize=8.5, fontweight="bold", color=YE)
            elif M[ti, rj] == 1: ax.text(rj, ti, "öteki", ha="center", va="center", fontsize=8.5, color=RE)
    labels = [f"{ti}: …{re.sub(r'pick_up_the_black_bowl_|_and_place_it_on_the_plate', '', n).replace('_', ' ')}" for ti, n in enumerate(names)]
    ax.set_yticks(range(10)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticks(range(len(regions))); ax.set_xticklabels(regions, rotation=35, ha="right", fontsize=9)
    ax.set_xticks(np.arange(-.5, len(regions), 1), minor=True); ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
    ax.grid(which="minor", color="#dddddd", lw=0.8); ax.tick_params(which="minor", length=0)
    for ti in (7, 9):
        ax.add_patch(Rectangle((-0.5, ti - 0.5), len(regions), 1, fill=False, ec="k", lw=2.2))
    ax.text(len(regions) - 0.5, 8.0, "görev 7 ve 9: AYNI sahne (ocak + dolap üstü),\nhedef ters — yalnızca komut ayırır", ha="right", va="center", fontsize=9.5,
            bbox=dict(boxstyle="round", fc="white", ec="k"))
    ax.set_title("LIBERO-Spatial: hangi görevde iki kâse nerede? (BDDL :init, gerçek; satır = görev, sütun = bölge)\n"
                 "10 görevin 9'u sahneden ayrışır → komutu okumayan policy'nin üst sınırı 9/10; görev 0'ın 'wrong' komutu (= görev 1) bowl_2'nin bölgesini tarif eder", fontsize=10.5)
    fig.tight_layout(); fig.savefig(out / "04-spatial-scene-matrix.png", dpi=150); plt.close(fig)


def fig05(out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    suites = list(CF["π0"])
    for ax, (model, d) in zip(axes, CF.items()):
        xs = np.arange(len(suites)); w = 0.38
        f = [d[s][0] for s in suites]; b = [d[s][1] for s in suites]
        ax.bar(xs - w / 2, f, w, color=G, edgecolor=GE, label="faithful: komuttaki nesneye dokundu")
        ax.bar(xs + w / 2, b, w, color=R, edgecolor=RE, label="biased: eğitimdeki görevin nesnesine dokundu")
        for x, (fv, bv) in enumerate(zip(f, b)):
            ax.text(x - w / 2, fv + 1.5, f"{fv:.0f}", ha="center", fontsize=8.5); ax.text(x + w / 2, bv + 1.5, f"{bv:.0f}", ha="center", fontsize=8.5)
        ax.set_xticks(xs); ax.set_xticklabels(suites, fontsize=9); ax.set_title(model, fontsize=12)
        ax.set_ylim(0, 108); ax.grid(axis="y", alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("grounding (%)")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("LIBERO-CF (Fang vd. 2602.17659, Tablo III, CAG yok): karşı-olgusal komutta gripper hangi nesneye gitti?\n"
                 "Bizim 'wrong' satırımızdaki grasped_bowl etiketi bunun elle sayılmış hâli: bowl_2 = faithful, bowl_1 = biased", fontsize=10)
    fig.tight_layout(rect=[0, 0.06, 1, 0.9]); fig.savefig(out / "05-cf-faithful-biased.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--libero", required=True); ap.add_argument("--out", default="../../lectures/images/lecture-19")
    a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig02(out); fig03(out, a.libero); fig04(out, a.libero); fig05(out)
    print("ok:", sorted(p.name for p in out.glob("*.png")))
```

## conditions_goal8.json

```json
{
  "clean": null,
  "empty": "",
  "wrong": "put the bowl on the stove",
  "tr": "kaseyi tabağın üstüne koy"
}
```

## t4_goal_config.yaml

```yaml
# Ders 19 — aynı dört koşullu deney LIBERO-Goal görev 8'de ("put the bowl on the plate"): tek sahne, 10 hedef → komutsuz taban ≈ %10.
# Ders 18'in t4_cond_config.yaml'ından yalnızca iki satır farklı: checkpoint ve task_suite_name. Yamalar aynı (Ders 12 + Ders 18).
# Checkpoint: hf download DravenALG/VLANeXt VLANeXt_libero_goal.pt --local-dir checkpoints   (16.4 GB; Spatial checkpoint'i Goal'da KULLANILMAZ)
# Kullanım (VLANeXt kökünde):
#   VLANEXT_TASK_IDS=8 VLANEXT_CONDITIONS="$(cat conditions_goal8.json)" VLANEXT_RUN_TAG=_goal8_4cond \
#   python -m scripts.libero_bench_eval --config config/t4_goal_config.yaml
# wrong = Goal görev 1'in komutu ("put the bowl on the stove"): aynı sahnede uygulanabilir, hedef farklı → success(wrong) yalnızca
# orijinal hedefi (bowl on plate) ölçer; ≈ 0 çıkarsa model komutu okuyor, ≈ clean çıkarsa sahne komutu eziyor (Ders 19 §6).
data:
  augmentation:
    center_crop: true
    center_crop_ratio: 0.9

model:
  diffusion_steps: 5
  scheduler_type: "flow_match"

project:
  use_wandb: false

eval:
  finetuned_checkpoint: "checkpoints/VLANeXt_libero_goal.pt"
  task_suite_name: "libero_goal"
  num_steps_wait: 10
  num_steps_execute: 8
  num_parallel_envs: 2
  num_trials_per_task: 10
  seed: 42
  image_size: 256
  resume_episodes: 0
  resume_successes: 0
```
