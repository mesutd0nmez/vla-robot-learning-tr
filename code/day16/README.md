# day16

Ders 16 — Faz 2 sentezi + LIBERO-Plus: ayrı runtime'da `PYTHONPATH` ile kurulum, Language görevlerinin kod yolu, BDDL ayrıştırıcısının Türkçe'ye etkisi, VLANeXt ile `rewritten` / `empty` (isteğe bağlı `clean` / `tr`) kapalı döngü denemesi.

| Dosya | Ne | Durum |
|---|---|---|
| `colab_libero_plus_t4.ipynb` | 14 hücrelik Colab/Kaggle T4 notebook'u (yeni runtime!) | Colab'da çalıştırılmadı (GPU/HF yok) |
| `select_plus_tasks.py` | `task_classification.json` + task map → suite × kategori özeti, temel görevin varyantları, `VLANEXT_TASK_IDS` | çalıştırıldı (gerçek veri) |
| `bddl_lang_probe.py` | Türkçe komut `bddl.parsing.scan_tokens`'tan geçince ne olur (`;`, `.lower()`, parantez) | gerçek `bddl==1.0.1` ile çalıştırıldı |
| `vlanext_plus_patch.py` | 5 yama: json yolu, `VLANEXT_INSTR_OVERRIDE`, `VLANEXT_RUN_TAG`, log satırı, `np.float_` | VLANeXt d911e6b + LIBERO-plus 4976dc3 kopyasında doğrulandı (idempotent) |
| `parse_plus_log.py` | LIBERO-Plus `eval.log` → tablo + `results/runs.csv` | `--selftest` (sentetik log) |
| `t4_plus_config.yaml` | Küçük bütçe eval config'i | — |
| `plot_day16.py` | Ders görselleri 02, 04, 05, 06 | çalıştırıldı |

Kullanım (Mac'te, model gerekmez):
```bash
git clone --depth 1 https://github.com/sylvestf/LIBERO-plus ~/vla-work/LIBERO-plus
curl -O https://raw.githubusercontent.com/mesutd0nmez/vla-robot-learning-tr/main/code/day16/select_plus_tasks.py
python select_plus_tasks.py ~/vla-work/LIBERO-plus --base 0 --category Language --pick 3
pip install bddl==1.0.1 future && curl -O https://raw.githubusercontent.com/mesutd0nmez/vla-robot-learning-tr/main/code/day16/bddl_lang_probe.py
python bddl_lang_probe.py
```
Colab: <https://colab.research.google.com/github/mesutd0nmez/vla-robot-learning-tr/blob/main/code/day16/colab_libero_plus_t4.ipynb>


## select_plus_tasks.py

```python
"""Ders 16 — LIBERO-Plus görev haritasını oku, bir temel görevin varyantlarını seç, VLANEXT_TASK_IDS üret.

Model / GPU / asset GEREKMEZ: yalnızca LIBERO-Plus klonundaki iki metin dosyasını okur
(kodu çalıştırmaz; task map'i `ast.literal_eval` ile ayrıştırır).

Kullanım:
  python select_plus_tasks.py <LIBERO-plus kökü>                      # suite × kategori özeti
  python select_plus_tasks.py <kök> --suite libero_spatial --base 0 --category Language --pick 3
  python select_plus_tasks.py <kök> --base 0 --category Language --pick 3 --json secim.json

--base N   : orijinal LIBERO görev sırası (Ders 10'daki list_tasks.py ile aynı, 0–9)
--pick K   : en kolay K varyant (difficulty_level küçükten büyüğe; eşitlikte id sırası)
Çıktıdaki indeksler 0 tabanlıdır = `task_suite.get_task(i)` = VLANEXT_TASK_IDS'e yazılacak sayı.
task_classification.json'daki "id" alanı 1 tabanlıdır (id - 1 = indeks; bu script bunu doğrular).
"""
import argparse, ast, json, pathlib, re, sys
from collections import Counter

# Orijinal LIBERO-Spatial / Object / Goal / 10 görev sırası (LIBERO libero_suite_task_map.py; Ders 10)
BASE = {
    "libero_spatial": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ],
}
SHORT = {"Language": "Language Instructions", "Camera": "Camera Viewpoints", "Robot": "Robot Initial States",
         "Light": "Light Conditions", "Background": "Background Textures", "Noise": "Sensor Noise",
         "Layout": "Objects Layout"}


def load_task_map(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "libero_task_map":
            return ast.literal_eval(node.value)
    sys.exit(f"HATA: {path} içinde libero_task_map bulunamadı")


def bddl_language(path):
    """LIBERO-Plus'ın gördüğü komut: bddl.parsing.scan_tokens gibi ';' sonrası silinir, .lower(), boşlukla birleştirilir."""
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r";.*$", "", raw, flags=re.MULTILINE).lower()
    m = re.search(r"\(:language([^()]*)\)", raw)
    return " ".join(m.group(1).split()) if m else None


def variant_of(name, base):
    """Görev adının hangi temel göreve ait olduğu: base + '_' ile başlıyorsa (ya da tam eşitse)."""
    return name == base or name.startswith(base + "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="LIBERO-plus repo kökü (içinde libero/libero/benchmark/ olan)")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--base", type=int, default=None)
    ap.add_argument("--category", default=None, help="Language | Camera | Robot | Light | Background | Noise | Layout")
    ap.add_argument("--pick", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    lib = pathlib.Path(a.root) / "libero" / "libero"
    cls = json.loads((lib / "benchmark" / "task_classification.json").read_text(encoding="utf-8"))
    tmap = load_task_map(lib / "benchmark" / "libero_suite_task_map.py")

    # 1) Tutarlılık: json id-1 == task map sırası mı?
    for s in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        names, js = tmap[s], cls[s]
        ok = all(names[x["id"] - 1] == x["name"] for x in js)
        print(f"{s:15s} görev={len(names):5d}  json={len(js):5d}  id-1 == indeks: {ok}")
    print(f"TOPLAM {sum(len(cls[s]) for s in cls)} görev\n")

    if a.base is None:
        cats = list(SHORT.values())
        print(f"{'suite':15s} " + " ".join(f"{c.split()[0][:10]:>10s}" for c in cats))
        for s in cls:
            c = Counter(x["category"] for x in cls[s])
            print(f"{s:15s} " + " ".join(f"{c[k]:10d}" for k in cats))
        return

    base = BASE[a.suite][a.base]
    names = tmap[a.suite]
    info = {x["name"]: x for x in cls[a.suite]}
    cat = SHORT.get(a.category, a.category) if a.category else None
    rows = []
    for i, n in enumerate(names):
        if not variant_of(n, base):
            continue
        x = info[n]
        if cat and x["category"] != cat:
            continue
        rows.append((x["difficulty_level"] or 0, i, n, x["category"]))
    clean_in_map = base in names
    print(f"temel görev {a.base}: {base}")
    print(f"  bu görevin varyantları ({cat or 'tüm kategoriler'}): {len(rows)}  | bozulmamış hali benchmark'ta var mı: {clean_in_map}")
    print("  zorluk dağılımı:", dict(sorted(Counter(r[0] for r in rows).items())))
    rows.sort()
    pick = rows[: a.pick] if a.pick else rows
    out = []
    print(f"\n{'indeks':>6} {'zorluk':>6}  komut (LIBERO-Plus'ın modele verdiği hâl)")
    for d, i, n, c in pick:
        if "_language_" in n:
            bddl = lib / "bddl_files" / a.suite / (n.split("_view_")[0] + ".bddl")
        else:
            bddl = None
        lang = bddl_language(bddl) if bddl and bddl.exists() else " ".join(base.split("_"))
        print(f"{i:6d} {d:6d}  {lang}")
        out.append({"index": i, "difficulty": d, "category": c, "name": n, "language": lang})
    ids = ",".join(str(r["index"]) for r in out)
    clean = " ".join(base.split("_"))
    print(f"\nVLANEXT_TASK_IDS=\"{ids}\"")
    print(f"temiz komut (orijinal LIBERO task.language): \"{clean}\"")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps({"suite": a.suite, "base": a.base, "clean_language": clean,
                                                    "task_ids": ids, "tasks": out}, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
        print("yazıldı:", a.json)


if __name__ == "__main__":
    main()
```

## bddl_lang_probe.py

```python
"""Ders 16 — Türkçe bir komut LIBERO-Plus'ın BDDL ayrıştırıcısından geçince modele ne olarak ulaşır?

LIBERO-Plus, `_language_` görevlerinde komutu BDDL dosyasının `(:language ...)` alanından okur
(libero/libero/benchmark/__init__.py → grab_language_from_filename → bddl_utils.get_problem_info).
O yol `bddl.parsing.scan_tokens` kullanır: (1) ';' ile satır sonuna kadar her şeyi YORUM sayıp siler,
(2) tüm dosyayı Python `str.lower()` ile küçültür, (3) '(' ve ')' karakterlerini yapı olarak okur,
(4) kelimeleri tek boşlukla yeniden birleştirir.

Bu script gerçek `bddl` paketini (pip install bddl==1.0.1 future) kullanır; yoksa aynı 3 satırı taklit eder.
Kullanım:  python bddl_lang_probe.py [--csv cikti.csv]
"""
import argparse, csv, re, unicodedata

try:
    from bddl.parsing import scan_tokens          # LIBERO / LIBERO-Plus'ın kullandığı fonksiyon
    SRC = "bddl.parsing.scan_tokens (gerçek paket)"
except Exception:                                  # pragma: no cover
    SRC = "taklit (bddl kurulu değil)"

    def scan_tokens(string):
        s = re.sub(r";.*$", "", string, flags=re.MULTILINE).lower()
        stack, tokens = [], []
        for t in re.findall(r"[()]|[^\s()]+", s):
            if t == "(":
                stack.append(tokens); tokens = []
            elif t == ")":
                if not stack:
                    raise Exception("Missing open parenthesis")
                toks, tokens = tokens, stack.pop(); tokens.append(toks)
            else:
                tokens.append(t)
        if stack:
            raise Exception("Missing close parenthesis")
        return tokens[0]

TEMPLATE = "(define (problem LIBERO_Tabletop_Manipulation)\n  (:domain robosuite)\n  (:language {cmd})\n)\n"

CASES = [
    ("EN referans", "pick up the black bowl between the plate and the ramekin and place it on the plate"),
    ("TR küçük harf", "tabak ile ramekinin arasındaki siyah kaseyi al ve tabağın üstüne koy"),
    ("TR büyük I", "Işığın altındaki kâseyi al"),
    ("TR büyük İ", "İki tabağın arasındaki kâseyi al"),
    ("TR TAMAMI BÜYÜK", "SIRADAKİ KASEYİ AL"),
    ("noktalı virgül", "kâseyi al; tabağa koy"),
    ("parantez", "kâseyi (siyah olanı) tabağa koy"),
    ("kesme işareti", "ramekin'in yanındaki kâseyi al"),
    ("boş", ""),
]


def parse_language(cmd):
    tokens = scan_tokens(string=TEMPLATE.format(cmd=cmd))
    # libero/libero/envs/bddl_utils.py get_problem_info: tokens.pop(0)=="define", sondan gruplar, ":language" grubu
    tokens = list(tokens)
    assert tokens.pop(0) == "define"
    lang = ""
    while tokens:
        g = tokens.pop()
        if isinstance(g, list) and g and g[0] == ":language":
            lang = " ".join(g[1:])   # bddl_utils.py:83 ile aynı; iç içe liste varsa TypeError
    return lang


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv"); a = ap.parse_args()
    print("ayrıştırıcı:", SRC)
    rows = []
    for label, cmd in CASES:
        try:
            out = parse_language(cmd); err = ""
        except Exception as e:
            out, err = "", f"{type(e).__name__}: {e}"
        same = out == cmd
        extra = [f"U+{ord(c):04X}" for c in out if unicodedata.combining(c)]
        note = err or ("aynı" if same else "DEĞİŞTİ") + (f" (birleşik işaret {','.join(extra)})" if extra else "")
        rows.append({"durum": label, "girdi": cmd, "modele_giden": out, "len_in": len(cmd), "len_out": len(out), "not": note})
        print(f"- {label:16s} | {cmd!r}\n  {'':16s} → {out!r}  [{note}]")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print("yazıldı:", a.csv)


if __name__ == "__main__":
    main()
```

## vlanext_plus_patch.py

```python
"""Ders 16 — VLANeXt'in LIBERO-Plus eval'ini Colab/Kaggle'a ve deney koşullarına hazırla.
Önce Ders 12'nin yamasını uygula (sdpa, mmap, VLANEXT_TASK_IDS), sonra bunu:

  python vlanext_t4_patch.py   /content/VLANeXt
  python vlanext_plus_patch.py /content/VLANeXt [/content/VLANeXt/third_party/LIBERO-plus]

İdempotenttir; beklenen satırı bulamazsa durur ve söyler (repo değişmiş olabilir). Kaynak: VLANeXt d911e6b,
LIBERO-plus 4976dc3 (Kasım 2026 klonu).

Yamalar
 1. task_classification.json yolu   Eval script'i yolu yazarların sunucusuna sabitlemiş
                                    (/data/NTU_slab/...). Bulamayınca kategori istatistiği BOŞ kalır
                                    (yalnızca bir warning). → LIBERO_PLUS_ROOT ortam değişkeni.
 2. VLANEXT_INSTR_OVERRIDE          Tanımlıysa her episode'un komutu bu string olur ("" = boş komut).
                                    Değiştirilen tek satır: build_episode_specs içindeki
                                    `task_description = task.language` (Ders 13).
 3. VLANEXT_RUN_TAG                 Çıktı klasörüne ek; koşulların videoları/log'ları karışmasın.
 4. Log satırı                       Override değeri eval.log'a yazılır (parse_plus_log.py okur).
 5. np.float_ → np.float64          LIBERO-plus env_wrapper.py:105, numpy 2 (VLANeXt COMMON_ISSUES #3).
"""
import pathlib, sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
plus = pathlib.Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else root / "third_party" / "LIBERO-plus"
ev = root / "scripts" / "libero_plus_bench_eval.py"
par = root / "src" / "evaluation" / "libero_bench" / "parallel_eval_utils.py"
envw = plus / "libero" / "libero" / "envs" / "env_wrapper.py"
for p in (ev, par):
    if not p.exists():
        sys.exit(f"HATA: {p} yok — ilk argüman VLANeXt repo kökü olmalı")


def patch(path, old, new, label):
    src = path.read_text(encoding="utf-8")
    if new in src:
        print(f"  [=] {label}: zaten yamalı"); return
    if old not in src:
        sys.exit(f"HATA: {label}: beklenen satır bulunamadı ({path.name}). Aranan: {old!r}")
    path.write_text(src.replace(old, new, 1), encoding="utf-8")
    print(f"  [+] {label}: uygulandı")


print(f"VLANeXt kökü: {root}\nLIBERO-plus: {plus}")
# 1) json yolu
patch(ev,
      'json_path = Path("/data/NTU_slab/draven/proj/third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json")',
      'json_path = Path(os.environ.get("LIBERO_PLUS_ROOT", "third_party/LIBERO-plus")) / "libero/libero/benchmark/task_classification.json"  # Ders 16 yaması',
      "task_classification.json yolu")
# 2) komut override (parallel_eval_utils, Ders 12 yaması `import os` ekler; yoksa biz ekleriz)
if "\nimport os\n" not in "\n" + par.read_text(encoding="utf-8"):
    patch(par, "import numpy as np\n", "import os\nimport numpy as np\n", "import os (parallel_eval_utils)")
patch(par,
      "        task_description = task.language\n",
      '        task_description = os.environ["VLANEXT_INSTR_OVERRIDE"] if "VLANEXT_INSTR_OVERRIDE" in os.environ else task.language  # Ders 16 yaması\n',
      "VLANEXT_INSTR_OVERRIDE")
# 3) çıktı klasörü etiketi
patch(ev,
      '        f"_libero_plus"\n    )',
      '        f"_libero_plus"\n        + os.environ.get("VLANEXT_RUN_TAG", "")  # Ders 16 yaması\n    )',
      "VLANEXT_RUN_TAG")
# 4) override'ı log'a yaz
patch(ev,
      '    logger.info("Task suite: %s", cfg.eval.task_suite_name)\n',
      '    logger.info("Task suite: %s", cfg.eval.task_suite_name)\n'
      '    logger.info("Instruction override: %r", os.environ.get("VLANEXT_INSTR_OVERRIDE", "<yok: task.language>"))  # Ders 16 yaması\n',
      "override log satırı")
# 5) numpy 2
if envw.exists():
    patch(envw, "dtype=np.float_)", "dtype=np.float64)  # Ders 16 yaması (numpy 2)", "np.float_ -> np.float64")
else:
    print(f"  [!] {envw} yok — LIBERO-plus'ı klonladıktan sonra tekrar çalıştır")

print("Bitti. Kontrol:")
for p in (ev, par, envw):
    if p.exists():
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if "Ders 16 yaması" in line or "T4 yaması" in line:
                print(f"  {p.name}:{i}: {line.strip()[:100]}")
```

## parse_plus_log.py

```python
"""Ders 16 — VLANeXt LIBERO-Plus eval.log → episode tablosu + runs.csv satırları.

Kullanım:
  python parse_plus_log.py <eval.log> --run-id vlanext_plus_t4_01 --condition rewritten --lang en --csv results/runs.csv
  python parse_plus_log.py --selftest      # script'in kendi format string'iyle üretilmiş log'la sınar

Log biçimi scripts/libero_plus_bench_eval.py'dendir (Ders 12'deki libero_bench_eval'den FARKLI: trial/episode_time yok,
category var):
  Episode result: task=%d/%d category=%s success=%s rollout_steps=%d total_episodes=%d/%d total_successes=%d total_success_rate=%.1f%% video=%s
runs.csv sözleşmesi (CLAUDE.md): run_id, model, checkpoint, suite, task_id, lang, condition, n_episodes, success_rate
task_id = LIBERO-Plus indeksi (0 tabanlı, VLANEXT_TASK_IDS ile aynı).
"""
import argparse, csv, pathlib, re, sys, tempfile
from collections import defaultdict

EP = re.compile(r"Episode result: task=(\d+)/(\d+) category=(\S+) success=(True|False) rollout_steps=(\d+) "
                r"total_episodes=(\d+)/(\d+) total_successes=(\d+) total_success_rate=([\d.]+)% video=(\S+)")
OVR = re.compile(r"Instruction override: (.*)")
SUITE = re.compile(r"Task suite: (\S+)")
CKPT = re.compile(r"checkpoint=(\S+)")
HEADER = ["run_id", "model", "checkpoint", "suite", "task_id", "lang", "condition", "n_episodes", "success_rate"]


def parse(text):
    eps = [m.groups() for m in EP.finditer(text)]
    s = SUITE.search(text); c = CKPT.search(text); o = OVR.search(text)
    return {"suite": s.group(1) if s else "?", "checkpoint": pathlib.Path(c.group(1)).name if c else "?",
            "override": o.group(1).strip() if o else "(satır yok — yama 4 uygulanmamış olabilir)", "episodes": eps}


def report(info, run_id, lang, condition, csv_path=None):
    eps = info["episodes"]
    if not eps:
        sys.exit("Log'da 'Episode result' satırı yok — eval başlamadan düşmüş olabilir; log'un sonuna bak.")
    print(f"suite={info['suite']}  checkpoint={info['checkpoint']}  override={info['override']}  episode={len(eps)}")
    per = defaultdict(lambda: [0, 0]); cats = {}
    print(f"{'indeks':>6} {'kategori':>10} {'success':>7} {'adım':>5}  video")
    for task, n, cat, ok, steps, *_rest, video in eps:
        i = int(task) - 1; per[i][0] += 1; per[i][1] += ok == "True"; cats[i] = cat
        print(f"{i:6d} {cat:>10} {ok:>7} {int(steps):5d}  {pathlib.Path(video).name[:60]}")
    tn = sum(v[0] for v in per.values()); ts = sum(v[1] for v in per.values())
    print(f"Toplam: {ts}/{tn} = {100 * ts / tn:.1f}%")
    if csv_path:
        out = pathlib.Path(csv_path); out.parent.mkdir(parents=True, exist_ok=True)
        new = not out.exists()
        with out.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(HEADER)
            for i, (n, s) in sorted(per.items()):
                w.writerow([run_id, "VLANeXt-B", info["checkpoint"], info["suite"] + "_plus", i, lang, condition, n,
                            round(s / n, 4)])
        print(f"{len(per)} satır eklendi → {out}")
    return ts, tn


def selftest():
    fmt = ("Episode result: task=%d/%d category=%s success=%s rollout_steps=%d total_episodes=%d/%d "
           "total_successes=%d total_success_rate=%.1f%% video=%s")
    lines = ["2026-11-09 10:00:00 INFO Task suite: libero_spatial",
             "2026-11-09 10:00:00 INFO Instruction override: ''",
             "2026-11-09 10:00:01 INFO Eval setup: checkpoint=checkpoints/VLANeXt_libero_spatial.pt tasks=2402"]
    demo = [(989, True, 131), (995, False, 220), (996, True, 152)]
    for k, (t, ok, st) in enumerate(demo, 1):
        succ = sum(d[1] for d in demo[:k])
        lines.append("2026-11-09 10:0%d:00 INFO " % k + fmt % (t, 2402, "Language", ok, st, k, 3, succ, 100 * succ / k,
                                                              f"/x/episode={k}--success={ok}--task=.mp4"))
    with tempfile.TemporaryDirectory() as d:
        csvp = pathlib.Path(d) / "runs.csv"
        ts, tn = report(parse("\n".join(lines)), "selftest", "none", "empty", csvp)
        rows = list(csv.DictReader(csvp.open(encoding="utf-8")))
    assert (ts, tn) == (2, 3) and len(rows) == 3 and rows[0]["task_id"] == "988", rows
    print("SELFTEST OK (sentetik log; gerçek sonuç değildir)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?")
    ap.add_argument("--run-id", default="vlanext_plus_t4_01")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--condition", default="rewritten", help="clean | rewritten | empty | tr")
    ap.add_argument("--csv")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.log:
        selftest()
    else:
        report(parse(pathlib.Path(a.log).read_text(encoding="utf-8", errors="replace")), a.run_id, a.lang, a.condition, a.csv)
```

## plot_day16.py

```python
"""Ders 16 görselleri (02, 04, 05, 06).

Kullanım:  python plot_day16.py <çıktı_klasörü/> [--plus <LIBERO-plus kökü>] [--probe probe.csv]
  --plus verilirse 02 ve 04 sayıları task_classification.json'dan YENİDEN hesaplanır; verilmezse aşağıdaki gömülü
  sayılar kullanılır (LIBERO-plus 4976dc3 klonundan bu script'in --plus moduyla alındı).
  --probe: bddl_lang_probe.py --csv çıktısı (05 için).
06'daki sayılar LIBERO-plus README liderlik tablosundandır (elle aktarıldı).
"""
import argparse, csv, json, os, pathlib
from collections import Counter, defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--plus"); ap.add_argument("--probe")
a = ap.parse_args(); out = a.out.rstrip("/") + "/"; os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
CATS = ["Camera Viewpoints", "Robot Initial States", "Language Instructions", "Light Conditions",
        "Background Textures", "Sensor Noise", "Objects Layout"]
SHORT = ["Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout"]
COUNTS = {  # suite -> kategori sırası CATS
    "libero_spatial": [376, 350, 390, 292, 258, 351, 385],
    "libero_object": [396, 398, 354, 297, 248, 422, 403],
    "libero_goal": [408, 409, 410, 279, 281, 379, 425],
    "libero_10": [419, 393, 383, 274, 289, 449, 312],
}
LANG_DIFF = {  # suite -> Language kategorisinde zorluk 1..5 sayıları
    "libero_spatial": [115, 100, 58, 82, 35], "libero_object": [103, 67, 72, 63, 49],
    "libero_goal": [45, 49, 52, 79, 185], "libero_10": [65, 64, 77, 104, 73],
}
if a.plus:
    d = json.loads((pathlib.Path(a.plus) / "libero/libero/benchmark/task_classification.json").read_text(encoding="utf-8"))
    COUNTS = {s: [Counter(x["category"] for x in d[s])[c] for c in CATS] for s in SUITES}
    LANG_DIFF = {}
    for s in SUITES:
        c = Counter(x["difficulty_level"] for x in d[s] if x["category"] == "Language Instructions")
        LANG_DIFF[s] = [c[i] for i in range(1, 6)]
    print("yeniden hesaplandı:", COUNTS, LANG_DIFF)

# --- 02: 10.030 görev, suite × kategori
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(SUITES)); bottom = np.zeros(len(SUITES))
cols = ["#9ecae1", "#c6dbef", "#ffd166", "#fdd0a2", "#d9d9d9", "#e5c3e8", "#c7e9c0"]
for j, (name, c) in enumerate(zip(SHORT, cols)):
    v = np.array([COUNTS[s][j] for s in SUITES])
    ax.bar(x, v, 0.62, bottom=bottom, color=c, edgecolor="white", label=name, zorder=3)
    for xi, (b, val) in enumerate(zip(bottom, v)):
        ax.text(xi, b + val / 2, str(val), ha="center", va="center", fontsize=8.5,
                fontweight="bold" if name == "Language" else None)
    bottom += v
for xi, t in enumerate(bottom):
    ax.text(xi, t + 25, f"{int(t)}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(["Spatial", "Object", "Goal", "Long (10)"])
ax.set_ylabel("görev sayısı"); ax.set_ylim(0, 2800); ax.yaxis.grid(True, color="#eee", zorder=0)
ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=9)
ax.set_title(f"LIBERO-Plus: {int(bottom.sum())} görev = 4 suite × 7 bozulma — kaynak: task_classification.json (LIBERO-plus 4976dc3)",
             fontsize=10)
fig.text(0.5, 0.005, "Her görev tek bir bozulma örneğidir ve 1 episode koşulur (num_trials_per_task: 1). "
         "Bozulmamış orijinal görev bu listede YOK.", ha="center", fontsize=8.5, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out + "02-task-map.png", dpi=170); plt.close(fig)

# --- 04: Language görevlerinde zorluk dağılımı (zorluk = 4 modelden kaçının çözdüğü)
fig, ax = plt.subplots(figsize=(11, 4.6))
lv_cols = ["#d4f5d4", "#e8f5c8", "#fff3c4", "#ffe0c2", "#ffd6d6"]
labels = ["L1: 4 modelin 4'ü çözdü", "L2: 3'ü", "L3: 2'si", "L4: 1'i", "L5: hiçbiri"]
y = np.arange(len(SUITES))[::-1]; left = np.zeros(len(SUITES))
tot = np.array([sum(LANG_DIFF[s]) for s in SUITES], dtype=float)
for k in range(5):
    v = np.array([LANG_DIFF[s][k] for s in SUITES]) / tot * 100
    ax.barh(y, v, 0.6, left=left, color=lv_cols[k], edgecolor="#999", linewidth=0.5, label=labels[k])
    for yi, l, val, n in zip(y, left, v, [LANG_DIFF[s][k] for s in SUITES]):
        if val > 4:
            ax.text(l + val / 2, yi, f"{val:.0f}%\n({n})", ha="center", va="center", fontsize=8)
    left += v
ax.set_yticks(y); ax.set_yticklabels([f"{n} ({int(t)} görev)" for n, t in zip(["Spatial", "Object", "Goal", "Long"], tot)])
ax.set_xlim(0, 100); ax.set_xlabel("Language görevlerinin yüzdesi")
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, fontsize=8.5)
ax.set_title("Language bozulması: zorluk seviyesi dağılımı — kaynak: task_classification.json; seviye tanımı: arXiv:2510.13626 Ek C.3",
             fontsize=9.5)
fig.text(0.5, 0.005, "Zorluk görevin kendisinin değil, 4 modelin (OpenVLA-OFT, π0, π0-fast, UniVLA) sonucudur: "
         "Goal'da Language görevlerinin %45'ini hiçbiri çözemedi.", ha="center", fontsize=8.5, color="#555")
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(out + "04-language-difficulty.png", dpi=170); plt.close(fig)

# --- 05: Türkçe komut BDDL ayrıştırıcısından geçince (gerçek bddl 1.0.1 çıktısı)
if a.probe:
    rows = list(csv.DictReader(open(a.probe, encoding="utf-8")))
    import textwrap
    wr = lambda t, n: "\n".join(textwrap.wrap(t, n)) or t
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(rows) + 1.0)); ax.axis("off")
    cell = [[r["durum"], wr(r["girdi"] or "(boş)", 44), wr(r["modele_giden"] or "—", 44), wr(r["not"], 34)] for r in rows]
    tb = ax.table(cellText=cell, colLabels=["durum", "BDDL'e yazılan", "task.language (modele giden)", "not"],
                  loc="center", cellLoc="left", colWidths=[0.13, 0.33, 0.33, 0.27])
    tb.auto_set_font_size(False); tb.set_fontsize(8.6); tb.scale(1, 2.6)
    for (r, c), cl in tb.get_celld().items():
        cl.set_edgecolor("#bbb")
        if r == 0:
            cl.set_facecolor("#cfe8ff"); cl.set_text_props(fontweight="bold")
        elif c == 3:
            t = rows[r - 1]["not"]
            cl.set_facecolor("#d4f5d4" if t == "aynı" else "#ffd6d6")
        elif c == 2:
            cl.set_facecolor("#fffbea")
    ax.set_title("bddl.parsing.scan_tokens: ';' sonrası yorum, str.lower(), parantez = yapı — gerçek çıktı (bddl 1.0.1, bddl_lang_probe.py)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(out + "05-bddl-turkish-probe.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# --- 06: LIBERO-Plus liderlik tablosu: Language sütunu vs diğer 6 bozulmanın ortalaması (README)
LB = {  # model: Camera, Robot, Language, Light, Background, Noise, Layout, Total
    "OpenVLA": [0.8, 3.5, 23.0, 8.1, 34.8, 15.2, 28.5, 15.6],
    "OpenVLA-OFT": [56.4, 31.9, 79.5, 88.7, 93.3, 75.8, 74.2, 69.6],
    "OpenVLA-OFT_w": [10.4, 38.7, 70.5, 76.8, 93.6, 49.9, 69.9, 55.8],
    "NORA": [2.2, 37.0, 65.1, 45.7, 58.6, 12.8, 62.1, 39.0],
    "WorldVLA": [0.1, 27.9, 41.6, 43.7, 17.1, 10.9, 38.0, 25.0],
    "UniVLA": [1.8, 46.2, 69.6, 69.0, 81.0, 21.2, 31.9, 43.9],
    "π0": [13.8, 6.0, 58.8, 85.0, 81.4, 79.0, 68.9, 53.6],
    "π0-Fast": [65.1, 21.6, 61.0, 73.2, 73.2, 74.4, 68.8, 61.6],
    "RIPT-VLA": [55.2, 31.2, 77.6, 88.4, 91.6, 73.5, 74.2, 68.4],
    "OpenVLA-OFT_m": [55.6, 21.7, 81.0, 92.7, 91.0, 78.6, 68.7, 67.9],
    "OpenVLA-OFT+": [92.8, 30.3, 85.8, 94.9, 93.9, 89.3, 77.6, 79.6],
}
fig, ax = plt.subplots(figsize=(10.5, 4.8))
names = list(LB); lang = np.array([LB[n][2] for n in names]); others = np.array([np.mean(LB[n][:2] + LB[n][3:7]) for n in names])
rank = np.array([1 + sum(v > LB[n][2] for v in LB[n][:2] + LB[n][3:7]) for n in names])
order = np.argsort(lang)
yy = np.arange(len(names))
ax.hlines(yy, others[order], lang[order], color="#bbb", zorder=1)
ax.scatter(others[order], yy, color="#cfe8ff", edgecolor="#3d7fc4", s=60, zorder=3, label="diğer 6 bozulmanın ortalaması")
ax.scatter(lang[order], yy, color="#ffd166", edgecolor="#b07d00", s=70, zorder=3, label="Language bozulması")
for k, i in enumerate(order):
    ax.text(102, k, f"{rank[i]}.", va="center", fontsize=8.5, color="#555")
ax.text(2, -0.9, "sağ sütundaki sayı: Language, 7 bozulma içinde kaçıncı en kolay (1 = en kolay)", fontsize=8, color="#555", va="center")
ax.set_yticks(yy); ax.set_yticklabels([names[i] for i in order]); ax.set_xlim(0, 110)
ax.set_xlabel("success rate (%) — 4 suite ortalaması"); ax.xaxis.grid(True, color="#eee", zorder=0)
ax.set_ylim(-1.6, len(names) - 0.4)
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.set_title("LIBERO-Plus: komut yeniden yazılınca başarı, diğer bozulmalardan daha AZ düşüyor\nkaynak: LIBERO-plus README liderlik tablosu",
             fontsize=10)
fig.tight_layout(); fig.savefig(out + "06-leaderboard-language.png", dpi=170); plt.close(fig)
print("ok")
```

## t4_plus_config.yaml

```yaml
# Ders 16 — VLANeXt LIBERO-Plus eval, ücretsiz T4 için küçük bütçe.
# Orijinal: VLANeXt/config/libero_plus_bench_config.yaml (num_parallel_envs: 32, num_trials_per_task: 1, 2402 görev)
# Kullanım (VLANeXt kökünde; PYTHONPATH=third_party/LIBERO-plus, LIBERO_CONFIG_PATH=~/.libero_plus):
#   VLANEXT_TASK_IDS="988,994,995" python -m scripts.libero_plus_bench_eval --config config/t4_plus_config.yaml
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
  finetuned_checkpoint: "checkpoints/VLANeXt_libero_spatial.pt"   # Ders 12'deki dosya (LIBERO'da eğitildi)
  task_suite_name: "libero_spatial"
  num_steps_wait: 10
  num_steps_execute: 8
  num_parallel_envs: 1         # LIBERO-Plus env'leri ek asset yükler; T4 + 12.7 GB RAM'de 1 ile başla
  num_trials_per_task: 1       # LIBERO-Plus'ta her görev zaten tek bir bozulma örneği; README: 50 → 1
  seed: 42
  image_size: 256
  resume_episodes: 0
  resume_successes: 0
```
