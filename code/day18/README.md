# day18

Ders 18 — Aynı görev, dört komut: LIBERO-Spatial görev 0'da 10 init state × {clean, empty, wrong, tr} = 40 episode, tek model yüklemesiyle Colab/Kaggle T4'te. Koşul, eval kodunda `EpisodeSpec`'in bir alanı olur (Ders 18 yaması); analiz Wilson aralıkları + eşli (McNemar) ve eşsiz (Fisher) testlerle 4 satırlık tablo üretir. Bu klasördeki script'lerin hiçbiri Mac'te model/GPU istemez; koşu notebook'tadır.

| Dosya | Ne | Durum |
|---|---|---|
| `colab_4cond_t4.ipynb` | 13 hücrelik Colab/Kaggle T4 notebook'u: Ders 12 kurulumu + Ders 18 yaması + tek koşu + analiz + videolar | Colab'da çalıştırılmadı (GPU/HF yok) |
| `vlanext_cond_patch.py` | VLANeXt'e `VLANEXT_CONDITIONS` (JSON) desteği: `EpisodeSpec.condition`, init × koşul döngüsü, log satırına `condition=`, video adına koşul, `VLANEXT_RUN_TAG`, koşul log satırı (6 yama, idempotent) | depo kopyasında 3 senaryoda uygulandı (temiz / Ders 12 / Ders 12+16), 2. çalıştırma "zaten yamalı", `py_compile` temiz |
| `make_conditions.py` → `conditions_task0.json` | `datasets/01`'deki `commands.csv`'den koşul JSON'u (clean=null, empty="", wrong=distractor, tr=sp0_tr; `--extra para,tr2`) | çalıştırıldı (görev 0 ve 1) |
| `t4_cond_config.yaml` | `num_trials_per_task: 10`, `num_parallel_envs: 2` (Kaggle 4) | — |
| `spec_order_demo.py` | Yamalı döngünün ürettiği episode sırasını ve batch'leri modelsiz gösterir | çalıştırıldı |
| `analyze_4cond.py` | `eval.log` → `episodes.csv` + `runs.csv` + `table.md` (k/n, Wilson %95, Δ, uyumsuz çiftler, McNemar exact p, Fisher p, ortalama adım/süre); `--logs koşul=yol` ile B planı | `--selftest` (eval script'inin format string'iyle sentetik 40 satır) |
| `power_day18.py` | Görsel 03 (Wilson aralıkları, n = 10/20/50) ve 04 (Fisher / McNemar kesin güç eğrileri) | çalıştırıldı (saf hesap) |
| `plot_day18.py` | Görsel 02 (literatürde boş komut, Spatial), 06 (LIBERO-CF yanlış komut: faithful/biased), 07 (olası dört örüntü — TEMSİLİ) | çalıştırıldı (makale değerleri elle) |

Colab: <https://colab.research.google.com/github/mesutd0nmez/vla-robot-learning-tr/blob/main/code/day18/colab_4cond_t4.ipynb>

Mac'te (model gerekmez):
```bash
cd code/day18
python make_conditions.py --task 0            # conditions_task0.json
python spec_order_demo.py conditions_task0.json --parallel 2
python analyze_4cond.py --selftest
python power_day18.py ../../lectures/images/lecture-18/
python plot_day18.py  ../../lectures/images/lecture-18/
# Colab'dan indirdiğin eval.log ile:
python analyze_4cond.py eval.log --run-id vlanext_sp0_4cond_01 --out ../../projects/05-spatial0-4kosul/results/
```

## vlanext_cond_patch.py
```python
"""Ders 18 — VLANeXt'in LIBERO eval'ine "koşul" kavramını ekle: tek model yüklemesiyle, aynı init state'lerde,
birden fazla komut koşulu (clean / empty / wrong / tr) koş.

  python vlanext_t4_patch.py   /content/VLANeXt      # Ders 12 (sdpa, mmap, VLANEXT_TASK_IDS) — ÖNCE
  python vlanext_cond_patch.py /content/VLANeXt      # bu dosya

İdempotenttir; beklenen satırı bulamazsa durur ve söyler. Ders 16'nın yaması (VLANEXT_INSTR_OVERRIDE) uygulanmışsa
onunla da çalışır: koşul listesi tanımlıysa koşul kazanır, değilse Ders 16 davranışı sürer.
Kaynak: VLANeXt d911e6b (28 Eyl 2026), scripts/libero_bench_eval.py + src/evaluation/libero_bench/parallel_eval_utils.py.

Ortam değişkenleri (yamadan sonra)
  VLANEXT_CONDITIONS  JSON: {"clean": null, "empty": "", "wrong": "...", "tr": "..."}  ya da bir .json dosyasının yolu.
                      null = task.language (orijinal komut). Her init state için her koşuldan bir episode üretilir;
                      sıra: init 0 × {clean, empty, wrong, tr}, init 1 × {...}, ... → num_parallel_envs=4 ile aynı
                      init state'in dört koşulu AYNI batch'te koşar.
  VLANEXT_RUN_TAG     çıktı klasörü eki (koşuların log/videoları karışmasın; Ders 16'daki ile aynı ad).

Yamalar (6; hepsi "Ders 18 yaması" yorumuyla işaretli)
 1. EpisodeSpec.condition alanı                 parallel_eval_utils.py
 2. build_episode_specs: koşul × init state     parallel_eval_utils.py  (task_description satırı + for döngüsü + spec alanları)
 3. "Episode result" log satırına condition=    libero_bench_eval.py
 4. Video adının başına koşul adı               libero_bench_eval.py   (boş komutta "task=" boş kalmasın)
 5. VLANEXT_RUN_TAG                             libero_bench_eval.py
 6. Koşul listesini log'a yaz                   libero_bench_eval.py   (analyze_4cond.py okur)
"""
import pathlib, sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
ev = root / "scripts" / "libero_bench_eval.py"
par = root / "src" / "evaluation" / "libero_bench" / "parallel_eval_utils.py"
for p in (ev, par):
    if not p.exists():
        sys.exit(f"HATA: {p} yok — argüman VLANeXt repo kökü olmalı")


def patch(path, old, new, label, alt_old=None):
    src = path.read_text(encoding="utf-8")
    if new in src:
        print(f"  [=] {label}: zaten yamalı"); return
    if old not in src and alt_old and alt_old in src:
        old = alt_old
    if old not in src:
        sys.exit(f"HATA: {label}: beklenen satır bulunamadı ({path.name}); repo değişmiş olabilir. Aranan: {old!r}")
    path.write_text(src.replace(old, new, 1), encoding="utf-8")
    print(f"  [+] {label}: uygulandı")


print(f"VLANeXt kökü: {root}")

# --- parallel_eval_utils.py -------------------------------------------------------------------------------
if "\nimport os\n" not in "\n" + par.read_text(encoding="utf-8"):
    patch(par, "import numpy as np\n", "import os\nimport numpy as np\n", "import os (parallel_eval_utils)")
patch(par, "import numpy as np\n",
      "import numpy as np\n"
      "import json as _json  # Ders 18 yaması\n"
      "\n"
      "\n"
      "def _load_conditions(default_text):  # Ders 18 yaması: [(ad, komut)] ; null = task.language\n"
      '    raw = os.environ.get("VLANEXT_CONDITIONS")\n'
      "    if not raw:\n"
      '        return [(None, default_text)]\n'
      "    if raw.strip().endswith(\".json\") and os.path.exists(raw.strip()):\n"
      '        raw = open(raw.strip(), encoding="utf-8").read()\n'
      "    conds = _json.loads(raw)\n"
      "    return [(str(k), default_text if v is None else str(v)) for k, v in conds.items()]\n",
      "_load_conditions tanımı")
# 1) spec alanı
patch(par, "    task_category: str = None\n",
      "    task_category: str = None\n    condition: str = None  # Ders 18 yaması\n",
      "EpisodeSpec.condition")
# 2) koşul listesi: orijinal satır ya da Ders 16'nın override satırı
_ORIG = "        task_description = task.language\n"
_D16 = '        task_description = os.environ["VLANEXT_INSTR_OVERRIDE"] if "VLANEXT_INSTR_OVERRIDE" in os.environ else task.language  # Ders 16 yaması\n'
_line = _D16 if _D16 in par.read_text(encoding="utf-8") else _ORIG   # Ders 16 yaması varsa korunur (koşul listesi onun üstüne gelir)
patch(par, _line, _line + "        _conds = _load_conditions(task_description)  # Ders 18 yaması\n", "koşul listesi (task_description)")
patch(par,
      "        for episode_idx in range(num_trials_per_task):\n",
      "        for episode_idx, (_cond_name, _cond_text) in ((e, c) for e in range(num_trials_per_task) for c in _conds):  # Ders 18 yaması\n",
      "init state × koşul döngüsü")
patch(par,
      "                    task_description=task_description,\n                    task_category=task_category,\n",
      "                    task_description=_cond_text,  # Ders 18 yaması\n                    task_category=task_category,\n                    condition=_cond_name,  # Ders 18 yaması\n",
      "spec alanları")

# --- libero_bench_eval.py ---------------------------------------------------------------------------------
# 3) log satırı
patch(ev,
      '"Episode result: task=%d/%d trial=%d success=%s rollout_steps=%d',
      '"Episode result: task=%d/%d trial=%d condition=%s success=%s rollout_steps=%d',
      "log satırı (format)")
patch(ev,
      "                spec.task_id + 1,\n                num_tasks_in_suite,\n                spec.episode_idx,\n                result.success,\n",
      "                spec.task_id + 1,\n                num_tasks_in_suite,\n                spec.episode_idx,\n                spec.condition,  # Ders 18 yaması\n                result.success,\n",
      "log satırı (argüman)")
# 4) video adı
patch(ev,
      "                task_description=spec.task_description,\n",
      '                task_description=((spec.condition + "__") if spec.condition else "") + spec.task_description,  # Ders 18 yaması\n',
      "video adı")
# 5) çıktı klasörü etiketi
patch(ev,
      '        f"_libero"\n    )',
      '        f"_libero"\n        + os.environ.get("VLANEXT_RUN_TAG", "")  # Ders 18 yaması\n    )',
      "VLANEXT_RUN_TAG")
# 6) koşulları log'a yaz
patch(ev,
      '    logger.info("Task suite: %s", cfg.eval.task_suite_name)\n',
      '    logger.info("Task suite: %s", cfg.eval.task_suite_name)\n'
      '    logger.info("Conditions: %s", os.environ.get("VLANEXT_CONDITIONS", "<yok: task.language>"))  # Ders 18 yaması\n',
      "koşul log satırı")

print("Bitti. Kontrol:")
for p in (par, ev):
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if "Ders 18 yaması" in line:
            print(f"  {p.name}:{i}: {line.strip()[:110]}")
```

## make_conditions.py
```python
"""Ders 18 — datasets/01-libero-spatial-tr-mini/data/commands.csv'den bir görevin koşul JSON'unu üret.

  python make_conditions.py --task 0                      # → conditions_task0.json (clean/empty/wrong/tr)
  python make_conditions.py --task 0 --extra para,tr2     # 6 koşul: + EN parafraz + ikinci Türkçe yüzey formu
  python make_conditions.py --task 3 --csv ../../datasets/01-libero-spatial-tr-mini/data/commands.csv

Çıktı VLANEXT_CONDITIONS'ın beklediği biçimdir: {"ad": komut | null}. null = task.language (eval kodu doldurur;
böylece "clean" koşulunun metni dosyadan değil benchmark'tan gelir ve bir yazım hatası clean'i bozamaz).
Eşleme: clean ← en (yalnızca kontrol için yazılır, JSON'da null), empty ← empty (""), wrong ← distractor, tr ← tr.
Model / GPU gerekmez.
"""
import argparse, csv, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--task", type=int, default=0)
ap.add_argument("--csv", default=str(HERE / "../../datasets/01-libero-spatial-tr-mini/data/commands.csv"))
ap.add_argument("--extra", default="", help="virgülle: para,tr2 (commands.csv'deki condition adları)")
ap.add_argument("--out", default=None)
a = ap.parse_args()

rows = {r["condition"]: r for r in csv.DictReader(open(a.csv, encoding="utf-8")) if int(r["task_id"]) == a.task}
if not rows:
    sys.exit(f"HATA: commands.csv'de task_id={a.task} yok (0–4 var)")
need = {"en": "clean", "empty": "empty", "distractor": "wrong", "tr": "tr"}
missing = [k for k in need if k not in rows]
if missing:
    sys.exit(f"HATA: görev {a.task} için eksik koşul(lar): {missing}")

conds = {"clean": None, "empty": "", "wrong": rows["distractor"]["instruction"], "tr": rows["tr"]["instruction"]}
for name in [x for x in a.extra.split(",") if x.strip()]:
    if name not in rows:
        sys.exit(f"HATA: --extra {name} commands.csv'de yok; olanlar: {sorted(rows)}")
    conds[name] = rows[name]["instruction"]

for name, text in conds.items():          # Ders 16/17 kuralları: override yolu büyük harfi korur ama kontrol zararsız
    if text and any(ch in text for ch in ";()"):
        print(f"UYARI: {name} koşulunda ';' ya da parantez var — BDDL yoluna geçersen scan_tokens kırılır (Ders 16)")

out = pathlib.Path(a.out or HERE / f"conditions_task{a.task}.json")
out.write_text(json.dumps(conds, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"görev {a.task} — clean (task.language, JSON'da null): {rows['en']['instruction']}")
for k, v in conds.items():
    print(f"  {k:6s} → {v!r}")
print(f"{len(conds)} koşul → {out}")
print(f'Colab: os.environ["VLANEXT_CONDITIONS"] = open("{out.name}").read()')
```

## spec_order_demo.py
```python
"""Ders 18 — yamalı build_episode_specs'in ürettiği sırayı modelsiz göster (aynı döngü, sahte task_suite).

  python spec_order_demo.py conditions_task0.json [--trials 10] [--parallel 2]

Soruya cevap: "num_parallel_envs=2 iken hangi episode'lar aynı batch'te?" ve "global_episode_id hangi koşula denk geliyor?"
(Yamanın döngüsü: for episode_idx, (ad, metin) in ((e, c) for e in range(trials) for c in conds) → init state dış, koşul iç döngü.)
Model / GPU / LIBERO gerekmez.
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("conditions")
ap.add_argument("--trials", type=int, default=10)
ap.add_argument("--parallel", type=int, default=2)
ap.add_argument("--task-language", default="pick up the black bowl between the plate and the ramekin and place it on the plate")
a = ap.parse_args()

conds = [(k, a.task_language if v is None else v) for k, v in json.load(open(a.conditions, encoding="utf-8")).items()]
specs = []
seen = 0
for episode_idx, (cond_name, cond_text) in ((e, c) for e in range(a.trials) for c in conds):   # yamadaki satırın aynısı
    seen += 1
    specs.append((seen, episode_idx, cond_name, cond_text))

print(f"{len(conds)} koşul × {a.trials} init state = {len(specs)} episode; num_parallel_envs={a.parallel}\n")
print(f"{'batch':>5} {'episode':>7} {'init':>4} {'koşul':6}  komut")
for i in range(0, len(specs), a.parallel):
    for g, init, name, text in specs[i:i + a.parallel]:
        print(f"{i // a.parallel + 1:5d} {g:7d} {init:4d} {name:6s}  {text[:60]!r}")
    print()
same = all(len({init for _, init, _, _ in specs[i:i + a.parallel]}) == 1 for i in range(0, len(specs), a.parallel))
print("Her batch tek bir init state mi içeriyor?", "EVET" if same else "HAYIR (koşul sayısı parallel'e bölünmüyor)")
print(f"Log'da trial=<init>, condition=<ad>; video adı '<ad>__<komut>' — episode numarasından koşulu çıkarmak gerekmez.")
```

## analyze_4cond.py
```python
"""Ders 18 — 4 koşullu deneyin eval.log'unu oku → episodes.csv + runs.csv + Wilson aralıklı sonuç tablosu + eşli testler.

Kullanım:
  python analyze_4cond.py <eval.log> --run-id vlanext_sp0_4cond_01 --out results/
  python analyze_4cond.py --logs clean=a/eval.log empty=b/eval.log wrong=c/eval.log tr=d/eval.log --out results/
        (B planı: Ders 16'nın VLANEXT_INSTR_OVERRIDE'ıyla koşul başına ayrı koşu; log'larda condition= alanı yok)
  python analyze_4cond.py --selftest      # eval script'inin format string'iyle üretilmiş sentetik log'la sınar

Log biçimi (Ders 18 yaması 3): Episode result: task=%d/%d trial=%d condition=%s success=%s rollout_steps=%d episode_time=%s ...
trial = init state indeksi (build_episode_specs: initial_states[episode_idx]) → koşullar aynı trial'da AYNI sahneyi görür → eşli.

Çıktılar (--out klasörü):
  episodes.csv  run_id, suite, task_id, condition, init_state, success, rollout_steps, episode_time_s
  runs.csv      CLAUDE.md sözleşmesi: run_id, model, checkpoint, suite, task_id, lang, condition, n_episodes, success_rate (koşul başına 1 satır; varsa dosyaya eklenir)
  table.md      koşul | k/n | SR | Wilson %95 | Δ vs clean | uyumsuz çiftler (clean✓x✗ / clean✗x✓) | McNemar exact p | Fisher p
Model / GPU gerekmez; yalnızca scipy (Fisher) — yoksa Fisher sütunu "—" olur.
"""
import argparse, csv, math, pathlib, re, sys, tempfile
from collections import OrderedDict, defaultdict

EP = re.compile(r"Episode result: task=(\d+)/(\d+) trial=(\d+)(?: condition=(\S+))? success=(True|False) "
                r"rollout_steps=(\d+) episode_time=(\S+)")
SUITE = re.compile(r"Task suite: (\S+)")
CKPT = re.compile(r"checkpoint=(\S+)")
CONDS = re.compile(r"Conditions: (.*)")
OVR = re.compile(r"Instruction override: (.*)")
LANG = {"clean": "en", "empty": "none", "wrong": "en", "para": "en", "tr": "tr", "tr2": "tr"}
HEADER_RUNS = ["run_id", "model", "checkpoint", "suite", "task_id", "lang", "condition", "n_episodes", "success_rate"]


def parse_duration(s):
    """format_duration (src/utils/logging_utils.py) çıktısı 'HH:MM:SS'; esneklik için '1m23s' / '45.2s' de kabul."""
    s = s.strip()
    m = re.fullmatch(r"(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:([\d.]+)s)?", s)
    if m and any(m.groups()):
        h, mi, se = m.groups()
        return 3600 * float(h or 0) + 60 * float(mi or 0) + float(se or 0)
    m = re.fullmatch(r"(\d+):(\d\d):(\d\d(?:\.\d+)?)", s)
    if m:
        return 3600 * int(m.group(1)) + 60 * int(m.group(2)) + float(m.group(3))
    try:
        return float(s)
    except ValueError:
        return float("nan")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def binom_two_sided(b, c):
    """McNemar exact: uyumsuz çift sayıları b (clean✓, x✗) ve c (clean✗, x✓); H0: b ~ Binom(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def fisher(k1, n1, k2, n2):
    try:
        from scipy.stats import fisher_exact
    except ImportError:
        return float("nan")
    return fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]


def read_log(path, forced_condition=None):
    text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    s = SUITE.search(text); c = CKPT.search(text); cl = CONDS.search(text); o = OVR.search(text)
    eps = []
    for m in EP.finditer(text):
        task, _n, trial, cond, ok, steps, t = m.groups()
        cond = forced_condition or cond or "clean"
        eps.append(dict(task_id=int(task) - 1, init_state=int(trial), condition=cond, success=ok == "True",
                        rollout_steps=int(steps), episode_time_s=parse_duration(t)))
    return {"suite": s.group(1) if s else "?", "checkpoint": pathlib.Path(c.group(1)).name if c else "?",
            "conditions_line": (cl.group(1).strip() if cl else None), "override": (o.group(1).strip() if o else None),
            "episodes": eps}


def summarize(eps, order=None):
    by = OrderedDict()
    for e in eps:
        by.setdefault(e["condition"], {})[(e["task_id"], e["init_state"])] = e
    conds = order or list(by)
    if "clean" in conds:
        conds = ["clean"] + [c for c in conds if c != "clean"]
    rows = []
    base = by.get("clean", {})
    for cnd in conds:
        d = by[cnd]; n = len(d); k = sum(e["success"] for e in d.values())
        lo, hi = wilson(k, n)
        r = dict(condition=cnd, k=k, n=n, sr=k / n if n else float("nan"), lo=lo, hi=hi,
                 mean_steps=sum(e["rollout_steps"] for e in d.values()) / n if n else float("nan"),
                 mean_time=sum(e["episode_time_s"] for e in d.values()) / n if n else float("nan"))
        if cnd != "clean" and base:
            common = sorted(set(base) & set(d))
            b = sum(base[key]["success"] and not d[key]["success"] for key in common)
            c = sum((not base[key]["success"]) and d[key]["success"] for key in common)
            kb = sum(base[key]["success"] for key in common); nb = len(common)
            r.update(delta=(k / n - kb / nb) if n and nb else float("nan"), b=b, c=c, n_pairs=len(common),
                     p_mcnemar=binom_two_sided(b, c), p_fisher=fisher(kb, nb, k, n))
        rows.append(r)
    return rows


def fmt_p(p):
    return "—" if p != p else ("<0.001" if p < 0.001 else f"{p:.3f}")


def table_md(rows, suite, task_id):
    out = [f"Suite `{suite}`, görev {task_id}; n = init state sayısı; Wilson %95; eşli testler clean'e göre (aynı init state).", "",
           "| koşul | k/n | SR | Wilson %95 | Δ vs clean | uyumsuz (clean✓x✗ / clean✗x✓) | McNemar exact p | Fisher p | ort. adım | ort. süre |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["condition"] == "clean":
            d = "—"; u = "—"; pm = "—"; pf = "—"
        else:
            d = f"{100 * r['delta']:+.0f}"; u = f"{r['b']} / {r['c']} (n={r['n_pairs']})"; pm = fmt_p(r["p_mcnemar"]); pf = fmt_p(r["p_fisher"])
        out.append(f"| {r['condition']} | {r['k']}/{r['n']} | {100 * r['sr']:.0f}% | {100 * r['lo']:.0f}–{100 * r['hi']:.0f} | {d} | {u} | {pm} | {pf} | "
                   f"{r['mean_steps']:.0f} | {r['mean_time']:.0f} s |")
    return "\n".join(out)


def write_outputs(info, eps, rows, run_id, out_dir, model="VLANeXt-B"):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    task_ids = sorted({e["task_id"] for e in eps})
    with (out / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "suite", "task_id", "condition", "init_state", "success", "rollout_steps", "episode_time_s"])
        for e in sorted(eps, key=lambda e: (e["task_id"], e["init_state"], e["condition"])):
            w.writerow([run_id, info["suite"], e["task_id"], e["condition"], e["init_state"], int(e["success"]), e["rollout_steps"],
                        f"{e['episode_time_s']:.1f}"])
    runs = out / "runs.csv"; new = not runs.exists()
    with runs.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER_RUNS)
        for r in rows:
            w.writerow([run_id, model, info["checkpoint"], info["suite"], ",".join(map(str, task_ids)), LANG.get(r["condition"], "?"),
                        r["condition"], r["n"], f"{r['sr']:.3f}"])
    md = table_md(rows, info["suite"], ",".join(map(str, task_ids)))
    (out / "table.md").write_text(md + "\n", encoding="utf-8")
    return md


def selftest():
    # Ders 18 yaması 3'ün format string'i (libero_bench_eval.py) ile üretilmiş sahte log: 10 init × 4 koşul; sonuçlar TEMSİLİ
    fmt = ("Episode result: task=%d/%d trial=%d condition=%s success=%s rollout_steps=%d episode_time=%s total_episodes=%d/%d "
           "total_successes=%d task_success_rate=%.1f%% total_success_rate=%.1f%% elapsed=%s avg_episode=%.2fs eta=%s video=%s")
    pattern = {"clean": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0], "empty": [1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
               "wrong": [1, 1, 1, 0, 1, 1, 1, 0, 1, 0], "tr": [1, 0, 0, 1, 0, 0, 1, 0, 1, 0]}
    lines = ["2026-11-11 07:00:00 INFO Task suite: libero_spatial",
             '2026-11-11 07:00:00 INFO Conditions: {"clean": null, "empty": "", "wrong": "pick up ...", "tr": "tabak ile ..."}',
             "2026-11-11 07:00:01 INFO Eval setup: checkpoint=checkpoints/VLANeXt_libero_spatial.pt tasks=10 trials_per_task=10 ..."]
    g = 0; succ = 0
    for init in range(10):
        for cnd in ["clean", "empty", "wrong", "tr"]:
            ok = bool(pattern[cnd][init]); g += 1; succ += ok
            lines.append("2026-11-11 07:05:00 INFO " + fmt % (1, 10, init, cnd, ok, 96 if ok else 220, "00:01:31" if ok else "00:03:04",
                                                             g, 40, succ, 0.0, 100 * succ / g, "00:10:00", 91.0, "00:30:00", "x.mp4"))
    lines.append("2026-11-11 08:00:00 INFO Eval finished: total_episodes=40 total_successes=%d final_success_rate=%.2f%% elapsed=01:02:00" % (succ, 2.5 * succ))
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "eval.log"; p.write_text("\n".join(lines), encoding="utf-8")
        info = read_log(p); eps = info["episodes"]
        assert len(eps) == 40, len(eps)
        rows = summarize(eps)
        assert [r["condition"] for r in rows] == ["clean", "empty", "wrong", "tr"]
        assert rows[0]["k"] == 9 and rows[1]["k"] == 5 and rows[3]["k"] == 4
        tr = rows[3]; assert (tr["b"], tr["c"]) == (5, 0), (tr["b"], tr["c"])       # clean✓tr✗ 5, clean✗tr✓ 0
        assert abs(tr["p_mcnemar"] - 2 * 0.5 ** 5) < 1e-9                          # 0.0625: 5 uyumsuz çift tek yönlüyse bile p = 0.0625
        assert abs(wilson(9, 10)[0] - 0.5958) < 1e-3 and abs(wilson(9, 10)[1] - 0.9821) < 1e-3
        assert abs(parse_duration("00:01:31") - 91) < 1e-9 and abs(parse_duration("1m23s") - 83) < 1e-9
        assert abs(rows[0]["mean_time"] - (9 * 91 + 184) / 10) < 1e-9
        md = write_outputs(info, eps, rows, "selftest", pathlib.Path(d) / "results")
        assert (pathlib.Path(d) / "results" / "runs.csv").read_text().count("\n") == 5
        # B planı: condition= alanı olmayan log + zorlanmış koşul adı
        p2 = pathlib.Path(d) / "e2.log"; p2.write_text("\n".join(lines).replace(" condition=tr", "").replace(" condition=clean", ""), encoding="utf-8")
        info2 = read_log(p2, forced_condition="empty"); assert {e["condition"] for e in info2["episodes"]} == {"empty"}
        print(md)
        print("\nselftest OK (sayılar TEMSİLİ; gerçek log'la çalıştır)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?")
    ap.add_argument("--logs", nargs="*", default=[], help="B planı: koşul=yol çiftleri")
    ap.add_argument("--run-id", default="vlanext_sp0_4cond_01")
    ap.add_argument("--out", default="results")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.log:
        info = read_log(a.log); eps = info["episodes"]
        if eps and all(e["condition"] == "clean" for e in eps) and info["conditions_line"] in (None, "<yok: task.language>"):
            print("UYARI: log'da condition= alanı yok — Ders 18 yaması uygulanmamış ya da VLANEXT_CONDITIONS boş; hepsi 'clean' sayıldı")
    elif a.logs:
        info = None; eps = []
        for item in a.logs:
            cnd, _, path = item.partition("=")
            inf = read_log(path, forced_condition=cnd); eps += inf["episodes"]; info = info or inf
    else:
        ap.error("eval.log ya da --logs ver (ya da --selftest)")
    if not eps:
        sys.exit("Log'da 'Episode result' satırı yok — eval başlamadan düşmüş olabilir; log'un sonuna bak.")
    print(f"suite={info['suite']}  checkpoint={info['checkpoint']}  episode={len(eps)}  koşullar={info['conditions_line'] or info['override'] or '—'}")
    rows = summarize(eps)
    print(write_outputs(info, eps, rows, a.run_id, a.out))
    print(f"\n→ {a.out}/episodes.csv, runs.csv ({len(rows)} satır eklendi), table.md")
    inits = defaultdict(set)
    for e in eps:
        inits[e["condition"]].add(e["init_state"])
    if len({frozenset(v) for v in inits.values()}) > 1:
        print("UYARI: koşulların init state kümeleri farklı — eşli test yalnızca ortak init state'lerde hesaplandı")


if __name__ == "__main__":
    main()
```

## power_day18.py
```python
"""Ders 18 — görsel 03 ve 04: 10 episode ne gösterir, ne göstermez? (saf matematik; model/GPU gerekmez)

  python power_day18.py <çıktı_klasörü/>

03-wilson-n10.png : k/n başarıda Wilson %95 aralığı, n = 10 / 20 / 50 (tüm k değerleri). Gerçek hesap.
04-power.png      : clean koşulunun gerçek başarı olasılığı p_clean = 0.95 iken, ikinci koşulun gerçek olasılığı p_x
                    düşükse iki test bu farkı α = 0.05'te yakalar mı?
                    (a) eşsiz: Fisher exact (bağımsız iki binom; tüm (k1,k2) çiftleri üzerinden kesin güç),
                    (b) eşli: McNemar exact — aynı init state'te iki koşul; modelde "clean başarır, x başaramaz" çifti
                        olasılığı p_clean − p_x, tersi 0 varsayımıyla (en iyimser eşli durum: koşul yalnızca kaybettirir).
                    Güç = p < 0.05 çıkan sonuçların toplam olasılığı. Kesin hesap (simülasyon değil).
Kaynak yok (hesap); eğitsel varsayımlar başlıkta yazılı.
"""
import math, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, fisher_exact

out = (sys.argv[1] if len(sys.argv) > 1 else ".").rstrip("/") + "/"; os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
YD, BD, GD, RD = "#c79a00", "#3b6ea8", "#2e8b57", "#c0392b"


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


# ---------------- 03: Wilson ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax, n, col in zip(axes, [10, 20, 50], [RD, YD, GD]):
    ks = np.arange(0, n + 1)
    lo = np.array([wilson(k, n)[0] for k in ks]); hi = np.array([wilson(k, n)[1] for k in ks])
    x = ks / n * 100
    ax.fill_between(x, lo * 100, hi * 100, color=col, alpha=0.18, lw=0)
    ax.plot(x, lo * 100, color=col, lw=1.6); ax.plot(x, hi * 100, color=col, lw=1.6)
    ax.plot(x, x, color="#555", lw=0.8, ls="--")
    ax.set_title(f"n = {n} episode", fontsize=11)
    ax.set_xlabel("gözlenen success rate (%)")
    ax.grid(alpha=0.25)
    for k in ([5, 9] if n == 10 else [10, 18] if n == 20 else [25, 45]):
        l, h = wilson(k, n)
        high = k / n > 0.8
        ax.annotate(f"{k}/{n}: {100*l:.0f}–{100*h:.0f}", (k / n * 100, (l * 100 - 7) if high else (h * 100 + 2)),
                    ha="right" if high else "center", fontsize=9, color=col)
    # 10 episode: 9/10 ile 5/10 aralıkları kesişiyor mu?
    if n == 10:
        l9, h9 = wilson(9, 10); l5, h5 = wilson(5, 10)
        ax.axhspan(l9 * 100, h5 * 100, color=R, alpha=0.5, lw=0)
        ax.text(2, (l9 + h5) / 2 * 100, f"9/10 ile 5/10'un\naralıkları çakışıyor\n({100*l9:.0f}–{100*h5:.0f})", fontsize=8.5, color=RD, va="center")
axes[0].set_ylabel("Wilson %95 aralığı (%)")
fig.suptitle("Ders 18 — k/n için Wilson %95 güven aralığı: 10 episode 'yön' verir, 'sayı' vermez (kesin hesap)", fontsize=12)
fig.tight_layout(); fig.savefig(out + "03-wilson-n10.png", dpi=160); plt.close(fig)


# ---------------- 04: güç ----------------
def power_fisher(p1, p2, n, alpha=0.05):
    pw = 0.0
    for k1 in range(n + 1):
        q1 = binom.pmf(k1, n, p1)
        if q1 < 1e-12:
            continue
        for k2 in range(n + 1):
            q2 = binom.pmf(k2, n, p2)
            if q2 < 1e-12:
                continue
            if fisher_exact([[k1, n - k1], [k2, n - k2]])[1] < alpha:
                pw += q1 * q2
    return pw


def mcnemar_p(b, c):
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)


def power_mcnemar(p1, p2, n, alpha=0.05):
    # çift türleri: (✓,✓) p2 ; (✓,✗) p1-p2 ; (✗,✗) 1-p1 ; (✗,✓) 0  — iyimser eşli model
    pb = p1 - p2
    pw = 0.0
    for b in range(n + 1):  # b uyumsuz çift (clean✓, x✗); c = 0 → p = 2 * 0.5**b
        pr = binom.pmf(b, n, pb)
        if mcnemar_p(b, 0) < alpha:
            pw += pr
    return pw


P1 = 0.95
px = np.round(np.arange(0.05, 0.951, 0.05), 2)
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
for ax, (fn, title) in zip(axes, [(power_fisher, "(a) eşsiz: Fisher exact (iki bağımsız grup)"),
                                  (power_mcnemar, "(b) eşli: McNemar exact (aynı init state; iyimser model: tek yönlü uyumsuzluk)")]):
    for n, col, ls in [(10, RD, "-"), (20, YD, "-"), (50, GD, "-")]:
        pw = [fn(P1, p, n) for p in px]
        ax.plot((P1 - px) * 100, np.array(pw) * 100, color=col, lw=2, ls=ls, label=f"n = {n}")
        # bu n için gücün %80'i geçtiği en küçük fark
        ok = [(P1 - p) * 100 for p, w in zip(px, pw) if w >= 0.8]
        if ok:
            ax.annotate(f"≥%80 güç: Δ ≥ {min(ok):.0f} puan", (min(ok), 80), xytext=(min(ok) + 3, 80 - (0 if n == 10 else 9 if n == 20 else 18)),
                        fontsize=9, color=col, arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
    ax.axhline(80, color="#777", lw=0.8, ls=":")
    ax.set_xlabel("gerçek fark: p_clean − p_x (puan); p_clean = 0.95")
    ax.set_title(title, fontsize=10.5); ax.grid(alpha=0.25); ax.legend(frameon=False, loc="lower right")
    for ref, yy, lab in [(0.95 - 0.66, 30, "π0.5 Spatial boş komut\nΔ≈29 (97.5→66, Chen vd.)"), (0.95 - 0.80, 8, "OpenVLA-OFT Spatial\nboş Δ≈14 (93.5→80)")]:
        ax.axvline(ref * 100, color="#999", lw=0.8, ls="--"); ax.text(ref * 100 + 0.8, yy, lab, fontsize=7.5, color="#555")
axes[0].set_ylabel("güç: farkı α=0.05'te yakalama olasılığı (%)")
fig.suptitle("Ders 18 — 10 / 20 / 50 episode hangi farkı güvenilir yakalar? (kesin güç hesabı; p_clean = 0.95 varsayımı)", fontsize=12)
fig.tight_layout(); fig.savefig(out + "04-power.png", dpi=160); plt.close(fig)

# metin özeti (derste kullanılan sayılar)
for n in (10, 20, 50):
    f29 = power_fisher(P1, 0.66, n); m29 = power_mcnemar(P1, 0.66, n)
    f15 = power_fisher(P1, 0.80, n); m15 = power_mcnemar(P1, 0.80, n)
    f55 = power_fisher(P1, 0.40, n); m55 = power_mcnemar(P1, 0.40, n)
    print(f"n={n:2d}  Δ=15: Fisher {100*f15:4.0f}%  McNemar {100*m15:4.0f}% | Δ=29: Fisher {100*f29:4.0f}%  McNemar {100*m29:4.0f}% | Δ=55: Fisher {100*f55:4.0f}%  McNemar {100*m55:4.0f}%")
print("Wilson 9/10: %.3f–%.3f  5/10: %.3f–%.3f  10/10: %.3f–%.3f  0/10: %.3f–%.3f" % (*wilson(9, 10), *wilson(5, 10), *wilson(10, 10), *wilson(0, 10)))
print("McNemar en küçük p, n=10, b uyumsuz: " + ", ".join(f"b={b}: {mcnemar_p(b,0):.4f}" for b in (3, 4, 5, 6, 10)))
```

## plot_day18.py
```python
"""Ders 18 — görsel 02 ve 06: literatürdeki 'boş komut' ve 'yanlış komut' sayıları (elle aktarıldı; model/GPU gerekmez).

  python plot_day18.py <çıktı_klasörü/>

02-empty-literature.png  LIBERO-Spatial'da doğru İngilizce komut vs boş komut (Ø), model başına.
   Kaynaklar: Chen vd. 2026 (arXiv:2606.15714) Tablo 6 — Spatial sütunları en / Ø (OpenVLA-OFT satırı makalede ABot-M0 ile
   aynı; makalenin kendi Tablo 1'iyle çelişiyor → 'şüpheli' işareti); Zhan vd. 2026 (arXiv:2601.04052) Tablo 8/9 — π0 ve π0.5
   correct / blank; Fang vd. 2026 (arXiv:2602.17659) Tablo I — π0.5 vision-only (dil girdisi kaldırılmış) Spatial 68.2.
06-wrong-literature.png  LIBERO-CF CF-Spatial (Fang vd. Tablo III, CAG'siz): yanlış (karşı-olgusal) komutta model hangi görevi yapıyor?
   Grounding/Success × Faithful (komuta sadık) / Biased (eğitimdeki görevi yapıyor).
"""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = (sys.argv[1] if len(sys.argv) > 1 else ".").rstrip("/") + "/"; os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
YD, BD, GD, RD = "#c79a00", "#3b6ea8", "#2e8b57", "#c0392b"

# ---------- 02 ----------
# (etiket, en, empty, kaynak kısaltması, not)
ROWS = [
    ("OpenVLA-OFT", 93.5, 80.0, "Chen T6", "satır şüpheli"),
    ("π0.5", 97.5, 66.0, "Chen T6", ""),
    ("π0.5", 95.4, 65.0, "Zhan T8/9", ""),
    ("π0.5 (vision-only)", 97.2, 68.2, "Fang T1", "dil girdisi yok"),
    ("π0", 96.8, 29.8, "Zhan T8/9", ""),
    ("Cosmos Policy", 96.5, 55.5, "Chen T6", ""),
    ("Qwen3-VL-OFT", 92.5, 34.0, "Chen T6", ""),
    ("Qwen3-VL-FAST", 89.0, 12.5, "Chen T6", ""),
    ("Qwen3-VL-GR00T", 94.0, 64.5, "Chen T6", ""),
    ("Qwen3-VL-π", 93.0, 41.0, "Chen T6", ""),
    ("Qwen2.5-VL-OFT", 91.5, 66.5, "Chen T6", ""),
    ("Qwen2.5-VL-FAST", 86.5, 2.0, "Chen T6", ""),
    ("Qwen2.5-VL-GR00T", 95.5, 62.5, "Chen T6", ""),
    ("Qwen2.5-VL-π", 88.0, 52.5, "Chen T6", ""),
]
fig, ax = plt.subplots(figsize=(15, 6.4))
y = np.arange(len(ROWS))[::-1]
for yi, (lab, en, em, src, note) in zip(y, ROWS):
    ax.plot([em, en], [yi, yi], color="#bbb", lw=3, solid_capstyle="round", zorder=1)
    ax.scatter(en, yi, s=80, color=GD, zorder=3, edgecolor="white", lw=1)
    ax.scatter(em, yi, s=80, color=RD, zorder=3, edgecolor="white", lw=1)
    ax.text(en + 1.2, yi, f"{en:.1f}", va="center", fontsize=9, color=GD)
    ax.text(em - 1.2, yi, f"{em:.1f}", va="center", ha="right", fontsize=9, color=RD)
    ax.text(103, yi, f"Δ −{en - em:.0f}  ·  {src}" + (f"  ({note})" if note else ""), va="center", fontsize=8.5, color="#555")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in ROWS])
ax.set_xlim(-2, 135); ax.set_xlabel("LIBERO-Spatial success rate (%)")
ax.axvspan(29.8, 68.2, color=R, alpha=0.25, lw=0)
ax.set_ylim(-1.6, len(ROWS) - 0.4)
ax.text(49, -1.1, "boş komutla Spatial'da 'görüntüden çözülebilen' pay: çoğu modelde %30–68", ha="center", fontsize=9, color=RD)
ax.scatter([], [], s=80, color=GD, label="doğru İngilizce komut"); ax.scatter([], [], s=80, color=RD, label="boş komut (Ø) / dil girdisi yok")
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.grid(axis="x", alpha=0.25)
ax.set_title("Ders 18 — literatürde 'boş komut' etkisi, LIBERO-Spatial (gerçek makale değerleri; Chen vd. 2606.15714 T6, Zhan vd. 2601.04052 T8–9, Fang vd. 2602.17659 T1)",
             fontsize=10.5)
fig.tight_layout(); fig.savefig(out + "02-empty-literature.png", dpi=160); plt.close(fig)

# ---------- 06 ----------
CF = {  # CF-Spatial, CAG'siz: (grounding faithful, grounding biased, success faithful, success biased)
    "OpenVLA-OFT": (6.8, 84.7, 1.1, 80.7),
    "π0": (70.9, 37.1, 34.4, 19.1),
    "π0.5": (39.3, 61.3, 24.4, 56.9),
}
fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
for ax, (title, i_f, i_b) in zip(axes, [("Grounding: gripper hangi nesneye dokundu?", 0, 1), ("Success: hangi görev tamamlandı?", 2, 3)]):
    x = np.arange(len(CF)); w = 0.36
    f = [v[i_f] for v in CF.values()]; b = [v[i_b] for v in CF.values()]
    ax.bar(x - w / 2, f, w, color=B, edgecolor=BD, label="Faithful — yanlış komutun hedefi (komutu okudu)")
    ax.bar(x + w / 2, b, w, color=R, edgecolor=RD, label="Biased — eğitimdeki orijinal görev (görüntüden gitti)")
    for xi, (fv, bv) in enumerate(zip(f, b)):
        ax.text(xi - w / 2, fv + 1.5, f"{fv:.1f}", ha="center", fontsize=9, color=BD)
        ax.text(xi + w / 2, bv + 1.5, f"{bv:.1f}", ha="center", fontsize=9, color=RD)
    ax.set_xticks(x); ax.set_xticklabels(list(CF)); ax.set_ylim(0, 100); ax.set_title(title, fontsize=11); ax.grid(axis="y", alpha=0.25)
axes[0].set_ylabel("%"); axes[0].legend(frameon=False, fontsize=9, loc="upper right")
fig.suptitle("Ders 18 — 'yanlış komut' verilince ne olur? LIBERO-CF, CF-Spatial (Fang vd. 2026, arXiv:2602.17659, Tablo III, CAG'siz)\n"
             "Bizim 'wrong' koşulunda success = Biased sütunu; Faithful'ı ancak videodan (hangi kâse?) sayabilirsin", fontsize=10.5)
fig.tight_layout(); fig.savefig(out + "06-wrong-literature.png", dpi=160); plt.close(fig)
print("02-empty-literature.png, 06-wrong-literature.png →", out)

# ---------- 07: olası örüntüler (TEMSİLİ — gerçek veri değil; Ders 19'un okuma anahtarı) ----------
PAT = [
    ("A) Dil görmezden geliniyor\n(saf shortcut)", [90, 85, 85, 85], "clean ≈ empty ≈ wrong ≈ tr\n→ Spatial görev 0 sahneden çözülüyor"),
    ("B) Dil okunuyor, Türkçe okunmuyor", [90, 40, 40, 40], "tr ≈ empty: Türkçe'den hiçbir şey çıkmıyor\n(Chen vd.: OFT/π0.5 örüntüsü)"),
    ("C) Yanlış komut 'biased'", [90, 40, 80, 40], "wrong ≈ clean ama empty düşük:\nkomut 'bir kâse al' diyor, sahne hangisini seçiyor"),
    ("D) Çok dilli omurga işe yarıyor", [90, 40, 40, 75], "tr > empty: Qwen3-VL Türkçe'den\nen azından nesne/eylemi çıkarıyor (hipotez)"),
]
fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True)
cols = [GD, RD, RD, YD]
for ax, (title, vals, note) in zip(axes, PAT):
    ax.bar(["clean", "empty", "wrong", "tr"], vals, color=[G, R, R, Y], edgecolor=cols, lw=1.2)
    ax.set_ylim(0, 128); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_title(title, fontsize=10.5); ax.grid(axis="y", alpha=0.25)
    ax.text(1.5, 124, note, ha="center", va="top", fontsize=8.5, color="#444")
axes[0].set_ylabel("success rate (%) — TEMSİLİ")
fig.suptitle("Ders 18 — 4 satırlık tablonun olası dört örüntüsü (TEMSİLİ değerler; gerçek sayılar senin koşundan gelecek, yorum Ders 19)", fontsize=11)
fig.tight_layout(); fig.savefig(out + "07-expected-patterns.png", dpi=160); plt.close(fig)
print("07-expected-patterns.png →", out)
```

## conditions_task0.json
```json
{
  "clean": null,
  "empty": "",
  "wrong": "pick up the black bowl next to the ramekin and place it on the plate",
  "tr": "tabak ile ramekinin arasındaki siyah kaseyi al ve tabağın üstüne koy"
}
```

## t4_cond_config.yaml
```yaml
# Ders 18 — 4 koşullu deney: LIBERO-Spatial görev 0, 10 init state × {clean, empty, wrong, tr}, ücretsiz T4.
# Ders 12'nin t4_bench_config.yaml'ından farkı: num_trials_per_task 1 → 10 (init state 0..9), num_parallel_envs 2 (Kaggle 29 GB RAM: 4 →
# aynı init state'in dört koşulu aynı batch'te koşar). Koşullar config'de değil ortam değişkeninde: VLANEXT_CONDITIONS (Ders 18 yaması).
# Kullanım (VLANeXt kökünde, Ders 12 + Ders 18 yamaları uygulanmış):
#   VLANEXT_TASK_IDS=0 VLANEXT_CONDITIONS="$(cat conditions_task0.json)" VLANEXT_RUN_TAG=_4cond \
#   python -m scripts.libero_bench_eval --config config/t4_cond_config.yaml
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
  finetuned_checkpoint: "checkpoints/VLANeXt_libero_spatial.pt"
  task_suite_name: "libero_spatial"
  num_steps_wait: 10
  num_steps_execute: 8
  num_parallel_envs: 2         # Colab 12.7 GB RAM: 2; Kaggle: 4 (koşul sayısıyla aynı → eşli batch)
  num_trials_per_task: 10      # init state 0..9; her biri 4 koşulla → görev başına 40 episode
  seed: 42
  image_size: 256
  resume_episodes: 0
  resume_successes: 0
```
