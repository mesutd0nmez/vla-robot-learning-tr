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
