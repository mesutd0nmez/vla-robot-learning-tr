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
