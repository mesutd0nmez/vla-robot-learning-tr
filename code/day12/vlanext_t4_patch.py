"""Gün 12 — VLANeXt kod tabanını ücretsiz T4 (Colab/Kaggle) için yamala.
Kullanım:  python vlanext_t4_patch.py /content/VLANeXt        (idempotent; ikinci çalıştırma "zaten yamalı" der)

Üç yama (hepsi tek satırlık, kaynak: VLANeXt repo, Kasım 2026):
 1. flash_attention_2 -> sdpa      T4 (sm75) FlashAttention-2'yi desteklemez; eval kodunda sabit yazılı.
 2. torch.load(..., mmap=True)     Checkpoint 16.4 GB; Colab'ın 12.7 GB RAM'ine sığmaz. mmap ile dosya
                                   diskten parça parça okunur, RAM'e tamamı yüklenmez.
 3. VLANEXT_TASK_IDS ortam değişkeni  Varsayılan eval 10 görevin hepsini koşar; "0,2" dersen yalnızca o görevler.
"""
import os, sys, re, pathlib

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
utils = root / "src/evaluation/libero_bench/VLANeXt_utils.py"
par   = root / "src/evaluation/libero_bench/parallel_eval_utils.py"
for p in (utils, par):
    if not p.exists():
        sys.exit(f"HATA: {p} yok — argüman VLANeXt repo kökü olmalı")

def patch(path, old, new, label):
    src = path.read_text(encoding="utf-8")
    if new in src:
        print(f"  [=] {label}: zaten yamalı"); return
    if old not in src:
        sys.exit(f"HATA: {label}: beklenen satır bulunamadı; repo değişmiş olabilir. Aranan: {old!r}")
    path.write_text(src.replace(old, new, 1), encoding="utf-8")
    print(f"  [+] {label}: uygulandı")

print(f"VLANeXt kökü: {root}")
# 1) T4: FlashAttention-2 yok -> PyTorch SDPA
patch(utils,
      'attn_implementation = "flash_attention_2"',
      'attn_implementation = os.environ.get("VLANEXT_ATTN", "sdpa")  # T4 yaması (orijinal: flash_attention_2)',
      "attn_implementation -> sdpa")
if "import os" not in utils.read_text(encoding="utf-8").split("\n\n")[0]:
    patch(utils, "from pathlib import Path", "import os\nfrom pathlib import Path", "import os (VLANeXt_utils)")

# 2) 16.4 GB .pt dosyasını RAM'e kopyalamadan aç
patch(utils,
      'ckpt = torch.load(str(path), map_location="cpu")',
      'ckpt = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)  # T4 yaması: RAM',
      "torch.load mmap=True")

# 3) Görev filtresi: VLANEXT_TASK_IDS="0,2,5"
patch(par,
      "import numpy as np\n",
      'import os\nimport numpy as np\n_TASK_FILTER = {int(x) for x in os.environ.get("VLANEXT_TASK_IDS", "").split(",") if x.strip()}  # T4 yaması\n',
      "VLANEXT_TASK_IDS tanımı")
patch(par,
      "    for task_id in range(num_tasks):\n        task = task_suite.get_task(task_id)",
      "    for task_id in range(num_tasks):\n        if _TASK_FILTER and task_id not in _TASK_FILTER:\n            continue  # T4 yaması\n        task = task_suite.get_task(task_id)",
      "görev filtresi")

print("Bitti. Kontrol:")
for p in (utils, par):
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if "T4 yaması" in line:
            print(f"  {p.name}:{i}: {line.strip()[:90]}")
