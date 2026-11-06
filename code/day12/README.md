# day12

Ders 12 — VLANeXt'i ücretsiz T4'te (Colab/Kaggle) LIBERO Spatial üzerinde çalıştırma. Sıra: notebook → yama → config → log ayrıştırma.

| Dosya | Ne |
|---|---|
| `colab_vlanext_t4.ipynb` | 8 hücrelik notebook: donanım kontrolü, kurulum, yamalar, checkpoint, eval, sonuç |
| `vlanext_t4_patch.py` | VLANeXt kaynağına 3 tek satırlık yama (sdpa, mmap, görev filtresi); idempotent |
| `t4_bench_config.yaml` | `libero_bench_config.yaml`'ın küçük bütçeli hâli (2 env, görev başına 1 episode) |
| `parse_eval_log.py` | `eval.log` → episode tablosu + `results/runs.csv` satırları |
| `plot_paper_results.py` | Dersteki 03/04 grafiklerini makale sayılarından üretir |

Notebook'u Colab'da açmak için: https://colab.research.google.com/github/mesutd0nmez/vla-robot-learning-tr/blob/main/code/day12/colab_vlanext_t4.ipynb

## vlanext_t4_patch.py

```python
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
```

## parse_eval_log.py

```python
"""Gün 12 — VLANeXt eval.log dosyasını oku; episode tablosu + runs.csv satırı üret.
Kullanım:  python parse_eval_log.py <eval.log> [--run-id vlanext_spatial_t4_01] [--lang en] [--condition correct] [--csv results/runs.csv]
Log biçimi scripts/libero_bench_eval.py'deki "Episode result: ..." ve "Eval finished: ..." satırlarıdır.
runs.csv sütunları repo CLAUDE.md'deki sözleşmeyle aynıdır:
  run_id, model, checkpoint, suite, task_id, lang, condition, n_episodes, success_rate
"""
import re, sys, csv, argparse, pathlib
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument("log")
p.add_argument("--run-id", default="vlanext_t4_01")
p.add_argument("--lang", default="en")
p.add_argument("--condition", default="correct", help="correct | empty | wrong | paraphrase | tr")
p.add_argument("--csv", default=None, help="varsa satırları bu CSV'ye ekler (başlık yoksa yazar)")
args = p.parse_args()

text = pathlib.Path(args.log).read_text(encoding="utf-8", errors="replace")
ep_re = re.compile(r"Episode result: task=(\d+)/(\d+) trial=(\d+) success=(True|False) rollout_steps=(\d+) episode_time=(\S+)")
suite = re.search(r"Task suite: (\S+)", text)
ckpt  = re.search(r"checkpoint=(\S+)", text)
suite = suite.group(1) if suite else "?"
ckpt  = pathlib.Path(ckpt.group(1)).name if ckpt else "?"

rows, per_task = [], defaultdict(lambda: [0, 0])
for m in ep_re.finditer(text):
    task, n_tasks, trial, ok, steps, t = m.groups()
    task_id = int(task) - 1
    ok = ok == "True"
    per_task[task_id][0] += 1; per_task[task_id][1] += int(ok)
    rows.append((task_id, int(trial), ok, int(steps), t))

if not rows:
    sys.exit("Log'da 'Episode result' satırı yok — eval başlamadan düşmüş olabilir; dosyanın sonuna bak.")

print(f"suite={suite}  checkpoint={ckpt}  episode={len(rows)}")
print(f"{'task':>4} {'trial':>5} {'success':>7} {'steps':>5} {'süre':>8}")
for task_id, trial, ok, steps, t in rows:
    print(f"{task_id:>4} {trial:>5} {str(ok):>7} {steps:>5} {t:>8}")
tot_n = sum(v[0] for v in per_task.values()); tot_s = sum(v[1] for v in per_task.values())
print(f"\nToplam: {tot_s}/{tot_n} = {100*tot_s/tot_n:.1f}%")
fin = re.search(r"Eval finished: .*final_success_rate=([\d.]+)%.*elapsed=(\S+)", text)
if fin:
    print(f"Script'in kendi özeti: SR={fin.group(1)}%  süre={fin.group(2)}")

if args.csv:
    out = pathlib.Path(args.csv); out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run_id", "model", "checkpoint", "suite", "task_id", "lang", "condition", "n_episodes", "success_rate"])
        for task_id in sorted(per_task):
            n, s = per_task[task_id]
            w.writerow([args.run_id, "VLANeXt-B", ckpt, suite, task_id, args.lang, args.condition, n, f"{s/n:.3f}"])
    print(f"{len(per_task)} satır eklendi → {out}")
```

## plot_paper_results.py

```python
"""Gün 12 — VLANeXt makalesindeki LIBERO ve LIBERO-plus (Language) sayılarını çizer.
Kullanım: python plot_paper_results.py [çıktı_klasörü/]   → 03-libero-results.png, 04-libero-plus-language.png
Sayılar: arXiv:2602.18532v2 (VLANeXt) tabloları; elle aktarıldı.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
import os, sys
out = (sys.argv[1] if len(sys.argv) > 1 else "./") ; os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# --- 03: LIBERO per-suite, VLANeXt paper Table (arXiv 2602.18532v2)
suites = ["Spatial", "Object", "Goal", "Long", "Ort."]
models = [("OpenVLA (2024)", "#8c6bb1", [84.7, 88.4, 79.2, 53.7, 76.5]),
          ("OpenVLA-OFT (2025)", "#3a9a5c", [97.6, 98.4, 97.9, 94.5, 97.1]),
          ("π0.5 (2025)", "#3d7fc4", [98.8, 98.2, 98.0, 92.4, 96.9]),
          ("VLANeXt-B (2026)", "#e09a00", [99.0, 99.2, 96.6, 94.8, 97.4])]
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(suites)); w = 0.2
for i, (name, c, v) in enumerate(models):
    b = ax.bar(x + (i - 1.5) * w, v, w * 0.92, color=c, label=name, zorder=3)
    for r, val in zip(b, v):
        ax.text(r.get_x() + r.get_width() / 2, val + 0.8, f"{val:.1f}", ha="center", va="bottom", fontsize=8, color="#333")
ax.set_xticks(x); ax.set_xticklabels(suites)
ax.set_ylim(40, 104); ax.set_ylabel("Success rate (%)")
ax.yaxis.grid(True, color="#e5e5e5", zorder=0)
ax.set_title("LIBERO: 4 suite, 500 episode/suite — kaynak: VLANeXt makalesi Tablo (arXiv:2602.18532v2)", fontsize=10)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False, fontsize=9)
fig.text(0.5, 0.005, "Her suite 10 görev × 50 episode = 500 episode. Bugün koşacağın 3 episode bu sayıların yanında yalnızca 'çalışıyor mu' testidir.",
        ha="center", fontsize=8.5, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out + "03-libero-results.png", dpi=180); plt.close(fig)

# --- 04: LIBERO-plus Language perturbation per suite, OFT vs VLANeXt (same paper, LIBERO-plus table)
suites = ["Spatial", "Object", "Goal", "Long"]
oft   = [80.5, 99.0, 53.2, 87.0]
vla   = [86.9, 98.6, 51.5, 90.3]
clean_oft = [97.6, 98.4, 97.9, 94.5]; clean_vla = [99.0, 99.2, 96.6, 94.8]
fig, ax = plt.subplots(figsize=(10, 5.2))
x = np.arange(4); w = 0.36
b1 = ax.bar(x - w/2, oft, w * 0.94, color="#3a9a5c", label="OpenVLA-OFT, Language bozulması", zorder=3)
b2 = ax.bar(x + w/2, vla, w * 0.94, color="#e09a00", label="VLANeXt-B, Language bozulması", zorder=3)
ax.scatter(x - w/2, clean_oft, marker="_", s=700, color="#222", zorder=4, linewidths=2, label="aynı model, temiz LIBERO (çizgi)")
ax.scatter(x + w/2, clean_vla, marker="_", s=700, color="#222", zorder=4, linewidths=2)
for bars, vals in ((b1, oft), (b2, vla)):
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width()/2, v - 6, f"{v:.1f}", ha="center", fontsize=9, color="white", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(suites); ax.set_ylim(0, 105); ax.set_ylabel("Success rate (%)")
ax.yaxis.grid(True, color="#e5e5e5", zorder=0)
ax.set_title("LIBERO-plus 'Language' bozulması: komut değiştirilince başarı ne kadar düşüyor? — kaynak: arXiv:2602.18532v2", fontsize=10)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=9)
fig.text(0.5, 0.005, "Goal'da iki model de ~%50'ye düşüyor (dil gerçekten kullanılıyor); Object'te ~%99 kalıyor (komut neredeyse gereksiz, Ders 3–4'teki kısayol).",
         ha="center", fontsize=8.5, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out + "04-libero-plus-language.png", dpi=180); plt.close(fig)
print("ok")
```

## t4_bench_config.yaml

```yaml
# Gün 12 — VLANeXt LIBERO eval, ücretsiz T4 için küçük bütçe.
# Orijinal: VLANeXt/config/libero_bench_config.yaml (num_parallel_envs: 32, num_trials_per_task: 50)
# Kullanım: VLANeXt kökünde  python -m scripts.libero_bench_eval --config config/t4_bench_config.yaml
data:
  augmentation:
    center_crop: true          # eğitimde random crop vardı; eval'de merkez kırpma (orijinal ayar)
    center_crop_ratio: 0.9

model:
  diffusion_steps: 5           # flow-matching adımı; orijinal 5. Hızlandırmak için düşürme, kalite düşer
  scheduler_type: "flow_match"

project:
  use_wandb: false             # wandb hesabı isteme

eval:
  finetuned_checkpoint: "checkpoints/VLANeXt_libero_spatial.pt"   # hf download ile indirdiğin dosya (16.4 GB)
  task_suite_name: "libero_spatial"   # libero_spatial | libero_object | libero_goal | libero_10
  num_steps_wait: 10           # ilk 10 adım no-op (nesneler yerleşsin) — OpenVLA protokolüyle aynı
  num_steps_execute: 8         # 8'lik chunk'ın 8'i de uygulanır, sonra model yeniden çağrılır
  num_parallel_envs: 2         # T4 + 12 GB RAM: 1–2. Her env ayrı bir süreç (~1 GB RAM)
  num_trials_per_task: 1       # görev başına 1 episode (orijinal 50). VLANEXT_TASK_IDS ile görevleri de daralt
  seed: 42
  image_size: 256
  resume_episodes: 0
  resume_successes: 0
```

## colab_vlanext_t4.ipynb (hücreler)

# Ders 12 — VLANeXt'i ücretsiz T4'te LIBERO üzerinde çalıştır

**Önce:** Çalışma zamanı → Çalışma zamanı türünü değiştir → **T4 GPU**. Kaggle'daysan Settings → Accelerator → **GPU T4 x2** (RAM 29 GB; Colab'ın 12.7 GB'ına göre daha güvenli).

Akış: (1) donanımı doğrula → (2) sistem kütüphaneleri → (3) VLANeXt + LIBERO kur → (4) T4 yamaları → (5) checkpoint indir (16.4 GB) → (6) 2–3 episode eval → (7) sonucu oku, videoyu izle, `runs.csv` satırı üret.

Toplam süre (hipotez, T4): kurulum ~10 dk + indirme ~5 dk + 3 episode ~5 dk. Hücreleri sırayla çalıştır; hata alırsan hücre çıktısının **tamamını** chat'e yapıştır.

```python
# 1) Donanım kontrolü — compute capability 7.5 = T4 (FlashAttention-2 YOK, bf16 donanım desteği YOK → yamalar gerekli)
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
import torch, psutil, shutil
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| GPU", torch.cuda.get_device_name(0), "| cc", torch.cuda.get_device_capability(0))
print("RAM %.1f GB | disk boş %.0f GB" % (psutil.virtual_memory().total/1e9, shutil.disk_usage("/").free/1e9))
print("bf16 (emülasyon dahil):", torch.cuda.is_bf16_supported())
```

```python
# 2) MuJoCo'nun ekransız (EGL) render için istediği sistem kütüphaneleri — VLANeXt COMMON_ISSUES #1
!apt-get -qq update && apt-get -qq install -y libgl1-mesa-dev libegl1-mesa-dev libxrandr2 libxcursor1 libxinerama1 libxrender1 > /dev/null
import os; os.environ["MUJOCO_GL"] = "egl"; os.environ["PYOPENGL_PLATFORM"] = "egl"
print("ok")
```

```python
# 3) VLANeXt + LIBERO. requirements.txt'in tamamını DEĞİL, eval için gerekenleri kuruyoruz
#    (torch 2.4 / flash-attn / deepspeed / dm-reverb yok). mujoco 3.8.1 VLANeXt'in kendi pinidir; Linux'ta robosuite 1.4.1 ile çalışır.
%cd /content
!git clone -q https://github.com/DravenALG/VLANeXt && echo "clone ok"
%cd /content/VLANeXt
!mkdir -p third_party && git clone -q https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
!pip install -q "transformers==5.5.0" "diffusers==0.36.0" accelerate huggingface_hub sentencepiece h5py opencv-python "imageio[ffmpeg]" \
    "robosuite==1.4.1" "mujoco==3.8.1" bddl easydict cloudpickle gym pyyaml wandb tqdm tensorflow-datasets rlds 2>&1 | tail -2
!pip install -q ./third_party/LIBERO 2>&1 | tail -1
# LIBERO ilk import'ta "custom path? (Y/N)" sorar; N cevabını borudan veriyoruz
!printf 'N\n' | python -c "import libero.libero; print('LIBERO ok')"
```

```python
# 4) T4 yamaları: flash_attention_2 → sdpa, torch.load mmap=True, VLANEXT_TASK_IDS görev filtresi
!curl -sO https://raw.githubusercontent.com/mesutd0nmez/vla-robot-learning-tr/main/code/day12/vlanext_t4_patch.py
!curl -s -o config/t4_bench_config.yaml https://raw.githubusercontent.com/mesutd0nmez/vla-robot-learning-tr/main/code/day12/t4_bench_config.yaml
!python vlanext_t4_patch.py /content/VLANeXt
```

```python
# 5) Checkpoint (16.4 GB; Spatial suite için eğitilmiş). Qwen3-VL-2B omurgası ilk model yüklemesinde otomatik iner (~4.5 GB).
!hf download DravenALG/VLANeXt VLANeXt_libero_spatial.pt --local-dir checkpoints
!ls -lh checkpoints
```

```python
# 6) Eval: yalnızca görev 0, 2 ve 8 (between / table center / next to the plate), görev başına 1 episode.
#    İlk model yüklemesi birkaç dakika sürer (Qwen indirme + 16.4 GB state_dict kopyalama).
import os
os.environ["VLANEXT_TASK_IDS"] = "0,2,8"
os.environ["PYTHONPATH"] = "/content/VLANeXt/third_party/LIBERO"
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
os.environ.pop("LIBERO_CONFIG_PATH", None)     # COMMON_ISSUES #6
!CUDA_VISIBLE_DEVICES=0 python -m scripts.libero_bench_eval --config config/t4_bench_config.yaml 2>&1 | grep -v Warning | tee eval_stdout.txt
```

```python
# 7) Sonuç: log'u tabloya çevir, runs.csv satırı üret, videoları listele
!curl -sO https://raw.githubusercontent.com/mesutd0nmez/vla-robot-learning-tr/main/code/day12/parse_eval_log.py
import glob
logs = sorted(glob.glob("checkpoints/libero_spatial_checkpoint_*/eval.log"))
print(logs)
!python parse_eval_log.py {logs[-1]} --run-id vlanext_spatial_t4_01 --csv results/runs.csv
!ls checkpoints/libero_spatial_checkpoint_*/*.mp4
```

```python
# 8) Bir rollout videosunu notebook içinde izle (başarısız olanı seç: nerede takılıyor?)
from IPython.display import Video
import glob
vids = sorted(glob.glob("checkpoints/libero_spatial_checkpoint_*/*.mp4"))
Video(vids[0], embed=True, width=384)
```

## Sonra
- `results/runs.csv` ve `eval.log`'u indir (Dosyalar paneli) → repoda `projects/` altında bir deney klasörüne koy.
- Ders 13'te bu kodun içini okuyacağız: komut nerede giriyor (`spec.task_description`), action nasıl decode ediliyor (`_denormalize_action`), success nerede hesaplanıyor (`env.step` → `done`).
- Türkçe deney için değiştirilecek tek satır `build_episode_specs` içindeki `task_description = task.language` — ama onu Ders 18'de yapacağız.

