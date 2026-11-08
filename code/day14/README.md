# day14

Ders 14 — OpenVLA-7B'yi Mac'te MPS ile yükleyip tek görüntü + İngilizce komuttan 7-DoF action almak.

| Dosya | Ne |
|---|---|
| `openvla_mps_single_action.py` | OpenVLA-7B (LIBERO-Spatial fine-tune) Mac'te MPS/bf16 ile yükler; tek görüntü + komut → 7 token, bin, [-1,1], gerçek action, süre; `action_tokens.json` yazar (ayrı `openvla` env) |
| `render_libero_frame.py` | LIBERO Spatial görev/başlangıç için 180° döndürülmüş 256×256 agentview karesi üretir (`vla` env, Ders 10) |
| `preprocess_demo.py` | Görüntü ön işleme zincirini (döndür → 224 → center crop) gerçek bir robosuite/LIBERO karesinde 4 panel olarak çizer (model gerekmez) |
| `decode_action_tokens.py` | 7 action token'ını elle decode eder: token → bin → [-1,1] → q01/q99 (gerçek `libero_spatial` istatistikleri gömülü; model/indirme gerekmez) |
| `plot_day14.py` | Dersteki 02 (token→action), 03 (dizi düzeni + EN/TR token sayıları), 04 (M2 bellek bütçesi) görselleri |

Kullanım:
```bash
# 1) model GEREKMEDEN: decode zinciri ve ön işleme (vla env yeter)
conda activate vla && cd ~/vla-work && mkdir -p day14 && cd day14
for f in decode_action_tokens.py preprocess_demo.py render_libero_frame.py openvla_mps_single_action.py; do
  curl -O https://raw.githubusercontent.com/mesutd0nmez/vla-robot-learning-tr/main/code/day14/$f; done
python decode_action_tokens.py
python preprocess_demo.py --libero --out preprocess.png     # LIBERO yoksa --libero'yu kaldır
python render_libero_frame.py --suite libero_spatial --task 0 --out frame_task0.png

# 2) model: AYRI env (transformers 4.40.1, Ders 13'ün Qwen3-VL sürümüyle çakışır)
conda create -n openvla python=3.10 -y && conda activate openvla
pip install torch torchvision "transformers==4.40.1" "tokenizers==0.19.1" "timm==0.9.10" accelerate sentencepiece pillow "numpy<2"
python openvla_mps_single_action.py --image frame_task0.png \
  --instruction "pick up the black bowl between the plate and the ramekin and place it on the plate"
python decode_action_tokens.py --from-json action_tokens.json   # aynı sayıları elle doğrula
```

## openvla_mps_single_action.py

```python
"""Ders 14 — OpenVLA-7B (LIBERO-Spatial fine-tune) Mac'te MPS ile: tek görüntü + komut → 7-DoF action.

Kurulum (ayrı conda env; Ders 13'ün transformers>=4.57'si ile çakışır):
    conda create -n openvla python=3.10 -y && conda activate openvla
    pip install torch torchvision "transformers==4.40.1" "tokenizers==0.19.1" "timm==0.9.10" \
                accelerate sentencepiece pillow "numpy<2"
Kullanım:
    python openvla_mps_single_action.py --image frame_task0.png \
        --instruction "pick up the black bowl between the plate and the ramekin and place it on the plate"
    python openvla_mps_single_action.py --image frame_task0.png --instruction "..." --no-center-crop
İlk çalıştırmada ~15 GB checkpoint iner (~/.cache/huggingface). Model yüklemesi dakikalar sürer; sonraki
çağrılar hızlıdır. Çıktı: 7 token id, bin indeksleri, [-1,1] değerleri, gerçek action, süre; action_tokens.json.
"""
import argparse
import json
import math
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

CKPT = "openvla/openvla-7b-finetuned-libero-spatial"
UNNORM_KEY = "libero_spatial"      # dataset_statistics.json'daki tek anahtar
DIMS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def center_crop_like_training(img: Image.Image, crop_scale=0.9) -> Image.Image:
    """openvla_utils.crop_and_resize'ın TF'siz karşılığı: alanın %90'ı (kenar sqrt(0.9)) merkezden kesilir,
    sonra eski boyuta geri büyütülür. LIBERO fine-tune'ları random-crop augmentasyonu ile eğitildiği için
    eval'de bu ZORUNLU (run_libero_eval.py:94 assert)."""
    w, h = img.size
    s = math.sqrt(crop_scale)
    cw, ch = round(w * s), round(h * s)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="224x224 (ya da 256x256) RGB PNG; 180° döndürülmüş LIBERO agentview")
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--no-center-crop", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--out", default="action_tokens.json")
    args = ap.parse_args()

    device = pick_device()
    dtype = getattr(torch, args.dtype)
    print(f"[*] device={device} dtype={dtype} attn={args.attn}")

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        CKPT,
        attn_implementation=args.attn,      # flash_attention_2 Mac'te yok
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    vla.eval()
    n_params = sum(p.numel() for p in vla.parameters())
    print(f"[*] yüklendi: {n_params/1e9:.2f} B parametre, {time.time()-t0:.0f} s")
    print(f"[*] norm_stats anahtarları: {list(vla.norm_stats.keys())}")

    # --- görüntü ---
    img = Image.open(args.image).convert("RGB")
    if img.size != (224, 224):
        img = img.resize((224, 224), Image.LANCZOS)   # eval kodu lanczos3 kullanır (libero_utils.py:44)
    if not args.no_center_crop:
        img = center_crop_like_training(img)
    img.save("model_input_224.png")

    # --- prompt (openvla_utils.py:163) ---
    prompt = f"In: What action should the robot take to {args.instruction.lower()}?\nOut:"
    inputs = processor(prompt, img).to(device, dtype=dtype)   # BatchFeature.to yalnızca float tensörleri cast eder
    ids = inputs["input_ids"][0].tolist()
    print(f"[*] prompt token sayısı (BOS dahil): {len(ids)}; son token: {ids[-1]} "
          f"(29871 '▁' değilse predict_action ekler)")
    print(f"[*] pixel_values: {tuple(inputs['pixel_values'].shape)}  (6 kanal = DINOv2 3 + SigLIP 3)")

    # --- action ---
    t1 = time.time()
    with torch.inference_mode():
        action = vla.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
    dt = time.time() - t1

    # predict_action token id'lerini döndürmez; aynı generate'i bir kez daha çağırarak id'leri alıyoruz
    # (deterministik: do_sample=False). Sadece inceleme için; eval döngüsünde yapılmaz.
    input_ids = inputs["input_ids"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat((input_ids, torch.tensor([[29871]], device=device)), dim=1)
    with torch.inference_mode():
        gen = vla.generate(input_ids=input_ids, pixel_values=inputs["pixel_values"],
                           attention_mask=torch.ones_like(input_ids), max_new_tokens=7, do_sample=False)
    tok = gen[0, -7:].cpu().numpy()
    disc = np.clip(vla.vocab_size - tok - 1, 0, vla.bin_centers.shape[0] - 1)
    norm = vla.bin_centers[disc]

    print(f"\n[*] predict_action süresi: {dt:.2f} s\n")
    print(f"{'boyut':8s} {'token':>6s} {'bin':>4s} {'[-1,1]':>8s} {'action':>8s}")
    for i, d in enumerate(DIMS):
        print(f"{d:8s} {int(tok[i]):6d} {int(disc[i]):4d} {norm[i]:8.4f} {action[i]:8.4f}")
    g = -np.sign(2 * action[6] - 1)   # run_libero_eval.py:219-225
    print(f"\nLIBERO'ya gidecek gripper: {g:+.0f} ({'kapat' if g > 0 else 'aç'})")

    json.dump({"checkpoint": CKPT, "instruction": args.instruction, "prompt": prompt,
               "prompt_token_ids": ids, "action_token_ids": tok.tolist(), "bins": disc.tolist(),
               "normalized": norm.tolist(), "action": action.tolist(), "seconds": dt,
               "device": str(device), "dtype": args.dtype}, open(args.out, "w"), indent=1)
    print(f"[*] yazıldı: {args.out}, model_input_224.png")


if __name__ == "__main__":
    main()
```

## render_libero_frame.py

```python
"""Ders 14 — LIBERO'dan tek bir agentview karesi üretir (OpenVLA'nın göreceği biçimde: 180° döndürülmüş).

Ders 10'un `vla` ortamında çalışır (mujoco 2.3.7, robosuite 1.4.1, LIBERO). GPU gerekmez.
    conda activate vla
    python render_libero_frame.py --suite libero_spatial --task 0 --init 0 --out frame_task0.png
Çıktı: 256x256 RGB PNG (`frame_task0.png`), yanında ham/çevrilmiş karşılaştırma `frame_task0_raw_vs_flipped.png`
ve komut metni (task.language) ekrana.
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl" if os.uname().sysname == "Linux" else "glfw")

import numpy as np
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--init", type=int, default=0, help="50 kayıtlı başlangıçtan hangisi")
    ap.add_argument("--settle", type=int, default=10, help="OpenVLA eval gibi 10 no-op adım bekle")
    ap.add_argument("--out", default="frame_task0.png")
    args = ap.parse_args()

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(args.task)[args.init])
    for _ in range(args.settle):                     # run_libero_eval.py: num_steps_wait = 10
        obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
    raw = obs["agentview_image"]
    flipped = raw[::-1, ::-1]                        # libero_utils.get_libero_image: 180° döndür
    Image.fromarray(flipped).save(args.out)
    side = np.concatenate([raw, flipped], axis=1)
    Image.fromarray(side).save(args.out.replace(".png", "_raw_vs_flipped.png"))
    print("komut (task.language):", task.language)
    print("yazıldı:", args.out, "| ham (sol) vs çevrilmiş (sağ):", args.out.replace(".png", "_raw_vs_flipped.png"))
    env.close()


if __name__ == "__main__":
    main()
```

## preprocess_demo.py

```python
"""Ders 14 — OpenVLA'nın görüntü ön işleme zincirini gerçek bir robosuite karesinde gösterir (model GEREKMEZ).

ham agentview (256x256, robosuite ters verir) → img[::-1, ::-1] (180°) → 224x224 (lanczos) → center crop %90 → 224x224
Kaynak: experiments/robot/libero/libero_utils.py:50-58 ve experiments/robot/openvla_utils.py:132-156.
LIBERO kuruluysa (Ders 10) `--libero` ile aynı zinciri LIBERO Spatial görev 0'da çalıştırır; yoksa robosuite Lift.
    python preprocess_demo.py --out 05-preprocess-pipeline.png
"""
import argparse
import math
import os

os.environ.setdefault("MUJOCO_GL", "egl" if os.uname().sysname == "Linux" else "glfw")

import numpy as np
from PIL import Image


def get_frame_robosuite():
    import robosuite as suite
    env = suite.make("Lift", robots="Panda", has_renderer=False, has_offscreen_renderer=True,
                     use_camera_obs=True, camera_names=["agentview"], camera_heights=256, camera_widths=256,
                     control_freq=20)
    obs = env.reset()
    for _ in range(10):
        obs, _, _, _ = env.step(np.zeros(env.action_dim))
    env.close()
    return obs["agentview_image"], "robosuite Lift (Panda)"


def get_frame_libero(task_id=0):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    s = benchmark.get_benchmark_dict()["libero_spatial"]()
    t = s.get_task(task_id)
    env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file),
                             camera_heights=256, camera_widths=256)
    env.reset()
    obs = env.set_init_state(s.get_task_init_states(task_id)[0])
    for _ in range(10):
        obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
    env.close()
    return obs["agentview_image"], f"LIBERO Spatial görev {task_id}: {t.language}"


def center_crop(img: Image.Image, crop_scale=0.9):
    w, h = img.size
    s = math.sqrt(crop_scale)
    cw, ch = round(w * s), round(h * s)
    l, t = (w - cw) // 2, (h - ch) // 2
    return img.crop((l, t, l + cw, t + ch)).resize((w, h), Image.BILINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero", action="store_true")
    ap.add_argument("--out", default="05-preprocess-pipeline.png")
    a = ap.parse_args()
    raw, title = get_frame_libero() if a.libero else get_frame_robosuite()
    flipped = raw[::-1, ::-1]
    r224 = Image.fromarray(flipped).resize((224, 224), Image.LANCZOS)
    cropped = center_crop(r224)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, im, cap in zip(axes, [raw, flipped, np.array(r224), np.array(cropped)],
                           ["1) ham obs['agentview_image'] 256x256\n(robosuite ters verir)",
                            "2) img[::-1, ::-1] — 180° döndür\n(libero_utils.py:56)",
                            "3) lanczos → 224x224\n(libero_utils.py:44)",
                            "4) center crop %90 alan → 224x224\n(openvla_utils.py:147)  → modele giden"]):
        ax.imshow(im)
        ax.set_title(cap, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"OpenVLA görüntü ön işleme — {title}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(a.out, dpi=150)
    Image.fromarray(np.array(cropped)).save(a.out.replace(".png", "_model_input.png"))
    print("yazıldı:", a.out)


if __name__ == "__main__":
    main()
```

## decode_action_tokens.py

```python
"""Ders 14 — OpenVLA'nın 7 action token'ını elle decode eder (model/GPU/indirme GEREKMEZ).

OpenVLA'nın `predict_action` fonksiyonundaki üç adımı birebir uygular:
    token_id → bin indeksi → [-1, 1] bin merkezi → q01/q99 ile gerçek action
Kaynak: prismatic/extern/hf/modeling_prismatic.py:518-534 (openvla/openvla, commit c8f03f4).
İstatistikler: openvla/openvla-7b-finetuned-libero-spatial / dataset_statistics.json ("libero_spatial").

Kullanım:
    python decode_action_tokens.py                      # örnek 7 token
    python decode_action_tokens.py 31890 31812 31999 31870 31875 31872 31744
    python decode_action_tokens.py --from-json action_tokens.json   # openvla_mps_single_action.py çıktısı
"""
import json
import sys

import numpy as np

# LLaMA-2 tokenizer: 32.000 token. OpenVLA, son 256 token'ı (31744..31999) action için kullanır.
# HF config'te text_config.vocab_size = 32064 (64'e yuvarlama); kod bunu geri alır: 32064 - 64 = 32000.
VOCAB_SIZE = 32000
N_BINS = 256
BINS = np.linspace(-1, 1, N_BINS)              # 256 kenar
BIN_CENTERS = (BINS[:-1] + BINS[1:]) / 2.0     # 255 merkez (!)

# dataset_statistics.json → "libero_spatial" → "action" (gerçek değerler)
LIBERO_SPATIAL = {
    "q01": [-0.7454732114076613, -0.6616071462631226, -0.9375,
            -0.1071428582072258, -0.20678570866584778, -0.1842857152223587, 0.0],
    "q99": [0.9375, 0.8758928775787354, 0.9321428537368774,
            0.1039285734295845, 0.17678570747375488, 0.14571428298950195, 1.0],
    "mask": [True, True, True, True, True, True, False],
}
DIMS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def decode(token_ids, stats=LIBERO_SPATIAL):
    token_ids = np.asarray(token_ids, dtype=np.int64)
    assert token_ids.shape == (7,), "7 token bekleniyor"
    # 1) token → bin indeksi (modeling_prismatic.py:522-523)
    discretized = VOCAB_SIZE - token_ids                       # 1..256
    discretized = np.clip(discretized - 1, 0, BIN_CENTERS.shape[0] - 1)   # 0..254
    # 2) bin indeksi → [-1, 1]
    normalized = BIN_CENTERS[discretized]
    # 3) [-1, 1] → gerçek action (q01/q99 ile), gripper'a dokunma (mask=False)
    lo, hi, mask = (np.array(stats[k]) for k in ("q01", "q99", "mask"))
    action = np.where(mask, 0.5 * (normalized + 1) * (hi - lo) + lo, normalized)
    return discretized, normalized, action


def libero_gripper(action):
    """run_libero_eval.py:219-225 — gripper [0,1] → [-1,1], işaretle ikilileştir, sonra OpenVLA için ters çevir."""
    g = 2 * action[6] - 1
    g = np.sign(g)
    return -g   # invert_gripper_action: OpenVLA verisinde 1 = aç, LIBERO'da +1 = kapat


def main(argv):
    if len(argv) >= 2 and argv[0] == "--from-json":
        toks = json.load(open(argv[1]))["action_token_ids"]
    elif len(argv) == 7:
        toks = [int(x) for x in argv]
    else:
        # örnek: hafif geri, sağa, hızlı aşağı; küçük dönüş; gripper "1 = aç" (OpenVLA verisinde gripper 0/1'dir:
        # 0 → bin 127 → token 31872, 1 → bin 254 → token 31744; model eğitimde başka gripper değeri görmedi)
        toks = [31890, 31812, 31999, 31870, 31875, 31872, 31744]
    disc, norm, act = decode(toks)
    print(f"{'boyut':8s} {'token':>6s} {'bin':>4s} {'[-1,1]':>8s} {'q01':>8s} {'q99':>8s} {'action':>8s}")
    for i, d in enumerate(DIMS):
        print(f"{d:8s} {toks[i]:6d} {disc[i]:4d} {norm[i]:8.4f} {LIBERO_SPATIAL['q01'][i]:8.4f} "
              f"{LIBERO_SPATIAL['q99'][i]:8.4f} {act[i]:8.4f}")
    print(f"\nLIBERO'ya giden gripper (normalize + binarize + invert): {libero_gripper(act):+.0f}  "
          f"({'kapat' if libero_gripper(act) > 0 else 'aç'})")
    step = (np.array(LIBERO_SPATIAL['q99']) - np.array(LIBERO_SPATIAL['q01'])) / (N_BINS - 1)
    print("Bir bin'in gerçek karşılığı (q99-q01)/255:", np.round(step[:6], 4))


if __name__ == "__main__":
    main(sys.argv[1:])
```

## plot_day14.py

```python
"""Ders 14 görselleri: 02 token→action eşlemesi (gerçek libero_spatial q01/q99), 03 dizi düzeni + EN/TR token
sayıları (LLaMA-2 tokenizer, llama-tokenizer-js ile sayıldı), 04 M2 bellek bütçesi (parametre sayıları yaklaşık).
    python plot_day14.py --out ../../lectures/images/lecture-14
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
DIMS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]
Q01 = np.array([-0.7454732114076613, -0.6616071462631226, -0.9375,
                -0.1071428582072258, -0.20678570866584778, -0.1842857152223587])
Q99 = np.array([0.9375, 0.8758928775787354, 0.9321428537368774,
                0.1039285734295845, 0.17678570747375488, 0.14571428298950195])
BINS = np.linspace(-1, 1, 256)
CENTERS = (BINS[:-1] + BINS[1:]) / 2


def fig02(out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    b = np.arange(255)
    for i, c in zip([0, 2, 3], ["#e07a00", "#1f77b4", "#2ca02c"]):
        act = 0.5 * (CENTERS + 1) * (Q99[i] - Q01[i]) + Q01[i]
        ax1.plot(b, act, lw=2.2, color=c, label=f"{DIMS[i]}  (q01={Q01[i]:.3f}, q99={Q99[i]:.3f})")
    ax1.axvline(127, color="grey", ls="--", lw=1)
    ax1.text(129, -0.85, "bin 127 → [-1,1]'de 0.0\n(token 31872)", fontsize=9, color="grey")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_xlabel("bin indeksi (= 32000 − token id − 1)")
    ax1.set_ylabel("action (LIBERO birimi, normalize delta)")
    ax1.set_title("Token → bin → action: doğrusal ama simetrik DEĞİL", fontsize=11)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(alpha=0.3)

    x = np.arange(6)
    ax2.bar(x, Q99, color=G, edgecolor="k", label="q99 (bin 254'ün karşılığı)")
    ax2.bar(x, Q01, color=B, edgecolor="k", label="q01 (bin 0'ın karşılığı)")
    zero_act = 0.5 * (0 + 1) * (Q99 - Q01) + Q01
    ax2.scatter(x, zero_act, color="red", zorder=5, s=45, label="\"nötr\" token 31872'nin ürettiği action")
    for i in range(6):
        ax2.text(i, zero_act[i] + 0.04, f"{zero_act[i]:+.3f}", ha="center", fontsize=8, color="red")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xticks(x, DIMS)
    ax2.set_title("libero_spatial q01/q99 (dataset_statistics.json, gerçek)", fontsize=11)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(alpha=0.3, axis="y")
    fig.suptitle("OpenVLA action decode — openvla/openvla-7b-finetuned-libero-spatial", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "02-token-to-action.png"), dpi=160)
    plt.close(fig)


def fig03(out):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.2), gridspec_kw={"height_ratios": [1, 1.6]})
    # üst: dizi düzeni (gerçek sayılar: BOS 1, patch 256, EN prompt 33, '▁' 1, action 7)
    segs = [("BOS", 1, B), ("256 görüntü patch token'ı (DINOv2+SigLIP → projector)", 256, Y),
            ("prompt 33", 33, B), ("▁", 1, R), ("action", 7, G)]
    pos = 0
    for name, n, c in segs:
        w = max(n, 12)
        ax1.barh(0, w, left=pos, color=c, edgecolor="k", height=0.6)
        ax1.text(pos + w / 2, 0, name if n > 12 else f"{name}\n{n}", ha="center", va="center", fontsize=9)
        pos += w
    ax1.set_xlim(0, pos)
    ax1.set_yticks([])
    ax1.set_xlabel("konum (LLaMA-2 girdi dizisi: 1 + 256 + 33 + 1 = 291 token, sonra 7 token üretilir)")
    ax1.set_title("Spatial görev 0 için OpenVLA dizisi — 'pick up the black bowl between the plate and the ramekin "
                  "and place it on the plate'", fontsize=10)
    # alt: EN vs TR token sayıları (llama-tokenizer-js, LLaMA-2 BPE; BOS hariç)
    labels = ["EN komut", "TR komut", "EN tam prompt", "TR tam prompt"]
    counts = [20, 32, 33, 45]
    colors = [B, R, B, R]
    ax2.barh(labels[::-1], counts[::-1], color=colors[::-1], edgecolor="k")
    for i, v in enumerate(counts[::-1]):
        ax2.text(v + 0.5, i, str(v), va="center", fontsize=10)
    ax2.set_xlim(0, 60)
    ax2.set_xlabel("token sayısı (LLaMA-2 tokenizer, BOS hariç)")
    ax2.set_title("Aynı komut Türkçe: 20 → 32 token (×1.6). "
                  "'arasındaki' = ▁ar|as|ı|nd|aki (5), 'üstüne' = ▁ü|st|ü|ne (4), 'tabağın' = ▁tab|a|ğ|ı|n (5)",
                  fontsize=10)
    ax2.text(35.5, 2.0, "TR: tabak ile ramekinin arasındaki siyah kaseyi al ve tabağın üstüne koy", fontsize=8.5,
             va="center")
    ax2.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "03-sequence-tokens.png"), dpi=160)
    plt.close(fig)


def fig04(out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    names = ["LLaMA-2 7B", "SigLIP So400m/14", "DINOv2 ViT-L/14", "Projector MLP"]
    vals = [6.74, 0.43, 0.30, 0.07]
    ax1.barh(names[::-1], vals[::-1], color=Y, edgecolor="k")
    for i, v in enumerate(vals[::-1]):
        ax1.text(v + 0.08, i, f"{v:.2f} B", va="center", fontsize=10)
    ax1.set_xlim(0, 8)
    ax1.set_xlabel("parametre (milyar) — toplam ≈ 7.5 B (HF kartı: \"8B params\"); parça değerleri yaklaşık")
    ax1.set_title("OpenVLA-7B: parametreler nerede? (%90'ı dil modelinde)", fontsize=11)
    ax1.grid(alpha=0.3, axis="x")

    cats = ["bf16 ağırlıklar (7.5 B × 2 bayt)", "KV cache + aktivasyon (298 token, b=1)", "macOS + Python + diğer"]
    vals2 = [15.1, 0.5, 4.0]
    cols = [Y, G, B]
    bottom = 0
    for c_, v, col in zip(cats, vals2, cols):
        ax2.bar(0, v, bottom=bottom, color=col, edgecolor="k", width=0.5)
        ax2.annotate(f"{c_}: ≈ {v} GB", xy=(0.25, bottom + v / 2), xytext=(0.45, bottom + v / 2),
                     fontsize=8.5, va="center", arrowprops=dict(arrowstyle="-", color="grey"))
        bottom += v
    ax2.axhline(32, color="k", lw=1.2)
    ax2.text(-0.2, 32.6, "M2 32 GB birleşik bellek", fontsize=9)
    ax2.axhline(21.8, color="red", ls="--", lw=1.2)
    ax2.text(-0.2, 23.0, "Metal recommendedMaxWorkingSetSize ≈ 21.8 GB (tahmin);\n"
             "PyTorch MPS tavanı = 1.7 × bu (PYTORCH_MPS_HIGH_WATERMARK_RATIO)", fontsize=8, color="red")
    ax2.set_ylim(0, 36)
    ax2.set_xlim(-0.3, 2.6)
    ax2.set_xticks([])
    ax2.set_ylabel("GB")
    ax2.set_title("Mac'te bellek bütçesi (tahmin; Activity Monitor ile ölç)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "04-memory-budget.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig02(a.out); fig03(a.out); fig04(a.out)
    print("yazıldı:", a.out)
```
