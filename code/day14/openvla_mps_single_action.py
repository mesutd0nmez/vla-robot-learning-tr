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
        device_map={"": device},            # shard'ları doğrudan MPS'e koy; önce CPU'ya yükleyip .to() yapmak
        trust_remote_code=True,             # birleşik bellekte iki kopya (~30 GB) riski taşır (hipotez; ölç)
    )
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
