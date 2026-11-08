"""Ders 15 — Tatar vd. mini-reprodüksiyonu: 5 LIBERO-Spatial karesi × 6 komut koşulu → OpenVLA-7B (MPS) → 7-DoF action.

Ders 14'ün `openvla_mps_single_action.py` zinciri, model BİR kez yüklenip 30 çağrı için döngüye alınmış hâli.
Girdi : frames/frame_task{0..4}.png  (render_frames.py; 180° döndürülmüş 256x256 agentview)
        datasets/01-libero-spatial-tr-mini/data/commands.csv  (make_commands.py)
Çıktı : results/actions.csv  — satır başına: id, task_id, condition, lang, instruction, prompt_tokens,
        tok0..tok6 (token id), bin0..bin6, n0..n6 ([-1,1]), a0..a6 (LIBERO birimi), gripper_libero, seconds
        results/json/<id>.json — Ders 14 formatında tam kayıt
Kullanım (openvla env, Ders 14):
    python run_tatar_mini.py --frames frames --commands ../../datasets/01-libero-spatial-tr-mini/data/commands.csv \
        --out results/actions.csv
    python run_tatar_mini.py ... --only sp0_en,sp0_tr      # yalnızca seçili satırlar (hızlı deneme)
    python run_tatar_mini.py ... --repeat 2                # her satırı 2 kez: greedy ile aynı çıkmalı (determinizm kontrolü)
Greedy decode (do_sample=False): Tatar vd.'nin sıcaklık 0.3 örneklemesi yerine deterministik; EN–EN gürültüsü 0 olmalı,
dil-dışı gürültü kontrolünü `para` koşulu üstlenir.
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

CKPT = "openvla/openvla-7b-finetuned-libero-spatial"
UNNORM_KEY = "libero_spatial"
DIMS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def center_crop_like_training(img, crop_scale=0.9):
    w, h = img.size
    s = math.sqrt(crop_scale)
    cw, ch = round(w * s), round(h * s)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)


def load_model(dtype, attn):
    device = pick_device()
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(CKPT, attn_implementation=attn, torch_dtype=dtype,
                                                low_cpu_mem_usage=True, device_map={"": device}, trust_remote_code=True)
    vla.eval()
    print(f"[*] model yüklendi ({time.time()-t0:.0f} s), device={device}")
    return processor, vla, device


def prepare_image(path):
    img = Image.open(path).convert("RGB")
    if img.size != (224, 224):
        img = img.resize((224, 224), Image.LANCZOS)
    return center_crop_like_training(img)


def one_call(processor, vla, device, dtype, img, instruction):
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inputs = processor(prompt, img).to(device, dtype=dtype)
    ids = inputs["input_ids"][0].tolist()
    t1 = time.time()
    with torch.inference_mode():
        action = vla.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
    dt = time.time() - t1
    input_ids = inputs["input_ids"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat((input_ids, torch.tensor([[29871]], device=device)), dim=1)
    with torch.inference_mode():
        gen = vla.generate(input_ids=input_ids, pixel_values=inputs["pixel_values"],
                           attention_mask=torch.ones_like(input_ids), max_new_tokens=7, do_sample=False)
    tok = gen[0, -7:].cpu().numpy()
    disc = np.clip(vla.vocab_size - tok - 1, 0, vla.bin_centers.shape[0] - 1)
    norm = vla.bin_centers[disc]
    g = -np.sign(2 * action[6] - 1)
    return dict(prompt=prompt, prompt_token_ids=ids, action_token_ids=tok.tolist(), bins=disc.tolist(),
                normalized=norm.tolist(), action=[float(x) for x in action], gripper_libero=float(g), seconds=dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="frames")
    ap.add_argument("--commands", required=True)
    ap.add_argument("--out", default="results/actions.csv")
    ap.add_argument("--only", default="", help="virgülle ayrılmış id listesi")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.commands, encoding="utf-8")))
    if args.only:
        keep = set(args.only.split(","))
        rows = [r for r in rows if r["id"] in keep]
    dtype = getattr(torch, args.dtype)
    processor, vla, device = load_model(dtype, args.attn)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    jdir = os.path.join(os.path.dirname(args.out) or ".", "json")
    os.makedirs(jdir, exist_ok=True)
    cols = (["id", "task_id", "condition", "lang", "instruction", "repeat", "prompt_tokens"]
            + [f"tok{i}" for i in range(7)] + [f"bin{i}" for i in range(7)]
            + [f"n{i}" for i in range(7)] + [f"a{i}" for i in range(7)] + ["gripper_libero", "seconds"])
    images = {}
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            t = int(r["task_id"])
            if t not in images:
                images[t] = prepare_image(os.path.join(args.frames, f"frame_task{t}.png"))
            for rep in range(args.repeat):
                res = one_call(processor, vla, device, dtype, images[t], r["instruction"])
                rec = dict(id=r["id"], task_id=t, condition=r["condition"], lang=r["lang"], instruction=r["instruction"],
                           repeat=rep, prompt_tokens=len(res["prompt_token_ids"]), gripper_libero=res["gripper_libero"],
                           seconds=round(res["seconds"], 3))
                for i in range(7):
                    rec[f"tok{i}"] = res["action_token_ids"][i]
                    rec[f"bin{i}"] = res["bins"][i]
                    rec[f"n{i}"] = round(res["normalized"][i], 5)
                    rec[f"a{i}"] = round(res["action"][i], 5)
                w.writerow(rec)
                f.flush()
                json.dump(dict(checkpoint=CKPT, **{k: r[k] for k in r}, **res, device=str(device), dtype=args.dtype),
                          open(os.path.join(jdir, f"{r['id']}_r{rep}.json"), "w"), indent=1, ensure_ascii=False)
                print(f"{r['id']:16s} rep{rep} tokens={res['action_token_ids']} g={res['gripper_libero']:+.0f} "
                      f"{res['seconds']:.2f}s  [{rec['prompt_tokens']} prompt token]")
    print(f"[*] yazıldı: {args.out}  → şimdi: python analyze_day15.py --actions {args.out}")


if __name__ == "__main__":
    main()
