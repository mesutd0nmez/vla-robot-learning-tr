# day14

Lecture 14 — Loading OpenVLA-7B on the Mac with MPS and getting a 7-DoF action from a single image + English command.

| File | What |
|---|---|
| `openvla_mps_single_action.py` | Loads OpenVLA-7B (LIBERO-Spatial fine-tune) on the Mac with MPS/bf16; single image + command → 7 tokens, bins, [-1,1], real action, time; writes `action_tokens.json` (separate `openvla` env) |
| `render_libero_frame.py` | Produces a 256×256 agentview frame rotated by 180° for a LIBERO Spatial task/initial state (`vla` env, Lecture 10) |
| `preprocess_demo.py` | Plots the image preprocessing chain (rotate → 224 → center crop) on a real robosuite/LIBERO frame as 4 panels (no model needed) |
| `decode_action_tokens.py` | Decodes the 7 action tokens by hand: token → bin → [-1,1] → q01/q99 (real `libero_spatial` statistics embedded; no model/download needed) |
| `plot_day14.py` | The lecture's figures 02 (token→action), 03 (sequence layout + EN/TR token counts), 04 (M2 memory budget) |

Usage:
```bash
# 1) WITHOUT a model: decode chain and preprocessing (the vla env is enough)
conda activate vla && cd ~/vla-work && mkdir -p day14 && cd day14
for f in decode_action_tokens.py preprocess_demo.py render_libero_frame.py openvla_mps_single_action.py; do
  curl -O https://raw.githubusercontent.com/mesutd0nmez/vla-robot-learning-tr/main/code/day14/$f; done
python decode_action_tokens.py
python preprocess_demo.py --libero --out preprocess.png     # if LIBERO is missing, drop --libero
python render_libero_frame.py --suite libero_spatial --task 0 --out frame_task0.png

# 2) model: SEPARATE env (transformers 4.40.1, conflicts with Lecture 13's Qwen3-VL version)
conda create -n openvla python=3.10 -y && conda activate openvla
pip install torch torchvision "transformers==4.40.1" "tokenizers==0.19.1" "timm==0.9.10" accelerate sentencepiece pillow "numpy<2"
python openvla_mps_single_action.py --image frame_task0.png \
  --instruction "pick up the black bowl between the plate and the ramekin and place it on the plate"
python decode_action_tokens.py --from-json action_tokens.json   # verify the same numbers by hand
```

## openvla_mps_single_action.py

```python
"""Lecture 14 — OpenVLA-7B (LIBERO-Spatial fine-tune) on the Mac with MPS: a single image + command → 7-DoF action.

Setup (separate conda env; conflicts with Lecture 13's transformers>=4.57):
    conda create -n openvla python=3.10 -y && conda activate openvla
    pip install torch torchvision "transformers==4.40.1" "tokenizers==0.19.1" "timm==0.9.10" \
                accelerate sentencepiece pillow "numpy<2"
Usage:
    python openvla_mps_single_action.py --image frame_task0.png \
        --instruction "pick up the black bowl between the plate and the ramekin and place it on the plate"
    python openvla_mps_single_action.py --image frame_task0.png --instruction "..." --no-center-crop
The first run downloads a ~15 GB checkpoint (~/.cache/huggingface). Loading the model takes minutes; later
calls are fast. Output: 7 token ids, bin indices, [-1,1] values, real action, time; action_tokens.json.
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
UNNORM_KEY = "libero_spatial"      # the only key in dataset_statistics.json
DIMS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def center_crop_like_training(img: Image.Image, crop_scale=0.9) -> Image.Image:
    """TF-free equivalent of openvla_utils.crop_and_resize: 90% of the area (side sqrt(0.9)) is cropped from the center,
    then scaled back up to the original size. Since the LIBERO fine-tunes were trained with random-crop augmentation,
    this is MANDATORY at eval time (run_libero_eval.py:94 assert)."""
    w, h = img.size
    s = math.sqrt(crop_scale)
    cw, ch = round(w * s), round(h * s)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="224x224 (or 256x256) RGB PNG; LIBERO agentview rotated by 180°")
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
        attn_implementation=args.attn,      # flash_attention_2 is not available on the Mac
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": device},            # put the shards directly on MPS; loading to CPU first and then calling .to()
        trust_remote_code=True,             # risks two copies in unified memory (~30 GB) (hypothesis; measure)
    )
    vla.eval()
    n_params = sum(p.numel() for p in vla.parameters())
    print(f"[*] loaded: {n_params/1e9:.2f} B parameters, {time.time()-t0:.0f} s")
    print(f"[*] norm_stats keys: {list(vla.norm_stats.keys())}")

    # --- image ---
    img = Image.open(args.image).convert("RGB")
    if img.size != (224, 224):
        img = img.resize((224, 224), Image.LANCZOS)   # the eval code uses lanczos3 (libero_utils.py:44)
    if not args.no_center_crop:
        img = center_crop_like_training(img)
    img.save("model_input_224.png")

    # --- prompt (openvla_utils.py:163) ---
    prompt = f"In: What action should the robot take to {args.instruction.lower()}?\nOut:"
    inputs = processor(prompt, img).to(device, dtype=dtype)   # BatchFeature.to casts only float tensors
    ids = inputs["input_ids"][0].tolist()
    print(f"[*] prompt token count (including BOS): {len(ids)}; last token: {ids[-1]} "
          f"(predict_action appends 29871 '▁' if it is not there)")
    print(f"[*] pixel_values: {tuple(inputs['pixel_values'].shape)}  (6 channels = DINOv2 3 + SigLIP 3)")

    # --- action ---
    t1 = time.time()
    with torch.inference_mode():
        action = vla.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
    dt = time.time() - t1

    # predict_action does not return the token ids; we call the same generate once more to get the ids
    # (deterministic: do_sample=False). For inspection only; not done in the eval loop.
    input_ids = inputs["input_ids"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat((input_ids, torch.tensor([[29871]], device=device)), dim=1)
    with torch.inference_mode():
        gen = vla.generate(input_ids=input_ids, pixel_values=inputs["pixel_values"],
                           attention_mask=torch.ones_like(input_ids), max_new_tokens=7, do_sample=False)
    tok = gen[0, -7:].cpu().numpy()
    disc = np.clip(vla.vocab_size - tok - 1, 0, vla.bin_centers.shape[0] - 1)
    norm = vla.bin_centers[disc]

    print(f"\n[*] predict_action time: {dt:.2f} s\n")
    print(f"{'dim':8s} {'token':>6s} {'bin':>4s} {'[-1,1]':>8s} {'action':>8s}")
    for i, d in enumerate(DIMS):
        print(f"{d:8s} {int(tok[i]):6d} {int(disc[i]):4d} {norm[i]:8.4f} {action[i]:8.4f}")
    g = -np.sign(2 * action[6] - 1)   # run_libero_eval.py:219-225
    print(f"\nGripper that goes to LIBERO: {g:+.0f} ({'close' if g > 0 else 'open'})")

    json.dump({"checkpoint": CKPT, "instruction": args.instruction, "prompt": prompt,
               "prompt_token_ids": ids, "action_token_ids": tok.tolist(), "bins": disc.tolist(),
               "normalized": norm.tolist(), "action": action.tolist(), "seconds": dt,
               "device": str(device), "dtype": args.dtype}, open(args.out, "w"), indent=1)
    print(f"[*] written: {args.out}, model_input_224.png")


if __name__ == "__main__":
    main()
```

## render_libero_frame.py

```python
"""Lecture 14 — Produces a single agentview frame from LIBERO (in the form OpenVLA will see it: rotated by 180°).

Runs in Lecture 10's `vla` environment (mujoco 2.3.7, robosuite 1.4.1, LIBERO). No GPU needed.
    conda activate vla
    python render_libero_frame.py --suite libero_spatial --task 0 --init 0 --out frame_task0.png
Output: a 256x256 RGB PNG (`frame_task0.png`), next to it the raw/flipped comparison `frame_task0_raw_vs_flipped.png`,
and the command text (task.language) on screen.
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
    ap.add_argument("--init", type=int, default=0, help="which of the 50 stored initial states")
    ap.add_argument("--settle", type=int, default=10, help="wait 10 no-op steps like the OpenVLA eval")
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
    flipped = raw[::-1, ::-1]                        # libero_utils.get_libero_image: rotate by 180°
    Image.fromarray(flipped).save(args.out)
    side = np.concatenate([raw, flipped], axis=1)
    Image.fromarray(side).save(args.out.replace(".png", "_raw_vs_flipped.png"))
    print("command (task.language):", task.language)
    print("written:", args.out, "| raw (left) vs flipped (right):", args.out.replace(".png", "_raw_vs_flipped.png"))
    env.close()


if __name__ == "__main__":
    main()
```

## preprocess_demo.py

```python
"""Lecture 14 — Shows OpenVLA's image preprocessing chain on a real robosuite frame (NO model needed).

raw agentview (256x256, robosuite returns it upside down) → img[::-1, ::-1] (180°) → 224x224 (lanczos) → center crop 90% → 224x224
Source: experiments/robot/libero/libero_utils.py:50-58 and experiments/robot/openvla_utils.py:132-156.
If LIBERO is installed (Lecture 10), `--libero` runs the same chain on LIBERO Spatial task 0; otherwise robosuite Lift.
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
    return obs["agentview_image"], f"LIBERO Spatial task {task_id}: {t.language}"


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
                           ["1) raw obs['agentview_image'] 256x256\n(robosuite returns it upside down)",
                            "2) img[::-1, ::-1] — rotate by 180°\n(libero_utils.py:56)",
                            "3) lanczos → 224x224\n(libero_utils.py:44)",
                            "4) center crop 90% area → 224x224\n(openvla_utils.py:147)  → goes to the model"]):
        ax.imshow(im)
        ax.set_title(cap, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"OpenVLA image preprocessing — {title}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(a.out, dpi=150)
    Image.fromarray(np.array(cropped)).save(a.out.replace(".png", "_model_input.png"))
    print("written:", a.out)


if __name__ == "__main__":
    main()
```

## decode_action_tokens.py

```python
"""Lecture 14 — Decodes OpenVLA's 7 action tokens by hand (NO model/GPU/download needed).

Applies the three steps in OpenVLA's `predict_action` function one to one:
    token_id → bin index → [-1, 1] bin center → real action via q01/q99
Source: prismatic/extern/hf/modeling_prismatic.py:518-534 (openvla/openvla, commit c8f03f4).
Statistics: openvla/openvla-7b-finetuned-libero-spatial / dataset_statistics.json ("libero_spatial").

Usage:
    python decode_action_tokens.py                      # example 7 tokens
    python decode_action_tokens.py 31890 31812 31999 31870 31875 31872 31744
    python decode_action_tokens.py --from-json action_tokens.json   # output of openvla_mps_single_action.py
"""
import json
import sys

import numpy as np

# LLaMA-2 tokenizer: 32,000 tokens. OpenVLA uses the last 256 tokens (31744..31999) for actions.
# In the HF config text_config.vocab_size = 32064 (rounded up to a multiple of 64); the code undoes this: 32064 - 64 = 32000.
VOCAB_SIZE = 32000
N_BINS = 256
BINS = np.linspace(-1, 1, N_BINS)              # 256 edges
BIN_CENTERS = (BINS[:-1] + BINS[1:]) / 2.0     # 255 centers (!)

# dataset_statistics.json → "libero_spatial" → "action" (real values)
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
    assert token_ids.shape == (7,), "expected 7 tokens"
    # 1) token → bin index (modeling_prismatic.py:522-523)
    discretized = VOCAB_SIZE - token_ids                       # 1..256
    discretized = np.clip(discretized - 1, 0, BIN_CENTERS.shape[0] - 1)   # 0..254
    # 2) bin index → [-1, 1]
    normalized = BIN_CENTERS[discretized]
    # 3) [-1, 1] → real action (via q01/q99), do not touch the gripper (mask=False)
    lo, hi, mask = (np.array(stats[k]) for k in ("q01", "q99", "mask"))
    action = np.where(mask, 0.5 * (normalized + 1) * (hi - lo) + lo, normalized)
    return discretized, normalized, action


def libero_gripper(action):
    """run_libero_eval.py:219-225 — gripper [0,1] → [-1,1], binarize by sign, then invert for OpenVLA."""
    g = 2 * action[6] - 1
    g = np.sign(g)
    return -g   # invert_gripper_action: in the OpenVLA data 1 = open, in LIBERO +1 = close


def main(argv):
    if len(argv) >= 2 and argv[0] == "--from-json":
        toks = json.load(open(argv[1]))["action_token_ids"]
    elif len(argv) == 7:
        toks = [int(x) for x in argv]
    else:
        # example: slightly back, to the right, fast down; small rotation; gripper "1 = open" (in the OpenVLA data the gripper is 0/1:
        # 0 → bin 127 → token 31872, 1 → bin 254 → token 31744; the model saw no other gripper value in training)
        toks = [31890, 31812, 31999, 31870, 31875, 31872, 31744]
    disc, norm, act = decode(toks)
    print(f"{'dim':8s} {'token':>6s} {'bin':>4s} {'[-1,1]':>8s} {'q01':>8s} {'q99':>8s} {'action':>8s}")
    for i, d in enumerate(DIMS):
        print(f"{d:8s} {toks[i]:6d} {disc[i]:4d} {norm[i]:8.4f} {LIBERO_SPATIAL['q01'][i]:8.4f} "
              f"{LIBERO_SPATIAL['q99'][i]:8.4f} {act[i]:8.4f}")
    print(f"\nGripper that goes to LIBERO (normalize + binarize + invert): {libero_gripper(act):+.0f}  "
          f"({'close' if libero_gripper(act) > 0 else 'open'})")
    step = (np.array(LIBERO_SPATIAL['q99']) - np.array(LIBERO_SPATIAL['q01'])) / (N_BINS - 1)
    print("Real-valued size of one bin (q99-q01)/255:", np.round(step[:6], 4))


if __name__ == "__main__":
    main(sys.argv[1:])
```

## plot_day14.py

```python
"""Lecture 14 figures: 02 token→action mapping (real libero_spatial q01/q99), 03 sequence layout + EN/TR token
counts (LLaMA-2 tokenizer, counted with llama-tokenizer-js), 04 M2 memory budget (parameter counts approximate).
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
    ax1.text(129, -0.85, "bin 127 → 0.0 in [-1,1]\n(token 31872)", fontsize=9, color="grey")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_xlabel("bin index (= 32000 − token id − 1)")
    ax1.set_ylabel("action (LIBERO unit, normalized delta)")
    ax1.set_title("Token → bin → action: linear but NOT symmetric", fontsize=11)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(alpha=0.3)

    x = np.arange(6)
    ax2.bar(x, Q99, color=G, edgecolor="k", label="q99 (value of bin 254)")
    ax2.bar(x, Q01, color=B, edgecolor="k", label="q01 (value of bin 0)")
    zero_act = 0.5 * (0 + 1) * (Q99 - Q01) + Q01
    ax2.scatter(x, zero_act, color="red", zorder=5, s=45, label="\"neutral\" token 31872 produces this action")
    for i in range(6):
        ax2.text(i, zero_act[i] + 0.04, f"{zero_act[i]:+.3f}", ha="center", fontsize=8, color="red")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xticks(x, DIMS)
    ax2.set_title("libero_spatial q01/q99 (dataset_statistics.json, real)", fontsize=11)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(alpha=0.3, axis="y")
    fig.suptitle("OpenVLA action decode — openvla/openvla-7b-finetuned-libero-spatial", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "02-token-to-action.png"), dpi=160)
    plt.close(fig)


def fig03(out):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.2), gridspec_kw={"height_ratios": [1, 1.6]})
    # top: sequence layout (real numbers: BOS 1, patch 256, EN prompt 33, '▁' 1, action 7)
    segs = [("BOS", 1, B), ("256 image patch tokens (DINOv2+SigLIP → projector)", 256, Y),
            ("prompt 33", 33, B), ("▁", 1, R), ("action", 7, G)]
    pos = 0
    for name, n, c in segs:
        w = max(n, 12)
        ax1.barh(0, w, left=pos, color=c, edgecolor="k", height=0.6)
        ax1.text(pos + w / 2, 0, name if n > 12 else f"{name}\n{n}", ha="center", va="center", fontsize=9)
        pos += w
    ax1.set_xlim(0, pos)
    ax1.set_yticks([])
    ax1.set_xlabel("position (LLaMA-2 input sequence: 1 + 256 + 33 + 1 = 291 tokens, then 7 tokens are generated)")
    ax1.set_title("OpenVLA sequence for Spatial task 0 — 'pick up the black bowl between the plate and the ramekin "
                  "and place it on the plate'", fontsize=10)
    # bottom: EN vs TR token counts (llama-tokenizer-js, LLaMA-2 BPE; excluding BOS)
    labels = ["EN command", "TR command", "EN full prompt", "TR full prompt"]
    counts = [20, 32, 33, 45]
    colors = [B, R, B, R]
    ax2.barh(labels[::-1], counts[::-1], color=colors[::-1], edgecolor="k")
    for i, v in enumerate(counts[::-1]):
        ax2.text(v + 0.5, i, str(v), va="center", fontsize=10)
    ax2.set_xlim(0, 60)
    ax2.set_xlabel("token count (LLaMA-2 tokenizer, excluding BOS)")
    ax2.set_title("The same command in Turkish: 20 → 32 tokens (×1.6). "
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
    ax1.set_xlabel("parameters (billions) — total ≈ 7.5 B (HF card: \"8B params\"); component values approximate")
    ax1.set_title("OpenVLA-7B: where are the parameters? (90% in the language model)", fontsize=11)
    ax1.grid(alpha=0.3, axis="x")

    cats = ["bf16 weights (7.5 B × 2 bytes)", "KV cache + activations (298 tokens, b=1)", "macOS + Python + other"]
    vals2 = [15.1, 0.5, 4.0]
    cols = [Y, G, B]
    bottom = 0
    for c_, v, col in zip(cats, vals2, cols):
        ax2.bar(0, v, bottom=bottom, color=col, edgecolor="k", width=0.5)
        ax2.annotate(f"{c_}: ≈ {v} GB", xy=(0.25, bottom + v / 2), xytext=(0.45, bottom + v / 2),
                     fontsize=8.5, va="center", arrowprops=dict(arrowstyle="-", color="grey"))
        bottom += v
    ax2.axhline(32, color="k", lw=1.2)
    ax2.text(-0.2, 32.6, "M2 32 GB unified memory", fontsize=9)
    ax2.axhline(21.8, color="red", ls="--", lw=1.2)
    ax2.text(-0.2, 23.0, "Metal recommendedMaxWorkingSetSize ≈ 21.8 GB (estimate);\n"
             "PyTorch MPS ceiling = 1.7 × this (PYTORCH_MPS_HIGH_WATERMARK_RATIO)", fontsize=8, color="red")
    ax2.set_ylim(0, 36)
    ax2.set_xlim(-0.3, 2.6)
    ax2.set_xticks([])
    ax2.set_ylabel("GB")
    ax2.set_title("Memory budget on the Mac (estimate; measure with Activity Monitor)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "04-memory-budget.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig02(a.out); fig03(a.out); fig04(a.out)
    print("written:", a.out)
```
