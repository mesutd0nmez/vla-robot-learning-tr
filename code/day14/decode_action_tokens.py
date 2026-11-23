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
