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
