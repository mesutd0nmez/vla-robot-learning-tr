"""Lecture 13 — Show the text template that the VLANeXt eval sends to Qwen3-VL, with an EN and a TR command.

No GPU needed, no model download; only the tokenizer (~10 MB) is downloaded. On the Mac:
    conda activate vla
    pip install "transformers>=4.57"
    python show_prompt.py
Output: the full string produced by apply_chat_template + text token counts + how the Turkish command is split into pieces.
(Eval code: VLANeXt_utils.get_vla_action, is_qwen branch — messages = [image, image, text].)
"""
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-VL-2B-Instruct"   # VLANeXt libero_train_config.yaml: model.lmm_path
PAIRS = [
    ("pick up the black bowl between the plate and the ramekin and place it on the plate",
     "tabak ile ramekin arasındaki siyah kaseyi al ve tabağın üstüne koy"),
    ("pick up the black bowl next to the ramekin and place it on the plate",
     "ramekinin yanındaki siyah kaseyi al ve tabağın üstüne koy"),
    ("pick up the black bowl from table center and place it on the plate",
     "masanın ortasındaki siyah kaseyi al ve tabağın üstüne koy"),
]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    en0 = PAIRS[0][0]
    # Same structure as in the eval code: two images (agentview + wrist) + command, add_generation_prompt=True.
    # The tokenizer alone writes a single <|image_pad|> per image; the processor expands it to
    # (256/32)^2 = 64 pad tokens per image (patch 16 x merge 2 = 32 px/token).
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": en0}]}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("=== apply_chat_template output (EN, 2 images) ===")
    print(repr(text))
    print()
    print(f"{'EN token':>9} {'TR token':>9}  command")
    for en, tr in PAIRS:
        n_en = len(tok.encode(en, add_special_tokens=False))
        n_tr = len(tok.encode(tr, add_special_tokens=False))
        print(f"{n_en:>9} {n_tr:>9}  {en}")
        print(f"{'':>9} {'':>9}  {tr}")
    print()
    tr0 = PAIRS[0][1]
    ids = tok.encode(tr0, add_special_tokens=False)
    print("=== Pieces of the Turkish command (Qwen tokenizer) ===")
    print(" | ".join(tok.decode([i]) for i in ids))


if __name__ == "__main__":
    main()
