"""Ders 13 — VLANeXt eval'inin Qwen3-VL'e gönderdiği metin şablonunu, EN ve TR komutla göster.

GPU gerekmez, model inmez; yalnızca tokenizer (~10 MB) iner. Mac'te:
    conda activate vla
    pip install "transformers>=4.57"
    python show_prompt.py
Çıktı: apply_chat_template'in ürettiği tam string + metin token sayıları + Türkçe komutun parçalanışı.
(Eval kodu: VLANeXt_utils.get_vla_action, is_qwen dalı — messages = [image, image, text].)
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
    # Eval kodundaki ile aynı yapı: iki görüntü (agentview + wrist) + komut, add_generation_prompt=True.
    # Tokenizer tek başına her görüntü için tek bir <|image_pad|> yazar; processor bunu görüntü başına
    # (256/32)^2 = 64 pad token'a açar (patch 16 x merge 2 = 32 px/token).
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": en0}]}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("=== apply_chat_template çıktısı (EN, 2 görüntü) ===")
    print(repr(text))
    print()
    print(f"{'EN token':>9} {'TR token':>9}  komut")
    for en, tr in PAIRS:
        n_en = len(tok.encode(en, add_special_tokens=False))
        n_tr = len(tok.encode(tr, add_special_tokens=False))
        print(f"{n_en:>9} {n_tr:>9}  {en}")
        print(f"{'':>9} {'':>9}  {tr}")
    print()
    tr0 = PAIRS[0][1]
    ids = tok.encode(tr0, add_special_tokens=False)
    print("=== Türkçe komutun parçaları (Qwen tokenizer) ===")
    print(" | ".join(tok.decode([i]) for i in ids))


if __name__ == "__main__":
    main()
