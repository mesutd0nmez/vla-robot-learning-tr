"""Lecture 22 — edge cases: questions put to each of the two tokenizers (the pitfalls in lecture §4 come from here).

  python probe_edge_cases.py --llama tokenizer.model --qwen Qwen/Qwen3-VL-2B-Instruct

1. What does OpenVLA's `.lower()` do to the Turkish 'İ'? ('İçine koy' → 'i̇çine koy': i + U+0307 → a byte fragment in LLaMA)
2. Turkish letters on their own and at the start of a word (is there a ▁ç?) — the reason for LLaMA's standalone ▁ tokens
3. The sentence-initial / sentence-medial difference in Qwen3 (whether there is a space prefix)
4. OpenVLA action range (LLaMA-2 ids 31744–31999): overlap in Turkish, Korean, Japanese, Chinese, Russian, Arabic sentences
5. Tatar et al.'s 'Kal|dır' example (Lecture 5/15): what does the real LLaMA-2 tokenizer give?
6. False friends: Turkish 'it' (to push) = English 'it' (id 372); the 'sold' inside 'soldaki'
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenize_set import HFTokenizer, LLAMA_ACTION_IDS, load_llama  # noqa: E402


def show(t, s, note=""):
    ids, pieces, _ = t.encode(s)
    print(f"  [{t.name:6s}] {len(ids):2d}  {' | '.join(pieces)}   {note}")
    return ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama", default="openvla/openvla-7b")
    ap.add_argument("--qwen", default="Qwen/Qwen3-VL-2B-Instruct")
    a = ap.parse_args()
    L, Q = load_llama(a.llama), HFTokenizer(a.qwen, "qwen3")

    print("1) lower() and İ")
    for s in ("İçine koy", "İçine koy".lower(), "içine koy"):
        print(f"  {s!r}")
        show(L, s); show(Q, s)

    print("\n2) Turkish letters: standalone id / is there a word-initial (▁+letter) piece?")
    for ch in "çğıöşüâ":
        has = L.sp.piece_to_id("▁" + ch) != L.sp.unk_id()
        print(f"  {ch}: LLaMA '{ch}'={L.sp.piece_to_id(ch)}  '▁{ch}' {'YES' if has else 'NO -> standalone ▁ + ' + ch}   Qwen ' {ch}' → {Q.encode(' ' + ch)[1]}")

    print("\n3) Qwen3 sentence-initial / medial")
    for w in ("arasındaki", "soldaki", "öndeki", "kaseyi", "tabağın", "kaldır"):
        print(f"  {w:12s} initial {Q.encode(w)[1]}   medial {Q.encode(' ' + w)[1]}")

    print("\n4) Overlap with the action range (31744–31999)")
    tests = {"tr": "kaseyi tabağın üstüne koy", "ko": "검은 그릇을 접시 위에 놓으세요", "ja": "黒いボウルを皿の上に置いてください",
             "zh": "把黑色的碗放在盘子上", "ru": "положи чёрную миску на тарелку", "ar": "ضع الوعاء الأسود على الطبق",
             "en": "put the bowl on the plate"}
    for k, s in tests.items():
        ids, pieces, _ = L.encode(s)
        hit = [(i, p) for i, p in zip(ids, pieces) if i in LLAMA_ACTION_IDS]
        print(f"  {k}: LLaMA {len(ids):2d} tokens, {len(hit)} in the action range {hit[:4]}   Qwen {len(Q.encode(s)[0])} token")
    print("  first 10 pieces of the range:", " ".join(L.id_to_piece(i) for i in list(LLAMA_ACTION_IDS)[:10]))

    print("\n5) Tatar vd. 'Kal|dır'")
    for s in ("Kırmızı küpü kaldır", "Kırmızı küpü Kaldır", "kaldır", "Kaldır"):
        print(f"  {s!r}"); show(L, s); show(Q, s)

    print("\n6) False friends")
    for s in ("tabağı ocağın önüne it", "place it on the plate", "soldaki", "sold", "sol", "al", "et al."):
        print(f"  {s!r}")
        ids = show(L, s)
        print(f"           LLaMA ids {ids}")
        show(Q, s)
