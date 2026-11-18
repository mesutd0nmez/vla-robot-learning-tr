"""Ders 22 — kenar durumları: iki tokenizer'a tek tek sorular (ders §4'teki tuzaklar buradan).

  python probe_edge_cases.py --llama tokenizer.model --qwen Qwen/Qwen3-VL-2B-Instruct

1. OpenVLA'nın `.lower()`'ı Türkçe 'İ'ye ne yapar? ('İçine koy' → 'i̇çine koy': i + U+0307 → LLaMA'da bayt kırıntısı)
2. Türkçe harfler tek başına ve sözcük başında (▁ç var mı?) — LLaMA'nın tek başına ▁ token'larının nedeni
3. Qwen3'te cümle başı / cümle ortası farkı (boşluk önekli olup olmaması)
4. OpenVLA action aralığı (LLaMA-2 id 31744–31999): Türkçe, Korece, Japonca, Çince, Rusça, Arapça cümlelerde çakışma
5. Tatar vd.'nin 'Kal|dır' örneği (Ders 5/15): gerçek LLaMA-2 tokenizer ne veriyor?
6. Yanlış arkadaşlar: Türkçe 'it' (itmek) = İngilizce 'it' (id 372); 'soldaki' içindeki 'sold'
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

    print("1) lower() ve İ")
    for s in ("İçine koy", "İçine koy".lower(), "içine koy"):
        print(f"  {s!r}")
        show(L, s); show(Q, s)

    print("\n2) Türkçe harfler: tek başına id / sözcük başında (▁+harf) var mı?")
    for ch in "çğıöşüâ":
        has = L.sp.piece_to_id("▁" + ch) != L.sp.unk_id()
        print(f"  {ch}: LLaMA '{ch}'={L.sp.piece_to_id(ch)}  '▁{ch}' {'VAR' if has else 'YOK → tek başına ▁ + ' + ch}   Qwen ' {ch}' → {Q.encode(' ' + ch)[1]}")

    print("\n3) Qwen3 cümle başı / ortası")
    for w in ("arasındaki", "soldaki", "öndeki", "kaseyi", "tabağın", "kaldır"):
        print(f"  {w:12s} başta {Q.encode(w)[1]}   ortada {Q.encode(' ' + w)[1]}")

    print("\n4) Action aralığı (31744–31999) çakışması")
    tests = {"tr": "kaseyi tabağın üstüne koy", "ko": "검은 그릇을 접시 위에 놓으세요", "ja": "黒いボウルを皿の上に置いてください",
             "zh": "把黑色的碗放在盘子上", "ru": "положи чёрную миску на тарелку", "ar": "ضع الوعاء الأسود على الطبق",
             "en": "put the bowl on the plate"}
    for k, s in tests.items():
        ids, pieces, _ = L.encode(s)
        hit = [(i, p) for i, p in zip(ids, pieces) if i in LLAMA_ACTION_IDS]
        print(f"  {k}: LLaMA {len(ids):2d} token, action aralığında {len(hit)} {hit[:4]}   Qwen {len(Q.encode(s)[0])} token")
    print("  aralığın ilk 10 parçası:", " ".join(L.id_to_piece(i) for i in list(LLAMA_ACTION_IDS)[:10]))

    print("\n5) Tatar vd. 'Kal|dır'")
    for s in ("Kırmızı küpü kaldır", "Kırmızı küpü Kaldır", "kaldır", "Kaldır"):
        print(f"  {s!r}"); show(L, s); show(Q, s)

    print("\n6) Yanlış arkadaşlar")
    for s in ("tabağı ocağın önüne it", "place it on the plate", "soldaki", "sold", "sol", "al", "et al."):
        print(f"  {s!r}")
        ids = show(L, s)
        print(f"           LLaMA id'leri {ids}")
        show(Q, s)
