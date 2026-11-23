# Ders 28 — yüzey ablation kontrolü (20 çift)

- tr_a_seg tirelerden arındırılınca tr_a: 20/20
- harf adımı (tr_a → tr_ascii) yalnız harf değiştiriyor: 20/20
- ek adımı (tr_a → tr_bare) yalnız ek soyuyor (kök, gerekirse ünsüz sertleşmesi geri alınarak): 20/20
- sözcük sayısı tr_a = tr_bare: 20/20
- ek sayısı tr_a: toplam 111, çift başına 5.5 (tr_bare'de 0)
- Türkçe harf: tr_a 97, tr_bare 31, tr_ascii 0, tr_bare_ascii 0
- datasets/05 tokens.csv çapraz kontrol (4 yüzey × 20 × 2 tokenizer): uyuşmayan 0

## 2×2 token tablosu (çift başına ortalama token; parantez min–max)

| tokenizer | en | tr_a (TR, ek) | tr_ascii (ASCII, ek) | tr_bare (TR, eksiz) | tr_bare_ascii (ASCII, eksiz) | harf etkisi | ek etkisi | etkileşim |
|---|---|---|---|---|---|---|---|---|
| llama2 | 13.6 (7–21) | 22.4 (12–36) | 17.6 (7–30) | 14.9 (7–25) | 14.3 (6–24) | +2.7 | +5.4 | +4.3 |
| qwen3 | 12.3 (6–19) | 16.9 (8–28) | 16.0 (8–26) | 12.2 (6–20) | 12.2 (6–20) | +0.4 | +4.2 | +0.9 |

harf etkisi = TR harfli − ASCII (iki ek düzeyinin ortalaması); ek etkisi = ekli − eksiz; etkileşim = harf etkisinin ekli ve eksizdeki farkı.
