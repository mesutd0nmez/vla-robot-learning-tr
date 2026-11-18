# TR komut setinin tokenizasyonu — 100 komut × 2 tokenizer (LLaMA-2 / Qwen3)

**Tür:** kendi ürettiğim (datasets/04 üstüne türetilmiş) · **Sürüm:** v0.1 (Qwen3-VL tokenizer kimliği HF'den doğrulanmayı bekliyor) · **Tarih:** Ders 22 · **Lisans:** MIT (tablolar); tokenizer dosyaları repoya girmez (Meta LLaMA 2 Community License / Apache 2.0) · **İlgili proje/ders:** Ders 22, `code/day22/`; girdi `datasets/04-tr-spatial-commands/` v0.1; Ders 15'in `token_counts.json`'unun devamı

## Ne içeriyor
`datasets/04`'ün 100 komutu (20 EN–TR çift × 5 yüzey) iki VLA'nın tokenizer'ından geçirildi: **LLaMA-2** (OpenVLA; SentencePiece BPE, 32.000 parça, Meta `tokenizer.model`, sha256 doğrulandı) ve **Qwen3** (VLANeXt'in Qwen3-VL-2B omurgası; byte-level BPE, 151.643 + 26 özel). Satır başına token/sözcük/bayt sayıları, parçalar ve id'ler; ilişki sözcüğünün (`rel_span`) parçaları ve biçimbirim–token sınır örtüşmesi; ilişki ekini taşıyan token biçimleri; LLaMA-2'de OpenVLA action aralığına (id 31744–31999) düşen token kontrolü; Ders 15'teki llama-tokenizer-js sayımının SentencePiece ile çapraz kontrolü (100/100 aynı). Model çalıştırılmadı; bütün sayılar tokenizer çıktısıdır.

## Dosyalar
| Dosya | İçerik |
|---|---|
| `data/tokens.csv` | 200 satır (100 komut × 2 tokenizer): `n_tokens`, `n_words`, `fertility`, `bytes_per_token`, `n_space_alone` (tek başına ▁), `n_byte_fragments`, `n_in_action_range`, `prompt_tokens` (LLaMA: OpenVLA prompt'u BOS dahil; Qwen: chat şablonu metni + 1 görüntü yer tutucusu), `pieces`, `ids` |
| `data/rel_span.csv` | 40 satır (20 çift × 2): ilişki sözcüğünün cümle içindeki parçaları, `recall`/`precision` (biçimbirim sınırı ↔ token sınırı), `last_suffix_separate` |
| `data/seg_align.csv` | 40 satır: `tr_a_seg`'deki bütün biçimbirim sınırları için örtüşme sayıları |
| `data/words.csv` | 274 satır: setteki her ayrı sözcük × 2 tokenizer, cümle başı ve cümle ortası parçaları |
| `data/suffix_forms.csv` | `-ki` ve yönelme (`-nA/-A`) eklerini taşıyan token biçimleri ve sayıları |
| `data/action_collision.csv` | action aralığına düşen token listesi (boş) + aralığın örnek içeriği |
| `data/crosscheck.md`, `data/js_ids.json` | llama-tokenizer-js id dizileri ve SentencePiece ile karşılaştırma |
| `data/probe_edge_cases.txt` | `probe_edge_cases.py` çıktısı: `lower()`/İ, Türkçe harf başları, Qwen cümle başı farkı, action aralığı (tr/ko/ja/zh/ru/ar), `Kal|dır`, yanlış arkadaşlar |
| `data/summary.md` | özet tablolar (ders metnindeki sayıların kaynağı) |

## Kaynak ve elde etme
- LLaMA-2 tokenizer: `tokenizer.model` (499.723 bayt), `curl -L -o tokenizer.model https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model`; `sha256 = 9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347` (Meta'nın LLaMA / LLaMA 2 dağıtımındaki `tokenizer_checklist.chk` ile aynı). OpenVLA bu dosyadan türetilmiş `LlamaTokenizerFast`'i kullanır (`openvla/openvla-7b`; eklenen `<PAD>` id 32000 metni etkilemez). `sentencepiece==0.2.3`, `add_dummy_prefix` (her metnin başına ▁).
- Qwen3 tokenizer: `tokenizer.json` (byte-level BPE, 151.387 merge; `normalizer: NFC`; 26 eklenmiş özel token: `<|im_start|>`, `<|vision_start|>`, `<|image_pad|>`, `<think>`…), bu oturumda npm paketi `@lenml/tokenizer-qwen3@3.7.2` içinden alındı (HF erişimi yoktu); `"Hello world" → [9707, 1879]` bilinen Qwen2/2.5/3 id'leriyle doğrulandı. **Hipotez:** `Qwen/Qwen3-VL-2B-Instruct`'ın `tokenizer.json`'u ile bayt-aynı (aynı `Qwen2Tokenizer` sınıfı, aynı vocab) — Mesut Mac'te HF ile `tokens.csv`'deki `ids` sütununu karşılaştıracak (v0.2).
- Uyarı: npm'deki `@lenml/tokenizer-llama2` paketinin `tokenizer.json`'u Meta vocab'ı **değil** (id 29871 ▁ yerine 'ข'; farklı merge sırası) — kullanılmadı.
- Komutlar: `datasets/04-tr-spatial-commands/data/commands.csv` v0.1 (anadil okuması girmemiş hâli).
- Üretim: `code/day22/tokenize_set.py --llama tokenizer.model --qwen tokenizer.json --jsids js_ids.json`; `count_js_ids.mjs` (llama-tokenizer-js 1.2.2); `probe_edge_cases.py`.

## Şema (`data/tokens.csv`)
| Alan | Tip | Açıklama | Örnek |
|---|---|---|---|
| id, pair_id, block, variant, lang | str | datasets/04'teki kimlikler | `B07_tr_a`, `B07`, `B`, `tr_a`, `tr` |
| tokenizer | str | `llama2` / `qwen3` | `llama2` |
| n_tokens | int | yalnızca komut metni (BOS/şablon yok) | 21 |
| n_words, n_chars, n_bytes | int | boşlukla ayrılmış sözcük, karakter, UTF-8 bayt | 8, 45, 51 |
| fertility | float | n_tokens / n_words | 2.625 |
| bytes_per_token | float | n_bytes / n_tokens | 2.43 |
| n_space_alone | int | tek başına ▁ token'ı (LLaMA'da `▁ç`, `▁ş`, `▁ğ`, `▁ı` vocab'da yok) | 0 |
| n_byte_fragments | int | tek başına geçerli UTF-8 olmayan parça | 0 |
| n_in_action_range | int | LLaMA-2 id ≥ 31744 (OpenVLA action bin'i) | 0 |
| prompt_tokens | int | LLaMA: `In: What action should the robot take to {komut.lower()}?\nOut:` BOS dahil; Qwen: chat şablonu metni + 1 görüntü yer tutucusu | 35 |
| pieces | str | parçalar `\|` ile; ▁ = sözcük başı boşluğu | `▁sold\|aki\|▁si\|y\|ah\|…` |
| ids | str | token id'leri boşlukla | `5239 9940 1354 …` |

`rel_span.csv`: `pair_id, rel_span, rel_seg, n_morphemes, tokenizer, n_pieces, pieces, ids, morph_cuts, token_cuts, cuts_hit, recall, precision, last_suffix, last_suffix_separate`. `words.csv`: `lang, word, count, tokenizer, n_pieces_mid, pieces_mid, ids_mid, n_pieces_initial, pieces_initial, occurs_initial`.

## Uygulanan dönüşümler
- Komut metni değiştirilmedi (NFC, küçük harf — datasets/04 zaten öyle). LLaMA prompt sayımında OpenVLA'nın yaptığı gibi `.lower()` uygulandı (setteki satırlarda etkisiz).
- Qwen'de sözcük parçaları konuma göre: cümle başındaki sözcük boşluksuz, diğerleri `' ' + sözcük` olarak tokenize edildi (VLANeXt şablonunda komut `<|vision_end|>`'den hemen sonra gelir — hipotez, Ders 13).
- Biçimbirim sınırları `datasets/04`'ün `tr_a_seg` / `rel_seg` sütunlarından (kural tabanlı `tr_morph.py` ayrımı); token sınırları karakter offset'lerinden.

## Bilinen sınırlılıklar
- Qwen3-VL'nin tokenizer'ı HF'den değil npm paketinden (v0.1); OpenVLA'nın HF tokenizer'ı yerine Meta `tokenizer.model` + sentencepiece (HF `legacy=True` davranışıyla aynı olmalı; özel token bitişikliği gibi kenar durumları test edilmedi).
- `tr_b` ayrımı yok (elle yazıldığı için `seg_align` yalnızca `tr_a`).
- Token sayısı ≠ anlama: bu set yalnızca girdinin biçimini ölçer; etkisinin success rate'e yansıması Ders 18 koşusuna bağlı.

## Kullanım
```python
import pandas as pd
t = pd.read_csv("data/tokens.csv")
t.pivot_table(index="variant", columns="tokenizer", values="n_tokens", aggfunc="mean")
```

## Değişiklik günlüğü
- v0.1 — Ders 22: ilk sürüm (datasets/04 v0.1 üstünde; Qwen3 tokenizer npm'den, LLaMA-2 Meta tokenizer.model sha256 doğrulandı)
