# Veri setleri

İki tür kayıt tutulur:

- **Kendi ürettiğim veri** (Türkçe komut setleri, EN–TR eşleri, deney sonuç tabloları): dosyanın kendisi repoda (`data/` altında CSV/JSON/JSONL, UTF-8, başlık satırlı). Tezin ekleri ve ileride Hugging Face veri seti bunlardan üretilir.
- **Dış veri** (LIBERO HDF5, checkpoint'ler, RLDS): dosya repoya girmez; yalnızca **veri kartı** (`dataset-card.md`) tutulur — kaynak, sürüm, lisans, indirme komutu, şema, sha256, uygulanan dönüşümler.

Klasör adı: `NN-kisa-ad`. Her klasörde `dataset-card.md` zorunlu; `data/` yalnızca küçük dosyalar için (< 10 MB).

| # | Veri seti | Tür | Durum | Özet |
|---|---|---|---|---|
| 01 | [LIBERO-Spatial TR mini](01-libero-spatial-tr-mini/dataset-card.md) | kendi ürettiğim | v0.1 (çeviri doğrulaması bekliyor) | 5 Spatial görevi × 6 komut koşulu (en/para/tr/tr2/distractor/empty), 30 satır + LLaMA-2 token sayıları — Ders 15 |
| 02 | [LIBERO-Plus Spatial 0 yeniden yazımları + TR BDDL](02-libero-plus-spatial0-rewrites/dataset-card.md) | kendi ürettiğim (dış metin üstüne etiket) | v0.1 (tür etiketi kural tabanlı) | Görev 0'ın 50 `_language_K` yeniden yazımı: benchmark üyeliği, indeks, zorluk, R1/R2 türü, korunan kelimeler; 10 Türkçe `_language_trK.bddl` + task map satırları; 7 bozulmanın kod tablosu — Ders 17 |
| 03 | [LIBERO-Spatial 0 — 4 koşullu komut deneyi](03-libero-spatial0-4cond/dataset-card.md) | kendi ürettiğim (tasarım) | v0.1 (sonuç bekliyor) | Görev 0 için clean/empty/wrong/tr koşul JSON'u + 10 init × 4 koşul episode şeması; sonuçlar Colab koşusundan gelecek — Ders 18 |
