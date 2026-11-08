# Veri setleri

İki tür kayıt tutulur:

- **Kendi ürettiğim veri** (Türkçe komut setleri, EN–TR eşleri, deney sonuç tabloları): dosyanın kendisi repoda (`data/` altında CSV/JSON/JSONL, UTF-8, başlık satırlı). Tezin ekleri ve ileride Hugging Face veri seti bunlardan üretilir.
- **Dış veri** (LIBERO HDF5, checkpoint'ler, RLDS): dosya repoya girmez; yalnızca **veri kartı** (`dataset-card.md`) tutulur — kaynak, sürüm, lisans, indirme komutu, şema, sha256, uygulanan dönüşümler.

Klasör adı: `NN-kisa-ad`. Her klasörde `dataset-card.md` zorunlu; `data/` yalnızca küçük dosyalar için (< 10 MB).

| # | Veri seti | Tür | Durum | Özet |
|---|---|---|---|---|
| 01 | [LIBERO-Spatial TR mini](01-libero-spatial-tr-mini/dataset-card.md) | kendi ürettiğim | v0.1 (çeviri doğrulaması bekliyor) | 5 Spatial görevi × 6 komut koşulu (en/para/tr/tr2/distractor/empty), 30 satır + LLaMA-2 token sayıları — Ders 15 |
