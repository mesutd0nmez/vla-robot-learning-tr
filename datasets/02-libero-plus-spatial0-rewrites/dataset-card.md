# LIBERO-Plus Spatial görev 0 — 50 yeniden yazımın türü + 10 Türkçe BDDL varyantı

**Tür:** kendi ürettiğim (dış verinin üstüne etiket) · **Sürüm:** v0.1 · **Tarih:** Ders 17 · **Lisans:** MIT (etiketler, Türkçe BDDL'ler); yeniden yazım metinleri LIBERO-Plus'ın MIT lisansı altında · **İlgili proje/ders:** Ders 17, `code/day17/`, Ders 18 deneyi

## Ne içeriyor
1. `data/rewrites_task0.csv` — LIBERO-Plus'ın Spatial temel görev 0 (*pick up the black bowl between the plate and the ramekin and place it on the plate*) için ürettiği **50** `_language_K.bddl` yeniden yazımı; her satırda benchmark'ta olup olmadığı (46 evet, 4 hayır: 16, 17, 29, 35), LIBERO-Plus görev indeksi (0 tabanlı; `VLANEXT_TASK_IDS` için), zorluk seviyesi, **kural tabanlı tür etiketi** (R1 dikkat dağıtma 26 / R2 sağduyu tam 16 / R2 sağduyu hafif 6 / R0 düz parafraz 2), kelime sayısı, *bowl* / *plate* / *between* kelimelerinin korunup korunmadığı, büyük harf sayısı (`scan_tokens.lower()` bunları değiştirir).
2. `data/tr_variants/` — LIBERO-Plus şablonuyla üretilmiş **10 Türkçe** `_language_trK.bddl` dosyası (Spatial görev 0–4 × {tr1 = `datasets/01` `tr`, tr2 = `tr2`}); temel BDDL'den yalnızca `:language` satırında ayrılırlar. Yanında `task_map_lines.txt` (task map'e eklenecek satırlar) ve `task_classification_entries.json`.
3. `data/perturbation_table_spatial.md` — 7 bozulmanın koddan çıkarılmış özet tablosu (taşıyıcı, gerçek parametre aralığı, kod yeri, BDDL/asset gereksinimi).

## Kaynak ve elde etme
- Metinler: [sylvestf/LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) commit `4976dc3` (21 Oca 2026), `libero/libero/bddl_files/libero_spatial/*_language_K.bddl` `:language` alanı; benchmark üyeliği, indeks ve zorluk `libero/libero/benchmark/task_classification.json`'dan.
- Tür etiketi: `code/day17/classify_rewrites.py` — makalenin Ek A.6 tanımlarına (R1 Distraction, R2 Common Sense, R3 Reasoning Chain) göre **regex kuralları**; JSON'da tür alanı yoktur, dolayısıyla bu etiket yazarların etiketi değil, bizim okumamızdır (hipotez). R3 Spatial'da tek adımlı görevlerde beklenmediği için görülmedi.
- Türkçe varyantlar: `code/day17/make_tr_variant.py --commands code/day17/tr_commands.csv`; komutlar `datasets/01-libero-spatial-tr-mini` v0.1'den (anadil doğrulaması hâlâ bekliyor). Her dosya gerçek `bddl==1.0.1` ayrıştırıcısından geçirildi; modele gidecek metin yazılanla birebir aynı (10/10).
- Perturbation tablosu: `code/day17/perturbation_table.py` (görev adları, BDDL dosyaları, `mounted_panda.py`, `problems/libero_tabletop_manipulation.py` okunarak).

## Şema (`data/rewrites_task0.csv`)
| Alan | Tip | Açıklama | Örnek |
|---|---|---|---|
| k | int | `_language_K` numarası (1–50) | 5 |
| in_benchmark | bool | task map / JSON'da var mı | True |
| task_index | int/boş | LIBERO-Plus Spatial görev indeksi (0 tabanlı) | 988 |
| difficulty | int/boş | L1–L5 (4 referans modelden kaçı çözdü; Ek C.3) | 1 |
| type | str | `R1 dikkat dağıtma` / `R2 sağduyu (tam)` / `R2 sağduyu (hafif)` / `R0 düz parafraz` (kural tabanlı) | `R1 dikkat dağıtma` |
| words | int | kelime sayısı (orijinal 17) | 22 |
| keeps_bowl / keeps_plate / keeps_between | bool | kelime metinde geçiyor mu | True |
| relation_word | str | ilişkiyi taşıyan ifade (`between`, `amid`, `separating`, `midway`) | `between` |
| conversational / descriptive | bool | R1 / R2 kural tetiklendi mi | True / False |
| lowercase_changes | int | büyük harf sayısı (`.lower()` ile değişecek karakter) | 1 |
| text | str | BDDL'deki ham `:language` metni (noktalamasız) | `Would you mind helping me by …` |

## Uygulanan dönüşümler
Yok; metinler BDDL'den olduğu gibi (büyük harfli). LIBERO-Plus eval sırasında `bddl.parsing.scan_tokens` tüm dosyayı `.lower()` yapar; modelin gördüğü hâl `text.lower()`'dır. Türkçe varyantlar baştan küçük harfle yazıldı (Ders 16 kuralı), bu yüzden onlarda dönüşüm etkisizdir.

## Bilinen sınırlılıklar
- Tür etiketi kural tabanlı; sınır durumlar (ör. 11, 18, 21, 31, 46: `black bowl→dark bowl`, `plate→dish`) "R2 hafif" sayıldı — ikinci okuma Mesut'tan.
- Yalnızca görev 0; öteki 9 Spatial görevinin 450 yeniden yazımı etiketlenmedi (script `--base N` ile üretir).
- Türkçe varyantlar 5 görev × 2; init state 0 ile aynı sahne. Benchmark'a eklendiklerinde zorluk seviyeleri yok (4 referans modelle ölçülmedi).
- 20 ve 44 numaralı yeniden yazımlar sahnede olmayan "a plate / another plate" ifadesi kullanıyor (tek tabak var) — LLM yeniden yazımının küçük bir sahne hatası.

## Kullanım
```python
import pandas as pd
df = pd.read_csv("data/rewrites_task0.csv")
df.groupby("type")[["words", "keeps_between"]].mean()
df[df.in_benchmark & (df.difficulty == 1)][["k", "task_index", "type", "text"]]   # Ders 16'daki 988/994/995 burada
```

## Değişiklik günlüğü
- v0.1 — ilk sürüm (Ders 17): 50 etiketli satır, 10 Türkçe BDDL, perturbation tablosu.
