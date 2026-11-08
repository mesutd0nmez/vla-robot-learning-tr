# LIBERO-Spatial TR mini — 5 görev × 6 komut koşulu

**Tür:** kendi ürettiğim · **Sürüm:** v0.1 · **Tarih:** Ders 15 · **Lisans:** MIT (komut metinleri; LIBERO komutları LIBERO'nun MIT lisansı altında) · **İlgili proje/ders:** Ders 15, `code/day15/`, (proje: `projects/03-tatar-mini/` — Mesut açacak)

## Ne içeriyor
30 komut satırı: LIBERO-Spatial'ın ilk 5 görevi (0–4) için 6 koşul — `en` (LIBERO'nun kendi `task.language` metni, referans), `para` (aynı anlamda İngilizce parafraz), `tr` (Türkçe çeviri, LIBERO stili `-daki` sıfat-fiil yapısı, "kase"), `tr2` (ikinci Türkçe yüzey formu: "duran" yapısı, "kâse", "kaldır … yerleştir"), `distractor` (aynı sahnedeki **öteki** siyah kâseyi tarif eden İngilizce komut), `empty` (boş komut). Her satırda ilişki türü (Ders 6 kalkülüsleri) ve Türkçe'de mekânsal ilişkiyi taşıyan ekler işaretli. Ayrıca `token_counts.json`: her satırın OpenVLA prompt'unun LLaMA-2 token sayısı ve parça listesi.

## Kaynak ve elde etme
- `en`: LIBERO `benchmark.get_benchmark_dict()["libero_spatial"]().get_task(i).language` (Ders 10 çıktısı; LIBERO commit: ana dal, Kas 2026 klonu).
- `distractor`: LIBERO BDDL dosyalarının `(:init …)` bloğu okunarak belirlendi — her Spatial sahnesinde `akita_black_bowl_1` (hedef) ve `akita_black_bowl_2` (çeldirici) var; `bowl_2`'nin bölgesini tarif eden Spatial komutu seçildi (görev 0→1, 1→6, 2→8, 3→9, 4→9).
- `para`, `tr`, `tr2`: Claude tarafından yazıldı (Ders 7'nin ek şeması ve Ders 14'ün `tabak ile ramekinin arasındaki…` çevirisiyle tutarlı); **ikinci okuma Mesut tarafından yapılacak** (anadil doğrulaması bekliyor — v0.2).
- Üretim script'i: `code/day15/make_commands.py`; token sayıları: `code/day15/count_tokens.mjs` (llama-tokenizer-js; Mac'te HF tokenizer ile doğrulanacak).

## Şema (`data/commands.csv`)
| Alan | Tip | Açıklama | Örnek |
|---|---|---|---|
| id | str | benzersiz kimlik `sp{task}_{condition}` | `sp0_tr` |
| suite | str | LIBERO suite | `libero_spatial` |
| task_id | int | görev indeksi (Ders 10 sırası) | 0 |
| lang | str | `en` / `tr` / `none` | `tr` |
| condition | str | `en` / `para` / `tr` / `tr2` / `distractor` / `empty` | `tr` |
| instruction | str | komut metni (OpenVLA prompt'una `.lower()` ile girer) | `tabak ile ramekinin arasındaki siyah kaseyi al ve tabağın üstüne koy` |
| relation | str | mekânsal ilişki türü | `arasında (2 referans)` |
| frame | str | bakış açısı | `camera` |
| suffixes | str | TR'de ilişkiyi taşıyan ekler, `;` ile | `-in;ara-sı-nda-ki;-ın;üst-ü-ne` |
| variant | str | yüzey formu | `a` / `b` |
| note | str | açıklama | `task.language (referans)` |

`data/token_counts.json`: `rows[]` (id, prompt_tokens_with_bos, instruction_tokens, instruction_pieces) ve `phrases[]` (mekânsal ifadelerin ve tek kelimelerin parçalanışı).

## Uygulanan dönüşümler
Yok (ham metin). OpenVLA prompt şablonu çağrı anında uygulanır: `In: What action should the robot take to {instruction.lower()}?\nOut:` — boş komutta `to ?` kalır (bilinçli; LIBERO-Plus'ın boş-komut koşuluyla aynı mantık).

## Bilinen sınırlılıklar
- 5 görev, tek başlangıç durumu (init 0): istatistiksel güç yok (n=5'te kesin Wilcoxon p ≥ 0.0625). Pilot set.
- Çeviriler tek yazarlı (Claude); anadil kontrolü bekliyor. `kase/kâse` ve `üstüne/üzerine` gibi yazım farkları bilinçli olarak iki form arasında bölüştürüldü.
- `distractor` görev 3 ve 4'te aynı komut (ikisinde de `bowl_2` dolabın üstünde).
- `frame=camera` bu sette yön (sağ/sol/ön/arka) içeren komut olmadığı için etkisiz.

## Kullanım
```python
import pandas as pd
df = pd.read_csv("data/commands.csv")
df[df.task_id == 0][["condition", "instruction"]]
```

## Değişiklik günlüğü
- v0.1 — ilk sürüm (Ders 15): 30 satır, token sayıları.
