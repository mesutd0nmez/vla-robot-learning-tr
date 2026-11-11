# LIBERO-Spatial görev 0 — 4 koşullu komut deneyi (clean / empty / wrong / tr): koşul tanımı + sonuç şeması

**Tür:** kendi ürettiğim (deney tasarımı; sonuçlar Mesut'un Colab koşusundan gelecek) · **Sürüm:** v0.1 · **Tarih:** Ders 18 · **Lisans:** CC BY 4.0 (komut metinleri), sonuçlar aynı · **İlgili proje/ders:** Ders 18 (tasarım), Ders 19 (analiz), `projects/05-spatial0-4kosul/` (açılacak)

## Ne içeriyor
`data/conditions_task0.json`: LIBERO-Spatial görev 0 (*pick up the black bowl between the plate and the ramekin and place it on the plate*) için dört komut koşulu — `clean` (null = benchmark'ın `task.language`'ı), `empty` (`""`), `wrong` (aynı sahnedeki **öteki** kâseyi tarif eden İngilizce komut: *next to the ramekin* = `akita_black_bowl_2`; `datasets/01`'in `distractor` satırı), `tr` (`datasets/01`'in `sp0_tr` satırı). `data/episodes_schema.csv`: koşu sonuçlarının episode düzeyinde şeması (2 örnek satır, **temsili**); gerçek 40 satır `projects/05-…/results/episodes.csv`'ye gelecek ve kart v0.2'de buraya kopyalanacak. Deney: 10 init state (`initial_states[0..9]`) × 4 koşul = 40 episode, VLANeXt-B LIBERO-Spatial checkpoint'i, T4.

## Kaynak ve elde etme
- Komutlar: `datasets/01-libero-spatial-tr-mini/data/commands.csv` → `code/day18/make_conditions.py --task 0` (otomatik; `--extra para,tr2` ile 6 koşul).
- Init state'ler: orijinal LIBERO `init_files/libero_spatial/<görev>.pruned_init` dosyasının ilk 10'u (VLANeXt `build_episode_specs`: `init_state = initial_states[episode_idx]`); koşullar arasında birebir aynı → eşli tasarım.
- Model: `DravenALG/VLANeXt` HF deposundan `VLANeXt_libero_spatial.pt` (16.4 GB; Ders 12 kartı/notebook'u). Eval kodu VLANeXt `d911e6b` + Ders 12 + Ders 18 yamaları (`code/day18/vlanext_cond_patch.py`).

## Şema (`data/conditions_task0.json`)
| Alan | Tip | Açıklama | Örnek |
|---|---|---|---|
| anahtar | str | koşul adı; log'da `condition=`, video adında `<ad>__` öneki | `tr` |
| değer | str/null | modele giden komut; `null` → `task.language` | `tabak ile ramekinin arasındaki siyah kaseyi al ve tabağın üstüne koy` |

## Şema (`data/episodes_schema.csv` = `analyze_4cond.py` çıktısı `episodes.csv`)
| Alan | Tip | Açıklama | Örnek |
|---|---|---|---|
| run_id | str | koşu kimliği (`runs.csv` ile aynı) | `vlanext_sp0_4cond_01` |
| suite | str | LIBERO suite | `libero_spatial` |
| task_id | int | görev indeksi (0 tabanlı) | 0 |
| condition | str | `clean` / `empty` / `wrong` / `tr` (+ `para`, `tr2`) | `empty` |
| init_state | int | `trial` = init state indeksi (0–9); eşleştirme anahtarı | 3 |
| success | int | 0/1 — BDDL hedefi `(On akita_black_bowl_1 plate_1)` sağlandı mı (koşuldan bağımsız, hep orijinal hedef) | 1 |
| rollout_steps | int | üretilen action sayısı (max 220; başarısızlar genelde 220) | 96 |
| episode_time_s | float | log'daki `episode_time` (HH:MM:SS → saniye) | 91.0 |

`runs.csv` (koşul başına 1 satır) repo sözleşmesindedir: `run_id, model, checkpoint, suite, task_id, lang, condition, n_episodes, success_rate`; `lang`: clean/wrong → `en`, empty → `none`, tr → `tr`.

## Uygulanan dönüşümler
1. `make_conditions.py`: CSV → JSON; `;`/parantez uyarısı (BDDL yoluna geçilirse `scan_tokens` kuralı, Ders 16).
2. Eval: komut `VLANEXT_CONDITIONS` ile `EpisodeSpec.task_description`'a yazılır; büyük/küçük harf korunur (override yolu; BDDL yolu değil — Ders 17 §6 karşılaştırması).
3. `analyze_4cond.py`: `eval.log` regex → episodes/runs/table; Wilson %95; eşli McNemar exact (uyumsuz çiftler), eşsiz Fisher exact.

## Bilinen sınırlılıklar
- **Henüz sonuç yok** (v0.1): bu ortamda GPU/HF erişimi olmadığı için koşu yapılmadı; şema satırları temsilidir.
- n = 10 yön verir, anlamlılık vermez: 9/10'un Wilson aralığı 60–98; eşli testte 5 uyumsuz çift bile p = 0.0625 (Ders 18 görsel 03–04). 20–50 init state'e çıkarmak `num_trials_per_task` ile tek satırlık değişikliktir.
- `wrong` koşulunda `success` yalnızca "orijinal görevi yaptı mı" (LIBERO-CF'in *biased* ölçüsü); "yanlış komuta sadık mı" (*faithful*) için video etiketi gerekir (sütun yok; v0.2'de `grasped_bowl` eklenecek).
- Tek görev, tek model, tek Türkçe yüzey formu; `tr2`/`para` isteğe bağlı.

## Kullanım
```python
import json, pandas as pd
conds = json.load(open("data/conditions_task0.json"))
df = pd.read_csv("data/episodes_schema.csv")     # gerçek veri: projects/05-spatial0-4kosul/results/episodes.csv
df.pivot(index="init_state", columns="condition", values="success")
```

## Değişiklik günlüğü
- v0.1 — koşul tanımı + şema (Ders 18); sonuç yok
