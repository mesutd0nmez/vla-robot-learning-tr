# TR mekânsal komut seti — 20 EN–TR çift × 5 yüzey

**Tür:** kendi ürettiğim · **Sürüm:** v0.1 (anadil okuması bekliyor) · **Tarih:** Ders 20 · **Lisans:** MIT (komut metinleri; İngilizce eğitim komutları ve BDDL şablonları LIBERO'nun MIT lisansı altında) · **İlgili proje/ders:** Ders 20, `code/day20/`; devamı Ders 21 (tokenizasyon), Ders 26–27 (deney matrisi)

## Ne içeriyor
20 EN–TR komut çifti, dört blokta: **A** (6) LIBERO-Spatial'ın eğitilmiş *seçim* komutları (`…-ndeki siyah kaseyi al`), **B** (6) aynı Spatial sahnelerinde hiçbir eğitim cümlesine denk düşmeyen tarifler (sol/sağ/ön; robota/tabağa yakın–uzak), **C** (6) LIBERO-Goal'ın eğitilmiş *yerleştirme* komutları (`…-nın üstüne koy`), **D** (2) eğitilmemiş yerleştirme (sağına/soluna; predicate yok). Her çift 5 yüzeyle gelir → `commands.csv` 100 satır: `en`, `tr_a` (ek üreticiyle üretilmiş ana biçim), `tr_b` (elle yazılmış ikinci biçim: *duran / üzerine / alıp*, farklı sözcük sırası, *kâse*), `tr_ascii` (Türkçe karaktersiz), `tr_bare` (ekleri atılmış sözcük torbası). B bloğunda komutun **hangi kâseyi** kastettiği üç çerçevede (LIBERO adlandırması / modelin gördüğü görüntü / robot) 50 init state'in gerçek koordinatlarından hesaplanmıştır.

## Dosyalar
| Dosya | İçerik |
|---|---|
| `data/pairs.csv` | 20 satır, çift başına tek satır (tasarım değişkenleri + bütün yüzeyler + referans etiketleri) |
| `data/commands.csv` | 100 satır, yüzey başına tek satır (şablon şemasıyla uyumlu; deney koşucusunun okuyacağı dosya) |
| `data/referents.csv` | 10 Spatial sahnesi × 9 tarif (sol/ön × 3 çerçeve, tabağa yakın, robota yakın, daha yüksekte): kastedilen kâse, 50 sahnedeki en küçük fark, eşik üstü mü |
| `data/bddl_cf/*.bddl` + `index.csv` | 5 karşı-olgusal BDDL: hedef `(On akita_black_bowl_2 plate_1)`; sahne ve bölgeler orijinalle aynı |
| `data/human_check.csv` | Mesut'un dolduracağı anadil okuması şablonu (doğallık 1–5, düzeltme, B07–B09 için "sen hangisini alırdın") |

## Kaynak ve elde etme
- `en` (A, C): LIBERO görev adları (`libero_suite_task_map.py`; modele giden metin dosya adından türetilir — Ders 13). `en` (B, D): Claude tarafından yazıldı; LIBERO'nun 130 görev adının hiçbirinde geçmediği `make_tr_set.py --libero` ile doğrulandı (`en_seen_libero = hayır`).
- `tr_a`: `code/day20/make_tr_set.py` — sözlük + şablon + `tr_morph.py` (ünlü uyumu, kaynaştırma, yumuşama, zamir n'si); ayrımı `tr_a_seg`. A01–A04 `datasets/01`'in `sp0/1/2/4_tr` satırlarıyla birebir aynıdır.
- `tr_b`: Claude tarafından elle yazıldı. **Anadil okuması Mesut tarafından yapılacak** (`human_check.csv`, v0.2).
- `ref_*`: `code/day20/scene_frames.py` — LIBERO `.pruned_init` (50 init state) + agentview kamera pozu (`libero_tabletop_manipulation.py`); LIBERO commit: ana dal, Kas 2026 `--depth 1` klonu.
- `bddl_cf/`: `code/day20/make_cf_bddl.py`; her dosya `bddl==1.0.1` `scan_tokens` ile geri okundu (dil birebir, hedef bowl_2, `:init`/`:regions` değişmemiş).

## Şema (`data/pairs.csv`)
| Alan | Tip | Açıklama | Örnek |
|---|---|---|---|
| pair_id | str | blok harfi + sıra | `B07` |
| block | str | `A` / `B` / `C` / `D` | `B` |
| tier | int | 1 = mevcut checkpoint + mevcut predicate; 2 = mevcut checkpoint + karşı-olgusal BDDL; 3 = yeni predicate + fine-tune | 2 |
| suite, scene_task_id | str, int | komutun koşulacağı sahne (resmî görev sırası) | `libero_spatial`, 1 |
| relation | str | ilişki türü (Ders 6 kalkülüsleri) | `yön (sol)` |
| rel_noun | str | ilişki adının yalın biçimi (`—`: ilişki adı yok) | `sol` |
| case | str | ilişki sözcüğünün hâli | `LOC+ki (iyeliksiz)` |
| frame_dependent | str | `evet` / `hayır` | `evet` |
| en, tr_a, tr_b, tr_ascii, tr_bare | str | beş yüzey | `soldaki siyah kaseyi al ve tabağın üstüne koy` |
| tr_a_seg | str | `tr_a`'nın ek ayrımı | `sol-da-ki siyah kase-yi al ve tabağ-ın üst-ü-ne koy` |
| rel_span, rel_seg | str | ilişkiyi taşıyan sözcük ve ayrımı (Ders 21 bunu tek başına tokenize eder) | `soldaki`, `sol-da-ki` |
| ref_span | str | referans ad (varsa) | `ramekinin` |
| ref_libero, ref_image, ref_robot | str | üç çerçevede kastedilen kâse (A: eğitimdeki hedef; C/D: `n/a`) | `bowl_2`, `bowl_1`, `bowl_1` |
| ref_min_margin | str | 50 init state'teki en küçük fark (cm / px) | `36.5 cm \| 111.6 px \| 36.5 cm` |
| en_seen_suite, en_seen_libero | str | EN komut aynı suite'in / herhangi bir LIBERO suite'inin görev adı mı | `hayır` |
| goal_predicate | str | success predicate'i (BDDL `:goal`) | `(On akita_black_bowl_1 plate_1)` |
| note | str | tasarım notu | |

`data/commands.csv`: `id` (`B07_tr_a`), `pair_id`, `suite`, `task_id`, `lang`, `variant`, `instruction`, `relation`, `frame`, `suffixes`, `tier`.

## Çerçeve tanımları (`referents.csv`, `ref_*`)
| Çerçeve | sol | ön | Dayanak |
|---|---|---|---|
| `libero` | küçük y (−y) | büyük x (+x) | LIBERO-90/10 BDDL bölge adları: 5/5 left/right çifti, 2/2 front/back çifti (`scene_frames.py --conventions`) |
| `image` | küçük u | büyük v (görüntüde altta) | `obs[::-1, ::-1]` uygulanmış 256×256 kare; kamera quaternion'undan projeksiyon, Mesut'un project-2 render'ıyla örtüşüyor |
| `robot` | büyük y (+y) | robot tabanına yakın | taban x = −0.66, +x'e bakıyor (`mounted_panda.py`) |

"Belirsiz değil" eşiği: 50 sahnenin hepsinde aynı işaret **ve** en az 5 cm / 20 piksel fark.

## Uygulanan dönüşümler
Yok (ham metin). Bütün satırlar BDDL-güvenlidir (küçük harf, `;` ve parantez yok, NFC) — `validate_set.py` denetler.

## Bilinen sınırlılıklar
- Tek yazarlı çeviri; anadil okuması yok (v0.1). `tr_b` doğallığı ve B bloğundaki tariflerin insan yorumu `human_check.csv` ile gelecek.
- 20 çift bir **tasarım iskeleti**dir, istatistiksel örneklem değil: her hücrede 1–2 komut var. Tez deneyi için blok başına çoğaltılmalı (şablon + sözlük bunu ucuzlatır).
- B bloğu yalnızca iki kâseli üç sahneyi (görev 0, 1, 8) kullanır; sol/sağ için tek sahne (görev 1) var.
- Karşı-olgusal BDDL'ler gerçek LIBERO ortamında yüklenmedi (bu oturumda MuJoCo/LIBERO yok); yalnızca ayrıştırıcıdan geçti.
- `image` çerçevesi, modelin eğitim görüntülerinin de aynı 180° döndürmeyle üretildiği varsayımına dayanır (Ders 10/12 kod okuması).
- D bloğunun predicate'i yok; bu satırlar bugün yalnızca dil tarafı (tokenizasyon) için var.
- `tr_bare` üretimi tamlamaları olduğu gibi bırakır (*şarap şişesi*, *üst çekmece*).

## Kullanım
```python
import pandas as pd
p = pd.read_csv("data/pairs.csv")
p[p.block == "B"][["pair_id", "tr_a", "ref_libero", "ref_image", "ref_robot"]]
c = pd.read_csv("data/commands.csv")
c[(c.pair_id == "C13")][["variant", "instruction"]]
```

## Değişiklik günlüğü
- v0.1 — ilk sürüm (Ders 20): 20 çift, 100 yüzey, 90 referans satırı, 5 karşı-olgusal BDDL.
