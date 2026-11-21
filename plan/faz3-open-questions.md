# Faz 3 açık soruları (Ders 25)

Faz 3'ün (Ders 17–24) sentezinden çıkan, Faz 4'e (Ders 26 RQ formülasyonu) giren liste. `Durum` sütunu Mesut'un; bir soru kapanınca satır silinmez, tarih + bir satır sonuç yazılır. Ders 26–27 buradan okur.

**Öncelik:** S1 kritik yolda — Ders 24'ün karar ağacı (hangi çözüm tez olur) Ders 18'in örüntüsüne bağlı. S1 açıkken RQ'lar *yazılır*, *sıralanmaz*.

## Mesut'un koşusu bekliyor (script hazır)

| # | Soru | Araç / girdi | Kabul ölçütü | Beslediği RQ | Durum |
|---|---|---|---|---|---|
| S1 | Ders 18 tablosu: `clean / empty / wrong / tr` × 10 init state (VLANeXt-B, Spatial görev 0) — hangi örüntü? | `code/day18/colab_4cond_t4.ipynb` → `analyze_4cond.py` → `code/day19/read_patterns.py episodes.csv --features …` | `projects/05-spatial0-4kosul/` README başında örüntü harfi (A/B/C/D); `results/runs.csv` satırları (`seed`, `date` dahil) | RQ-a | bekliyor |
| S2 | MT round-trip: Opus-MT tr-en ile `datasets/04` 80 Türkçe satır → birebir eşleşme, "yanlış ama geçerli" sayısı, medyan gecikme | `code/day24/mt_layer_mac.py --model Helsinki-NLP/opus-mt-tr-en` | üç sayı + `wrong_en` örnekleri (Ders 24 notu) | RQ-c | bekliyor |
| S3 | Qwen3-VL tokenizer kimliği: HF `Qwen/Qwen3-VL-2B-Instruct` tokenizer'ı ile npm `@lenml/tokenizer-qwen3` `tokenizer.json` bayt-aynı mı; OpenVLA HF tokenizer'ı SentencePiece ile birebir mi? | `code/day22/tokenize_set.py` iki kez (`--llama tokenizer.model` / `--llama openvla/openvla-7b`; `--qwen Qwen/Qwen3-VL-2B-Instruct`) → `tokens.csv` ilk 7 sütun `diff` | `diff` boş → iki hipotez kapanır; değilse `datasets/05` kart v0.2 | RQ-a | bekliyor |
| S4 | Anadil okuması: görev 1 render'ında "soldaki / sağdaki" ve görev 8'de "öndeki" hangi kâse; `tr_a` / `tr_b` doğallık 1–5 + düzeltme | `datasets/04-tr-spatial-commands/data/human_check.csv` | 20 satır dolu; `hangi_kase_mesut` ↔ `ref_*` sütunları karşılaştırması bir cümle | RQ-b | bekliyor |
| S5 | Webots R2025a launcher M2'de çalışıyor mu; `ders21_masa.wbt` ilk gerçek render | `code/day21/check_webots_env.py`, `ders21_kol.py --out out` | `view_01_bowl.jpg`, `joints.csv` `max_err_rad` → `projects/06-webots-ilk-sahne/` | S8 | bekliyor |
| S6 | Parser held-out: kendi Türkçenle +5 cümle; hata türü (parse yok / yanlış ama geçerli) | `code/day24/heldout_tr.csv` + `tr_parse.py` | kaçı geçti + hata türleri bir satır | RQ-c | bekliyor |

## Danışmana (karar)

| # | Soru | Dayanak | Seçenekler | Beslediği RQ | Durum |
|---|---|---|---|---|---|
| S7 | Ana suite: Spatial (CDC ilişkileri; komutsuz tavan 9/10 → shortcut riski) mi, Goal (topoloji; tavan 1/10; token tarafı en temiz Türkçe) mi — yoksa ikisi de mi? | Ders 19 §2–3, Ders 23 §5 | Spatial 50'lik / Goal 50'lik / ikisi 10'luk pilot sonra karar | RQ-b | sorulmadı |
| S8 | Webots ikinci ortam olarak gerekli mi (oda ölçeği, robota göre yön, RoomSpace tarzı set); ROS 2 / gerçek robot planı var mı? | Ders 21 §6 karar notu + 3 soru | LIBERO'da kal / Webots küçük set / ikisi | RQ-b (dolaylı) | sorulmadı |
| S9 | Türkçe LoRA / co-training bütçesi: Kaggle 30 sa/hafta ile 2–3 hafta (tahmin 65–98 T4 saati, 4-bit) kabul edilebilir mi, yoksa tez eval-only mi kalsın? | Ders 24 §5, `code/day24/qlora_budget.py` | eval-only / 4-bit co-training / üniversite GPU'su | RQ-c | sorulmadı |
| S10 | Katkı ölçütü: zayıf okuma (Türkçe komutla doğru iş; MT/parser yeter) mi, güçlü okuma (B bloğu + karşı-olgusal BDDL'de `empty` üstü; omurga/LoRA gerekir) mi? | Ders 24 §1, §7 | zayıf yeter / güçlü şart / zayıf taban + güçlü hedef | RQ-c | sorulmadı |

## Ders 26'ya giden RQ tohumları (ad düzeyinde; soru biçimi Ders 26'da)

| Tohum | Ad | Girdileri | Bağlı sorular |
|---|---|---|---|
| RQ-a | **Fark var mı?** Aynı sahnede `tr − clean` ve `tr − empty`; iki sistem (OpenVLA-Spatial LoRA, VLANeXt-B); Wilson + PI | Ders 18–19, 22 | S1, S3 |
| RQ-b | **Nerede kırılıyor?** İlişki ailesi (CDC / topoloji / mesafe) × yüzey (`tr_a/tr_b/tr_ascii/tr_bare`) × çerçeve; suite seçimi | Ders 19–20, 23 | S4, S7, (S8) |
| RQ-c | **Nasıl kapanır?** Parser/MT tabanı vs EN+TR co-training; ölçüt B bloğu | Ders 24 | S2, S6, S9, S10 |

## Hipotez olarak kalanlar (Faz 4'te "ön çalışma" ile kapatılacak)
- Eğitim görüntülerinin de `[::-1, ::-1]` ile döndürülmüş olduğu (Ders 20 §3).
- VLANeXt'in koşudan koşuya değişkenliği (flow-matching `torch.randn`, seed; Ders 23 §4) — S1'in ≥3 tekrarı bunu ölçer.
- QLoRA'nın T4'te ≈ 7,7 GB tutması ve süre (Ders 24 §5).
- Türkçe'nin Qwen3-VL'de Çince tarafında mı Arapça tarafında mı olduğu (Chen Tablo 1 örüntüsü; Ders 24 §4) — S1'in `tr` satırı ilk kanıt.
- VLANeXt şablonunda `<|vision_end|>` sonrası boşluk (Ders 22 §4; `code/day13/show_prompt.py`).

## Ders 25 görevi (Mesut doldurur)
```
phase3_smoke.py: __ OK / __ HATA / __ ATLANDI   (tarih: ______)
tokens.csv == datasets/05?: ______
Öncelik 1 işaretlenen sorular: S__, S__, S__
```
