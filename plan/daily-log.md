# Günlük Çalışma Logu

Bu dosya her çalışma gününün sonunda güncellenir. Yeni güne başlarken Claude önce bunu okur.
Dersler tarihe değil sıraya bağlıdır (`lectures/lecture-NN.md`); bir günde birden fazla ders çalışılabilir. Girişler ders numarasına göredir. Ders 12'den itibaren her sabah 07:00'de (TR) zamanlanmış görev sıradaki dersi yazar; bu dosyaya girişini ekler.

## Ders 1 — Kaynak haritası + robot learning döngüsü
**Durum:** Başlandı
**İşlenen:**
- Tüm kaynaklar Claude tarafından okundu ve haritalandı (bkz. lecture-plan.md "Kaynak haritası").
- Çekirdek döngü: observation → policy → action → world changes → new observation. Policy = karar veren fonksiyon; VLA bir policy türü.
- Behavior cloning = supervised learning (X=observation, y=expert action). Robot manipülasyonunda RL yerine imitation learning'in baskın olma nedeni: gerçek robotta deneme maliyeti.

**Ders:** `lectures/lecture-01.md` (6 görsel, 4 kendini sına sorusu)
**Görev (Mesut):** Dersi oku, soruları cevaplara bakmadan cevapla, takıldığın yeri chat'e yaz.
**Notlar / sorular:** —

## Ders 2 — OpenVLA'nın içi
**Durum:** Ders hazır
**Ders:** `lectures/lecture-02.md` — SigLIP+DINOv2, projector, LLaMA-2, 256-bin action tokenizasyonu, LoRA fine-tune, OpenVLA-OFT farkı (4 görsel, 5 soru)
**Görev (Mesut):** Pipeline'ı görsele bakmadan çiz; soruları cevapla.
**Notlar / sorular:** —

## Ders 3 — Benchmark ≠ simülatör
**Durum:** Ders hazır
**Ders:** `lectures/lecture-03.md` — MuJoCo/robosuite/LIBERO/eval yığını, 4 suite, BDDL + 50 demo, eval protokolü (50 ep/task), liderlik tablosunun doyması, LIBERO-Plus'ın 7 bozulması ve "model dili görmezden geliyor" bulgusu (4 görsel, 5 soru)
**Görev (Mesut):** 4 suite tablosu + Spatial komutlarındaki ilişki kelimelerinin Türkçe ek karşılıkları.
**Notlar / sorular:** —

## Ders 4 — Language grounding ve shortcut learning
**Durum:** Ders hazır
**Ders:** `lectures/lecture-04.md` — grounding türleri, shortcut mekanizması, 5 koşullu ablation matrisi, iki sonuç örüntüsü, tokenizasyon ablation'ı (3 görsel, 5 soru)
**Görev (Mesut):** Spatial/Goal tablosuna shortcut riski + boş-komut tabanı sütunları.
**Notlar / sorular:** —

## Ders 5 — Tatar vd. derin okuma
**Durum:** Ders hazır
**Ders:** `lectures/lecture-05.md` — 10 soru şablonu, yöntem (L2/cos/EBO), bulgular, gripper ve tokenizer gözlemleri, 3 güçlü / 3 zayıf, tezin ayrışma tablosu (4 görsel, 5 soru)
**Görev (Mesut):** 5 komutu LIBERO stiline EN/TR yeniden yaz, ekleri işaretle.
**Notlar / sorular:** —

## Ders 6 — Mekânsal akıl yürütme temelleri
**Durum:** Ders hazır
**Ders:** `lectures/lecture-06.md` — sıkıştırma metaforu, CDC, RCC-8, 1-cross/TPCC, mesafe, bAbI/StepGame/SpartQA, LLM+ASP (4 görsel, 5 soru)
**Görev (Mesut):** 10 Türkçe eki kalkülüs eksenlerine yerleştir.
**Notlar / sorular:** —

## Ders 7 — RoomSpace → LIBERO
**Durum:** Ders hazır
**Ders:** `lectures/lecture-07.md` — Figure 1 okuma, 5 adımlı üretim hattı, CSP ⟨n,d,m,p⟩, ASP altın etiket, sonuçlar, 4 tasarım ilkesinin LIBERO'ya taşınması, 10 Türkçe komut taslağı (3 görsel, 5 soru)
**Görev (Mesut):** 10 komut için ikinci yüzey formu + robota-göre versiyonlar.
**Notlar / sorular:** —

## Ders 8 — Hafta 1 sentezi + model aileleri
**Durum:** Ders hazır
**Ders:** `lectures/lecture-08.md` — mindmap, awesome-vla-wam 6 kategori, VLANeXt (Qwen3-VL) vs OpenVLA (LLaMA-2) tokenizer kontrolü, VLANeXt ailesi, Faz 2 haftalık plan (4 görsel, 5 soru)
**Görev (Mesut):** Haritayı kendi düzeninle çiz; 3 ilgili makale seç.
**Notlar / sorular:** —

## Ders 9 — Mac: conda + MuJoCo + ilk sahne
**Durum:** Ders hazır
**Ders:** `lectures/lecture-09.md` — Miniconda arm64, vla env (py3.10), mujoco/torch(MPS), check_env.py, first_scene.py (gerçek render) (2 görsel, 5 soru)
**Görev (Mesut):** Kurulumu yap, first_scene.png üret, XML'de oyna.
**Notlar / sorular:** —

## Ders 10 — Mac: LIBERO kurulumu + ilk episode
**Durum:** Ders hazır
**Ders:** `lectures/lecture-10.md` — mujoco 2.3.7 pin (doğrulandı), robosuite 1.4.1, LIBERO editable, config.yaml, list_tasks.py (Spatial 10 komut), libero_first_episode.py, obs/action sözleşmesi, 180° döndürme (3 görsel, 5 soru)
**Görev (Mesut):** Episode'u çalıştır, x ekseni yönünü not et.
**Notlar / sorular:** —

## Ders 11 — LIBERO HDF5 demo verisi
**Durum:** Ders hazır
**Ders:** `lectures/lecture-11.md` — indirme, HDF5 ağacı, inspect_hdf5.py (doğrulandı), action grafiği, gripper işareti uyarısı, 128px sınırlılığı (3 görsel, 5 soru)
**Görev (Mesut):** Spatial demo indir, 3 demo'yu karşılaştır.
**Notlar / sorular:** —

## Ders 12 — Colab T4'te VLANeXt: ilk success rate
**Durum:** Ders hazır
**Ders:** `lectures/lecture-12.md` — Colab vs Kaggle (RAM), kodun 4 engeli (FA2/sm75, 16.4 GB checkpoint vs 12.7 GB RAM, 500-episode bütçe, bağımlılıklar) ve 3 tek-satırlık yama, 8 hücrelik notebook, eval döngüsü protokolü (10 no-op, chunk 8, max 220, 180° döndürme, gripper eşiği, done→success), 3 episode'un güven aralığı, makale LIBERO + LIBERO-plus Language grafikleri ve tez bağlantısı (4 görsel, 5 soru). Doğrulananlar: patch script (idempotent), log parser, robosuite 1.4.1 + mujoco 3.8.1 + numpy 2 (Linux). Hipotez: T4'te gerçek koşu süreleri/bellek (GPU erişimi yoktu). Ders 8'deki yazar hatası düzeltildi (Wu vd.).
**Kod:** `code/day12/` — `colab_vlanext_t4.ipynb`, `vlanext_t4_patch.py`, `t4_bench_config.yaml`, `parse_eval_log.py`, `plot_paper_results.py`
**Görev (Mesut):** Notebook'u koş; parse tablosu + runs.csv + başarısız episode teşhisi → `projects/01-vlanext-t4-ilk-eval/`.
**Notlar / sorular:** —

## Ders 13 — Kod okuma: komut → action → success zinciri
**Durum:** Ders hazır
**Ders:** `lectures/lecture-13.md` — VLANeXt + LIBERO eval zincirinin 22 kritik satırı (`trace_eval_chain.py` gerçek çıktısı); `task.language`'ın BDDL `:language` alanından değil **dosya adından** türediği; Qwen chat şablonu (eğitim/eval aynı yapı); dizi kuruluşu (8 proprio token başa, 16 meta query sona, 29 hidden state); `soft` bağlantı: policy bloğu i ↔ VLM katmanı i cross-attention (`MoEBlock`); flow-matching 5 Euler adımı (sigma 1→0.75→0.5→0.25→0.001→0, VLM bir kez çalışır); `_denormalize_action` gerçek min/max (legacy/mixed); success = `(On akita_black_bowl_1 plate_1)` → `check_ontop`: temas + xy < 3 cm, nesne adıyla; makale Tablo 1 bağlantı türü × LIBERO-plus Language (74.6/81.0/89.2) (5 görsel, 5 soru). Düzeltme: Ders 12'deki "16 query'lik policy head" anlatımı eksikti. Hipotez: Qwen3-VL şablonunun tam string'i (HF erişimi yoktu; `show_prompt.py` çalıştırılmadı).
**Kod:** `code/day13/` — `trace_eval_chain.py` (doğrulandı), `show_prompt.py` (py_compile), `plot_day13.py`
**Görev (Mesut):** VLANeXt'i klonla, `trace_eval_chain.py` + `show_prompt.py` çalıştır; şablon string'i, EN/TR token sayıları ve "arasındaki/üstüne" parçalanışını `projects/01-vlanext-t4-ilk-eval/README.md` sonuna "Ders 13 notu" olarak ekle.
**Notlar / sorular:** —

## Ders 14 — OpenVLA-7B Mac'te: tek görüntü + komut → 7 token → 7-DoF action
**Durum:** Ders hazır
**Ders:** `lectures/lecture-14.md` — `openvla-7b-finetuned-libero-spatial` (LoRA r=32, Spatial 84.7 ± 0.9; 432 trajectory / 52.970 transition, no-op ve başarısız demo'lar elenmiş); görüntü yolu (180° döndür → lanczos 224 → %90 center crop zorunlu → 6 kanallı `pixel_values` → DINOv2+SigLIP 256 patch → 3 katmanlı projector); komut yolu (`In: What action should the robot take to …?\nOut:` + `29871 "▁"` eklenmesi; dizi `[BOS][256 patch][33 metin][▁]` = 291 token); 7 token greedy üretim, action token'ları `31744..31999` (dağarcığın son 256'sı, 255 merkez), `vocab_size 32064 → 32000`; decode `bin → [-1,1] → q01/q99` gerçek `libero_spatial` değerleriyle ve asimetri ("nötr token" dx için +0.096), gripper `2g−1 → sign → invert`; Mac sapmaları (sdpa, bf16 MPS macOS 14+, `device_map={"": "mps"}`, ayrı `openvla` env — transformers 4.40.1 pinleri Ders 13'ün Qwen3-VL sürümüyle çakışır); OpenVLA vs VLANeXt 9 satırlık karşılaştırma; EN komut 20 → TR 32 token (×1.6), *arasındaki* 5 / *üstüne* 4 parça (6 görsel, 5 soru). Doğrulananlar: OpenVLA kodu (`c8f03f4`) satır referansları, HF `dataset_statistics.json`/`config.json`, token sayıları (llama-tokenizer-js), `decode_action_tokens.py` gerçek çıktı, görsel 05 gerçek robosuite render'ı. Hipotez: MPS süre (~1.5–3 s/action) ve bellek; model bu ortamda çalıştırılamadı (HF/Apple GPU yok). Düzeltme: Ders 2'de "2 katmanlı projector" denmişti; fused backbone'da 3 katman.
**Kod:** `code/day14/` — `openvla_mps_single_action.py` (py_compile), `render_libero_frame.py` (py_compile), `preprocess_demo.py` (doğrulandı, robosuite), `decode_action_tokens.py` (doğrulandı), `plot_day14.py`
**Görev (Mesut):** `openvla` env'ini kur; `frame_task0.png` ile 3 çağrı (görev 0 EN, görev 8 EN, görev 0 TR); token/action/süre tablosunu `projects/02-openvla-mac-tek-kare/README.md`'ye yaz; ilk `predict_action` süresini ve `model_input_224.png`'yi ekle.
**Notlar / sorular:** —

## Ders 15 — Tatar mini-reprodüksiyonu: 5 kare × 6 koşul, L2 + cosine
**Durum:** Ders hazır
**Ders:** `lectures/lecture-15.md` — tasarım: LIBERO-Spatial görev 0–4 × {en, para (EN parafraz), tr (-daki/kase), tr2 (duran/kâse/kaldır), distractor (aynı sahnedeki **öteki** kâseyi tarif eden EN komut — BDDL `:init`'teki `akita_black_bowl_2` bölgesinden türetildi: 0→1, 1→6, 2→8, 3→9, 4→9), empty} = 30 greedy çağrı; metrik geometrisi (gripper `mask=False` → flip L2'ye ≈2.0 katar, Tatar'ın "L2>0.99 ⇔ gripper farklı" gözlemi bu uzayda totolojiye yakın; ana metrik 6 boyut normalize L2/cos + ayrı gripper uyumu + tok_mismatch / max_bin / within5); gerçek LLaMA-2 token sayıları (prompt EN 29–35, TR 42–50, TR-2 47–54; *çekmecesindeki* 7 parça; *kâse* 3 vs *kase* 2); n=5'te kesin Wilcoxon min p 0.0625 (anlamlılık imkânsız → yön/büyüklük raporu); yorum ağacı (önce d(en,distractor) vs d(en,para), sonra d(en,tr)'nin para/empty/distractor'a göre yeri); Haziran 2026 literatürü: Chen vd. 2606.15714 (OpenVLA-OFT'de "komut yok" −27.5 ≈ yabancı dil −27.8…−29.8) ve Dong vd. 2606.11906 (Spatial OFT 97.6 → ko 49.2; adım-bazlı hidden-state cosine) (6 görsel, 5 soru). Doğrulananlar: token sayıları (llama-tokenizer-js), BDDL okuması, q01/q99 geometrisi, `analyze_day15.py` sentetik veriyle uçtan uca (görsel 04 **temsili**, başlıkta yazılı). Hipotez: Mac'te 30 çağrının süresi; Tatar'ın `Kal|dır` bölünmesi (bu tokenizer uyarlamasında `k|ald|ı|r`). Model bu ortamda çalıştırılamadı (HF/GPU yok); LIBERO kurulamadı (frame render Mac'te).
**Veri:** `datasets/01-libero-spatial-tr-mini/` — ilk veri kartı; `data/commands.csv` (30 satır), `data/token_counts.json`
**Kod:** `code/day15/` — `make_commands.py` (çalıştırıldı), `count_tokens.mjs` (çalıştırıldı), `analyze_day15.py` (sentetik ile çalıştırıldı), `plot_day15.py`, `run_tatar_mini.py` (py_compile), `render_frames.py` (py_compile)
**Görev (Mesut):** `render_frames.py` → `run_tatar_mini.py` (30 çağrı + `--repeat 2` determinizm) → `analyze_day15.py`; `projects/03-tatar-mini/` aç, `summary.md` tablosunu ve yorum ağacındaki dalı README'ye yaz; HF tokenizer ile `Kırmızı küpü kaldır` parçalanışını kontrol et; TR çevirilerini doğrula (kart v0.2).
**Notlar / sorular:** —

## Ders 16 — Faz 2 sentezi + LIBERO-Plus: ayrı runtime, komutun kod yolu, boş-komut denemesi
**Durum:** Ders hazır
**Ders:** `lectures/lecture-16.md` — Faz 2 bilançosu (Mac açık döngü vs Colab kapalı döngü; ölçebildiklerim/ölçemediklerim tablosu); LIBERO-Plus içeriden: 10.030 görev (Spatial 2402 / Object 2518 / Goal 2591 / Long 2519; Language toplam 1.537), **bozulmamış orijinal görev listede yok**, görev adı = bozulma parametreleri (`_view_0_0_100_0_0_initstate_0` = varsayılan kamera + init 0 → Language görevi = temiz sahne + yalnızca komut), JSON `id` 1 tabanlı / görev indeksi 0 tabanlı (dört suite'te doğrulandı), zorluk L1–L5 = 4 referans modelden kaçının çözdüğü (Ek C.3; Goal Language'ın %45'i L5); komutun kod yolu: `_language_` görevlerinde `task.language` dosya adından değil BDDL `:language`'dan (Ders 13 kuralının istisnası), dil BDDL'leri orijinalden yalnızca `:language` satırında ayrılıyor (`diff`), görev 0'ın 46 varyantı, R1/R2/R3 türleri (etiket JSON'da yok); `bddl.parsing.scan_tokens` Türkçe tuzakları (gerçek bddl 1.0.1: `Işığın`→`işığın`, `İ`→`i`+U+0307, `;` → Missing close parenthesis, parantez → TypeError; kural: küçük harf, `;`/parantez yok); ayrı runtime gerekçesi (iki `libero`, `~/.libero_plus` config, wand/ImageMagick) ve `PYTHONPATH` stratejisi; 5 yama (json yolu, `VLANEXT_INSTR_OVERRIDE`, `VLANEXT_RUN_TAG`, log, `np.float_`); deney: görev 988/994/995 (L1) × {rewritten, empty} + isteğe bağlı {clean, tr}; liderlik tablosu (11 modelin 10'unda Language diğer bozulmaların ortalamasından az zarar veriyor) (6 görsel, 5 soru). Doğrulananlar: `select_plus_tasks.py` gerçek veri, `bddl_lang_probe.py` gerçek paket, patch idempotent + py_compile, parser selftest. Hipotez: asset'lerin açılmış boyutu, Colab süreleri, duman testinin geçmesi (HF/GPU yoktu; asset'ler 6.4 GB indirilemedi).
**Kod:** `code/day16/` — `colab_libero_plus_t4.ipynb`, `select_plus_tasks.py` (çalıştırıldı), `bddl_lang_probe.py` (çalıştırıldı), `vlanext_plus_patch.py` (doğrulandı), `parse_plus_log.py` (selftest), `t4_plus_config.yaml`, `plot_day16.py`
**Görev (Mesut):** Notebook'u YENİ runtime'da koş; duman testi + `du -sh` + `rewritten`/`empty` tabloları (vakit varsa `clean`/`tr`) → `projects/04-libero-plus-kurulum/` (README: hücre süreleri, asset boyutu, hangi dal, başarısız video teşhisi; `results/runs.csv`).
**Notlar / sorular:** —

## Ders 17 — LIBERO-Plus'ın yedi bozulması kodda: üç taşıyıcı, kamera matematiği, 500 Panda, R1/R2, Türkçe varyant
**Durum:** Ders hazır
**Ders:** `lectures/lecture-17.md` — bozulmanın üç taşıyıcısı (görev adı: `_view_`/`_initstate_`/`_noise_` → env_wrapper.py:207-233; BDDL dosyası: `_language_`/`_add_`/`_levelL_sampleM`; problem sınıfı → scene XML: `_light_`/`_table_`/`_tb_`, 1.525 üretilmiş alt sınıf) ve Spatial için gerçek sayılarla perturbation tablosu; kamera: V→H→S→ER→EV sırası, pivot (0,0,0.8), 376 görev = 238 rotation (yatay ±75°, 130'unda dikey 15°) + 76 zoom (×1.15–2.0) + 62 endpoint (±10°), varsayılan agentview robotun karşısında, bakış masayı x=−0.345'te kesiyor; robot: 500 `MountedPandaK` sınıfı yalnızca `init_qpos`, **‖Δ‖ tam olarak {0.1,…,0.5} rad × 100**, zorluk büyüklükle artıyor (L1 medyan 0.2, L5 0.4); layout: `_add_` 300 dosya × 3 çeldirici (13 tür), `_level` kayması ≤ L cm; gürültü: 5 aile × 10 şiddet, yalnızca `agentview_image` (bilek temiz; OFT_w Noise 49.9); Language: görev 0'ın 50 yeniden yazımı kural tabanlı R1 26 / R2 tam 16 / R2 hafif 6 / R0 2 — **R1 hiçbir dayanağı silmiyor, R2 nesne adını siliyor ama *between* 48/50 duruyor** (benchmark ilişki ifadesini test etmiyor); 988/995 R1, 994 R2 hafif; makale Tablo 7 ile JSON'da Spatial Language 354 vs 390 satır karışıklığı (hipotez); VLANeXt Tablo III 7 bozulma × suite (Spatial: Robot ve Camera en kırıcı; Goal: Language en kötü 53/52); Türkçe `_language_trK.bddl` boru hattı: kural kontrolü → BDDL → `scan_tokens` geri okuma → task map sonuna + JSON girdisi → `VLANEXT_TASK_IDS=2402,…` (9 görsel, 5 soru). Doğrulananlar: `perturbation_table.py`, `robot_init_spread.py`, `layout_shift.py`, `classify_rewrites.py` gerçek dosyalardan; `camera_geometry.py` LIBERO-plus fonksiyonlarını çalıştırdı; görsel 04 gerçek robosuite render (Stack sahnesi, LIBERO değil); `make_tr_variant.py` gerçek bddl 1.0.1 ile 10/10 birebir, kötü örneklerde beklenen hatalar. Hipotez: ışık XML içeriği (asset yok), R1/R2 etiketleri, JSON girdisi eksik görevde eval davranışı. LIBERO depo kodu bu ortamda çalıştırılamadı (kısıt); gürültü fonksiyonları parametre tablolarıyla yeniden yazıldı.
**Veri:** `datasets/02-libero-plus-spatial0-rewrites/` — `rewrites_task0.csv` (50 satır), `tr_variants/` (10 Türkçe BDDL + task map satırları + JSON girdileri), `perturbation_table_spatial.md`
**Kod:** `code/day17/` — `perturbation_table.py`, `camera_geometry.py`, `robot_init_spread.py`, `render_perturbations.py`, `layout_shift.py`, `classify_rewrites.py`, `noise_on_frame.py`, `make_tr_variant.py` + `tr_commands.csv`, `plot_day17.py` (hepsi çalıştırıldı)
**Görev (Mesut):** `classify_rewrites.py` → 6 "R2 hafif" + 2 "R0" satırını kendin etiketle (`type_mesut` sütunu, kart v0.2); `camera_poses.csv`'den kind × zorluk dağılımı (bir cümle); vakit varsa görev 0 için `tr3` (robota-göre yön) ve `tr4` (*çanak*) komutlarını `make_tr_variant.py` ile BDDL'e çevir.
**Notlar / sorular:** —
