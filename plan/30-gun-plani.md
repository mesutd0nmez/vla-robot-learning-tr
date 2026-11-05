# VLA / Robot Learning — 30 Günlük Çalışma Planı

**Başlangıç:** 5 Kasım 2026 (Gün 1) · **Bitiş:** 4 Aralık 2026 (Gün 30)
**Günlük süre:** 30–60 dk · **Hedef:** 30 gün sonunda LIBERO üzerinde bir VLA'yı çalıştırabilmek, dil/mekânsal robustness deneyi tasarlayabilmek ve "bir proje yapalım" dendiğinde araştırma sorusu + deney planı üretebilmek.

## Her gün nasıl çalışıyoruz
1. Chat'e "Gün N" yaz. Claude `gunluk-log.md` dosyasını okur, o günün konusunu anlatır, görevi verir.
2. Gün sonunda kısa çıktıyı (3–5 cümle, bir tablo, bir komut çıktısı) chat'e yapıştır. Claude log'a işler.
3. Takıldığın yerde plan kayar; kaybolmaz. Plan esnek, sıra sabit.

## Kaynak haritası (danışmanın gönderdikleri)
| Kaynak | Ne? | Neden gönderildi? |
|---|---|---|
| `robot_learning_vla_giris.md` | Türkçe kavram notu (60 başlık) | Alanın zihinsel haritası; Faz 1'in omurgası |
| Tatar vd. 2026 (SIU) — OpenVLA'da Türkçe komut | 4 sayfalık IEEE bildirisi | Türkçe + VLA çizgisinin tek mevcut örneği; metodolojisi tartışmaya açık |
| Li, F. 2025 — Leeds PhD tezi (167 s.) | LLM'lerde spatial reasoning benchmark'ları: StepGame, RoomSpace | "Dil ≠ dünya" fikri; mekânsal ilişkilerin (sağ/sol/ön/arka/üstünde/içinde) nasıl benchmark'landığı |
| Hallac ↔ Cohn e-postaları | Türkçe spatial reasoning değerlendirmesi fikri + 7 arXiv makalesi | Tezin muhtemel ekseni: Türkçe dilde mekânsal/eylemsel anlama |
| Slayt fotoğrafı (Pat Grady alıntısı) | "Language is a compression of the world, not the world" | Tezin motivasyon cümlesi |
| RoomSpace Figure 1 | Top-down + north-facing görünüm, story + soru | Benchmark örneği; LIBERO'ya taşınabilecek ilişki türleri |
| `DravenALG/VLANeXt` (ICML 2026) | Kod: VLA eğitimi + LIBERO / LIBERO-Plus eval | Faz 2–3'ün pratik aracı |
| `DravenALG/awesome-vla-wam` | Okuma listesi: VLA, WAM, Policy, LAM, benchmark, donanım | Literatür haritası |
| paperswithcode.co/methods/vla | 2025–26 VLA makaleleri | Güncel yön: WAM, JEPA, RL-for-VLA |

**Olası tez ekseni (henüz danışmanla netleştirilecek):** Türkçe (özellikle mekânsal) komutların VLA modellerinde grounding'i — LIBERO/LIBERO-Plus üzerinde ölçülmesi ve iyileştirilmesi.

---

## FAZ 1 — Kavramlar (Gün 1–8)

| Gün | Tarih | Konu | Kaynak | Çıktı |
|---|---|---|---|---|
| 1 | 5 Kas | Kaynak haritası + robot learning döngüsü (obs → policy → action) | Not §1–9 | Kaynakları kendi cümlelerinle 1'er satır tanımla |
| 2 | 6 Kas | VLA nedir, VLM–VLA farkı, OpenVLA mimarisi (SigLIP+DINOv2 → projector → LLaMA-2 7B → 256-bin action token, 7-DoF) | Not §10–14, Tatar §III.A | OpenVLA pipeline'ını elle çiz |
| 3 | 7 Kas | Benchmark ≠ simülatör: LIBERO (4 suite), LIBERO-Plus, MuJoCo/robosuite | Not §15–17, §22–24 | LIBERO suite tablosu (ne ölçer, kaç task) |
| 4 | 8 Kas | Language grounding, shortcut learning, OpenVLA-OFT ≠ OpenVLA | Not §18–21, §36, §49–51 | "Dili gerçekten kullanıyor mu?" için 5 deney fikri |
| 5 | 9 Kas | Tatar vd. derin okuma: deney düzeni, L2/cosine, tokenizasyon ("Kal"+"dır"), EBO %72 | Makale tamamı | Makalenin 3 güçlü, 3 zayıf yönü |
| 6 | 10 Kas | Spatial reasoning temelleri: niteliksel mekânsal ilişkiler (yön, mesafe, topoloji), "dil dünyanın sıkıştırmasıdır" | Li tezi Böl. 1, §2.2–2.3; slayt | Alıntıyı VLA'ya bağlayan 1 paragraf |
| 7 | 11 Kas | RoomSpace benchmark: 3D oda, bakış açısı, soru türleri, tutarlılık kontrolü | Li tezi Böl. 5; Figure 1 | RoomSpace ilişkilerinden LIBERO'ya taşınabilecek Türkçe komut listesi (10 adet) |
| 8 | 12 Kas | Haftalık sentez + awesome-vla-wam taraması (VLA / WAM / Policy / LAM; π0, OpenVLA-OFT, VLANeXt, DreamZero) | Repo README | 1 sayfalık kendi kavram haritan |

## FAZ 2 — Ortam ve ilk çalıştırma (Gün 9–16)

**Donanım kararı:** MacBook Pro M2 (32 GB, 19 çekirdek GPU). CUDA yok → iki katman:
- **Mac'te (lokal):** LIBERO simülasyonu (MuJoCo arm64 çalışır), HDF5 dataset inceleme, OpenVLA-7B tek-görüntü inference (bf16, MPS; 14 GB RAM; yavaş ama çalışır), tokenizer analizi.
- **Colab/Kaggle ücretsiz T4'te:** VLANeXt / OpenVLA-OFT ile LIBERO eval (flash-attn + CUDA gerekiyor; Mac'te çalışmaz). Az episode ile.
- Kurulum günleri **komut komut** rehberle gelir (Python/conda temel düzey).

| Gün | Tarih | Konu | Nerede | Çıktı |
|---|---|---|---|---|
| 9 | 13 Kas | Miniconda (arm64) + Python 3.10 env; `pip install mujoco`; ilk MuJoCo sahnesi | Mac | `python -c "import mujoco"` çalışıyor, bir sahne render |
| 10 | 14 Kas | LIBERO kurulumu (robosuite arm64 notlarıyla); env `reset()` / `step()` / görüntü kaydet | Mac | 1 episode frame'leri (png) |
| 11 | 15 Kas | LIBERO demo HDF5'i aç: obs, 7-DoF action, episode/trajectory; action dizisini çiz | Mac | Action grafiği |
| 12 | 16 Kas | Colab T4: VLANeXt kurulumu (TUTORIAL.md), HF checkpoint; `libero_bench_eval` 2–3 episode | Colab | İlk success rate sayısı + kurulum notları |
| 13 | 17 Kas | Eval kodunu oku: instruction nereden giriyor, action nasıl decode ediliyor, success nasıl hesaplanıyor | Mac (kod okuma) | Kod akış şeması |
| 14 | 18 Kas | OpenVLA-7B'yi Mac'te MPS ile yükle; tek görüntü + İngilizce prompt → 7-DoF action | Mac | İlk action vektörü |
| 15 | 19 Kas | Tatar deneyi mini reprodüksiyon: 5 komut EN/TR, L2 + cosine | Mac | Mini tablo |
| 16 | 20 Kas | Haftalık sentez; LIBERO-Plus'ı Colab'da ayrı env'de kur (asset indirme) | Colab | LIBERO-Plus çalışıyor |

## FAZ 3 — Deney tasarımı ve robustness (Gün 17–24)

| Gün | Tarih | Konu | Çıktı |
|---|---|---|---|
| 17 | 21 Kas | LIBERO-Plus perturbation türleri; language perturbation kodda nasıl uygulanıyor | Perturbation tablosu |
| 18 | 22 Kas | Deney: aynı task, instruction = doğru / boş / yanlış / Türkçe → success rate (10 episode) | 4 satırlık sonuç tablosu |
| 19 | 23 Kas | Sonuç analizi: shortcut learning var mı? Hangi suite'te? | 1 paragraf yorum |
| 20 | 24 Kas | Türkçe mekânsal komut seti tasarımı (sağına/soluna/önüne/üstüne/içine; ek farkları) | 20 komutluk EN–TR eşli set |
| 21 | 25 Kas | Tokenizasyon analizi: VLANeXt/LLaMA tokenizer ile Türkçe komutlar (token sayısı, anlamsız bölünmeler) | Tablo |
| 22 | 26 Kas | Cohn makaleleri (abstract düzeyi): 2401.03991 AAAI-24, 2405.15064 IJCAI-24, 2406.16528 COSIT-24, 2410.03492, 2507.12059 QR-25, 2511.02627 TMLR, 2605.18380 | Her biri için 2 cümle |
| 23 | 27 Kas | Çözüm yaklaşımları: çeviri katmanı vs çok dilli VLM backbone vs Türkçe LoRA fine-tune — artı/eksi/maliyet | Karşılaştırma tablosu |
| 24 | 28 Kas | Haftalık sentez | Faz 3 özeti |

## FAZ 4 — Proje üzerine akıl yürütme (Gün 25–30)

| Gün | Tarih | Konu | Çıktı |
|---|---|---|---|
| 25 | 29 Kas | 3 aday araştırma sorusu (RQ) formüle et | RQ listesi |
| 26 | 30 Kas | Her RQ için: veri, model, benchmark, metrik, hesaplama maliyeti | Deney matrisi |
| 27 | 1 Ara | Baseline + ablation tasarımı | Ablation planı |
| 28 | 2 Ara | 1 sayfalık tez önerisi taslağı | Taslak (TR) |
| 29 | 3 Ara | Danışman toplantısı için soru/sunum listesi | Soru listesi |
| 30 | 4 Ara | Final: "bir proje yapalım" simülasyonu — Claude proje önerir, sen tasarlarsın | Değerlendirme + sonraki 30 gün |

---

## Açık kararlar (netleşince buraya işlenecek)
- GPU: MacBook Pro M2 32 GB (CUDA yok) + ücretsiz Colab/Kaggle T4 (karar: 5 Kas)
- Tez konusu: henüz netleşmedi, kaynaklar keşif amaçlı → Faz 4'te RQ'lar birlikte üretilecek (5 Kas)
- Python/conda: temel düzey → kurulum günleri komut komut rehberli (5 Kas)
- Günlük çalışma saati: ?
