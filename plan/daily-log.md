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
