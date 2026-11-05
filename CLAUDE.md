# CLAUDE.md — bu repoda çalışırken

Bu repo Mesut Bahtiyar Dönmez'in yüksek lisans tez günlüğüdür (ALKU, Bilgisayar Mühendisliği): Robot Learning,
Vision-Language-Action (VLA) modelleri, LIBERO benchmark'ı ve Türkçe mekânsal komutların grounding'i.
Türkçe açık kaynak; site: https://mesutd0nmez.github.io/vla-robot-learning-tr/

## Önce oku
1. `tools/LECTURE_GUIDE.md` — ders formatı ve tüm kurallar (zorunlu)
2. `plan/lecture-plan.md` — Ders 1–30 konu tablosu ve kaynak haritası
3. `plan/daily-log.md` — ilerleme; Mesut'un notları/soruları buraya düşer
4. Son ders: `lectures/lecture-NN.md` (en büyük NN)

## Yapı
- `lectures/lecture-NN.md` — dersler (sıra numaralı, tarihsiz); görseller `lectures/images/lecture-NN/`
- `code/dayNN/` — o dersin script'leri + README.md (her script `## ad.py` + python bloğu)
- `projects/NN-kisa-ad/` — Mesut'un yerelde Claude Code ile yaptığı demo/deney/araştırma projeleri; `projects/_template/README.md` şablonuyla başlanır, `projects/README.md` tablosuna satır eklenir. Dersler burayı değiştirmez; projeler derslere atıf yapabilir.
- `datasets/NN-kisa-ad/` — veri kartı (`dataset-card.md`, zorunlu) + küçük veri dosyaları (`data/`, <10 MB). Kendi ürettiğimiz Türkçe komut setleri ve sonuç tabloları burada; dış veriler (LIBERO HDF5, checkpoint) yalnızca kartla (kaynak, sürüm, lisans, sha256, dönüşümler).
- `resources/`, `notes/`, `plan/`; ileride `thesis/`, `slides/`
- Site: `index.html` + `assets/site.css` (Docsify, derleme yok), `home.md` (ana sayfa), `_sidebar.md` (üretilir)

## Otomasyon
- Her sabah 06:51 (TR) bulutta zamanlanmış görev sıradaki dersi yazıp `main`'e push eder.
  Yerelde çalışmaya başlamadan önce **`git pull`**; aynı ders numarasını iki kez yazma.
- Yeni ders/klasör ekledikten sonra: `python tools/gen_sidebar.py`, `home.md` ızgarasına kart, `README.md` tablosuna satır.

## Yerelde proje yaparken (Claude Code)
- `git pull` → `cp -r projects/_template projects/NN-kisa-ad` → README'yi doldur → kod/sonuç/görsel aynı klasöre → `projects/README.md` tablosunu güncelle → `python tools/gen_sidebar.py` → commit/push.
- Her deney koşusu `results/runs.csv`'ye bir satır (run_id, model, checkpoint, suite, task_id, lang, condition, n_episodes, success_rate). Üretilen/kullanılan her veri için `datasets/` altında kart aç.
- Büyük dosyalar (checkpoint, HDF5, video) repoya girmez (`.gitignore`); sonuçları küçük CSV/PNG olarak kaydet.

## Çalışma kuralları
- Dil Türkçe, teknik terimler İngilizce; Mesut'a "sen". Her iddia kaynaklı (arXiv/GitHub bağlantısı).
- Ders ≤ 1 saat; görsel zorunlu (Mermaid `.mmd` + PNG, matplotlib); çalıştırılabilen şey gerçekten çalıştırılır.
- Donanım: MacBook Pro M2 32 GB (CUDA yok, MPS var) + Colab/Kaggle T4. Pinler: Python 3.10, `numpy<2`, `mujoco==2.3.7`, `robosuite==1.4.1`.
- Telifli PDF/DOCX repoya girmez (`.gitignore`); yalnızca bağlantı.
- Commit: `Ders NN: <konu>` ya da kısa Türkçe açıklama; doğrudan `main`'e push serbest (Mesut'un izni var).

## Tez ekseni (aday — danışmanla netleşmedi)
Türkçe mekânsal komutların VLA'larda grounding'i: OpenVLA (LLaMA-2 tokenizer) vs VLANeXt (Qwen3-VL) karşılaştırması,
LIBERO Spatial/Goal + LIBERO-Plus, success rate + boş-komut tabanı (doğru/boş/yanlış/parafraz/Türkçe koşulları).

## Yerel site önizleme
`python3 -m http.server 8765` → http://localhost:8765
