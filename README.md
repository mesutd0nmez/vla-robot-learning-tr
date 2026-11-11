# Robot Learning ve VLA — Yüksek Lisans Tez Günlüğü (Türkçe)

Bilgisayar Mühendisliği yüksek lisans tezimin (ALKU) **tüm çalışma sürecinin** açık kaydı:
kavramları öğrenmekten literatür taramasına, kurulumlardan deneylere, araştırma sorusundan tez yazımına kadar
her adım bu repoda Türkçe olarak tutulur.

Alan: **Vision-Language-Action (VLA)** modelleri, robot learning, dil ve mekânsal komutların robot davranışına
bağlanması (language grounding). Türkçe kaynak çok az olduğu için notlar, başkalarının da kullanabileceği
şekilde yazılır.

**Site:** https://mesutd0nmez.github.io/vla-robot-learning-tr/ — sol menüden derslere ulaşılır (GitHub Pages + Docsify; derleme yok, markdown tarayıcıda render edilir).

## Aşamalar

| Aşama | Kapsam | Durum |
|---|---|---|
| 0. Başlangıç — Hızlandırılmış giriş (Ders 1–30) | Kavramlar → ortam kurulumu → ilk deneyler → proje üzerine akıl yürütme (`plan/lecture-plan.md`) | 🟡 Ders 1–17 hazır |
| 1. Literatür ve konu seçimi | Tez ekseninin danışmanla netleştirilmesi, kaynak taraması | ⬜ |
| 2. Deney altyapısı | Benchmark, model, metrik ve veri setinin kurulması | ⬜ |
| 3. Deneyler | Ana deneyler, ablasyonlar, sonuç analizi | ⬜ |
| 4. Tez yazımı ve savunma | Tez metni, makale, sunum | ⬜ |

Aşamalar netleştikçe bu tablo ve `plan/` klasörü güncellenir.

## Yapı

| Klasör | İçerik |
|---|---|
| `plan/` | Ders planı (`lecture-plan.md`) ve `daily-log.md` (her çalışma gününün durumu, görevi, çıktısı) |
| `lectures/` | Dersler `lecture-NN.md` (görseller `lectures/images/lecture-NN/`); her dersin sonunda kendini sına soruları ve cevapları |
| `resources/` | Kaynak listesi ve Türkçe giriş notları (telifli PDF'ler yerine bağlantılar) |
| `notes/` | Konu bazlı birikimli notlar (OpenVLA, LIBERO, spatial reasoning, tokenizasyon…) |
| `projects/` | Elle yürütülen demo, deney ve araştırma projeleri (klasör başına bir proje, `_template` şablonu) |
| `datasets/` | Veri kartları ve küçük veri dosyaları (Türkçe komut setleri, sonuç tabloları; dış veriler yalnızca kartla) |
| `code/` | Kurulum scriptleri, deney kodları, Colab notebook'ları (her klasörde README ile kaynak kodu) |
| `index.html`, `_sidebar.md`, `assets/` | Dokümantasyon sitesi (Docsify); `tools/gen_sidebar.py` menüyü üretir |

İleride eklenecek: `experiments/` (deney kayıtları ve sonuçlar), `thesis/` (tez metni taslakları), `slides/`.

## Nasıl çalışıyorum

- Günde en fazla 1 saat. Her gün bir konu, bir küçük çıktı. Ders notları Claude tarafından araştırılıp yazılır;
  çıktılar ve cevaplar benim.
- Her ders aynı iskelette: ~35 dk okuma (görsellerle) → "Kendini sına" açık uçlu sorular (cevaplar dersin
  sonunda) → çoktan seçmeli sorular (tek/çok seçimli; onay kutuları + açılır cevap) → kısa görev.

## Dersler

| # | Konu | Faz |
|---|---|---|
| [1](lectures/lecture-01.md) | Robot learning döngüsü ve kaynak haritası | 1 |
| [2](lectures/lecture-02.md) | OpenVLA'nın içi: görüntü + dil → action | 1 |
| [3](lectures/lecture-03.md) | Benchmark ≠ simülatör: LIBERO, MuJoCo, LIBERO-Plus | 1 |
| [4](lectures/lecture-04.md) | Language grounding ve shortcut learning | 1 |
| [5](lectures/lecture-05.md) | Makale derin okuma: Tatar vd. (2026) | 1 |
| [6](lectures/lecture-06.md) | Mekânsal akıl yürütmenin temelleri | 1 |
| [7](lectures/lecture-07.md) | RoomSpace → LIBERO | 1 |
| [8](lectures/lecture-08.md) | Hafta 1 sentezi ve VLA model aileleri | 1 |
| [9](lectures/lecture-09.md) | Mac: conda + MuJoCo + ilk sahne | 2 |
| [10](lectures/lecture-10.md) | Mac: LIBERO kurulumu + ilk episode | 2 |
| [11](lectures/lecture-11.md) | LIBERO HDF5 demo verisi | 2 |
| [12](lectures/lecture-12.md) | Colab T4'te VLANeXt: ilk success rate | 2 |
| [13](lectures/lecture-13.md) | Kod okuma: komut → action → success zinciri | 2 |
| [14](lectures/lecture-14.md) | OpenVLA-7B Mac'te (MPS): görüntü + komut → 7 token → action | 2 |
| [15](lectures/lecture-15.md) | Tatar mini-reprodüksiyonu: 5 kare × 6 koşul, L2 + cosine | 2 |
| [16](lectures/lecture-16.md) | Faz 2 sentezi + LIBERO-Plus: ayrı runtime, komutun kod yolu, boş-komut denemesi | 2 |
| [17](lectures/lecture-17.md) | LIBERO-Plus'ın yedi bozulması kodda: üç taşıyıcı, kamera matematiği, 500 Panda, R1/R2, Türkçe varyant | 3 |
| [18](lectures/lecture-18.md) | Aynı görev, dört komut: clean / empty / wrong / tr × 10 init state — deney tasarımı, Ders 18 yaması, Wilson + güç hesabı | 3 |
- Her ders `lectures/lecture-NN.md` olarak, `plan/daily-log.md` ile birlikte commit'lenir; bir günde birden fazla ders çalışılabilir; böylece kaldığım yer hiçbir zaman kaybolmaz.
- Takılınca plan kayar, atlanmaz.

## Donanım

MacBook Pro M2 (32 GB) — simülasyon (LIBERO/MuJoCo) ve OpenVLA inference lokal (MPS); CUDA gerektiren
eval ve eğitimler Colab/Kaggle T4 ya da kiralık GPU'da.

## Kaynaklar

Bkz. [`resources/README.md`](resources/README.md).

## Lisans

Notlar ve kod: MIT. Üçüncü taraf makale ve tezler bu repoya dahil edilmemiştir; yalnızca bağlantı verilmiştir.
