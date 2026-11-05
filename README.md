# Robot Learning ve VLA — Yüksek Lisans Tez Günlüğü (Türkçe)

Bilgisayar Mühendisliği yüksek lisans tezimin (ALKU) **tüm çalışma sürecinin** açık kaydı:
kavramları öğrenmekten literatür taramasına, kurulumlardan deneylere, araştırma sorusundan tez yazımına kadar
her adım bu repoda Türkçe olarak tutulur.

Alan: **Vision-Language-Action (VLA)** modelleri, robot learning, dil ve mekânsal komutların robot davranışına
bağlanması (language grounding). Türkçe kaynak çok az olduğu için notlar, başkalarının da kullanabileceği
şekilde yazılır.

## Aşamalar

| Aşama | Kapsam | Durum |
|---|---|---|
| 0. Başlangıç — 30 günlük hızlandırılmış giriş | Kavramlar → ortam kurulumu → ilk deneyler → proje üzerine akıl yürütme (`plan/30-day-plan.md`) | 🟡 Gün 1 / 30 |
| 1. Literatür ve konu seçimi | Tez ekseninin danışmanla netleştirilmesi, kaynak taraması | ⬜ |
| 2. Deney altyapısı | Benchmark, model, metrik ve veri setinin kurulması | ⬜ |
| 3. Deneyler | Ana deneyler, ablasyonlar, sonuç analizi | ⬜ |
| 4. Tez yazımı ve savunma | Tez metni, makale, sunum | ⬜ |

Aşamalar netleştikçe bu tablo ve `plan/` klasörü güncellenir.

## Yapı

| Klasör | İçerik |
|---|---|
| `plan/` | Aşama planları (`30-day-plan.md`) ve `daily-log.md` (her çalışma gününün durumu, görevi, çıktısı) |
| `lectures/` | Günlük dersler `YYYY-MM-DD.md` (görseller `lectures/images/YYYY-MM-DD/`); her dosyanın sonunda sorular ve o günün çıktısı |
| `resources/` | Kaynak listesi ve Türkçe giriş notları (telifli PDF'ler yerine bağlantılar) |
| `notes/` | Konu bazlı birikimli notlar (OpenVLA, LIBERO, spatial reasoning, tokenizasyon…) |
| `code/` | Kurulum scriptleri, deney kodları, Colab notebook'ları |

İleride eklenecek: `experiments/` (deney kayıtları ve sonuçlar), `thesis/` (tez metni taslakları), `slides/`.

## Nasıl çalışıyorum

- Günde en fazla 1 saat. Her gün bir konu, bir küçük çıktı. Ders notları Claude tarafından araştırılıp yazılır;
  çıktılar ve cevaplar benim.
- Her ders aynı iskelette: ~35 dk okuma (görsellerle) → "Kendini sına" soruları (cevaplar dersin sonunda, önce
  bakmadan yazılır) → 15 dk'lık görev → dosyanın sonundaki "Çıktı" bölümü.
- Her günün sonunda `lectures/YYYY-MM-DD.md` ve `plan/daily-log.md` commit'lenir; böylece kaldığım yer hiçbir zaman kaybolmaz.
- Takılınca plan kayar, atlanmaz.

## Donanım

MacBook Pro M2 (32 GB) — simülasyon (LIBERO/MuJoCo) ve OpenVLA inference lokal (MPS); CUDA gerektiren
eval ve eğitimler Colab/Kaggle T4 ya da kiralık GPU'da.

## Kaynaklar

Bkz. [`resources/README.md`](resources/README.md).

## Lisans

Notlar ve kod: MIT. Üçüncü taraf makale ve tezler bu repoya dahil edilmemiştir; yalnızca bağlantı verilmiştir.
