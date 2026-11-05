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
| 0. Başlangıç — 30 günlük hızlandırılmış giriş | Kavramlar → ortam kurulumu → ilk deneyler → proje üzerine akıl yürütme (`plan/30-gun-plani.md`) | 🟡 Gün 1 / 30 |
| 1. Literatür ve konu seçimi | Tez ekseninin danışmanla netleştirilmesi, kaynak taraması | ⬜ |
| 2. Deney altyapısı | Benchmark, model, metrik ve veri setinin kurulması | ⬜ |
| 3. Deneyler | Ana deneyler, ablasyonlar, sonuç analizi | ⬜ |
| 4. Tez yazımı ve savunma | Tez metni, makale, sunum | ⬜ |

Aşamalar netleştikçe bu tablo ve `plan/` klasörü güncellenir.

## Yapı

| Klasör | İçerik |
|---|---|
| `plan/` | Dönem/aşama planları ve `gunluk-log.md` (her çalışma gününün durumu, görevi, çıktısı) |
| `gunler/gun-NN/` | O günün ders notu (`ders.md`) ve kendi çıktım (`cikti.md`) |
| `kaynaklar/` | Kaynak listesi ve Türkçe giriş notları (telifli PDF'ler yerine bağlantılar) |
| `notlar/` | Konu bazlı birikimli notlar (OpenVLA, LIBERO, spatial reasoning, tokenizasyon…) |
| `kod/` | Kurulum scriptleri, deney kodları, Colab notebook'ları |

İleride eklenecek: `deneyler/` (deney kayıtları ve sonuçlar), `tez/` (tez metni taslakları), `sunumlar/`.

## Nasıl çalışıyorum

- Günde 30–60 dk. Her gün bir konu, bir küçük çıktı. Çalışma Claude ile birlikte yürütülür; ders notu ve plan
  oradan gelir, çıktılar benim.
- Her günün sonunda `gunler/gun-NN/` ve `plan/gunluk-log.md` commit'lenir; böylece kaldığım yer hiçbir zaman kaybolmaz.
- Takılınca plan kayar, atlanmaz.

## Donanım

MacBook Pro M2 (32 GB) — simülasyon (LIBERO/MuJoCo) ve OpenVLA inference lokal (MPS); CUDA gerektiren
eval ve eğitimler Colab/Kaggle T4 ya da kiralık GPU'da.

## Kaynaklar

Bkz. [`kaynaklar/README.md`](kaynaklar/README.md).

## Lisans

Notlar ve kod: MIT. Üçüncü taraf makale ve tezler bu repoya dahil edilmemiştir; yalnızca bağlantı verilmiştir.
