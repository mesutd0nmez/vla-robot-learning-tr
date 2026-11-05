# Veri seti adı

**Tür:** kendi ürettiğim / dış · **Sürüm:** v0.1 · **Tarih:** — · **Lisans:** — · **İlgili proje/ders:** —

## Ne içeriyor
Bir paragraf: kaç örnek, hangi diller, hangi görevler.

## Kaynak ve elde etme
- Dış veri: kaynak bağlantısı, sürüm/commit, indirme komutu, dosya boyutu, `sha256sum` çıktısı.
- Kendi verim: nasıl üretildi (elle yazım / çeviri / script), kim üretti, doğrulama (ikinci okuma, tokenizer kontrolü).

## Şema
| Alan | Tip | Açıklama | Örnek |
|---|---|---|---|
| id | str | benzersiz kimlik | `spatial_03_tr_a` |
| suite | str | LIBERO suite | `libero_spatial` |
| task_id | int | görev indeksi | 3 |
| lang | str | `en` / `tr` | `tr` |
| instruction | str | komut metni | `siyah kâseyi tabağın üstüne koy` |
| relation | str | mekânsal ilişki (EC/NTPP/1-cross/mesafe/arasına) | `EC` |
| frame | str | bakış açısı: `camera` / `robot` / `n/a` | `n/a` |
| suffixes | str | Türkçe ekler, `;` ile | `-ın;üst-ü-ne` |
| variant | str | parafraz varyantı | `a` |

## Uygulanan dönüşümler
Görüntü döndürme, gripper işareti, normalizasyon, filtreleme — sırayla, script adıyla.

## Bilinen sınırlılıklar
Örnek sayısı, bakış açısı belirsizliği, tek çevirmen vb.

## Kullanım
```python
import pandas as pd
df = pd.read_csv("data/commands.csv")
```

## Değişiklik günlüğü
- v0.1 — ilk sürüm
