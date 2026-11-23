# Deney matrisi — v0.1 (Ders 27)

Makine tarafından okunan dosya: `experiment-matrix.csv` (19 hücre). Bütçe: `python code/day27/budget.py` (orta senaryo pilot + çekirdek ≈ 258 T4 saati; yalın matris, görev başına 20 init ≈ 108 saat). Ayrıntı ve varsayımlar: `lectures/lecture-27.md`.

Hücre = sistem × suite × blok × koşul kümesi × görev başına init × tekrar. `tier`: pilot / core (çekirdek) / ext (genişletme). `status`: hazır ya da engelli (ön koşul yazılı).

Açık kararlar (Mesut / danışman):
- **İddia birimi:** RQ-a ana iddiası görev düzeyinde mi (n_init 50, MDE 23) blok düzeyinde mi (n_init 20, havuz n=120, MDE 13)?
- **Tekrar:** çekirdekte 1, PI yalnız pilotta (P1) — `rq-candidates.md` v0.2'ye girecek değişiklik önerisi.
- **S-OV = 4-bit NF4** (T4); K1 (yayımlanan 84,7 ile) ve K3 (Mac bf16 ile eşli) kalibrasyonları zorunlu.

İddia birimi kararı ve kota (Mesut doldurur):
```
iddia birimi: görev / blok   ; n_init (çekirdek): __   ; budget.py orta toplam: ___ sa ; Kaggle kotam: ___ sa/hafta
```

Sürüm: v0.1 — Ders 27 (otomatik taslak). Değişiklikler buraya tarihle eklenir.
