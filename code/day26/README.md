# day26

Ders 26 — Faz 4 başlangıcı: üç aday araştırma sorusu. Model yok; üç script: RQ ölçütünün güç/eşdeğerlik hesabı, RQ taslağının "definition of done" kontrolü, görseller. Çıktı dosyası `thesis/rq-candidates.md`.

| Dosya | Ne | Durum |
|---|---|---|
| `rq_power.py` | Eşli tasarımda (aynı init state, iki koşul) McNemar exact ile %80 güçle yakalanan en küçük fark (MDE) ve TOST eşdeğerlik için %90 CI yarı genişliği / gereken n; `--png` ile görsel 02 | çalıştırıldı (`results/rq_power.txt`) |
| `rq_lint.py` | `thesis/rq-candidates.md`'de her `## RQ-x` bölümünde 7 alan (Soru + 6), H0 etiketi, sayısal eşik, evet/hayır dalları, sıralama tablosu ve tahmin bloğu var mı; `--selftest` bozuk taslakta 7 eksiği yakalar; `--csv` matris | çalıştırıldı: 3 RQ × 7 alan **TAM**, selftest OK |
| `plot_day26.py` | Görsel 03 (deney uzayı: blok × koşul → RQ; blok sayıları `pairs.csv`'den) ve 05 (literatür haritası) | çalıştırıldı |
| `results/rq_power.txt`, `results/rq_matrix.csv` | Gerçek çıktılar | üretildi |

Mac'te (`vla` env; yalnızca `numpy scipy matplotlib`):
```bash
cd code/day26
python rq_power.py                                  # ~1 dk (kesin multinom toplamları)
python rq_lint.py --selftest
python rq_lint.py ../../thesis/rq-candidates.md --csv results/rq_matrix.csv
python plot_day26.py
```
Çıkış kodu `rq_lint.py`: 0 = TAM, 1 = EKSİK, 2 = selftest başarısız.

## rq_power.py
```python
"""Ders 26 — RQ'ların ölçütünü sayıya bağlamak: eşli tasarımda (aynı init state, iki koşul)
(a) kaç init state ile hangi büyüklükte fark yakalanır (McNemar exact, kesin güç),
(b) "fark YOK" demek için (TOST / eşdeğerlik) kaç init state gerekir.

  python rq_power.py                 # tabloyu yazdırır
  python rq_power.py --png <klasör>  # 02-mde-tost.png üretir

Model (Ders 18 ile aynı mantık, biraz daha gerçekçi):
  Aynı init state'te iki koşul koşulur. Uyumsuz çift olasılıkları
    P(clean başarır, x başaramaz) = Δ + r        (Δ = gerçek fark, r = "ters" gürültü oranı)
    P(clean başaramaz, x başarır) = r
  b, c bu iki olasılıkla multinom dağılır; McNemar exact p-değeri b+c uyumsuz çift içinde
  iki yönlü binom(0,5) testidir. Güç = p < 0,05 çıkan (b, c) çiftlerinin toplam olasılığı — kesin hesap.
  MDE = %80 güç veren en küçük Δ.
  Eşdeğerlik: TOST %90 güven aralığının yarı genişliği (eşli oranlar farkı için Wald SE =
  sqrt(b + c − (b − c)²/n)/n; planlamada b = c = q·n/2, q = toplam uyumsuzluk oranı) → yarı genişlik
  ≈ 1,645·sqrt(q/n). Eşdeğerlik sınırı δ için gereken n ≈ 1,645²·q/δ².
Kaynak: McNemar exact (klasik); TOST: Lakens 2017 (SPPS 8:355) — yöntem; sayılar bu script'in hesabı.
"""
import argparse, math, os
import numpy as np
from scipy.stats import binom, multinomial

ALPHA = 0.05
Z90 = 1.6449  # TOST için iki tek yönlü %5 test = %90 CI


def mcnemar_p(b, c):
    """İki yönlü exact McNemar p-değeri (uyumsuz çiftler içinde binom(0,5))."""
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * binom.cdf(k, m, 0.5))


def power_paired(n, delta, r):
    """Kesin güç: (b, c) multinom(n; Δ+r, r, 1−Δ−2r) üzerinden toplam."""
    p_b, p_c = delta + r, r
    p_rest = 1 - p_b - p_c
    if p_rest < 0:
        return float("nan")
    total = 0.0
    for b in range(n + 1):
        for c in range(n - b + 1):
            if mcnemar_p(b, c) < ALPHA:
                total += multinomial.pmf([b, c, n - b - c], n, [p_b, p_c, p_rest])
    return total


def mde(n, r, target=0.8, grid=np.arange(0.01, 0.96, 0.01)):
    for d in grid:
        if power_paired(n, float(d), r) >= target:
            return float(d)
    return float("nan")


def tost_halfwidth(n, q):
    return Z90 * math.sqrt(q / n)


def n_for_equivalence(delta, q):
    return math.ceil(Z90 ** 2 * q / delta ** 2)
# (tablo ve --png bölümleri dosyada; görsel 02'yi üretir)
```

## rq_lint.py
```python
"""Ders 26 — RQ taslağının "definition of done" kontrolü (ön kayıt linter'ı).

  python rq_lint.py ../../thesis/rq-candidates.md          # tablo + çıkış kodu (0 = tam, 1 = eksik)
  python rq_lint.py --selftest                              # bozuk bir taslakla eksikleri yakaladığını gösterir
  python rq_lint.py <dosya> --csv out.csv                   # matris CSV

Kural: her `## RQ-<harf>` bölümünde altı alan bulunmalı — **Soru**, **Null hipotez**, **Değişkenler**,
**Ölçüt ve eşik**, **Suite / blok / örneklem**, **Sistemler**, **Evet / hayır çıkarsa tez ne der**.
Ek kontroller: null hipotez bölümünde en az bir "H0" etiketi; ölçüt bölümünde bir sayısal eşik (δ/Δ/puan);
"Evet / hayır" bölümünde hem evet hem hayır dalı; sıralama tablosu var mı; tahmin bloğu var mı.
"""
import argparse, csv, re, sys

FIELDS = ["Soru", "Null hipotez", "Değişkenler", "Ölçüt ve eşik", "Suite / blok / örneklem", "Sistemler",
          "Evet / hayır çıkarsa tez ne der"]


def split_rqs(text):
    parts = re.split(r"^## (RQ-[a-z]) — (.+)$", text, flags=re.M)
    rqs = {}
    for i in range(1, len(parts), 3):
        body = parts[i + 2]
        body = re.split(r"^## ", body, flags=re.M)[0]
        rqs[parts[i]] = (parts[i + 1].strip(), body)
    return rqs


def field_text(body, field):
    m = re.search(r"\*\*" + re.escape(field) + r":?\*\*:?\s*(.+?)(?=\n\n\*\*|\n\n---|\Z)", body, flags=re.S)
    return m.group(1).strip() if m else None


def check(text):
    rows, problems = [], []
    rqs = split_rqs(text)
    if not rqs:
        problems.append("hiç `## RQ-x — …` bölümü yok")
    for rq, (title, body) in rqs.items():
        row = {"rq": rq, "title": title}
        for f in FIELDS:
            t = field_text(body, f)
            row[f] = len(t.split()) if t else 0
            if not t:
                problems.append(f"{rq}: '{f}' alanı yok")
        null = field_text(body, "Null hipotez") or ""
        if not re.search(r"H0", null):
            problems.append(f"{rq}: null hipotezde H0 etiketi yok")
        crit = field_text(body, "Ölçüt ve eşik") or ""
        if not re.search(r"(δ|Δ|κ)\s*[=≥<>]|\d+\s*puan|p\s*<\s*0,05", crit):
            problems.append(f"{rq}: ölçütte sayısal eşik yok")
        yn = field_text(body, "Evet / hayır çıkarsa tez ne der") or ""
        if not (re.search(r"\*Evet", yn) or re.search(r"etkisi var|reddi|κ ≥", yn)) or not (
                re.search(r"\*Hayır", yn) or re.search(r"Yok|reddedilemedi|κ <", yn)):
            problems.append(f"{rq}: evet/hayır dallarından biri eksik")
        row["H0_sayisi"] = len(set(re.findall(r"H0-[a-z]\d", null)))
        rows.append(row)
    if "## Sıralama kuralı" not in text:
        problems.append("sıralama tablosu (`## Sıralama kuralı`) yok")
    if "Önceden yazılan tahminler" not in text:
        problems.append("tahmin bloğu yok (ön kayıt için zorunlu)")
    return rows, problems
# (report / selftest / CSV bölümleri dosyada)
```

## plot_day26.py
```python
"""Ders 26 — görsel 03 ve 05.
03-rq-coverage.png   : Deney uzayı (blok × koşul) — hangi hücreyi hangi RQ ister; blok sayıları
                       datasets/04/data/pairs.csv'den GERÇEK (A 6 / B 6 / C 6 / D 2), tavan Ders 19'dan.
05-literature-map.png: Türkçe × VLA hücresinin literatürdeki yeri (konum = raporlanan özellikler; sayı değil).
"""
import csv, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

pairs = "../../datasets/04-tr-spatial-commands/data/pairs.csv"
blocks = {}
with open(pairs, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        b = row["block"]; blocks.setdefault(b, {"n": 0, "suite": row["suite"], "tier2": 0})
        blocks[b]["n"] += 1
        if row["tier"] == "2": blocks[b]["tier2"] += 1
# → {'A': 6, 'B': 6 (tier2 5), 'C': 6, 'D': 2}; devamı dosyada (03: blok × koşul ızgarası; 05: literatür haritası)
```
