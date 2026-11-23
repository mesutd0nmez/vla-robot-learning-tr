# Baseline + ablation planı — v0.1 (Ders 28)

Bu dosya `rq-candidates.md` (RQ'lar) ile `experiment-matrix.csv` (hücreler) arasındaki köprüdür: her RQ'nun sonucunu **neye göre** okuyacağımızı (baseline) ve hangi farkın **hangi tek faktöre** ait olduğunu (ablation) önceden yazar. Ayrıntı: `lectures/lecture-28.md`. Kod: `code/day28/`.

Kural: Bir karşılaştırma **ablation** sayılır ancak iki koşul arasında tek bir faktör değişiyorsa; birden fazla faktör birlikte değişiyorsa **betimsel karşılaştırma** diye yazılır ve nedensel cümle kurulmaz.

## Baseline'lar

| baseline | ne söyler | nereden | maliyet |
|---|---|---|---|
| Komutsuz tavan / 1/k | dil olmadan erişilebilecek en yüksek başarı (Spatial 9/10, Goal 1/10) | Ders 19, analitik | 0 |
| `empty` | dilsiz politikanın gerçek başarısı; "Türkçe okunmuyor" iddiasının referansı (H0-a2) | A1–A4, B5 | matriste |
| `wrong` | model dili izliyor mu (faithful) yoksa hedefi sahneden mi seçiyor (biased) | A1–A4 | matriste |
| `clean` | üst referans; Δ(tr − clean) | A1–A4 | matriste |
| EN-B | eğitimde olmayan İngilizce tarif; TR-B − EN-B = "Türkçe" bedeli, EN-B − clean = "yeni tarif" bedeli; dış referans LangGap Spatial Description %11 (π0.5) | B3/B4 | matriste |
| Yayımlanan sayılar | sistemlerin kalibrasyonu (S-OV Spatial 84,7 ± 0,9) | K1, K2 | ~4 sa |

## Ablation'lar

| id | soru (tek faktör) | karşılaştırma | tür | hücre | durum |
|---|---|---|---|---|---|
| ABL-harf | Türkçe harflerin bedeli | tr_a − tr_ascii (ekli) ve tr_bare − tr_bare_ascii (eksiz) | ablation (20/20 çiftte yalnız harf değişiyor) | A1–A4, B1–B2 (+ AB1) | hazır |
| ABL-ek | eklerin bedeli | **tr_ascii − tr_bare_ascii** (ASCII düzeyinde) | ablation — tr_a − tr_bare saf DEĞİL: 97 Türkçe harfin 66'sını da siliyor | AB1 (Goal C, çekirdek), AB2 (Spatial A, genişletme) | hazır |
| ABL-etkileşim | harf bedeli ekli sözcükte mi büyüyor (LLaMA-2 token: +4,3) | 2×2 etkileşimi | ablation, ama **betimsel güçte** (etkileşim için ~16× örneklem; Gelman) | AB1 | hazır |
| ABL-nesne | nesne adlarının Türkçe olmasının bedeli | tr_cs − tr_a (Türkçe iskelet, İngilizce nesne adı; Chen vd. Tablo 3) | ablation | AB3 (genişletme) | engelli: `tr_cs` yüzeyi yazılmadı (datasets/04 v0.3) |
| ABL-MT-kâhin | MT açığının ne kadarı çeviri hatası | tr_mt − clean (kâhin = insan çevirisi = `clean`, çünkü tr_a İngilizce'den üretildi) | yeniden okuma (yeni episode yok); S-OV greedy → episode düzeyinde kesin ayrıştırma | C1 + A1–A4 | C1 engelli (MT çıktısı) |
| ABL-parser-kâhin | parser açığı | tr_parse − clean | yeniden okuma; **A/C'de totoloji** (parser sete bakılarak yazıldı, 79/80 birebir) → asıl kanıt held-out | C2 | hazır |
| ABL-niceleme | 4-bit'in etkisi | S-OV bf16 (Mac) − S-OV 4-bit (T4), aynı 25 init | **iki faktör** (hassasiyet + cihaz/çekirdek) → "S-OV-4bit-T4 vs S-OV-bf16-MPS" diye yazılır | K3 | hazır |
| — sistem | omurga mı, action başı mı, veri mi | S-OV − S-VN | **betimsel** (üç faktör birlikte); parçalamak için StarVLA düzeni (aynı baş, iki omurga; Chen vd.) gerekir — bütçe dışı | A1–A4 | kapsam dışı |

## Bütçe (Ders 27 modeli, `code/day28/price_ablations.py`)

Matris v0.1 → v0.2: AB1 (çekirdek), AB2, AB3 (genişletme) eklendi; B6 yalnız `tr_b`'ye indirildi (`tr_bare` AB1/AB2'ye taşındı).

| matris | görev başına init | pilot + çekirdek T4 saati iyimser / orta / kötümser | Kaggle haftası (orta) | genişletme (orta) |
|---|---|---|---|---|
| v0.1 | 50 | 151 / 258 / 504 | 8,6 | 225 |
| v0.1 | 20 | 63 / 108 / 211 | 3,6 | 225 |
| v0.2 | 50 | 178 / 303 / 594 | 10,1 | 254 |
| v0.2 | 20 | 74 / 126 / 247 | 4,2 | 254 |

Önceden yazılan okuma kuralı: ABL-harf S-OV'de ≥ 15 puan ve S-VN'de |Δ| < 15 (TOST δ = 15) çıkarsa → "harf bedeli tokenizasyonla uyumlu" (Ders 22 hipotezinin kapalı-döngü kanıtı); iki sistemde de ≥ 15 → "tokenizasyon tek açıklama değil"; ikisinde de < 15 → "harfler önemli değil, düşüş başka yerde" (ek, sözcük, dil). ABL-ek aynı kuralla. Etkileşim yalnız yön kipinde raporlanır.

Sürüm: v0.1 — Ders 28 (otomatik taslak). Değişiklikler buraya tarihle eklenir.
