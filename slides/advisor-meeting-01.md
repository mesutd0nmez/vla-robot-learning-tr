# Danışman toplantısı 01 — gündem, sunum iskeleti, soru listesi, karar kaydı

Hazırlayan: Mesut Bahtiyar Dönmez · Taslak: Ders 30 (otomatik) · Toplantı tarihi: ____ · Süre: 30 dk
Ekler (e-postayla 2–3 gün önce): `thesis/proposal-v0.1.md` (Ders 29, 1 sayfa), bu dosya, `thesis/experiment-matrix.md`.
Soru kaynağı: `code/day30/questions.csv` (11 soru; her biri repoda dosya:satır izli, `code/day30/results/questions_raw.csv`).

---

## 1. Gündem (30 dk)

| dk | konu | slayt | tür |
|---|---|---|---|
| 0–3 | Durum: Aşama 0 bitti; boru hattı kurulu, kapalı-döngü Türkçe koşusu (S1) henüz yok | 1 | bağlam |
| 3–7 | Üç RQ ve deney matrisi; bütçe 108 / 258 T4 saati | 2–3 | bağlam |
| 7–12 | **Q1** ana suite (S7) + **Q2** iddia birimi | 4 | karar |
| 12–17 | **Q3** katkı ölçütü: zayıf mı güçlü okuma mı (S10) | 5 | karar |
| 17–21 | **Q4** Türkçe LoRA / GPU erişimi (S9) | 5 | karar |
| 21–24 | **Q11** tez önerisi takvimi ve formu | 7 | karar |
| 24–27 | Hızlı onay: **Q5** 4-bit OpenVLA, **Q6** ikinci okuyucu, **Q10** tekrar politikası | 6 | onay |
| 27–30 | Özet: her karar için sahip + tarih (aşağıdaki şablon) | — | kapanış |

Toplantı öncesi yazılı sorulacaklar (e-posta): **Q7** Webots rolü (S8), **Q8** Tatar vd. konumu, **Q9** Cohn hattı.

---

## 2. Sunum iskeleti (7 slayt + 1 yedek)

**Slayt 1 — Tez ekseni ve neredeyiz**
- Tek cümle: Türkçe mekânsal komutlar açık VLA'larda (OpenVLA, VLANeXt) aynı sahnede ne kadar düşüyor, neden, nasıl kapanır.
- Hazır olanlar: LIBERO / LIBERO-Plus boru hattı, T4 notebook'ları, 80 satırlık TR–EN mekânsal set (`datasets/04`), tokenizasyon analizi.
- Eksik: ilk kapalı-döngü dört-koşul tablosu (S1); model koşusu yapılmadı.
- Görsel: Ders 25 görsel 01 (Faz 3 haritası).

**Slayt 2 — Üç araştırma sorusu**
- RQ-a fark var mı (tr vs clean vs empty; iki sistem).
- RQ-b nerede kırılıyor (ilişki ailesi, yüzey, çerçeve; B bloğu).
- RQ-c nasıl kapanır (MT, parser, LoRA; zayıf/güçlü okuma).
- Görsel: Ders 26 görsel 03 (deney uzayı).

**Slayt 3 — Deney matrisi ve bütçe**
- 19 hücre; pilot + çekirdek ≈ 8.400 episode.
- Orta senaryo 258 T4 saati; iddia bloğa taşınırsa 108 saat (T4×2 ile ~2 Kaggle haftası).
- S-OV maliyetin %63'ü (chunk 1, T4'te 4-bit).
- Görsel: Ders 27 görsel 03 (hücre başına saat).

**Slayt 4 — Karar: suite ve iddia birimi (Q1, Q2)**
- Spatial: CDC ilişkileri ama komutsuz tavan 9/10 (shortcut riski); Goal: tavan 1/10, Türkçe token tarafı temiz.
- Görev başına 50 init: tek görevde en küçük görünen fark 23 puan; 6 görevlik blokta 13 puan (n=120).
- Önerimiz: A + C blokları, iddia blok düzeyinde.
- Görsel: Ders 27 görsel 04 (tersine güç).

**Slayt 5 — Karar: katkı ölçütü ve LoRA (Q3, Q4)**
- Zayıf okuma: Türkçe komutla doğru iş (MT/parser yeter). Güçlü okuma: eğitimde olmayan Türkçe tarifte `empty` üstü.
- Güçlü okuma kapatmak LoRA ister: 65–98 T4 saati (tahmin, 4-bit).
- Önerimiz: zayıf taban + güçlü hedef; bölüm GPU'su varsa C3 paralel.
- Görsel: Ders 30 görsel 01 (karar ağacı).

**Slayt 6 — Riskler ve hızlı onaylar (Q5, Q6, Q10)**
- OpenVLA T4'te yalnız 4-bit → K1/K3 kalibrasyonu; tezde "OpenVLA (4-bit)".
- Türkçe set tek yazarlı → ikinci okuyucu + uyum (κ), 20 satır.
- Tekrar yalnız pilotta (S-OV greedy; tekrar bağımsız örnek değil).
- Görsel: Ders 27 görsel 05 (nerede koşulur).

**Slayt 7 — Takvim ve sonraki 4 hafta**
- Hafta 1: S1 (P1) + K1/K2 → gerçek çağrı süresi → bütçe güncellenir.
- Hafta 2–3: birincil RQ çekirdek hücreleri; Hafta 4: ilk tablo + danışman toplantısı 02.
- Soru: tez önerisi teslim tarihi ve enstitü formu (Q11).
- Görsel: Ders 29 Gantt (proposal-v0.1).

**Yedek — Literatürdeki konum**
- Türkçe × kapalı döngü × boş-komut tabanı hücresi boş; Tatar vd. açık döngü tek kare.
- Çok dilli VLA (Chen 2606.15714, Dong 2606.11906) Türkçe içermiyor.
- Görsel: Ders 26 görsel 05 (literatür haritası).

---

## 3. Soru listesi (öncelik sırasıyla; etki × aciliyet, 1–5 öznel)

| # | soru | bağlam | seçenekler | önerimiz | neyi değiştirir | öncelik |
|---|---|---|---|---|---|---|
| Q1 | Ana suite Spatial mı, Goal mı, ikisi mi? (S7) | Spatial tavanı 9/10, Goal 1/10 | Spatial / Goal / ikisi | ikisi (A + C); Spatial'da `empty` ≥ 80 ise Goal ağırlıklı | A1–A4, B1–B2 | 25 |
| Q2 | İddia birimi görev mi blok mu? | görev MDE ≥ 23, blok n=120 → 13 | görev (258 sa) / blok (108 sa) | blok; görev Δ'ları "yön" kipinde | bütçe ×2,4, TOST eşikleri | 25 |
| Q3 | Katkı ölçütü zayıf mı güçlü okuma mı? (S10) | güçlü okuma = B bloğu + karşı-olgusal BDDL | zayıf / güçlü / ikisi | zayıf taban + güçlü hedef | RQ-c H0-c3, B3/B4/C4 | 20 |
| Q11 | Tez önerisinin takvimi ve formu? | repoda yazılı değil | — | toplantıda öğren | proposal-v0.1 biçimi | 16 |
| Q4 | LoRA yapılsın mı; GPU erişimi var mı? (S9) | 65–98 T4 sa tahmini | eval-only / Kaggle 4-bit / üniversite GPU | önce eval-only; GPU varsa C3 paralel | C3/C4 ≈ 150 sa | 12 |
| Q5 | OpenVLA 4-bit kabul mü? | bf16 T4'e sığmaz | 4-bit + K1/K3 / Mac bf16 / S-OV çıkar | 4-bit + kalibrasyon | sistem etiketi | 12 |
| Q6 | Türkçe set için ikinci okuyucu? | tek yazarlı; `human_check.csv` boş | tek / iki + κ / anket | iki okuyucu, 20 satır | `datasets/04` v0.2, H0-b4 | 9 |
| Q10 | Tekrar yalnız pilotta mı? | S-OV greedy | her hücre 3 / pilotta / hiç | pilotta | rq-candidates v0.2 | 6 |
| Q7 | Webots rolü? (S8) | Ders 21 karar notu + 3 soru | LIBERO / Webots küçük set / ikisi | LIBERO; Webots koşullu | yeni hücre grubu | 6 |
| Q8 | Tatar vd. ile ilişki? | açık döngü, null yok | genişletme / bağımsız / ortak | kapalı-döngü genişletmesi | giriş bölümü | 6 |
| Q9 | Cohn hattı tezin parçası mı? | metin LLM'leri, altın etiket çözücüden | parça / yan makale / dışı | yan ürün | literatür, ek makale | 4 |

---

## 4. Karar kaydı şablonu (ADR tarzı; toplantı bitmeden doldurulur)

```
### ADR-001 — <karar başlığı>            (soru: Q_)
Tarih: ____   Durum: kabul / ret / ertelendi
Karar: <tek cümle>
Gerekçe: <danışmanın gerekçesi, kendi sözcükleriyle>
Sonuç: <hangi dosya / hücre / RQ değişiyor — ör. experiment-matrix.csv A2 n_init 50 → 20>
Sahibi: Mesut   Son tarih: ____
Açık kalan: <varsa>
```

Toplantı sonrası özet e-postası (aynı gün): kararların tek satırlık listesi + "yanlış anladığım bir şey varsa düzeltir misiniz?" + bir sonraki toplantı tarihi önerisi.

Sürüm: v0.1 — Ders 30 (otomatik taslak). Mesut toplantıdan önce düzenler.
