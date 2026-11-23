<!-- Tez önerisi taslağı v0.1 — Ders 29. Bir sayfa: gövde ≤ 650 sözcük (Kaynaklar hariç). Her sayısal iddia bir etiket taşır: [Ders NN], [arXiv:…], [öneri] ya da [hesap]. Kontrol: python code/day29/proposal_lint.py thesis/proposal-v0.1.md -->

# Tez Önerisi (taslak v0.1)

**Başlık (TR):** Türkçe Mekânsal Komutların Görüntü-Dil-Eylem (VLA) Modellerinde Temellendirilmesi: Boş-Komut Tabanlı Kapalı-Döngü Bir Değerlendirme
**Title (EN):** Grounding Turkish Spatial Instructions in Vision-Language-Action Models: A Closed-Loop Evaluation Against an Empty-Instruction Baseline
**Öğrenci / Danışman:** Mesut Bahtiyar Dönmez / (danışman) · ALKÜ Lisansüstü Eğitim Enstitüsü, Bilgisayar Mühendisliği (tezli YL)

## 1. Problem ve önem

"Language is a beautiful compression of the world, but it is not the world" (P. Grady; danışmanın slaytı) [Ders 1]. VLA modelleri komutu dünyaya — kamera görüntüsüne ve robot eylemine — bağlamak için eğitilir, ama neredeyse yalnızca İngilizce veriyle. Çok dilli ölçümler sert düşüşler raporluyor: Korece komutla LIBERO-Spatial başarısı 97,6'dan 49,2'ye iniyor [arXiv:2606.11906]; OpenVLA-OFT'de yabancı dil kaybı (−27,8…−29,8) komutu hiç vermemekle (−27,5) aynı büyüklükte [arXiv:2606.15714]. Türkçe için tek çalışma açık döngüde, tek karede ölçüyor [Ders 5]. Bir modelin Türkçe "kâsenin solundaki" ifadesini gerçekten okuduğunu, boş komuttan ayrılmadan söyleyemeyiz.

## 2. Amaç

Türkçe mekânsal komutların iki açık VLA sisteminde kapalı döngüde ne kadar temellendirildiğini, boş-komut tabanına göre ölçmek; düşüşün nerede yoğunlaştığını bulmak ve girdi düzeyinde ucuz bir düzeltmenin açığı ne kadar kapattığını göstermek.

## 3. Araştırma soruları

- **RQ-a (fark var mı):** Aynı başlangıç sahnesinde Türkçe komut ile İngilizce komut arasında başarı farkı var mı; fark boş-komut tabanına göre nerede? [Ders 26]
- **RQ-b (nerede):** Düşüş ilişki ailesine, yüzeye (Türkçe harf, ek) ve referans çerçevesine bağlı mı; eğitimde olmayan Türkçe tarif boş komutun üstüne çıkıyor mu? [Ders 26]
- **RQ-c (nasıl kapanır):** Makine çevirisi ve kural tabanlı ayrıştırıcı açığın ne kadarını kapatıyor (kapanış oranı κ)? [Ders 26]

## 4. Yöntem

**Benchmark:** LIBERO-Spatial ve LIBERO-Goal; görev başına 50 başlangıç durumu [Ders 27]. **Veri:** 20 EN–TR eşli komut, dört blok (A eğitilmiş Spatial 6, B yeni tarif 6, C Goal 6, D 2) ve dört Türkçe yüzey [Ders 20]. **Sistemler:** S-OV = OpenVLA-7B LIBERO checkpoint'i (LLaMA-2 tokenizer; T4'te 4-bit) ve S-VN = VLANeXt (Qwen3-VL omurga) [Ders 27]; karşılaştırma sistem düzeyinde, nedensel değil [arXiv:2606.15714]. **Koşullar:** `clean`, `empty`, `wrong`, `tr` — aynı başlangıç durumunda eşli [Ders 18]. **Ölçüt:** eşli McNemar ve TOST eşdeğerlik; "fark var" için p < 0,05 ve en az 15 puan, eşikler koşudan önce yazılır (ön kayıt) [Ders 26]. Görev başına 50 durumla görünen en küçük fark 23 puan olduğundan ana iddia blok düzeyinde kurulur: 6 görev × 20 durum ile 13 puan [Ders 27]. Baseline ve ablation planı `thesis/ablation-plan.md`'de [Ders 28].

## 5. Beklenen katkılar

1. Türkçe mekânsal komut seti: eşli, blok ve yüzey etiketli, açık lisanslı [Ders 20].
2. Türkçe için ilk kapalı-döngü, boş-komut tabanlı VLA ölçümü; iki tokenizer'ın (Türkçe harf bedeli 4,9 ve 0,9 token) davranış karşılığı [Ders 22].
3. Ön kayıtlı, düşük bütçeli (ücretsiz T4) tekrarlanabilir değerlendirme protokolü ve kodu [Ders 27].

## 6. Takvim

Hesap birimi Kaggle T4 saati; yalın matris ≈ 108 saat, tam çekirdek ≈ 258 saat [hesap: Ders 27]; haftalık kota ≈ 30 saat (kullanıcı raporu) [Ders 27]. İş paketleri (hafta 1 = 30 Kasım 2026) [öneri]: İP1 pilot + kalibrasyon (hafta 1–2); İP2 RQ-a (hafta 2–5); İP3 RQ-b (hafta 5–7); İP4 RQ-c (hafta 7–8); İP5 analiz (hafta 8–10); İP6 yazım (hafta 9–16). Ara sınav: hafta 5'te RQ-a tablosu; son sınav: hafta 16'da tez taslağı [öneri].

## 7. Riskler ve B planları

| Risk | B planı |
|---|---|
| Model çağrısı tahminden yavaş (kötümser 497 saat) [Ders 27] | yalın matris; yalnız Spatial + Goal blok A/C |
| 4-bit S-OV yayımlanan 84,7'yi tutturmaz [Ders 27] | "OpenVLA (4-bit)" etiketi; Mac'te bf16 eşli alt küme |
| Spatial'da boş komut tavana yakın (komutsuz 9/10) [Ders 19] | ağırlık Goal'a (komutsuz 1/10) [Ders 19] |
| Karşı-olgusal BDDL yüklenmez | B bloğu betimsel, EN-B dış referansla [arXiv:2603.00592] |
| LoRA eğitimi onaylanmaz | RQ-c girdi düzeyinde (MT + ayrıştırıcı) kalır |

## Kaynaklar (kısa)

Kim vd. 2024, OpenVLA, arXiv:2406.09246 · Liu vd. 2023, LIBERO, arXiv:2306.03310 · Dong vd. 2026, arXiv:2606.11906 (ACL 2026) · Chen vd. 2026, arXiv:2606.15714 · Hou & Zhao 2026, LangGap, arXiv:2603.00592 · Tatar vd. 2026, SIU (IEEE) · VLANeXt: github.com/DravenALG/VLANeXt · Ayrıntı: `lectures/lecture-05…28`, `thesis/rq-candidates.md`, `thesis/experiment-matrix.csv`.

Sürüm: v0.1 — Ders 29 (otomatik taslak; danışmanla gözden geçirilecek). Değişiklikler buraya tarihle eklenir.
