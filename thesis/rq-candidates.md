# Aday araştırma soruları (RQ) — taslak v0.1 (Ders 26)

Üç aday soru; her biri aynı altı alanla yazılır. Bu dosya bir **ön kayıt** (pre-registration) taslağıdır: sayılar gelmeden önce soru, null hipotez, ölçüt ve eşikler buraya yazılır ve commit edilir; sonuç geldikten sonra eşik oynatılmaz, oynatılırsa tarih + gerekçeyle yazılır. Sıralama **koşulludur**: Ders 18 tablosu (S1) repoya girince `plan/faz3-open-questions.md`'deki örüntüye göre birincil RQ seçilir (Ders 26 §6 ağacı).

Kısaltmalar: `clean` = orijinal İngilizce komut, `empty` = boş komut, `wrong` = aynı sahnede başka hedefi tarif eden İngilizce komut, `tr_*` = `datasets/04` yüzeyleri (`tr_a` üretilmiş, `tr_b` elle, `tr_ascii` Türkçe harfsiz, `tr_bare` eksiz), blok A/B/C/D = `datasets/04/data/pairs.csv` (A: Spatial eğitilmiş 6, B: aynı sahnede eğitimde olmayan tarif 6, C: Goal eğitilmiş 6, D: eğitilmemiş yerleştirme 2). Sistemler: **S-OV** = `openvla/openvla-7b-finetuned-libero-spatial` (LLaMA-2 tokenizer, 256-bin action token), **S-VN** = VLANeXt-B (Qwen3-VL-2B omurga, flow-matching action başı). İki sistem arasındaki fark "omurga farkı" değil **sistem farkı**dır (omurga + action başı + eğitim verisi birlikte değişir; Chen vd. 2606.15714).

Ortak istatistik protokolü (Ders 18, 23): her hücre eşli (aynı init state'ler, aynı süreç); `n_koşu × n_episode` yazılır; koşu içi Wilson %95, koşudan koşuya tahmin aralığı (PI, ≥3 tekrar); eşli fark için McNemar exact; "fark yok" iddiası için TOST (%90 CI, önceden yazılmış δ). Eşikler (`code/day26/rq_power.py` ile hesaplandı): n=50 eşli init state, ters gürültü r=0,05 iken %80 güçle yakalanan en küçük fark ≈ 23 puan; Δ=15 için n≈100; eşdeğerlik δ=15 için n≈25, δ=10 için n≈55 (q=0,20 uyumsuz çift).

---

## RQ-a — Fark var mı?

**Soru:** LIBERO'da aynı başlangıç sahnesinde, Türkçe komut (`tr_a`) ile orijinal İngilizce komut (`clean`) arasında kapalı-döngü success rate farkı var mı; varsa bu fark boş komut tabanına (`empty`) göre nerede duruyor — ve bu örüntü S-OV ile S-VN'de aynı mı?

**Null hipotez:** H0-a1: aynı init state'lerde P(başarı | tr_a) = P(başarı | clean) (eşli fark 0; eşdeğerlik sınırı δ = 15 puan). H0-a2: P(başarı | tr_a) = P(başarı | empty) (δ = 10 puan; Ders 19'un "tr ≈ empty" eşiği). İki null birlikte okunur: a1 reddedilip a2 reddedilemezse "Türkçe okunmuyor" (örüntü B); ikisi de reddedilirse "okunuyor ama eksik" (D); a1 reddedilemezse `empty` satırı sonucu ikiye ayırır (A: dil görmezden geliniyor; D: Türkçe çalışıyor).

**Değişkenler:** Bağımsız: komut koşulu {clean, empty, wrong, tr_a} (sahne içi, eşli); sistem {S-OV, S-VN} (sahneler arası, aynı init state dosyaları). Bağımlı: episode başarı (0/1, LIBERO predicate). Ortak değişkenler: komutun token sayısı ve ilişki ekinin parça sayısı (`datasets/05/tokens.csv`), görevin komutsuz üst sınırı (Ders 19: Spatial 9/10, Goal 1/10). Sabit tutulan: init state kümesi, `num_steps_wait`, kamera/çözünürlük, checkpoint sürümü, seed listesi.

**Ölçüt ve eşik:** Birincil: eşli fark Δ(tr_a − clean) ve Δ(tr_a − empty), puan; McNemar exact p; Wilson %95 (koşu içi) + PI (≥3 tekrar). "Fark var" kararı: McNemar p < 0,05 **ve** |Δ| ≥ 15. "Fark yok" kararı: TOST %90 CI ⊂ (−δ, +δ). İkisi de değilse "belirsiz" (n artırılır, eşik oynatılmaz). İkincil: `grasped_bowl` etiketi (hangi nesneye gidildi) — başarısızlığın "yanlış hedef" mi "hiç hareket yok" mu olduğunu ayırır.

**Suite / blok / örneklem:** Spatial blok A (6 görev) + Goal blok C (6 görev). Pilot: Spatial görev 0 × 10 init state × 4 koşul × ≥3 tekrar (= Ders 18 koşusu + tekrarlar). Tez: görev başına 50 init state (S-VN için `.pruned_init`'teki 50'nin hepsi); Spatial'da `empty` tavana yakınsa (≥ 80) ağırlık Goal'a kayar (Ders 19 kuralı, S7). Toplam episode (tez): 12 görev × 50 init × 4 koşul × 2 sistem = 4.800 (+ tekrarlar); T4 bütçesi Ders 27'de.

**Sistemler:** S-OV (OpenVLA-7B Spatial LoRA; Goal için ayrı checkpoint `openvla-7b-finetuned-libero-goal`), S-VN (VLANeXt-B; Spatial ve Goal checkpoint'leri). Her sistem kendi eğitildiği suite'te değerlendirilir; sistemler arası karşılaştırma **betimseldir** (nedensel değil — sistem farkı).

**Evet / hayır çıkarsa tez ne der:** *Evet (a1 reddedildi, |Δ| ≥ 15):* "Türkçe komut, aynı sahnede, iki açık VLA'da başarıyı X / Y puan düşürüyor; düşüş `empty` tabanına [yaklaşıyor / yaklaşmıyor]" → tezin ana ölçüm katkısı; RQ-b'ye geçilir. *Hayır (eşdeğer):* `empty` ≈ `clean` ise (A) "bu görevlerde dil ölçülemiyor; Goal/B bloğuna geç"; `empty` ≪ `clean` ise (D) "Türkçe bu sistemde zaten okunuyor" → tez Qwen3-VL omurgasının Türkçe'yi nerede okuduğunu (RQ-b) ve LLaMA-2 ile farkını raporlar. *Belirsiz:* n ve tekrar artırılır; tez "yön" kipiyle yazılır. Hiçbir sonuç tezi boşa çıkarmaz; her dal bir sonraki soruyu seçer.

---

## RQ-b — Nerede kırılıyor?

**Soru:** Türkçe'deki düşüş (RQ-a'nın Δ'sı) komutun hangi özelliğine bağlı: ilişki ailesi (CDC/yön, RCC/topoloji, mesafe, bölge), yüzey (`tr_a / tr_b / tr_ascii / tr_bare`), referans çerçevesi (B07–B09: LIBERO / görüntü / robot) — ve eğitimde olmayan Türkçe tarif (blok B) modelde **boş komutun üstüne** çıkıyor mu?

**Null hipotez:** H0-b1: Δ(tr − clean) ilişki ailesine göre değişmez (aile × koşul etkileşimi yok). H0-b2: Δ yüzeye göre değişmez; özelde Δ(tr_ascii) = Δ(tr_a) her iki sistemde (Ders 22: LLaMA-2'de Türkçe harf bedeli 4,9 token, Qwen3'te 0,9 → H0-b2'nin S-OV'de reddedilip S-VN'de reddedilememesi tokenizasyon açıklamasının ilk kanıtı). H0-b3 (güçlü okuma): blok B'de P(başarı | tr_B) = P(başarı | empty) — Türkçe yeni tarif, boş komuttan farksız. H0-b4 (çerçeve): B07/B08'de (*soldaki/sağdaki*) gidilen kâse üç çerçeveden rastgele seçilmiş gibi dağılır.

**Değişkenler:** Bağımsız: ilişki ailesi (4 düzey; `pairs.csv` `relation`), yüzey (4 Türkçe + en), blok (A/B/C), çerçeve (B07–B09'da `ref_libero/ref_image/ref_robot`). Bağımlı: episode başarı; blok B'de ayrıca **gidilen nesne** (`grasped_bowl` ∈ {bowl_1, bowl_2, yok}) ve karşı-olgusal BDDL ile ölçülen başarı (`bddl_cf/`, hedef bowl_2). Ortak değişken: token sayısı, ilişki ekinin parça sayısı, `ref_min_margin` (cm/px). Sabit: RQ-a'nın init state'leri ve protokolü.

**Ölçüt ve eşik:** Hücre başına success + Wilson; aile ve yüzey için eşli fark tabloları (Δ ve %95 CI); etkileşim iddiası ancak hücre CI'ları ayrışıyorsa yazılır (küçük n: 6 çift × 50 init, aile başına 1–3 görev → **betimsel**, "yön" kipi; Ders 23). H0-b3 için: tr_B − empty ≥ 15 ve McNemar p < 0,05 → "güçlü okuma kanıtı"; TOST δ=10 içinde → "yok". H0-b4 için: gidilen kâsenin üç çerçeveden hangisiyle uyuştuğu, ki-kare yerine sayım tablosu + Wilson (n küçük).

**Suite / blok / örneklem:** Spatial blok A + **B** (B'nin EN sütunu da koşulur: EN-B − clean "yeni tarif" bedelini, TR-B − EN-B "Türkçe" bedelini ayırır; LangGap'in "Spatial Description" hücresi EN-B'nin dış referansı: π0.5 %11). Goal blok C (topoloji/EC, token tarafı en temiz Türkçe). Blok D (*sağına/soluna*) yalnızca dil tarafı (predicate yok) → bu RQ'da **yok**. Tier 2 satırlar (B07–B11) karşı-olgusal BDDL ister (`datasets/04/data/bddl_cf/`, 5 dosya hazır; gerçek env'de yükleme S-durumu: hipotez). Örneklem: görev başına 50 init; yüzey ekseni için önce pilot (10 init × 5 yüzey × 1 görev / blok). Suite ağırlığı danışman kararı (S7).

**Sistemler:** S-OV ve S-VN, RQ-a ile aynı checkpoint'ler. Yüzey karşılaştırması (özellikle `tr_ascii`) iki sistemde **zıt yön** beklenen tek hücre (Ders 22 hipotezi) → iki sistemin birlikte koşulması bu RQ'da zorunlu.

**Evet / hayır çıkarsa tez ne der:** *Aile/yüzey etkisi var:* "Türkçe düşüşü tekdüze değil; [topoloji/CDC] ailesinde ve [tr_ascii] yüzeyinde yoğunlaşıyor; S-OV'de harf bedeli görünüyor, S-VN'de görünmüyor" → tokenizasyon ile ön-eğitim açıklamaları ayrışır, Tatar vd.'nin hipotezi kapalı döngüde sınanmış olur. *Yok (tekdüze):* "Düşüş komutun biçiminden bağımsız, dilin kendisinden" → çözüm tarafında yüzey düzeltmeleri (ASCII'leştirme, eksiz yazım) elenir, RQ-c omurga/eğitim yoluna daralır. *H0-b3 reddedildi:* güçlü okuma kanıtı — tezin en iddialı cümlesi. *H0-b3 reddedilemedi:* "Türkçe yeni tarif boş komuta eşdeğer" → RQ-c'nin ölçütü zayıf okumaya iner (S10).

---

## RQ-c — Nasıl kapanır?

**Soru:** Modele dokunmayan iki müdahale (MT çeviri katmanı, kural tabanlı parser → İngilizce şablon) ve modele dokunan bir müdahale (EN+TR co-training LoRA, bütçe izin verirse) Türkçe açığını (RQ-a'nın Δ'sı) ne kadar kapatır — ve kapanış **zayıf okuma** (blok A/C'de `tr ≈ clean`) düzeyinde mi kalır, **güçlü okuma** (blok B'de `empty` üstü) düzeyine çıkar mı?

**Null hipotez:** H0-c1: müdahale M ∈ {MT, parser, LoRA} için P(başarı | tr, M) = P(başarı | tr, müdahale yok). H0-c2 (tavan): P(başarı | tr, M) = P(başarı | clean) — müdahale açığı tamamen kapatır (eşdeğerlik, δ = 15). H0-c3 (güçlü okuma, yalnız LoRA): blok B'de P(başarı | tr_B, LoRA) = P(başarı | empty). H0-c4 (yan etki, yalnız LoRA): P(başarı | clean, LoRA) = P(başarı | clean) — İngilizce bozulmuyor (Chen: E-FT/M-CT'de EN −2,7; δ = 5).

**Değişkenler:** Bağımsız: müdahale {yok, MT (Opus-MT tr-en), parser (`tr_parse.py` → EN şablon), LoRA (EN+TR karışık, 4-bit, T4)}; komut koşulu {clean, tr_a, tr_B, empty}. Bağımlı: episode başarı; MT/parser için ayrıca **ara çıktı doğruluğu** (çeviri/parse birebir mi, "yanlış ama geçerli" mi — Ders 24 H04) ve gecikme (ms). Ortak değişken: çeviri/parse hatası türü. Sabit: RQ-a protokolü; LoRA için eğitim verisi oranı EN:TR ve adım sayısı önceden yazılır.

**Ölçüt ve eşik:** Birincil: kapanış oranı κ = [SR(tr, M) − SR(tr)] / [SR(clean) − SR(tr)] (0 = hiç, 1 = tam), %95 CI bootstrap (eşli episode'lar üzerinden). "İşe yarıyor": κ ≥ 0,5 ve McNemar p < 0,05. "Tam kapatıyor": H0-c2 TOST içinde. Güçlü okuma: H0-c3 reddi (tr_B − empty ≥ 15). Yan etki: H0-c4 TOST δ=5 dışına çıkarsa raporlanır. MT/parser için hata ayrıştırması: açığın ne kadarı ara katman hatasından (çeviri yanlış → komut yanlış) geliyor (Ders 24'ün "yanlış ama geçerli" sayımı).

**Suite / blok / örneklem:** Zayıf okuma: Spatial A + Goal C (RQ-a hücreleri yeniden kullanılır; müdahale yalnızca yeni satır ekler). Güçlü okuma: Spatial **B** (yalnız LoRA; MT/parser B'de tanım gereği İngilizce'ye indirir, EN-B'nin sonucunu alır). Örneklem: RQ-a'nın 50 init'i; LoRA koşulu yalnızca S-OV'de (OpenVLA `finetune.py --use_quantization`; Ders 24 tahmini 65–98 T4 saati, S9 danışman kararı); MT/parser her iki sistemde (Mac'te saniye/ms).

**Sistemler:** MT ve parser: S-OV + S-VN (modele dokunmaz, ikisine de uygulanır). LoRA: S-OV (açık eğitim kodu; VLANeXt eğitimi T4'e sığmaz — hipotez, Ders 24). Sistem farkı uyarısı geçerli: "MT S-VN'de daha az kapatıyor" cümlesi sistem düzeyinde kalır.

**Evet / hayır çıkarsa tez ne der:** *MT/parser κ ≥ 0,5:* "Türkçe açığı büyük ölçüde girdi düzeyinde kapatılabilir; model Türkçe anlamıyor, İngilizce'ye indirilmiş komutla çalışıyor" (zayıf okuma; S10'a göre tez bununla biter ya da taban olur). *MT/parser κ < 0,5:* açığın bir kısmı ara katman hatasından değil (çeviri doğruyken de düşük) → model içi sorun; LoRA/omurga yolu gerekçelenir. *LoRA H0-c3 reddi:* "Türkçe mekânsal tarif, aynı sahnede, boş komutun üstüne çıktı" → güçlü okuma, tezin ana katkısı; H0-c4 yan etkisiyle birlikte raporlanır. *LoRA H0-c3 reddedilemedi:* "co-training eğitilmiş cümleleri öğretti, tarifi değil" → LangGap'in "veri tek başına yetmez" bulgusunun Türkçe karşılığı; tez bunu negatif sonuç olarak yazar.

---

## Sıralama kuralı (S1 örüntüsüne göre; Ders 26 §6)

| S1 örüntüsü (Ders 19) | Birincil RQ | Gerekçe | Hemen yapılacak |
|---|---|---|---|
| A — dil görmezden geliniyor | RQ-b (suite ekseni) | Spatial görev 0'da ölçek yok; Goal C ve Spatial B'ye geç | `conditions_goal8.json` koşusu |
| B — Türkçe okunmuyor (tr ≈ empty) | RQ-c (MT/parser önce) | Açık büyük, girdi düzeyi müdahale ucuz; güçlü okuma için LoRA S9'a bağlı | `mt_layer_mac.py`, Ders 18'e `tr_mt` koşulu |
| C — wrong biased | RQ-b (çerçeve + `grasped_bowl`) | Hedef sahneden seçiliyor; Türkçe'nin payı ancak B bloğunda görünür | 10 `wrong` videosunu etiketle |
| D — Türkçe okunuyor | RQ-a (iki sistem) → RQ-b (yüzey) | Fark küçükse eşdeğerlik testi, S-OV ile karşılaştırma öne çıkar | S-OV Spatial 4 koşul koşusu |
| belirsiz | RQ-a (n artır) | n=10 normal sonucu | 50 init + ≥3 tekrar |

## Önceden yazılan tahminler (Mesut doldurur — koşudan ÖNCE, commit ile)

```
RQ-a: Δ(tr_a − clean) S-VN: ___ puan   S-OV: ___ puan   ; tr − empty: ___
RQ-b: en çok kırılan aile: ______  yüzey: ______  ; tr_B > empty? evet / hayır
RQ-c: MT κ ≈ ___   parser κ ≈ ___   LoRA tr_B − empty ≈ ___
Birincil RQ tahminim: RQ-_   (gerekçe bir cümle: ______________________)
```

Sürüm: v0.1 — Ders 26 (otomatik taslak; sıralama koşullu). Değişiklikler bu bölüme tarihle eklenir.
