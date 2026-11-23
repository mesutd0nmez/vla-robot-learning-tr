# Proje tasarımı — P0 (adı: Goal görev 8'de `wrong` hedefi kaydırıyor mu — ÖRNEK, Claude doldurdu)

**Soru:** LIBERO Goal görev 8 sahnesinde (`put the bowl on the plate`), S-VN aynı init state'lerde `wrong` komutla (`put the bowl on the stove`) `clean`'e göre hedefi değiştiriyor mu, yoksa `empty` gibi davranıp komutu yok mu sayıyor?

**Null hipotez:** H0-1: P(başarı | wrong) = P(başarı | clean) (fark yok; δ = 15 puan). H0-2: P(başarı | wrong) = P(başarı | empty) (δ = 10 puan).

**Değişkenler:** Bağımsız: komut koşulu {clean, empty, wrong}. Bağımlı: episode başarı (görev 8 predicate'i) ve gidilen hedef (tabak / ocak / yok, video etiketi). Ortak: komutsuz tavan (Ders 19: Goal 1/10). Sabit: init state 0..49, checkpoint, `num_steps_wait` 10.

**Hücreler:**
- S-VN, libero_goal, C (görev 8), clean;empty;wrong, n_init 50, tekrar 1

**Bütçe:** 3,2 T4 sa (orta senaryo, budget.py: Goal clean 57 s + empty 88 s + wrong 88 s, × 50 init) · havuz n = 50 · MDE = 23 puan

**Başarı ölçütü:** "wrong hedefi kaydırıyor": McNemar p < 0,05 ve |Δ(wrong − clean)| ≥ 23 ve ocak etiketli episode ≥ 10/50. "Komutu yok sayıyor": TOST δ = 10 içinde wrong ≈ empty.

**Riskler:**
- Goal tavanı düşük (1/10) → `empty` ≈ 0; wrong − empty ayrımı yalnız gidilen hedef etiketiyle görülür (video etiketleme 50 × 3).
- Tek görev → genelleme yok; sonuç "görev 8'de" diye yazılır.

**Evet / hayır çıkarsa:** Evet: "S-VN Goal'da komutu okuyor ve yanlış hedefe gidiyor" → Türkçe `tr` koşulunun okunup okunmadığını ayırmak için `wrong` geçerli bir pozitif kontrol olur. Hayır: wrong ≈ empty → Goal'da da dil zayıf okunuyor; RQ-a'nın Goal C bloğu ölçümü bu sistemle anlamsızlaşır.
