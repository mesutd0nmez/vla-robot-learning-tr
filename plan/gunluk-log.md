# Günlük Çalışma Logu

Bu dosya her günün sonunda güncellenir. Yeni güne başlarken Claude önce bunu okur.

## Gün 1 — 5 Kasım 2026 — Kaynak haritası + robot learning döngüsü
**Durum:** Başlandı
**İşlenen:**
- Tüm kaynaklar Claude tarafından okundu ve haritalandı (bkz. 30-gun-plani.md "Kaynak haritası").
- Çekirdek döngü: observation → policy → action → world changes → new observation. Policy = karar veren fonksiyon; VLA bir policy türü.
- Behavior cloning = supervised learning (X=observation, y=expert action). Robot manipülasyonunda RL yerine imitation learning'in baskın olma nedeni: gerçek robotta deneme maliyeti.

**Görev (Mesut):** Not §1–9'u oku (~25 dk). Sonra kendi cümlelerinle: (a) 9 kaynağı 1'er satırda tanımla, (b) "policy" nedir, bir cümle. Chat'e yapıştır.

**Çıktı:** (bekleniyor)
**Notlar / sorular:** —
