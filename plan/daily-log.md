# Günlük Çalışma Logu

Bu dosya her günün sonunda güncellenir. Yeni güne başlarken Claude önce bunu okur.

## Gün 1 — 5 Kasım 2026 — Kaynak haritası + robot learning döngüsü
**Durum:** Başlandı
**İşlenen:**
- Tüm kaynaklar Claude tarafından okundu ve haritalandı (bkz. 30-day-plan.md "Kaynak haritası").
- Çekirdek döngü: observation → policy → action → world changes → new observation. Policy = karar veren fonksiyon; VLA bir policy türü.
- Behavior cloning = supervised learning (X=observation, y=expert action). Robot manipülasyonunda RL yerine imitation learning'in baskın olma nedeni: gerçek robotta deneme maliyeti.

**Ders:** `lectures/2026-11-05.md` (6 görsel, 4 kendini sına sorusu)
**Görev (Mesut):** Dersi oku, soruları cevaplara bakmadan cevapla, takıldığın yeri chat'e yaz.
**Notlar / sorular:** —

## Gün 2 — 6 Kasım 2026 — OpenVLA'nın içi
**Durum:** Ders hazır (5 Kas'ta yazıldı)
**Ders:** `lectures/2026-11-06.md` — SigLIP+DINOv2, projector, LLaMA-2, 256-bin action tokenizasyonu, LoRA fine-tune, OpenVLA-OFT farkı (4 görsel, 5 soru)
**Görev (Mesut):** Pipeline'ı görsele bakmadan çiz; soruları cevapla.
**Notlar / sorular:** —
