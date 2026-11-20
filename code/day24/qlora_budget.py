"""Ders 24 — Çözüm adayı 3 (Türkçe LoRA) için kaba bellek / süre hesabı. TAHMİNDİR; ölçüm değil.

Girdiler (gerçek): OpenVLA-7B ≈ 7,19 milyar parametre (makale Tablo 1: tam FT 7.188,1 M), LoRA r=32 → 97,6 M eğitilebilir
parametre (%1,4); openvla repo finetune.py: batch 16 ≈ 72 GB, '≥ ~27 GB' alt sınır; `use_quantization` bayrağı
bitsandbytes 4-bit (NF4) ile LoRA'ya izin verir. T4: 16 GB, bf16 yok (fp16), ~65 TFLOPS fp16 tensor (NVIDIA veri sayfası).

    python qlora_budget.py
"""
GB = 1024 ** 3


def mem(params=7.19e9, lora=97.6e6, bits_base=4, act_batch1_gb=1.5, vision_bf16_gb=1.2):
    base = params * bits_base / 8 / GB                   # dondurulmuş omurga (NF4 ≈ 0,5 bayt/param + ölçekler ≈ +%6)
    base *= 1.06
    lora_w = lora * 4 / GB                              # LoRA ağırlıkları fp32
    lora_grad = lora * 4 / GB
    adam = lora * 8 / GB                                # m ve v
    return {"omurga 4-bit": base, "görüntü omurgası (bf16/fp16)": vision_bf16_gb, "LoRA ağırlık": lora_w,
            "LoRA grad": lora_grad, "Adam durumları": adam, "aktivasyon (batch 1, 291 token, checkpointing)": act_batch1_gb}


def hours(a100_hours=(10, 15), t4_slowdown=5.0, quant_penalty=1.3):
    """OpenVLA makalesi: LoRA ile 'tek A100'de 10–15 saat' (yeni görev). T4 ≈ A100'ün 1/5'i (fp16 tensor 65 vs 312 TFLOPS)
    + 4-bit dequant maliyeti ≈ ×1,3 → saat aralığı."""
    return tuple(h * t4_slowdown * quant_penalty for h in a100_hours)


if __name__ == "__main__":
    m = mem()
    print("Bellek (tahmin), OpenVLA-7B + LoRA r=32 + 4-bit omurga, batch 1:")
    for k, v in m.items():
        print(f"  {k:<48} {v:5.2f} GB")
    tot = sum(m.values())
    print(f"  {'TOPLAM':<48} {tot:5.2f} GB   (T4 16 GB: {'sığar' if tot < 14 else 'riskli'}; pay ≈ {16 - tot:.1f} GB)")
    lo, hi = hours()
    print(f"\nSüre (tahmin): A100 10–15 sa → T4 {lo:.0f}–{hi:.0f} sa; Colab ücretsiz oturum ≤ ~12 sa → {lo/12:.0f}–{hi/12:.0f} oturum, "
          f"checkpoint/resume şart.\nKaggle: haftada 30 GPU saati → {lo/30:.1f}–{hi/30:.1f} hafta.")
    print("\nUyarı: bu bir hesap; gerçek sayı Mesut'un T4'te `finetune.py --use_quantization True --batch_size 1` ile ölçeceği değerdir.")
