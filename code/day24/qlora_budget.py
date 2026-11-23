"""Lecture 24 — rough memory / time estimate for solution candidate 3 (Turkish LoRA). It is an ESTIMATE; not a measurement.

Inputs (real): OpenVLA-7B ≈ 7.19 billion parameters (paper Table 1: full FT 7,188.1 M), LoRA r=32 -> 97.6 M trainable
parameters (1.4%); openvla repo finetune.py: batch 16 ≈ 72 GB, '≥ ~27 GB' lower bound; the `use_quantization` flag
allows LoRA with bitsandbytes 4-bit (NF4). T4: 16 GB, no bf16 (fp16), ~65 TFLOPS fp16 tensor (NVIDIA datasheet).

    python qlora_budget.py
"""
GB = 1024 ** 3


def mem(params=7.19e9, lora=97.6e6, bits_base=4, act_batch1_gb=1.5, vision_bf16_gb=1.2):
    base = params * bits_base / 8 / GB                   # frozen backbone (NF4 ≈ 0.5 byte/param + scales ≈ +6%)
    base *= 1.06
    lora_w = lora * 4 / GB                              # LoRA weights in fp32
    lora_grad = lora * 4 / GB
    adam = lora * 8 / GB                                # m and v
    return {"backbone 4-bit": base, "vision backbone (bf16/fp16)": vision_bf16_gb, "LoRA weights": lora_w,
            "LoRA grad": lora_grad, "Adam states": adam, "activations (batch 1, 291 tokens, checkpointing)": act_batch1_gb}


def hours(a100_hours=(10, 15), t4_slowdown=5.0, quant_penalty=1.3):
    """OpenVLA paper: with LoRA '10–15 hours on a single A100' (new task). T4 ≈ 1/5 of an A100 (fp16 tensor 65 vs 312 TFLOPS)
    + 4-bit dequant cost ≈ ×1.3 -> hour range."""
    return tuple(h * t4_slowdown * quant_penalty for h in a100_hours)


if __name__ == "__main__":
    m = mem()
    print("Memory (estimate), OpenVLA-7B + LoRA r=32 + 4-bit backbone, batch 1:")
    for k, v in m.items():
        print(f"  {k:<48} {v:5.2f} GB")
    tot = sum(m.values())
    print(f"  {'TOTAL':<48} {tot:5.2f} GB   (T4 16 GB: {'fits' if tot < 14 else 'risky'}; margin ≈ {16 - tot:.1f} GB)")
    lo, hi = hours()
    print(f"\nTime (estimate): A100 10–15 h -> T4 {lo:.0f}–{hi:.0f} h; Colab free session ≤ ~12 h -> {lo/12:.0f}–{hi/12:.0f} sessions, "
          f"checkpoint/resume is a must.\nKaggle: 30 GPU hours per week -> {lo/30:.1f}–{hi/30:.1f} weeks.")
    print("\nWarning: this is a calculation; the real number is the value you will measure on the T4 with `finetune.py --use_quantization True --batch_size 1`.")
