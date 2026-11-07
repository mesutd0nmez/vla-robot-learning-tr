"""Ders 13 görselleri (matplotlib).

03-denormalize.png   : VLANeXt'in [-1,1] çıktısını LIBERO action aralığına geri ölçekleme.
                       Sayılar src/datasets/libero_act.py LIBERO_ACTION_STATS["legacy"]'den (gerçek).
04-connection-language.png : VLANeXt makalesi Tablo 1, "VLM-Policy Connection" satırları (gerçek).
05-flow-matching-toy.png   : 5 adımlı flow-matching Euler çözücünün TEMSİLİ 1-B gösterimi
                       (sigma takvimi gerçek: diffusers FlowMatchEulerDiscreteScheduler, shift=1).
Kullanım: python plot_day13.py [çıktı_klasörü]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT.mkdir(parents=True, exist_ok=True)
YEL, BLU, GRN, RED = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
YEL_D, BLU_D, GRN_D, RED_D = "#b8860b", "#2f6fb3", "#2e8b57", "#c0392b"

# ---- libero_act.py LIBERO_ACTION_STATS["legacy"] (ilk 6 boyut; gripper ayrı) ----
STATS = {
    "mixed":   ([-0.9375, -0.9375, -0.9375, -0.24214286, -0.375, -0.36428571],
                [0.9375, 0.9375, 0.9375, 0.35571429, 0.375, 0.375]),
    "spatial": ([-0.9375, -0.9375, -0.9375, -0.18857142, -0.3675, -0.36],
                [0.9375, 0.9375, 0.9375, 0.19714285, 0.33642858, 0.375]),
}
DIMS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]


def denorm(raw, lo, hi):
    raw = np.asarray(raw, dtype=np.float32)
    out = raw.copy()
    out[:6] = (raw[:6] + 1) / 2 * (np.array(hi) - np.array(lo)) + np.array(lo)
    out[6] = 1.0 if raw[6] > 0 else -1.0
    return out


def fig03():
    lo, hi = STATS["mixed"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    x = np.arange(6)
    ax.bar(x - 0.18, [h - l for l, h in zip(lo, hi)], 0.36, bottom=lo, color=BLU, edgecolor="#333",
           label="legacy / mixed (checkpoint'in kullandığı)")
    lo_s, hi_s = STATS["spatial"]
    ax.bar(x + 0.18, [h - l for l, h in zip(lo_s, hi_s)], 0.36, bottom=lo_s, color=YEL, edgecolor="#333",
           label="legacy / spatial (karşılaştırma)")
    for i in range(6):
        ax.text(i - 0.18, hi[i] + 0.02, f"{hi[i]:+.3f}", ha="center", fontsize=7.5)
        ax.text(i - 0.18, lo[i] - 0.07, f"{lo[i]:+.3f}", ha="center", fontsize=7.5)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xticks(x, DIMS)
    ax.set_ylabel("LIBERO action birimi (OSC_POSE delta, [-1,1] ölçeği)")
    ax.set_title("Model çıktısı [-1,1] → bu aralıklara açılır\n(action_min/max: libero_act.py, legacy)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(-1.15, 1.15)

    ax = axes[1]
    raw = np.array([0.5, -0.2, 0.0, 0.1, -1.0, 1.0, 0.03])
    real = denorm(raw, lo, hi)
    names = DIMS + ["gripper"]
    y = np.arange(7)[::-1]
    ax.barh(y + 0.18, raw, 0.36, color=YEL, edgecolor="#333", label="ham çıktı")
    ax.barh(y - 0.18, real, 0.36, color=GRN, edgecolor="#333", label="denormalize")
    for yi, r, v in zip(y, raw, real):
        ax.text(1.12, yi, f"{r:+.2f} → {v:+.3f}", va="center", ha="left", fontsize=7.5)
    ax.set_yticks(y, names)
    ax.set_xlim(-1.25, 2.3)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_title("Örnek: 7-D ham çıktı → gerçek action\n(gripper 0.03 > 0 → +1 = kapat)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("_denormalize_action: ilk 6 boyut doğrusal, 7. boyut eşik", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "03-denormalize.png", dpi=170)
    plt.close(fig)


def fig04():
    # VLANeXt (Wu vd., ICML 2026) Tablo 1, "VLM-Policy Connection" satırları
    cols = ["LIBERO Spatial\n(orijinal)", "LIBERO-plus Spatial\nLanguage", "LIBERO-plus Spatial\nRobot", "LIBERO-plus Spatial\nToplam"]
    data = {
        "Loose (MetaQuery, bağlayıcı yok)": [90.0, 74.6, 42.0, 53.7],
        "Tight (katman katman, π0 tarzı)": [90.0, 81.0, 51.7, 55.4],
        "Soft (katman katman + 16 meta query)": [91.8, 89.2, 58.3, 56.2],
    }
    colors = [BLU, RED, YEL]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(cols))
    w = 0.26
    for i, (name, vals) in enumerate(data.items()):
        bars = ax.bar(x + (i - 1) * w, vals, w, color=colors[i], edgecolor="#333", label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(x, cols)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("VLM–policy bağlantı türü ve dil bozulmasına dayanıklılık\n"
                 "VLANeXt Tablo 1 (Wu vd., 2026; ablation modelleri, Qwen3-VL-2B, Spatial suite)", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "04-connection-language.png", dpi=170)
    plt.close(fig)


def fig05():
    # Gerçek sigma takvimi: linspace(1.0, 1/1000, 5) + [0]  (FlowMatchEulerDiscreteScheduler, shift=1)
    sigmas = np.append(np.linspace(1.0, 1 / 1000, 5), 0.0)
    rng = np.random.default_rng(0)
    x0 = 0.35                     # hedef action (temsili)
    noise = rng.normal(0, 1, 4)   # 4 farklı başlangıç gürültüsü
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for k, n in enumerate(noise):
        x = n
        xs = [x]
        for i in range(5):
            # Kusursuz model varsayımı: velocity = noise - x0 (doğrusal akış); gerçek modelde tahmin edilir
            v = (x - x0) / max(sigmas[i], 1e-6)   # x_t = (1-s) x0 + s*noise ⇒ v = noise - x0 = (x_t - x0)/s
            x = x + (sigmas[i + 1] - sigmas[i]) * v
            xs.append(x)
        ax.plot(sigmas, xs, marker="o", ms=5, lw=1.8, color=YEL_D if k == 0 else "#999",
                label="Euler yörüngesi (temsili)" if k == 0 else None)
    ax.axhline(x0, color=GRN_D, ls="--", lw=1.5, label="hedef action x₀ (temsili)")
    ax.invert_xaxis()
    ax.set_xticks([1.0, 0.75025, 0.5005, 0.25075, 0.0], ["1.0", "0.750", "0.500", "0.251", "0.001 → 0"])
    ax.set_xlabel("sigma (1 = saf gürültü → 0 = temiz action); 5 scheduler.step çağrısı")
    ax.set_ylabel("tek action boyutu, [-1,1] ölçeği")
    ax.set_title("Flow-matching çözücü: predict_action içindeki 5 adımlık döngü\n"
                 "(sigma takvimi gerçek — diffusers FlowMatchEuler, shift=1; yörünge TEMSİLİ, kusursuz model varsayımı)",
                 fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "05-flow-matching-toy.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig03(); fig04(); fig05()
    print("yazıldı:", *(sorted(p.name for p in OUT.glob("0[345]-*.png"))))
