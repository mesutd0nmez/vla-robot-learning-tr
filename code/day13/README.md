# day13

Ders 13 — VLANeXt + LIBERO eval zincirini kod üzerinden okuma (Mac, GPU gerekmez).

| Dosya | Ne |
|---|---|
| `trace_eval_chain.py` | Klonlanmış VLANeXt (+LIBERO) içinde 22 kritik satırı bulur, `dosya:satır` tablosu basar; upstream değişirse "BULUNAMADI" der |
| `show_prompt.py` | Eval'in Qwen3-VL'e gönderdiği chat şablonunu EN komutla basar; EN/TR token sayıları ve Türkçe komutun parçalanışı (yalnızca tokenizer iner) |
| `plot_day13.py` | Dersteki 03 (denormalizasyon, gerçek min/max), 04 (makale Tablo 1 bağlantı türleri), 05 (flow-matching sigma takvimi, temsili yörünge) görselleri |

Kullanım:
```bash
conda activate vla && cd ~/vla-work
git clone --depth 1 https://github.com/DravenALG/VLANeXt
curl -O https://raw.githubusercontent.com/mesutd0nmez/vla-robot-learning-tr/main/code/day13/trace_eval_chain.py
python trace_eval_chain.py ./VLANeXt ./LIBERO       # LIBERO Ders 10'da klonlanmıştı
curl -O https://raw.githubusercontent.com/mesutd0nmez/vla-robot-learning-tr/main/code/day13/show_prompt.py
python show_prompt.py
```

## trace_eval_chain.py

```python
"""Ders 13 — VLANeXt + LIBERO eval zincirinde kritik satırları bulur ve dosya:satır tablosu basar.

GPU gerekmez; yalnızca kaynak kodu okur. Kullanım (Mac):
    git clone --depth 1 https://github.com/DravenALG/VLANeXt ~/vla-work/VLANeXt
    python trace_eval_chain.py ~/vla-work/VLANeXt [~/vla-work/LIBERO]

Her satır: HANGİ SORU → hangi dosya → hangi satır → satırın kendisi. Upstream değişirse
"BULUNAMADI" yazar; o zaman desen eskimiş demektir, grep ile yenisini bul.
"""
import re
import sys
from pathlib import Path

# (soru, dosya, aranan düzenli ifade)
ANCHORS = [
    # --- 1. Komut nereden geliyor ---
    ("Komut string'i nereden geliyor?", "LIBERO:libero/libero/benchmark/__init__.py",
     r"def grab_language_from_filename"),
    ("Komutun episode'a bağlandığı yer (Türkçe için değişecek satır)",
     "src/evaluation/libero_bench/parallel_eval_utils.py", r"task_description = task\.language"),
    ("Komut modele kaç kez gönderiliyor? (her model çağrısında)",
     "src/evaluation/libero_bench/parallel_eval_utils.py", r"action_request_labels\.append\(specs\[env_id\]\.task_description\)"),
    # --- 2. Gözlem ---
    ("180 derece döndürme", "src/evaluation/libero_bench/libero_utils.py", r"img\[::-1, ::-1\]"),
    ("Proprio vektörü (eef_pos + axis-angle + gripper)",
     "src/evaluation/libero_bench/parallel_eval_utils.py", r"quat2axisangle\(obs\[\"robot0_eef_quat\"\]\), \[gripper_state\]"),
    ("Sablon (Qwen, image dalı): messages = [image, image, text]", "src/evaluation/libero_bench/VLANeXt_utils.py",
     r"^ {16}content\.append\(\{\"type\": \"text\", \"text\": label\}\)"),
    ("apply_chat_template(add_generation_prompt=True) (image dalı)", "src/evaluation/libero_bench/VLANeXt_utils.py",
     r"^ {16}texts\.append\(effective_processor\.apply_chat_template\(messages, tokenize=False, add_generation_prompt=True\)"),
    # --- 3. Model ---
    ("Proprio token'ları dizinin başına ekleniyor", "src/models/VLANeXt.py",
     r"inputs_embeds = torch\.cat\(\[proprio_embeds, inputs_embeds\], dim=1\)"),
    ("16 meta query dizinin sonuna ekleniyor", "src/models/VLANeXt.py",
     r"inputs_embeds = torch\.cat\(\[inputs_embeds, queries_embeds\], dim=1\)"),
    ("VLM tüm katman çıktılarıyla çağrılıyor (soft/tight)", "src/models/VLANeXt.py",
     r"\"output_hidden_states\": output_hidden_states_flag"),
    ("Policy bloğu i, VLM katmanı i'ye bakıyor", "src/models/modules/policies.py",
     r"relevant_vlm_states = vlm_hidden_states\[-len\(self\.blocks\):\]"),
    ("Cross-attention: key/value = [x, vlm_proj(hidden)]", "src/models/modules/policies.py",
     r"kv_list = \[x_norm, v_feat\]"),
    ("Flow-matching: gürültüden başla", "src/models/VLANeXt.py",
     r"action = torch\.randn\(B, self\.num_actions, self\.action_dim"),
    ("Flow-matching: Euler adımı", "src/models/VLANeXt.py",
     r"action = self\.noise_scheduler\.step\(output, t, action\)\.prev_sample"),
    # --- 4. Action decode ---
    ("Chunk'ın kaç adımı uygulanıyor (num_steps_execute)", "src/evaluation/libero_bench/parallel_eval_utils.py",
     r"steps_to_exec = min\(num_steps_execute, len\(chunk\)\)"),
    ("Denormalizasyon: ilk 6 boyut min/max", "src/evaluation/libero_bench/parallel_eval_utils.py",
     r"action\[:6\] = \(action\[:6\] \+ 1\) / 2 \* \(action_max - action_min\) \+ action_min"),
    ("Gripper eşiği: > 0 → +1, değilse −1", "src/evaluation/libero_bench/parallel_eval_utils.py",
     r"action\[6\] = 1\.0 if action\[6\] > 0 else -1\.0"),
    ("Min/max istatistikleri nereden (legacy/mixed)", "src/datasets/libero_act.py", r"\"legacy\": \{"),
    # --- 5. Success ---
    ("done → success=True (step fazı; :237 bekleme fazı)", "src/evaluation/libero_bench/parallel_eval_utils.py",
     r"^ {20}states\[env_id\]\[\"success\"\] = True"),
    ("LIBERO: done = _check_success()", "LIBERO:libero/libero/envs/bddl_base_domain.py",
     r"done = self\._check_success\(\)"),
    ("LIBERO: tüm goal predicate'lerinin VE'si", "LIBERO:libero/libero/envs/problems/libero_tabletop_manipulation.py",
     r"result = self\._eval_predicate\(state\) and result"),
    ("LIBERO: On predicate = temas + xy mesafe < 0.03 m", "LIBERO:libero/libero/envs/object_states/base_object_states.py",
     r"< 0\.03"),
]


def find(root: Path, rel: str, pattern: str):
    p = root / rel
    if not p.exists():
        return None, f"dosya yok: {p}"
    rx = re.compile(pattern)
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if rx.search(line):
            return i, line.strip()
    return None, "BULUNAMADI (desen eskimiş olabilir)"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    vla = Path(sys.argv[1]).expanduser()
    lib = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else None
    print(f"VLANeXt: {vla}\nLIBERO : {lib or '(verilmedi, LIBERO satırları atlanır)'}\n")
    print(f"{'#':>2} | {'Soru':62} | {'Dosya':64} | Satır")
    print("-" * 150)
    for n, (q, rel, pat) in enumerate(ANCHORS, 1):
        if rel.startswith("LIBERO:"):
            if lib is None:
                continue
            root, rel = lib, rel[len("LIBERO:"):]
        else:
            root = vla
        ln, text = find(root, rel, pat)
        loc = f"{rel}:{ln}" if ln else rel
        print(f"{n:>2} | {q:62} | {loc:64} | {text[:70]}")


if __name__ == "__main__":
    main()
```

## show_prompt.py

```python
"""Ders 13 — VLANeXt eval'inin Qwen3-VL'e gönderdiği metin şablonunu, EN ve TR komutla göster.

GPU gerekmez, model inmez; yalnızca tokenizer (~10 MB) iner. Mac'te:
    conda activate vla
    pip install "transformers>=4.57"
    python show_prompt.py
Çıktı: apply_chat_template'in ürettiği tam string + metin token sayıları + Türkçe komutun parçalanışı.
(Eval kodu: VLANeXt_utils.get_vla_action, is_qwen dalı — messages = [image, image, text].)
"""
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-VL-2B-Instruct"   # VLANeXt libero_train_config.yaml: model.lmm_path
PAIRS = [
    ("pick up the black bowl between the plate and the ramekin and place it on the plate",
     "tabak ile ramekin arasındaki siyah kaseyi al ve tabağın üstüne koy"),
    ("pick up the black bowl next to the ramekin and place it on the plate",
     "ramekinin yanındaki siyah kaseyi al ve tabağın üstüne koy"),
    ("pick up the black bowl from table center and place it on the plate",
     "masanın ortasındaki siyah kaseyi al ve tabağın üstüne koy"),
]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    en0 = PAIRS[0][0]
    # Eval kodundaki ile aynı yapı: iki görüntü (agentview + wrist) + komut, add_generation_prompt=True.
    # Tokenizer tek başına her görüntü için tek bir <|image_pad|> yazar; processor bunu görüntü başına
    # (256/32)^2 = 64 pad token'a açar (patch 16 x merge 2 = 32 px/token).
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": en0}]}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("=== apply_chat_template çıktısı (EN, 2 görüntü) ===")
    print(repr(text))
    print()
    print(f"{'EN token':>9} {'TR token':>9}  komut")
    for en, tr in PAIRS:
        n_en = len(tok.encode(en, add_special_tokens=False))
        n_tr = len(tok.encode(tr, add_special_tokens=False))
        print(f"{n_en:>9} {n_tr:>9}  {en}")
        print(f"{'':>9} {'':>9}  {tr}")
    print()
    tr0 = PAIRS[0][1]
    ids = tok.encode(tr0, add_special_tokens=False)
    print("=== Türkçe komutun parçaları (Qwen tokenizer) ===")
    print(" | ".join(tok.decode([i]) for i in ids))


if __name__ == "__main__":
    main()
```

## plot_day13.py

```python
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
```
