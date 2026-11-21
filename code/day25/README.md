# day25

Ders 25 — Faz 3 sentezi. Model yok; iki script: Faz 3 araç envanterinin kuru koşusu ve dersin görselleri.

| Dosya | Ne | Durum |
|---|---|---|
| `phase3_smoke.py` | `code/day17–24` altındaki 30 script/kontrolü kuru koşuda (model yok, GPU yok) çalıştırır; her satıra bağımlılık (yok / LIBERO klonu / LIBERO-plus klonu / bddl / sentencepiece+tokenizer / Webots / HF+GPU), sonuç (OK / HATA / ATLANDI), süre ve son çıktı satırı; Ders 22 için `tokens.csv`'nin `datasets/05` ile birebirliğini de kontrol eder | **bu oturumda koştu: 28 OK / 0 HATA / 2 ATLANDI** (`results/smoke.csv`, `results/smoke.md`) |
| `plot_day25.py` | Görsel 02 (bilanço matrisi), 03 (Faz 3'ün dört sayısı), 05 (kuru koşu sonuçları), 06 (LangGap Tablo VII–VIII) | çalıştırıldı |
| `results/smoke.csv`, `results/smoke.md` | Kuru koşunun gerçek çıktısı | üretildi |

Mac'te (`vla` env):
```bash
# dış girdiler (bir kez)
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO ~/src/LIBERO
git clone --depth 1 https://github.com/sylvestf/LIBERO-plus ~/src/LIBERO-plus
pip install sentencepiece tokenizers "bddl==1.0.1" future
curl -L -o ../day22/tokenizer.model https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model
shasum -a 256 ../day22/tokenizer.model     # 9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347
npm pack @lenml/tokenizer-qwen3 && tar xzf lenml-tokenizer-qwen3-*.tgz   # package/models/tokenizer.json

cd code/day25
python phase3_smoke.py --libero ~/src/LIBERO --plus ~/src/LIBERO-plus \
       --llama ../day22/tokenizer.model --qwen package/models/tokenizer.json
python phase3_smoke.py --only 18,23,24     # yalnızca bağımlılıksız dersler
python plot_day25.py
```
Çıkış kodu: 0 = HATA yok; 1 = en az bir HATA (ATLANDI hata sayılmaz).

## phase3_smoke.py

```python
"""Ders 25 — Faz 3 araç envanteri: code/day17–24 script'lerini kuru koşuda (model yok, GPU yok) çalıştır,
hangisi hangi dış girdiyle çalışıyor tabloya dök.

Kullanım (repo kökünden ya da code/day25 içinden):
    python phase3_smoke.py                              # dış klon yok: yalnızca "yok" bağımlılıklılar
    python phase3_smoke.py --libero ~/src/LIBERO --plus ~/src/LIBERO-plus \
                           --llama code/day22/tokenizer.model --qwen path/to/qwen3/tokenizer.json
    python phase3_smoke.py --only 19,20                 # yalnızca Ders 19 ve 20 script'leri
    python phase3_smoke.py --out results                # smoke.csv + smoke.md buraya

Her satır: ders, script, komut, bağımlılık (yok / LIBERO klonu / LIBERO-plus klonu / bddl paketi /
sentencepiece+tokenizer / Webots / HF+GPU), sonuç (OK / HATA / ATLANDI), süre, çıktının son satırı.
"ATLANDI" = bağımlılık verilmedi; "HATA" = çalıştırıldı ama çıkış kodu 0 değil. Hiçbir script değiştirilmez;
çıktı dosyaları geçici bir klasöre yazılır.
"""
import argparse, csv, os, pathlib, shutil, subprocess, sys, tempfile, time

HERE = pathlib.Path(__file__).resolve().parent
CODE = HERE.parent
ROOT = CODE.parent
DS = ROOT / "datasets"
PY = sys.executable


def manifest(libero, plus, tmp, llama=None, qwen=None):
    """(ders, script, argv, cwd, bağımlılık, kısa açıklama). argv None → atla (bağımlılık yok)."""
    L = str(libero) if libero else None
    P = str(plus) if plus else None
    T = (str(llama), str(qwen)) if (llama and qwen) else None
    try:
        import bddl.parsing  # noqa: F401  (pip install "bddl==1.0.1" future)
        B = True
    except Exception:
        B = False
    t = lambda name: str(tmp / name)
    d = lambda n: CODE / f"day{n}"
    M = []
    # Ders 17 — LIBERO-plus klonundan bozulma sayımları
    M += [(17, "perturbation_table.py", [P, "--json", t("pt.json")] if P else None, d(17), "LIBERO-plus klonu", "7 bozulma × Spatial sayımı"),
          (17, "camera_geometry.py", [P, t("cam")] if P else None, d(17), "LIBERO-plus klonu", "376 kamera pozu (V/H/S/ER/EV)"),
          (17, "robot_init_spread.py", [P, t("rob")] if P else None, d(17), "LIBERO-plus klonu", "500 Panda init_qpos, ‖Δ‖"),
          (17, "layout_shift.py", [P, t("lay")] if P else None, d(17), "LIBERO-plus klonu", "_add_ / _level kaymaları"),
          (17, "classify_rewrites.py", [P, t("cls")] if P else None, d(17), "LIBERO-plus klonu", "görev 0'ın 50 yeniden yazımı R0/R1/R2"),
          (17, "make_tr_variant.py", [P, "--commands", str(d(17) / "tr_commands.csv"), "--out", t("trv")] if (P and B) else None, d(17), "LIBERO-plus klonu + bddl", "Türkçe _language_trK BDDL üretimi (bddl 1.0.1 gerek)")]
    # Ders 18 — koşul dosyası, istatistik, log analizi (selftest)
    M += [(18, "make_conditions.py", ["--task", "0", "--out", t("cond.json")], d(18), "yok", "clean/empty/wrong/tr koşul JSON'u"),
          (18, "spec_order_demo.py", [str(d(18) / "conditions_task0.json")], d(18), "yok", "init × koşul sırası"),
          (18, "analyze_4cond.py", ["--selftest"], d(18), "yok", "eval.log → tablo (Wilson)"),
          (18, "power_day18.py", [t("power")], d(18), "yok", "Fisher / McNemar güç eğrileri")]
    # Ders 19 — okuma protokolü, BDDL yapısı, init state geometrisi
    M += [(19, "read_patterns.py", ["--selftest"], d(19), "yok", "A/B/C/D karar ağacı"),
          (19, "read_patterns.py --demo", ["--demo"], d(19), "yok", "örnek tabloyla paragraf"),
          (19, "suite_structure.py", ["--libero", L] if L else None, d(19), "LIBERO klonu", "komutsuz üst sınır 9/10/1/9"),
          (19, "init_state_geometry.py", ["--libero", L, "--task", "0"] if L else None, d(19), "LIBERO klonu", "50 init state'in geometrisi")]
    # Ders 20 — Türkçe set
    M += [(20, "tr_morph.py", ["--selftest"], d(20), "yok", "ek üretici 42 kontrol"),
          (20, "make_tr_set.py", ["--out", t("set"), "--referents", str(DS / "04-tr-spatial-commands/data/referents.csv")], d(20), "yok", "pairs.csv + commands.csv üretimi"),
          (20, "validate_set.py", [str(DS / "04-tr-spatial-commands/data"), "--d01", str(DS / "01-libero-spatial-tr-mini/data/commands.csv")], d(20), "yok", "set tutarlılığı"),
          (20, "scene_frames.py", ["--libero", L] if L else None, d(20), "LIBERO klonu", "üç çerçevede referans tablosu"),
          (20, "make_cf_bddl.py", ["--libero", L, "--pairs", str(DS / "04-tr-spatial-commands/data/pairs.csv"), "--out", t("cf")] if (L and B) else None, d(20), "LIBERO klonu + bddl", "karşı-olgusal BDDL (bddl 1.0.1 gerek)")]
    # Ders 21 — Webots
    M += [(21, "check_webots_env.py", None, d(21), "Webots", "WEBOTS_HOME + import controller"),
          (21, "ur10e_fk.py", [], d(21), "yok", "UR10e FK (scipy IK --solve ayrı)")]
    # Ders 22 — tokenizasyon
    M += [(22, "tokenize_set.py", ["--llama", T[0], "--qwen", T[1], "--out", t("tok")] if T else None, d(22), "sentencepiece+tokenizer", "100 komut × 2 tokenizer (tokenizer.model / tokenizer.json gerek)"),
          (22, "tokens.csv == datasets/05?", ["--diff", t("tok/tokens.csv"), str(DS / "05-tokenization-tr-commands/data/tokens.csv")] if T else None, HERE, "sentencepiece+tokenizer", "yeniden üretim: ilk 7 sütun birebir mi"),
          (22, "probe_edge_cases.py", ["--llama", T[0], "--qwen", T[1]] if T else None, d(22), "sentencepiece+tokenizer", "İ.lower, yanlış arkadaşlar")]
    # Ders 23 — tahmin aralığı
    M += [(23, "prediction_interval.py", ["--selftest"], d(23), "yok", "PI formülü 5 kontrol"),
          (23, "prediction_interval.py --scores", ["--scores", "90", "80", "90"], d(23), "yok", "3 tekrardan aralık")]
    # Ders 24 — parser, bütçe, MT
    M += [(24, "tr_parse.py --selftest", ["--selftest"], d(24), "yok", "parser 4 kontrol"),
          (24, "tr_parse.py", ["--csv", str(DS / "04-tr-spatial-commands/data/commands.csv"), "--out", t("parse.csv")], d(24), "yok", "80 satır + held-out 12"),
          (24, "qlora_budget.py", [], d(24), "yok", "4-bit LoRA bellek/süre tahmini"),
          (24, "mt_layer_mac.py", None, d(24), "HF+GPU", "Opus-MT / NLLB round-trip (model indirir)")]
    return M


def diff_first_cols(a, b, k=7):
    """İki CSV'nin ilk k sütununu karşılaştır (Ders 22 yeniden üretim kontrolü)."""
    ra = [row[:k] for row in csv.reader(open(a, encoding="utf-8"))]
    rb = [row[:k] for row in csv.reader(open(b, encoding="utf-8"))]
    bad = sum(1 for x, y in zip(ra, rb) if x != y) + abs(len(ra) - len(rb))
    print(f"{len(ra)} vs {len(rb)} satır, ilk {k} sütunda farklı satır: {bad} → {'BİREBİR AYNI' if bad == 0 else 'FARKLI'}")
    return 0 if bad == 0 else 1


def run_one(script, argv, cwd, timeout):
    name = script.split()[0]
    if argv and argv[0] == "--diff":                   # yerleşik kontrol, ayrı script değil
        cmd = [PY, str(HERE / "phase3_smoke.py")] + argv
    else:
        cmd = [PY, str(cwd / name)] + argv
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"})
        rc, out = p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired:
        rc, out = 124, "zaman aşımı"
    dt = time.time() - t0
    lines = [l for l in out.strip().splitlines() if l.strip()]
    last = lines[-1][:110] if lines else ""
    return rc, dt, last, cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--libero", help="LIBERO klon kökü (Ders 19/20 script'leri)")
    ap.add_argument("--plus", help="LIBERO-plus klon kökü (Ders 17 script'leri)")
    ap.add_argument("--llama", help="Meta LLaMA-2 tokenizer.model (Ders 22)")
    ap.add_argument("--qwen", help="Qwen3 tokenizer.json (Ders 22)")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"), help="(iç kullanım) iki CSV'nin ilk 7 sütununu karşılaştır")
    ap.add_argument("--only", help="virgülle ders numaraları, ör. 19,20")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    if a.diff:
        return diff_first_cols(*a.diff)
    only = {int(x) for x in a.only.split(",")} if a.only else None
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="faz3smoke_"))
    rows = []
    for day, script, argv, cwd, need, desc in manifest(a.libero and pathlib.Path(a.libero).expanduser(),
                                                      a.plus and pathlib.Path(a.plus).expanduser(), tmp,
                                                      a.llama and pathlib.Path(a.llama).expanduser(), a.qwen and pathlib.Path(a.qwen).expanduser()):
        if only and day not in only:
            continue
        if argv is None:
            rows.append(dict(ders=day, script=script, bagimlilik=need, sonuc="ATLANDI", sure_s="", son_satir="", aciklama=desc))
            print(f"[--] Ders {day:2d} {script:34s} ATLANDI ({need})")
            continue
        rc, dt, last, cmd = run_one(script, argv, cwd, a.timeout)
        res = "OK" if rc == 0 else "HATA"
        rows.append(dict(ders=day, script=script, bagimlilik=need, sonuc=res, sure_s=f"{dt:.1f}", son_satir=last, aciklama=desc))
        print(f"[{'OK' if rc == 0 else '!!'}] Ders {day:2d} {script:34s} {res:7s} {dt:6.1f}s  {last}")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "smoke.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    ok = sum(r["sonuc"] == "OK" for r in rows); err = sum(r["sonuc"] == "HATA" for r in rows); sk = sum(r["sonuc"] == "ATLANDI" for r in rows)
    with open(out / "smoke.md", "w", encoding="utf-8") as f:
        f.write(f"# Faz 3 kuru koşu — {ok} OK / {err} HATA / {sk} ATLANDI (toplam {len(rows)})\n\n")
        f.write("| Ders | Script | Bağımlılık | Sonuç | Süre | Son satır |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['ders']} | `{r['script']}` | {r['bagimlilik']} | {r['sonuc']} | {r['sure_s']} | {r['son_satir'].replace('|', '/')} |\n")
    print(f"\n{ok} OK / {err} HATA / {sk} ATLANDI → {out / 'smoke.csv'}, {out / 'smoke.md'}")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
```

## plot_day25.py

```python
"""Ders 25 — görseller 02, 03, 05, 06. Sayılar: Faz 3 derslerinin kendi çıktıları (başlıkta ders numarası),
makale tabloları (başlıkta arXiv) ve bu klasördeki results/smoke.csv (gerçek koşu). Temsili veri yok.

python plot_day25.py            # lectures/images/lecture-25/ altına yazar
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as _beta, norm

HERE = Path(__file__).resolve().parent
OUT = HERE / ".." / ".." / "lectures" / "images" / "lecture-25"
OUT.mkdir(parents=True, exist_ok=True)
Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"       # sarı model/anahtar, mavi girdi, yeşil çıktı, kırmızı sorun
YE, BE, GE, RE = "#b8860b", "#3b6fa0", "#2e8b57", "#c0392b"
GRAY = "#9a9a9a"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ---------------------------------------------------------------- 02: ölçüm durumu matrisi (Faz 3 bilançosu)
# Durumlar: 2 = bu ortamda gerçek çıktıyla ölçüldü, 1 = script hazır, koşu Mesut'ta, 0 = hipotez / ölçülmedi
ROWS = [
    # (satır, ders, gerçek, Mesut'ta, hipotez)  — Faz 3 günlük girişlerinden
    ("LIBERO-Plus bozulma sayımları (376 kamera, 500 Panda, 300 _add_)", 17, 2),
    ("Language 50 yeniden yazım: R1 26 / R2 22 / R0 2, 'between' 48/50", 17, 2),
    ("Türkçe _language_trK BDDL üretimi (bddl 1.0.1, 10/10)", 17, 2),
    ("Dört koşul × 10 init state success tablosu (T4)", 18, 1),
    ("Wilson / kesin güç hesapları (n=10, 20, 50)", 18, 2),
    ("Komutsuz üst sınır (BDDL): 9/10 · 10/10 · 1/10 · 9/10", 19, 2),
    ("50 init state geometrisi (.pruned_init): yayılım ≤ 3 cm", 19, 2),
    ("Örüntü A/B/C/D okuması — gerçek tabloyla", 19, 1),
    ("datasets/04: 20 çift × 5 yüzey, 90 referans hücresi, 5 CF BDDL", 20, 2),
    ("Anadil kontrolü: 'soldaki' hangi kâse, tr_a/tr_b doğallık", 20, 1),
    ("Eğitim görüntülerinin de 180° döndürülmüş olduğu", 20, 0),
    ("Webots: kurulum dokümanı, FK/IK, kuru koşu controller", 21, 2),
    ("Webots gerçek render (launcher M2'de çalışıyor mu)", 21, 1),
    ("Tokenizasyon: 100 komut × 2 tokenizer (şişme, ek, action aralığı)", 22, 2),
    ("Qwen3-VL tokenizer = npm Qwen3 tokenizer.json (bayt aynılığı)", 22, 1),
    ("PI formülü ve genişlik simülasyonu", 23, 2),
    ("VLANeXt'in koşudan koşuya değişkenliği (seed/flow-matching)", 23, 0),
    ("Parser: set içi 79/80, held-out 5/12", 24, 2),
    ("MT round-trip (Opus-MT): birebir eşleşme, gecikme", 24, 1),
    ("QLoRA T4 bellek 7,7 GB / 65–98 saat", 24, 0),
    ("Türkçe'nin Qwen3-VL'de yeri (Çince tarafı mı, Arapça tarafı mı)", 24, 0),
]


def fig02():
    fig, ax = plt.subplots(figsize=(13, 8.2))
    cols = ["Bu ortamda gerçek çıktı", "Script hazır, koşu Mesut'ta", "Hipotez / ölçülmedi"]
    colmap = {2: 0, 1: 1, 0: 2}
    colors = {0: G, 1: Y, 2: R}
    for i, (label, day, st) in enumerate(ROWS):
        y = len(ROWS) - 1 - i
        for j in range(3):
            ax.add_patch(plt.Rectangle((j, y), 1, 1, facecolor=colors[j] if colmap[st] == j else "#f7f7f7",
                                       edgecolor="white", linewidth=2))
        if colmap[st] == 0:
            ax.text(0.5, y + 0.5, "✓", ha="center", va="center", fontsize=12, color=GE, fontweight="bold")
        elif colmap[st] == 1:
            ax.text(1.5, y + 0.5, "→", ha="center", va="center", fontsize=12, color=YE, fontweight="bold")
        else:
            ax.text(2.5, y + 0.5, "?", ha="center", va="center", fontsize=12, color=RE, fontweight="bold")
        ax.text(-0.08, y + 0.5, f"D{day}  {label}", ha="right", va="center", fontsize=8.6)
    ax.set_xlim(0, 3); ax.set_ylim(0, len(ROWS))
    ax.set_xticks([0.5, 1.5, 2.5]); ax.set_xticklabels(cols, fontsize=9.5)
    ax.xaxis.tick_top(); ax.set_yticks([])
    for s in ("left", "bottom"): ax.spines[s].set_visible(False)
    n2 = sum(1 for r in ROWS if r[2] == 2); n1 = sum(1 for r in ROWS if r[2] == 1); n0 = sum(1 for r in ROWS if r[2] == 0)
    ax.set_title(f"Faz 3 bilançosu: {len(ROWS)} ölçüm — {n2} gerçek, {n1} Mesut'un koşusunda, {n0} hipotez\n"
                 "Kaynak: plan/daily-log.md Ders 17–24 girişleri; 'gerçek' = bu oturumda phase3_smoke.py ile yeniden üretildi (28/30 script)",
                 fontsize=10.5, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "02-measured-vs-pending.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 03: Faz 3'ün sayıları (dört küçük panel)
def fig03():
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    # (a) komutsuz üst sınır vs literatür boş-komut (Ders 19 §2–3)
    a = axes[0, 0]
    suites = ["Spatial", "Object", "Goal", "Long"]
    bound = [90, 100, 10, 90]
    a.bar(range(4), bound, color=B, edgecolor=BE, width=0.55, label="BDDL üst sınırı (suite_structure.py)")
    lit = {  # Ders 19 görsel 02: Chen Ø (T6), Zhan blank (T9), Fang vision-only (T1)
        "OpenVLA-OFT": ([80.0, 91.5, 10.0, 76.5], [83.2, 98.2, 10.0, 85.0]),
        "π0.5": ([66.0, 64.5, 9.0, 70.0], [65.0, 51.8, 11.4, 72.0], [68.2, 56.6, 10.2, 71.6]),
        "Qwen3-VL + OFT": ([34.0, 42.5, 8.5, 45.0],),
    }
    mk = {"OpenVLA-OFT": ("o", YE), "π0.5": ("s", GE), "Qwen3-VL + OFT": ("^", RE)}
    for name, series in lit.items():
        arr = np.array(series)
        lo, hi, md = arr.min(0), arr.max(0), arr.mean(0)
        x = np.arange(4) + {"OpenVLA-OFT": -0.12, "π0.5": 0.0, "Qwen3-VL + OFT": 0.12}[name]
        a.errorbar(x, md, yerr=[md - lo, hi - md], fmt=mk[name][0], color=mk[name][1], ms=6, capsize=3, lw=1.2, label=f"{name} boş komut")
    a.set_xticks(range(4)); a.set_xticklabels(suites); a.set_ylim(0, 105); a.set_ylabel("success rate (%)")
    a.legend(fontsize=7.5, loc="center", bbox_to_anchor=(0.66, 0.50), framealpha=0.95)
    a.set_title("(a) Komutsuz tavan sabit, kullanımı modele göre — Ders 19\nChen 2606.15714 T6 · Zhan 2601.04052 T9 · Fang 2602.17659 T1", fontsize=9.5)
    # (b) tokenizasyon (Ders 22, gerçek)
    b = axes[0, 1]
    surf = ["en", "tr_a", "tr_ascii", "tr_bare"]
    L = [13.6, 22.4, 17.6, 14.9]; Q = [12.3, 16.9, 16.0, 12.2]
    x = np.arange(4); w = 0.36
    b.bar(x - w / 2, L, w, color=Y, edgecolor=YE, label="LLaMA-2 (OpenVLA)")
    b.bar(x + w / 2, Q, w, color=B, edgecolor=BE, label="Qwen3 (VLANeXt)")
    for i in range(4):
        b.text(x[i] - w / 2, L[i] + 0.3, f"{L[i]:.1f}", ha="center", fontsize=8)
        b.text(x[i] + w / 2, Q[i] + 0.3, f"{Q[i]:.1f}", ha="center", fontsize=8)
    b.set_xticks(x); b.set_xticklabels(surf); b.set_ylabel("token / komut"); b.set_ylim(0, 26)
    b.legend(fontsize=8, loc="upper right")
    b.set_title("(b) Aynı 100 komut, iki tokenizer — Ders 22 (tokens.csv, bu oturumda birebir yeniden üretildi)\n"
                "tr_a/en şişme 1,63× vs 1,41×; Türkçe harf bedeli (tr_a − tr_ascii) 4,9 vs 0,9 token", fontsize=9.5)
    # (c) Chen Tablo 1: LLaMA-2 OFT vs Qwen3-VL OFT vs Qwen3-VL GR00T
    c = axes[1, 0]
    langs = ["zh", "fr", "ru", "ar", "∅"]
    rowsC = {"OpenVLA-OFT (LLaMA-2)": ([-28.6, -28.6, -27.8, -29.8, -27.5], Y, YE),
             "Qwen3-VL + OFT başı": ([-7.3, -39.1, -48.8, -47.1, -63.1], R, RE),
             "Qwen3-VL + GR00T başı": ([-6.1, -33.0, -38.8, -36.0, -48.0], B, BE)}
    x = np.arange(5); w = 0.26
    for k, (name, (v, col, ed)) in enumerate(rowsC.items()):
        c.bar(x + (k - 1) * w, v, w, color=col, edgecolor=ed, label=name)
    c.axhline(0, color="#333", lw=0.8); c.set_xticks(x); c.set_xticklabels(langs)
    c.set_ylabel("EN'e göre fark (puan)"); c.legend(fontsize=8, loc="lower left")
    c.set_title("(c) 'Çok dilli omurga' yalnız Çince'de iyi — Ders 24\nChen vd. 2606.15714 Tablo 1, LIBERO ort.; Türkçe ölçülmedi (S1 bunu ölçer)", fontsize=9.5)
    # (d) Wilson aralıkları n=10 vs n=50 (Ders 18)
    d = axes[1, 1]
    cases = [("9/10", 9, 10), ("5/10", 5, 10), ("45/50", 45, 50), ("25/50", 25, 50)]
    for i, (lab, k, n) in enumerate(cases):
        lo, hi = wilson(k, n)
        col = Y if n == 10 else G; ed = YE if n == 10 else GE
        d.barh(i, (hi - lo) * 100, left=lo * 100, color=col, edgecolor=ed, height=0.5)
        d.plot([k / n * 100], [i], "o", color="#333", ms=5)
        d.text(hi * 100 + 1, i, f"{lo*100:.0f}–{hi*100:.0f}", va="center", fontsize=8.5)
    d.set_yticks(range(4)); d.set_yticklabels([c[0] for c in cases]); d.set_xlim(0, 115)
    d.set_xlabel("success rate (%) — Wilson %95")
    d.set_title("(d) 10 episode yön söyler, 50 episode fark söyler — Ders 18 §5\n9/10 ve 5/10 aralıkları çakışır; 45/50 ve 25/50 ayrışır", fontsize=9.5)
    fig.suptitle("Faz 3'ün sayıları — dördü de gerçek (ders numarası ve kaynak panel başlığında)", fontsize=11.5)
    fig.tight_layout(); fig.savefig(OUT / "03-phase3-numbers.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 05: kuru koşu sonuçları (results/smoke.csv, gerçek)
def fig05():
    rows = list(csv.DictReader(open(HERE / "results" / "smoke.csv", encoding="utf-8")))
    days = sorted({int(r["ders"]) for r in rows})
    ok = [sum(1 for r in rows if int(r["ders"]) == d and r["sonuc"] == "OK") for d in days]
    er = [sum(1 for r in rows if int(r["ders"]) == d and r["sonuc"] == "HATA") for d in days]
    sk = [sum(1 for r in rows if int(r["ders"]) == d and r["sonuc"] == "ATLANDI") for d in days]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(days))
    ax.bar(x, ok, color=G, edgecolor=GE, label="OK (çalıştı, çıkış 0)")
    ax.bar(x, er, bottom=ok, color=R, edgecolor=RE, label="HATA")
    ax.bar(x, sk, bottom=np.array(ok) + np.array(er), color="#eeeeee", edgecolor=GRAY, label="ATLANDI (ikili/model yok)")
    for i, d in enumerate(days):
        names = [r["script"].split()[0] for r in rows if int(r["ders"]) == d and r["sonuc"] == "ATLANDI"]
        if names:
            ax.text(x[i], ok[i] + er[i] + sk[i] + 0.15, "\n".join(names), ha="center", va="bottom", fontsize=7.5, color="#555")
        ax.text(x[i], 0.2, f"{ok[i]}/{ok[i]+er[i]+sk[i]}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"Ders {d}" for d in days]); ax.set_ylabel("script / kontrol sayısı")
    ax.set_ylim(0, max(np.array(ok) + np.array(er) + np.array(sk)) + 2.2)
    ax.legend(fontsize=8.5, loc="upper right")
    t_ok, t_all = sum(ok), len(rows)
    ax.set_title(f"Faz 3 araç envanteri kuru koşuda: {t_ok}/{t_all} OK — phase3_smoke.py, bu oturum (model yok, GPU yok)\n"
                 "Girdiler: LIBERO + LIBERO-plus klonları (--depth 1), Meta tokenizer.model (sha256 OK), npm Qwen3 tokenizer.json, bddl 1.0.1",
                 fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "05-smoke.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 06: LangGap (Hou & Zhao 2026, arXiv:2603.00592) — π0.5, aynı sahne, yalnızca komut değişir
def fig06():
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
    suites = ["Spatial", "Goal", "Object"]
    orig = [97.0, 97.0, 100.0]; ext = [5.9, 30.0, 37.7]           # Tablo VII
    x = np.arange(3); w = 0.36
    a.bar(x - w / 2, orig, w, color=G, edgecolor=GE, label="Orijinal 40 görev")
    a.bar(x + w / 2, ext, w, color=R, edgecolor=RE, label="Genişletilmiş 59 görev (aynı sahne, komut değişti)")
    for i in range(3):
        a.text(x[i] - w / 2, orig[i] + 1, f"{orig[i]:.1f}", ha="center", fontsize=8.5)
        a.text(x[i] + w / 2, ext[i] + 1, f"{ext[i]:.1f}", ha="center", fontsize=8.5)
    a.set_xticks(x); a.set_xticklabels(suites); a.set_ylim(0, 125); a.set_ylabel("success rate (%) — π0.5")
    a.legend(fontsize=8, loc="upper center", ncol=2, framealpha=0.95)
    a.set_title("Tablo VII — suite başına: Spatial 97,0 → 5,9", fontsize=10)
    dims = ["Nesne\ndeğişti", "Hedef\ndeğişti", "Mekânsal\ntarif değişti", "Çekmece\neylemi"]
    vals = [29.3, 0.0, 11.0, 31.7]; ns = ["38 görev / 760 ep", "13 / 260", "5 / 100", "3 / 60"]
    cols = [B, R, Y, B]; eds = [BE, RE, YE, BE]
    b.bar(range(4), vals, color=cols, edgecolor=eds, width=0.6)
    for i in range(4):
        b.text(i, vals[i] + 1, f"{vals[i]:.1f}", ha="center", fontsize=9)
        b.text(i, -5.5, ns[i], ha="center", fontsize=7.5, color="#555")
    b.set_xticks(range(4)); b.set_xticklabels(dims, fontsize=8.5); b.set_ylim(-8, 45); b.axhline(0, color="#333", lw=0.8)
    b.set_ylabel("success rate (%) — π0.5, genişletilmiş görevler")
    b.set_title("Tablo VIII — hangi bileşen değişince düşüyor; 'Mekânsal tarif' 5 görev × 20 episode", fontsize=10)
    fig.suptitle("LangGap (Hou & Zhao, Şub 2026, arXiv:2603.00592): aynı sahnede komutun bir bileşeni değişince π0.5 93,8 → 21,4\n"
                 "Yalnızca İngilizce, Türkçe yok; 'Spatial Description' hücresi datasets/04 B bloğunun İngilizce karşılığı", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "06-langgap.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig02(); fig03(); fig05(); fig06()
    print("yazıldı:", OUT.resolve())
```
