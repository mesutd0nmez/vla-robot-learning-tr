# day29

Ders 29 — bir sayfalık tez önerisi taslağı. Model yok. Çıktı dosyası `thesis/proposal-v0.1.md`.

| Dosya | Ne | Durum |
|---|---|---|
| `proposal_lint.py` | Önerinin "definition of done" kontrolü: gövde ≤ 650 sözcük (Kaynaklar hariç), 8 zorunlu bölüm, sayı içeren her cümle/tablo satırında `[Ders NN]` / `[arXiv:…]` / `[öneri]` / `[hesap…]` etiketi; `--selftest` | çalıştırıldı: 568/650 sözcük, 12/12 sayısal iddia etiketli, **TAM**; selftest OK → `results/lint.txt`, `results/claims.csv` |
| `plot_day29.py` | Görsel 02 (Gantt; T4 saatleri `code/day27/results/budget*.csv`'den) ve 03 (risk matrisi; puanlar değerlendirme) | çalıştırıldı → `results/plot_stdout.txt` |

Mac'te (`vla` env; yalnızca `matplotlib`):
```bash
cd code/day29
python proposal_lint.py --selftest
python proposal_lint.py ../../thesis/proposal-v0.1.md --csv results/claims.csv
python plot_day29.py
```
Çıkış kodu `proposal_lint.py`: 0 = TAM, 1 = EKSİK, 2 = selftest başarısız.

## proposal_lint.py

```python
"""Ders 29 — Tez önerisi taslağının "definition of done" kontrolü.

  python proposal_lint.py ../../thesis/proposal-v0.1.md            # rapor, çıkış kodu 0 = TAM
  python proposal_lint.py ../../thesis/proposal-v0.1.md --max-words 650 --csv results/claims.csv
  python proposal_lint.py --selftest

Üç kontrol:
  1. Gövde sözcük sayısı ≤ sınır (Kaynaklar bölümü, HTML yorumları ve "Sürüm:" satırı hariç) — "bir sayfa".
  2. Zorunlu bölümler var: Problem, Amaç, Araştırma soruları, Yöntem, Katkı, Takvim, Risk, Kaynaklar.
  3. Sayısal iddia içeren her cümle/tablo satırı bir kaynak etiketi taşır:
     [Ders NN] · [arXiv:XXXX.XXXXX] · [öneri] · [hesap…]. Model/etiket adlarındaki rakamlar
     (RQ-a, S-OV, 4-bit, LLaMA-2, Qwen3-VL, T4, OpenVLA-7B, İP1, hafta adları yok) sayılmaz.
Çıkış kodu: 0 TAM, 1 EKSİK, 2 selftest başarısız.
"""
import argparse, csv, re, sys

REQUIRED = {"Problem": r"^##\s+\d*\.?\s*Problem", "Amaç": r"^##\s+\d*\.?\s*Amaç",
            "Araştırma soruları": r"^##\s+\d*\.?\s*Araştırma soru", "Yöntem": r"^##\s+\d*\.?\s*Yöntem",
            "Katkı": r"^##\s+\d*\.?\s*Beklenen katkı", "Takvim": r"^##\s+\d*\.?\s*Takvim",
            "Risk": r"^##\s+\d*\.?\s*Risk", "Kaynaklar": r"^##\s+Kaynaklar"}
TAG = re.compile(r"\[(Ders \d+(?:[–-]\d+)?|arXiv:\d{4}\.\d{4,5}|öneri|hesap[^\]]*)\]")
# Rakam içeren ama sayısal iddia olmayan adlar (önce silinir)
NAMES = re.compile(r"(RQ-[a-z]|S-OV|S-VN|\d-bit|bf16|LLaMA-2|Qwen3(-VL)?|OpenVLA-7B|OpenVLA-OFT|T4|M2|İP\d|H0-\w+|"
                   r"v\d+\.\d+|lecture-\d+|Ders \d+(?:[–-]\d+)?|arXiv:\d{4}\.\d{4,5}|EN-B|tr_\w+|\bK\d\b|\bS\d+\b)")
NUM = re.compile(r"\d")


def body_and_refs(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    lines = text.splitlines()
    out, refs, in_refs = [], [], False
    for ln in lines:
        if re.match(r"^##\s+Kaynaklar", ln):
            in_refs = True
        if ln.startswith("Sürüm:"):
            continue
        (refs if in_refs else out).append(ln)
    return "\n".join(out), "\n".join(refs)


def words(s):
    s = re.sub(r"[`*_#|>\[\]]", " ", s)
    return [w for w in re.split(r"\s+", s) if re.search(r"\w", w)]


def claim_units(body):
    """Cümle (paragraf/madde) ya da tablo satırı düzeyinde birimler."""
    units = []
    for ln in body.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("<!--") or re.match(r"^\|[-\s|]+\|$", s) or s.startswith("**Başlık") \
                or s.startswith("**Title") or s.startswith("**Öğrenci"):
            continue
        if s.startswith("|"):
            units.append(s); continue
        s = re.sub(r"^(\d+\.|[-*])\s+", "", s)   # liste işaretini at
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ\"“*(])", s):
            units.append(sent)
    return units


def lint(path, max_words):
    text = open(path, encoding="utf-8").read()
    body, refs = body_and_refs(text)
    problems, rows = [], []
    n = len(words(body))
    if n > max_words:
        problems.append(f"gövde {n} sözcük > {max_words} (bir sayfa sınırı)")
    for name, rx in REQUIRED.items():
        if not re.search(rx, text, flags=re.M):
            problems.append(f"'{name}' bölümü yok")
    untagged = 0
    for u in claim_units(body):
        stripped = NAMES.sub(" ", TAG.sub(" ", u))
        has_num = bool(NUM.search(stripped))
        tagged = bool(TAG.search(u))
        if has_num:
            rows.append({"tagged": int(tagged), "tags": ";".join(TAG.findall(u)), "text": u[:160]})
            if not tagged:
                untagged += 1
                problems.append(f"etiketsiz sayısal iddia: {u[:90]}…")
    return n, rows, problems


def selftest():
    bad = ("# T\n\n## 1. Problem\n\nBaşarı 97'den 49'a iniyor.\n\n## 2. Amaç\n\nx\n\n## 4. Yöntem\n\nn = 50 [Ders 27].\n\n"
           "## Kaynaklar\n\nA 2024\n")
    import tempfile, os
    p = os.path.join(tempfile.mkdtemp(), "bad.md"); open(p, "w", encoding="utf-8").write(bad)
    n, rows, probs = lint(p, 650)
    want = {"'Araştırma soruları' bölümü yok", "'Katkı' bölümü yok", "'Takvim' bölümü yok", "'Risk' bölümü yok"}
    ok = want <= set(probs) and any("etiketsiz" in x and "97" in x for x in probs) and not any("n = 50" in x for x in probs)
    print("selftest", "OK" if ok else "BAŞARISIZ", f"({len(probs)} sorun yakalandı: 4 eksik bölüm + 1 etiketsiz iddia beklenir)")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?"); ap.add_argument("--max-words", type=int, default=650)
    ap.add_argument("--csv"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 2)
    n, rows, probs = lint(a.path, a.max_words)
    print(f"gövde sözcük: {n} / {a.max_words}")
    print(f"sayısal iddia birimi: {len(rows)}, etiketli: {sum(r['tagged'] for r in rows)}")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["tagged", "tags", "text"]); w.writeheader(); w.writerows(rows)
    for p in probs:
        print("  -", p)
    print("SONUÇ:", "TAM" if not probs else f"EKSİK ({len(probs)})")
    sys.exit(0 if not probs else 1)
```

## plot_day29.py

```python
"""Ders 29 — görseller: 02 Gantt (T4 saatleri Ders 27 budget_lean.csv / budget.csv'den), 03 risk matrisi.

  python plot_day29.py [çıktı klasörü]
Gantt'taki hafta aralıkları ÖNERİ (thesis/proposal-v0.1.md §6); saatler Ders 27 bütçe modelinin orta senaryosu.
Risk matrisindeki olasılık/etki puanları bu dersin DEĞERLENDİRMESİ (ölçüm değil).
"""
import csv, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = sys.argv[1] if len(sys.argv) > 1 else "../../lectures/images/lecture-29"
D27 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "day27", "results")
Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
YD, BD, GD, RD = "#c79a00", "#3b6ea8", "#2e8b57", "#c0392b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def hours(fn):
    rows = list(csv.DictReader(open(os.path.join(D27, fn), encoding="utf-8")))
    h = lambda cells: sum(float(r["h_orta"]) for r in rows if r["cell"] in cells)
    return {"İP1": h({"P1", "P2", "K1", "K2"}), "İP2": h({"A1", "A2", "A3", "A4"}),
            "İP3": h({"B1", "B2", "B3", "B4", "B5"}), "İP4": h({"C1", "C2"})}


lean, full = hours("budget_lean.csv"), hours("budget.csv")
print("T4 saat (orta) yalın:", {k: round(v, 1) for k, v in lean.items()}, "toplam", round(sum(lean.values()), 1))
print("T4 saat (orta) tam  :", {k: round(v, 1) for k, v in full.items()}, "toplam", round(sum(full.values()), 1))

# --- 02 Gantt
wp = [("İP1 pilot + kalibrasyon (P1, P2, K1, K2; K3 Mac)", 1, 2, B, BD, "İP1"),
      ("İP2 RQ-a çekirdek (A1–A4)", 2, 5, Y, YD, "İP2"),
      ("İP3 RQ-b (B1–B5)", 5, 7, Y, YD, "İP3"),
      ("İP4 RQ-c (C1, C2)", 7, 8, Y, YD, "İP4"),
      ("İP5 analiz (McNemar / TOST / κ)", 8, 10, G, GD, None),
      ("İP6 tez yazımı", 9, 16, G, GD, None)]
fig, ax = plt.subplots(figsize=(13, 4.8))
for i, (lab, a, b, fc, ec, key) in enumerate(wp):
    yy = len(wp) - 1 - i
    ax.barh(yy, b - a + 1, left=a - 0.5, color=fc, edgecolor=ec, height=0.6)
    txt = lab if key is None else f"{lab} — {lean[key]:.0f} sa yalın / {full[key]:.0f} sa tam"
    ax.text(a - 0.4, yy + 0.38, txt, fontsize=8.5, va="bottom")
for wk, lab in [(5, "ara sınav: RQ-a tablosu"), (16, "son sınav: tez taslağı")]:
    ax.axvline(wk + 0.5, color=RD, ls="--", lw=1); ax.text(wk + 0.55, len(wp) - 0.3, lab, color=RD, fontsize=8.5)
ax.set_yticks([]); ax.set_xticks(range(1, 17)); ax.set_xlim(0.4, 17.6); ax.set_ylim(-0.6, len(wp) + 0.2)
ax.set_xlabel("hafta (hafta 1 = 30 Kasım 2026 — öneri)")
tl, tf = sum(lean.values()), sum(full.values())
ax.set_title(f"Ders 29 — Önerinin takvimi: deney İP'leri yalın matriste ≈ {tl:.0f} T4 sa (tam çekirdek ≈ {tf:.0f}); "
             f"Kaggle ~30 sa/hafta ile ≈ {tl/30:.1f} hafta (tam: {tf/30:.1f}) — saatler Ders 27 bütçe modeli (orta)", fontsize=9.5)
ax.grid(axis="x", alpha=0.25)
fig.tight_layout(); os.makedirs(OUT, exist_ok=True); fig.savefig(os.path.join(OUT, "02-gantt.png"), dpi=160); plt.close(fig)

# --- 03 risk matrisi (olasılık × etki, 1–5, değerlendirme)
risks = [("R1 çağrı süresi yavaş\n(kötümser 497 sa)", 3, 4), ("R2 4-bit S-OV\n84,7'yi tutmaz", 3, 3),
         ("R3 Spatial'da empty\ntavana yakın", 4, 3), ("R4 bddl_cf\nyüklenmez", 2, 3),
         ("R5 LoRA onayı\nyok (S9)", 3, 2), ("R6 VLANeXt T4'te\nkoşmaz", 2, 5), ("R7 MT çıktısı\nüretilemez (HF)", 2, 2)]
fig, ax = plt.subplots(figsize=(9.5, 7))
for x in range(1, 6):
    for y in range(1, 6):
        s = x * y
        c = G if s <= 4 else (Y if s <= 9 else R)
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=c, alpha=0.55, lw=0))
off = {}
for lab, p, imp in risks:
    k = (p, imp); n = off.get(k, 0); off[k] = n + 1
    ax.plot(p, imp + 0.18 - 0.36 * n, "o", color="#333", ms=5)
    ax.text(p + 0.06, imp + 0.18 - 0.36 * n, lab, fontsize=7.8, va="center")
ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5); ax.set_xticks(range(1, 6)); ax.set_yticks(range(1, 6))
ax.set_xlabel("olasılık (1 düşük – 5 yüksek)"); ax.set_ylabel("etki (1 küçük – 5 tezi durdurur)")
ax.legend(handles=[Patch(color=G, alpha=0.55, label="izle"), Patch(color=Y, alpha=0.55, label="B planı hazır olsun"),
                   Patch(color=R, alpha=0.55, label="önce bunu çöz")], loc="lower right", fontsize=8.5)
ax.set_title("Ders 29 — Risk matrisi (puanlar bu dersin değerlendirmesi, ölçüm değil);\nher riskin B planı thesis/proposal-v0.1.md §7 ve dersin §5'inde", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "03-risk-matrix.png"), dpi=160); plt.close(fig)
print("yazıldı: 02-gantt.png, 03-risk-matrix.png →", OUT)
```
