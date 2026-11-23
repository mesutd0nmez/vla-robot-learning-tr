# day31

Ders 31 — Final: "bir proje yapalım" simülasyonu ve Aşama 0 bilançosu. Model yok; üç script de bu oturumda koştu.

| Dosya | Ne | Durum |
|---|---|---|
| `phase0_inventory.py` | Repoyu tarar: ders/görsel/kod/veri kartı/tez dosyası sayıları, log'daki "çalıştırıldı" işaretleri, devreden işlerin repoda kanıtı; `--png` görsel 01 | çalıştırıldı → `results/inventory.csv`, `lectures.csv`, `handoff.md` |
| `design_lint.py` | Proje tasarım şablonunun 8 alanını denetler; bütçedeki n ile MDE'yi `code/day27/budget.py` güç fonksiyonuyla karşılaştırır; eksik alan için dönülecek dersi yazar | çalıştırıldı; selftest OK → `results/lint_demo.txt` |
| `plot_day31.py` | Görsel 04: yetkinlik radarı (ders kapsamı + senin elinden kanıt + öz-değerlendirme) | çalıştırıldı |
| `templates/` | `design-template.md` (boş), `P0-ornek.md` (Claude'un doldurduğu örnek, TAM), `P1.md`–`P3.md` (senin için) | — |
| `self_assessment.csv` | Yetkinlik ↔ ders eşlemesi ve kanıt yolu; `mesut_0_4` sütunu senin | — |

```bash
cd code/day31
python phase0_inventory.py --png ../../lectures/images/lecture-31
python design_lint.py --selftest
python design_lint.py templates/P1.md        # doldurduktan sonra
python plot_day31.py
```

## phase0_inventory.py

```python
"""Ders 31 — Aşama 0 bilançosu: repoyu tarar, gerçek sayıları çıkarır.

  python phase0_inventory.py                # repo kökü ../.. varsayılır
  python phase0_inventory.py --png ../../lectures/images/lecture-31

Çıktılar (results/):
  inventory.csv   metrik, değer (ders, görsel, script, veri kartı, tez dosyası, proje ...)
  lectures.csv    ders başına: faz, kelime, görsel, kod dosyası, log'da "çalıştırıldı" sayılan script
  handoff.md      Aşama 0'dan devreden açık işler + repoda var/yok kontrolü
Kaynak tek: repo dosyaları. "Çalıştırıldı" bilgisi plan/daily-log.md'nin **Kod:** satırlarından okunur
(o satırları dersleri yazan oturumlar yazdı; Mesut'un kendi koşuları ayrı: projects/).
"""
import argparse, csv, glob, os, re
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ap.add_argument("--png", default=None)
a = ap.parse_args()
R = os.path.abspath(a.root)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
J = lambda *p: os.path.join(R, *p)

PHASES = [(1, 8, "Faz 1 Kavramlar"), (9, 16, "Faz 2 Ortam"), (17, 25, "Faz 3 Deney tasarımı"), (26, 31, "Faz 4 Proje")]
phase = lambda n: next(p for lo, hi, p in PHASES if lo <= n <= hi)
CODE_EXT = (".py", ".ipynb", ".mjs")
RAN = re.compile(r"çalıştırıldı|doğrulandı|selftest|koştu|gerçek çıktı", re.I)

# --- daily-log: ders blokları ve **Kod:** satırları
log = open(J("plan", "daily-log.md"), encoding="utf-8").read()
blocks = {int(m.group(1)): m.group(0) for m in re.finditer(r"^## Ders (\d+) —.*?(?=^## |\Z)", log, re.M | re.S)}
ran_by_lec, comp_by_lec = {}, {}
for n, b in blocks.items():
    kod = " ".join(l for l in b.splitlines() if l.startswith("**Kod:**"))
    ran = comp = 0
    for m in re.finditer(r"`([\w./-]+\.(?:py|mjs|ipynb))`\s*(\(([^)]*)\))?", kod):
        st = m.group(3) or ""
        if RAN.search(st): ran += 1
        elif "py_compile" in st: comp += 1
    ran_by_lec[n], comp_by_lec[n] = ran, comp

rows = []
for f in sorted(glob.glob(J("lectures", "lecture-*.md"))):
    n = int(re.search(r"lecture-(\d+)", f).group(1))
    txt = open(f, encoding="utf-8").read()
    title = txt.splitlines()[0].lstrip("# ").strip()
    imgs = [p for p in glob.glob(J("lectures", "images", f"lecture-{n:02d}", "*.png"))]
    mmd = glob.glob(J("lectures", "images", f"lecture-{n:02d}", "*.mmd"))
    code = [p for p in glob.glob(J("code", f"day{n:02d}", "**", "*"), recursive=True) if p.endswith(CODE_EXT)]
    rows.append(dict(lecture=n, phase=phase(n), title=title[:90], words=len(txt.split()), png=len(imgs), mmd=len(mmd),
                     code_files=len(code), log_ran=ran_by_lec.get(n, 0), log_py_compile=comp_by_lec.get(n, 0),
                     hipotez=len(re.findall(r"hipotez", txt, re.I)), arxiv=len(set(re.findall(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", txt)))))
with open(os.path.join(OUT, "lectures.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

arxiv_all = set()
for f in glob.glob(J("lectures", "lecture-*.md")):
    arxiv_all |= set(re.findall(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", open(f, encoding="utf-8").read()))
cards = [d for d in glob.glob(J("datasets", "[0-9]*")) if os.path.exists(os.path.join(d, "dataset-card.md"))]
projects = [d for d in glob.glob(J("projects", "*")) if os.path.isdir(d) and not os.path.basename(d).startswith("_")]
thesis = sorted(os.path.basename(p) for p in glob.glob(J("thesis", "*")))
inv = [
    ("ders", len(rows)), ("ders_aralik", f"{min(r['lecture'] for r in rows)}–{max(r['lecture'] for r in rows)}"),
    ("toplam_kelime", sum(r["words"] for r in rows)), ("gorsel_png", sum(r["png"] for r in rows)),
    ("mermaid_kaynak", sum(r["mmd"] for r in rows)), ("kod_dosyasi", sum(r["code_files"] for r in rows)),
    ("kod_klasoru", len(glob.glob(J("code", "day*")))),
    ("logda_calistirildi", sum(r["log_ran"] for r in rows)), ("logda_yalniz_py_compile", sum(r["log_py_compile"] for r in rows)),
    ("hipotez_gecisi", sum(r["hipotez"] for r in rows)), ("farkli_arxiv_kimligi", len(arxiv_all)),
    ("veri_karti", len(cards)), ("tez_dosyasi", len(thesis)), ("tez_dosyalari", ";".join(thesis)),
    ("proje_klasoru_mesut", len(projects)), ("projeler", ";".join(os.path.basename(p) for p in sorted(projects))),
    ("log_ders_girisi", len(blocks)),
    ("log_gorev_satiri", len(re.findall(r"^\*\*Görev \(Mesut\):\*\*", log, re.M))),
]
by_phase = Counter()
for r in rows: by_phase[r["phase"]] += 1
for p, c in by_phase.items(): inv.append((f"ders_{p}", c))
with open(os.path.join(OUT, "inventory.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["metrik", "deger"]); w.writerows(inv)

# --- handoff: Aşama 0'dan devreden işler, repoda kanıtı var mı?
def has(path_glob): return bool(glob.glob(J(path_glob)))
def contains(path, pat):
    p = J(path); return os.path.exists(p) and re.search(pat, open(p, encoding="utf-8").read(), re.M) is not None
def s_open():
    p = J("plan", "faz3-open-questions.md")
    if not os.path.exists(p): return 10
    st = [l.rstrip().rstrip("|").split("|")[-1].strip() for l in open(p, encoding="utf-8") if re.match(r"^\| S\d+ ", l)]
    return sum(1 for x in st if x in ("bekliyor", "sorulmadı", ""))
items = [
    ("S1 — Ders 18 dört-koşul koşusu (T4)", "projects/05-spatial0-4kosul/ var", has("projects/05-*")),
    ("Ders 12 ilk T4 eval (VLANeXt)", "projects/01-vlanext-t4-ilk-eval/ var", has("projects/01-*")),
    ("Ders 14 OpenVLA Mac tek kare", "projects/02-openvla-mac-tek-kare/ var", has("projects/02-openvla*")),
    ("Ders 15 Tatar mini-reprodüksiyonu", "projects/03-tatar-mini/ var", has("projects/03-*")),
    ("Ders 16 LIBERO-Plus kurulumu", "projects/04-libero-plus-kurulum/ var", has("projects/04-*")),
    ("results/runs.csv (deney koşusu kaydı)", "results/runs.csv var", has("results/runs.csv")),
    ("Ders 25 S1–S10 Durum sütunu", f"faz3-open-questions.md: {s_open()} / 10 soru hâlâ 'bekliyor' ya da 'sorulmadı'", s_open() == 0),
    ("Ders 26 ön kayıt tahminleri", "rq-candidates.md tahmin bloğunda '___' kalmamış", not contains("thesis/rq-candidates.md", r"___")),
    ("Ders 27 iddia birimi + kota", "experiment-matrix.md'de '___' kalmamış", not contains("thesis/experiment-matrix.md", r"___")),
    ("K1/K2 kalibrasyon (gerçek t_call)", "code/day27/results/budget_stdout.txt'de 'ölçüm' senaryosu", contains("code/day27/results/budget_stdout.txt", r"ölçüm")),
    ("Ders 28 ablation planı", "thesis/ablation-plan.md var", has("thesis/ablation-plan.md")),
    ("Ders 29 tez önerisi taslağı", "thesis/proposal-v0.1.md var", has("thesis/proposal-v0.1.md")),
    ("Ders 30 danışman toplantısı paketi", "slides/advisor-meeting-01.md var", has("slides/advisor-meeting-01.md")),
]
with open(os.path.join(OUT, "handoff.md"), "w", encoding="utf-8") as fh:
    fh.write("| iş | kanıt (repoda) | durum |\n|---|---|---|\n")
    for name, ev, ok in items:
        fh.write(f"| {name} | {ev} | {'✔ var' if ok else '✘ yok'} |\n")

for k, v in inv: print(f"{k:28s} {v}")
print()
print(open(os.path.join(OUT, "handoff.md"), encoding="utf-8").read())

if a.png:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    pc = {"Faz 1 Kavramlar": "#cfe8ff", "Faz 2 Ortam": "#d4f5d4", "Faz 3 Deney tasarımı": "#ffd166", "Faz 4 Proje": "#ffd6d6"}
    ec = {"Faz 1 Kavramlar": "#3b6ea8", "Faz 2 Ortam": "#2e8b57", "Faz 3 Deney tasarımı": "#c79a00", "Faz 4 Proje": "#c0392b"}
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    x = np.array([r["lecture"] for r in rows])
    ax = axes[0]
    ax.bar(x, [r["words"] for r in rows], color=[pc[r["phase"]] for r in rows], edgecolor=[ec[r["phase"]] for r in rows])
    ax.set_ylabel("kelime / ders")
    ax.set_title(f"Ders 31 — Aşama 0 envanteri (phase0_inventory.py, gerçek tarama): {len(rows)} ders, "
                 f"{sum(r['words'] for r in rows):,} kelime, {sum(r['png'] for r in rows)} görsel, {sum(r['code_files'] for r in rows)} kod dosyası, "
                 f"{len(arxiv_all)} farklı arXiv kimliği", fontsize=10.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=pc[p], edgecolor=ec[p], label=p) for p in pc], fontsize=8.5, ncol=4, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax = axes[1]
    w = 0.27
    ax.bar(x - w, [r["png"] for r in rows], w, color="#cfe8ff", edgecolor="#3b6ea8", label="görsel (PNG)")
    ax.bar(x, [r["code_files"] for r in rows], w, color="#ffd166", edgecolor="#c79a00", label="kod dosyası (.py/.ipynb/.mjs)")
    ax.bar(x + w, [r["log_ran"] for r in rows], w, color="#d4f5d4", edgecolor="#2e8b57", label="log'da script başına 'çalıştırıldı/doğrulandı' (alt sınır)")
    ax.set_xticks(x); ax.set_xlabel("ders"); ax.set_ylabel("adet"); ax.legend(fontsize=8.5, loc="upper left"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(a.png, "01-inventory.png"), dpi=160); plt.close(fig)
    print("yazıldı:", os.path.join(a.png, "01-inventory.png"))
```

## design_lint.py

```python
"""Ders 31 — Proje tasarım şablonu denetçisi (rubriğin otomatik yarısı).

  python design_lint.py templates/P0-ornek.md          # TAM beklenir
  python design_lint.py templates/design-template.md   # EKSİK beklenir (boş şablon)
  python design_lint.py --selftest

Sekiz alanın her biri için yapısal kontrol + bütçe tutarlılığı: yazdığın havuz n ile MDE,
code/day27/budget.py'nin güç fonksiyonuyla (McNemar exact, r = 0,05) ±2 puan içinde uyuşmalı.
İçeriğin doğruluğunu (soru iyi mi?) denetlemez — o rubriğin insan yarısı.
Çıkış kodu: 0 TAM, 1 EKSİK, 2 selftest başarısız.
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "day27"))

FIELDS = ["Soru", "Null hipotez", "Değişkenler", "Hücreler", "Bütçe", "Başarı ölçütü", "Riskler", "Evet / hayır çıkarsa"]
BACK = {"Soru": "Ders 26 §1", "Null hipotez": "Ders 26 §2", "Değişkenler": "Ders 26 §3", "Hücreler": "Ders 27 §1",
        "Bütçe": "Ders 27 §4–5", "Başarı ölçütü": "Ders 26 §2, Ders 19", "Riskler": "Ders 27 §6, Ders 28",
        "Evet / hayır çıkarsa": "Ders 26 §1 (altıncı alan)"}


def sections(text):
    out, pat = {}, re.compile(r"^\*\*(.+?):\*\*", re.M)
    ms = list(pat.finditer(text))
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        out[m.group(1).strip()] = text[m.end():end].strip()
    return out


def num(pattern, s):
    m = re.search(pattern, s)
    return float(m.group(1).replace(",", ".")) if m else None


def lint(text, check_mde=True):
    S, problems = sections(text), []
    for f in FIELDS:
        body = S.get(f)
        if body is None:
            problems.append((f, "alan yok")); continue
        if not body or "___" in body:
            problems.append((f, "doldurulmamış ('___' duruyor)")); continue
        if f == "Null hipotez" and not (re.search(r"H0", body) and re.search(r"δ\s*=\s*\d", body)):
            problems.append((f, "H0 etiketi ya da δ = sayı yok"))
        if f == "Değişkenler" and not all(k in body for k in ("Bağımsız", "Bağımlı")):
            problems.append((f, "Bağımsız / Bağımlı ayrımı yok"))
        if f == "Hücreler" and not (re.search(r"S-(VN|OV)", body) and re.search(r"libero_\w+", body) and re.search(r"n_init\s*\d+", body)):
            problems.append((f, "sistem (S-VN/S-OV), suite (libero_*) ya da n_init sayısı eksik"))
        if f == "Bütçe":
            h, n, m = num(r"([\d,\.]+)\s*T4 sa", body), num(r"n\s*=\s*(\d+)", body), num(r"MDE\s*=\s*(\d+)", body)
            if None in (h, n, m):
                problems.append((f, "T4 saati, havuz n ya da MDE sayısı eksik"))
            elif check_mde:
                from budget import mde
                true = round(100 * mde(int(n)))
                if abs(true - m) > 2:
                    problems.append((f, f"n = {int(n)} için MDE {true} puan (r = 0,05), sen {int(m)} yazmışsın"))
        if f == "Başarı ölçütü" and not re.search(r"(p\s*<\s*0[,.]\d+|[Δκδ][^\n]{0,12}?[≥=]\s*[\d,]+|\d+\s*puan)", body):
            problems.append((f, "sayısal eşik yok (p < 0,05, |Δ| ≥ …, κ ≥ …, δ = …)"))
        if f == "Riskler" and len(re.findall(r"^\s*-\s+\S", body, re.M)) < 2:
            problems.append((f, "en az iki madde (- …) gerek"))
        if f == "Evet / hayır çıkarsa" and not (re.search(r"Evet", body) and re.search(r"Hayır", body)):
            problems.append((f, "Evet ve Hayır dallarının ikisi de yazılmalı"))
    return problems


def report(path):
    probs = lint(open(path, encoding="utf-8").read())
    ok = not probs
    print(f"{os.path.basename(path)}: {'TAM' if ok else 'EKSİK'} ({len(FIELDS) - len({p[0] for p in probs})}/{len(FIELDS)} alan)")
    for f, why in probs:
        print(f"  - {f}: {why}  → dön: {BACK[f]}")
    return ok


def selftest():
    good = open(os.path.join(HERE, "templates", "P0-ornek.md"), encoding="utf-8").read()
    assert not lint(good), lint(good)
    blank = open(os.path.join(HERE, "templates", "design-template.md"), encoding="utf-8").read()
    assert len(lint(blank, check_mde=False)) == len(FIELDS)
    bad = good.replace("MDE = 23", "MDE = 10")
    assert any("MDE 23" in w for _, w in lint(bad)), lint(bad)
    print("selftest OK: örnek TAM, boş şablon 8/8 eksik, yanlış MDE (10 yerine 23) yakalandı")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        try:
            selftest(); sys.exit(0)
        except AssertionError as e:
            print("selftest BAŞARISIZ:", e); sys.exit(2)
    sys.exit(0 if all(report(p) for p in sys.argv[1:]) else 1)
```

## plot_day31.py

```python
"""Ders 31 — görsel 04: yetkinlik radarı.
Üç çizgi: (1) ders kapsamı = yetkinliği işleyen ders sayısı / en çok (GERÇEK, self_assessment.csv'deki eşleme);
(2) senin elinden çıkan kanıt var mı (proje klasörü ya da doldurduğun blok; 'dolu:' = '___' kalmamış) — GERÇEK tarama;
(3) senin öz-değerlendirmen 0–4 (mesut_0_4 sütunu; boşsa çizilmez).
  python plot_day31.py [çıktı klasörü]
"""
import csv, glob, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "lectures", "images", "lecture-31")
rows = list(csv.DictReader(open(os.path.join(HERE, "self_assessment.csv"), encoding="utf-8")))
lab = [r["yetkinlik"] for r in rows]
cov = np.array([len(r["dersler"].split(";")) for r in rows], float); cov = cov / cov.max()
def kanit(spec):
    # "dolu:yol" = dosya var VE içinde doldurulmamış '___' kalmamış (Mesut'un doldurduğu blok); aksi glob
    if spec.startswith("dolu:"):
        p = os.path.join(ROOT, spec[5:])
        return float(os.path.exists(p) and "___" not in open(p, encoding="utf-8").read())
    return float(bool(glob.glob(os.path.join(ROOT, spec))))
ev = np.array([kanit(r["kanit"]) for r in rows])
me = [r["mesut_0_4"].strip() for r in rows]
ang = np.linspace(0, 2 * np.pi, len(rows), endpoint=False); close = lambda v: np.r_[v, v[:1]]; A = close(ang)
plt.rcParams.update({"font.size": 9.5})
fig = plt.figure(figsize=(10, 8.5)); ax = fig.add_subplot(111, polar=True)
ax.plot(A, close(cov), color="#c79a00", lw=2); ax.fill(A, close(cov), color="#ffd166", alpha=0.45, label="ders kapsamı (işleyen ders sayısı, normalize) — gerçek")
ax.plot(A, close(ev), color="#2e8b57", lw=2, marker="o"); ax.fill(A, close(ev), color="#d4f5d4", alpha=0.5, label="senin elinden kanıt: proje klasörü / doldurulmuş blok var (1) / yok (0) — gerçek tarama")
if all(m != "" for m in me):
    v = np.array([float(m) / 4 for m in me]); ax.plot(A, close(v), color="#3b6ea8", lw=2, ls="--", label="öz-değerlendirme (0–4) — Mesut")
else:
    ax.text(0.5, -0.12, "Öz-değerlendirme çizgisi boş: self_assessment.csv 'mesut_0_4' sütununu doldur ve yeniden çalıştır",
            transform=ax.transAxes, ha="center", fontsize=9, color="#c0392b")
ax.set_xticks(ang); ax.set_xticklabels([l.replace(" (", "\n(") for l in lab], fontsize=8.5); ax.tick_params(axis="x", pad=18)
ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["", "", "", ""]); ax.set_ylim(0, 1.05)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), fontsize=8.5)
fig.suptitle("Ders 31 — Yetkinlik haritası: anlatılan (sarı) ile elinle kanıtlanan (yeşil) arasındaki boşluk Aşama 0'ın devrettiği iş", fontsize=10.5, y=0.99)
fig.tight_layout(); os.makedirs(out, exist_ok=True); fig.savefig(os.path.join(out, "04-competence-radar.png"), dpi=160)
print("kanıt:", dict(zip(lab, ev.astype(int)))); print("yazıldı:", os.path.join(out, "04-competence-radar.png"))
```
