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
