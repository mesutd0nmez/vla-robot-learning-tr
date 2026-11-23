# day30

Ders 30 — danışman toplantısı hazırlığı. Model yok; iki script ve bir veri dosyası. Çıktı `slides/advisor-meeting-01.md`.

| Dosya | Ne | Durum |
|---|---|---|
| `collect_questions.py` | Repodaki `.md` dosyalarında "danışman", "S7–S10", "karar", "iddia birimi", "4-bit" geçen satırları dosya:satır ile CSV'ye yazar | çalıştırıldı: Ders 27 sonundaki repoda 184 satır / 37 dosya; Ders 28–31 eklenince 290 / 45 → `results/questions_raw.csv`, `collect_stdout.txt` |
| `questions.csv` | Elle süzülmüş 11 soru: bağlam, seçenekler, önerimiz, neyi değiştirir, kaynak, etki, aciliyet, yer (toplantı / hızlı / e-posta) | bu derste yazıldı (puanlar öznel) |
| `plot_day30.py` | Öncelik sırası (`results/priority.csv`), görsel 02 (etki × aciliyet) ve 03 (30 dk gündem) | çalıştırıldı |

Mac'te (`vla` env; yalnızca `matplotlib`):
```bash
cd code/day30
python collect_questions.py
python plot_day30.py        # questions.csv'yi değiştirdikten sonra yeniden çiz
```

## collect_questions.py

```python
"""Ders 30 — Danışmana gidecek her şeyi repodan topla: dosya:satır kaynaklı CSV.

Repodaki .md dosyalarında karar bekleyen satırları anahtar sözcüklerle tarar
("danışman", "S7", "S9", "S10", "karar" …), her eşleşmeyi kaynak dosya ve satır numarasıyla
yazar ve aynı satırı birden çok anahtar yakalarsa tek satırda birleştirir.
Amaç: toplantı listesindeki her sorunun repoda bir izi olsun (izi olmayan soru = uydurma soru).

  python collect_questions.py                    # repo kökü ../.. ; results/questions_raw.csv
  python collect_questions.py --root . --out x.csv
"""
import argparse, csv, os, re
from collections import Counter

KEYS = {
    "danisman": re.compile(r"danışman", re.I),
    "S7": re.compile(r"\bS7\b"),
    "S8": re.compile(r"\bS8\b"),
    "S9": re.compile(r"\bS9\b"),
    "S10": re.compile(r"\bS10\b"),
    "karar": re.compile(r"\bkarar", re.I),
    "iddia_birimi": re.compile(r"iddia birimi", re.I),
    "4bit": re.compile(r"4-bit", re.I),
}
SKIP_DIRS = {".git", "node_modules", "images", "slides"}  # slides: bu dersin çıktısı, kendini saymasın


def scan(root):
    rows = []
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d not in SKIP_DIRS)
        for fn in sorted(fns):
            if not fn.endswith(".md") or fn == "_sidebar.md":
                continue
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, root)
            with open(p, encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    hit = [k for k, rx in KEYS.items() if rx.search(line)]
                    if hit:
                        rows.append(dict(file=rel, line=i, keys="|".join(hit),
                                         text=re.sub(r"\s+", " ", line.strip())[:240]))
    return rows


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(here, "..", ".."))
    ap.add_argument("--out", default=os.path.join(here, "results", "questions_raw.csv"))
    a = ap.parse_args()
    rows = scan(os.path.abspath(a.root))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "line", "keys", "text"]); w.writeheader(); w.writerows(rows)
    kc = Counter(k for r in rows for k in r["keys"].split("|"))
    top = Counter(r["file"].split(os.sep)[0] if os.sep in r["file"] else r["file"] for r in rows)
    print(f"{len(rows)} satır, {len({r['file'] for r in rows})} dosya → {a.out}")
    print("anahtar başına:", dict(sorted(kc.items(), key=lambda x: -x[1])))
    print("klasör başına:", dict(top.most_common()))
    # Danışmana açıkça yönelen satırlar (danışman + karar/soru) — toplantı listesinin ham adayları
    cand = [r for r in rows if "danisman" in r["keys"] and re.search(r"karar|sor|konuş|\?", r["text"], re.I)]
    print(f"'danışman' + karar/soru içeren aday satır: {len(cand)}")
    for r in cand[:12]:
        print(f"  {r['file']}:{r['line']}  {r['text'][:110]}")


if __name__ == "__main__":
    main()
```

## plot_day30.py

```python
"""Ders 30 — görseller: (02) etki × aciliyet öncelik matrisi, (03) 30 dakikalık gündem zaman çizelgesi.
Girdi: questions.csv (11 soru; etki/aciliyet 1–5 bu dersin ÖZNEL puanı — danışmanla güncellenir).
  python plot_day30.py [çıktı klasörü]
Ayrıca results/priority.csv: öncelik = etki × aciliyet, sıralı; hangi soru toplantıda / hızlı / e-postada.
"""
import csv, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

here = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "..", "lectures", "images", "lecture-30")
os.makedirs(out, exist_ok=True)
Q = list(csv.DictReader(open(os.path.join(here, "questions.csv"), encoding="utf-8")))
for q in Q:
    q["impact"], q["urgency"] = int(q["impact"]), int(q["urgency"]); q["prio"] = q["impact"] * q["urgency"]
Q.sort(key=lambda q: (-q["prio"], q["qid"]))
os.makedirs(os.path.join(here, "results"), exist_ok=True)
with open(os.path.join(here, "results", "priority.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["rank", "qid", "short", "impact", "urgency", "priority", "slot"])
    for i, q in enumerate(Q, 1):
        w.writerow([i, q["qid"], q["short"], q["impact"], q["urgency"], q["prio"], q["slot"]])
for i, q in enumerate(Q, 1):
    print(f"{i:2d}. {q['qid']:4s} {q['short']:22s} etki {q['impact']} × aciliyet {q['urgency']} = {q['prio']:2d}  → {q['slot']}")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
SLOT = {"toplantı": (Y, "#c79a00", "toplantıda konuşulur (karar)"), "hızlı": (B, "#3b6ea8", "toplantı sonu hızlı onay"),
        "e-posta": (G, "#2e8b57", "e-postayla sorulur")}

# 02 — öncelik matrisi
fig, ax = plt.subplots(figsize=(11, 8))
ax.add_patch(Rectangle((3.5, 3.5), 2, 2, color=R, alpha=0.45, zorder=0))
ax.text(5.45, 5.42, "önce bunlar", ha="right", va="top", fontsize=9, color="#c0392b")
from collections import defaultdict
cell = defaultdict(list)
for q in Q:
    cell[(q["urgency"], q["impact"])].append(q)
for (u, i), qs in cell.items():
    for k, q in enumerate(qs):
        dx = (k - (len(qs) - 1) / 2) * 0.4
        fc, ec, _ = SLOT[q["slot"]]
        ax.scatter(u + dx, i, s=1500, color=fc, edgecolor=ec, lw=1.5, zorder=3)
        ax.text(u + dx, i + 0.02, q["qid"], ha="center", va="center", fontsize=10, weight="bold", zorder=4)
        ax.text(u + dx, i - 0.3, q["short"], ha="center", va="top", fontsize=7.5, zorder=4)
for s, (fc, ec, lab) in SLOT.items():
    ax.scatter([], [], s=150, color=fc, edgecolor=ec, label=lab)
ax.set_xlim(1.3, 5.6); ax.set_ylim(1.3, 5.6)
ax.set_xticks(range(2, 6)); ax.set_yticks(range(2, 6))
ax.set_xlabel("aciliyet (1–5): cevap gelmeden hangi hücre başlayamaz?")
ax.set_ylabel("etki (1–5): cevap bütçeyi / RQ sırasını / tez cümlesini ne kadar değiştirir?")
ax.grid(alpha=0.25); ax.legend(loc="lower right", fontsize=9)
ax.set_title("Ders 30 — Danışmana gidecek 11 soru: etki × aciliyet (ÖZNEL puan, bu dersin önerisi)\n"
             "üst-sağ dört soru (Q1 suite, Q2 iddia birimi, Q3 katkı ölçütü, Q11 takvim) toplantının ana gövdesi", fontsize=10.5)
fig.tight_layout(); fig.savefig(os.path.join(out, "02-priority-matrix.png"), dpi=160); plt.close(fig)

# 03 — gündem
agenda = [  # (başlangıç dk, süre, başlık, tür)
    (0, 3, "Durum\nS1 bekliyor\n(slayt 1)", "durum"),
    (3, 4, "3 RQ + matris\n+ bütçe\n(slayt 2–3)", "durum"),
    (7, 5, "Q1 suite (S7)\n+ Q2 iddia birimi", "karar"),
    (12, 5, "Q3 katkı ölçütü (S10)", "karar"),
    (17, 4, "Q4 LoRA / GPU (S9)", "karar"),
    (21, 3, "Q11 takvim\n+ öneri formu", "karar"),
    (24, 3, "hızlı onay:\nQ5, Q6, Q10", "hızlı"),
    (27, 3, "özet: kararlar,\nsahipler, tarih", "kapanış"),
]
col = {"durum": (B, "#3b6ea8"), "karar": (Y, "#c79a00"), "hızlı": ("#eee", "#888"), "kapanış": (G, "#2e8b57")}
fig, ax = plt.subplots(figsize=(15, 3.8))
for s, d, t, k in agenda:
    fc, ec = col[k]
    ax.barh(0, d, left=s, color=fc, edgecolor=ec, height=0.6)
    ax.text(s + d / 2, 0, t, ha="center", va="center", fontsize=8.2)
ax.set_xlim(0, 30); ax.set_ylim(-0.6, 0.6); ax.set_yticks([])
ax.set_xticks(range(0, 31, 3)); ax.set_xlabel("dakika")
ax.text(30, 0.48, "Q7 Webots, Q8 Tatar vd., Q9 Cohn → toplantı öncesi e-posta (yazılı cevap)", ha="right", fontsize=8.5, color="#2e8b57")
ax.set_title("Ders 30 — 30 dakikalık gündem: 17 dakika beş karara (Q1+Q2, Q3, Q4, Q11), 7 dakika bağlama; durum anlatımı kısa tutuluyor", fontsize=10.5)
fig.tight_layout(); fig.savefig(os.path.join(out, "03-agenda.png"), dpi=160); plt.close(fig)
print("yazıldı: 02-priority-matrix.png, 03-agenda.png →", out)
```
