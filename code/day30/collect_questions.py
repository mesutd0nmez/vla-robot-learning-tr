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
