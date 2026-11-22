"""Ders 26 — RQ taslağının "definition of done" kontrolü (ön kayıt linter'ı).

  python rq_lint.py ../../thesis/rq-candidates.md          # tablo + çıkış kodu (0 = tam, 1 = eksik)
  python rq_lint.py --selftest                              # bozuk bir taslakla eksikleri yakaladığını gösterir
  python rq_lint.py <dosya> --csv out.csv                   # matris CSV (plot_day26.py görsel 03 için)

Kural: her `## RQ-<harf>` bölümünde altı alan bulunmalı — **Soru**, **Null hipotez**, **Değişkenler**,
**Ölçüt ve eşik**, **Suite / blok / örneklem**, **Sistemler**, **Evet / hayır çıkarsa tez ne der** (Soru + 6 = 7 satır).
Ek kontroller: null hipotez bölümünde en az bir "H0" etiketi; ölçüt bölümünde bir sayısal eşik (δ/Δ/puan);
"Evet / hayır" bölümünde hem evet hem hayır dalı; sıralama tablosu (`## Sıralama kuralı`) var mı; tahmin bloğu var mı.
Model/GPU gerekmez.
"""
import argparse, csv, re, sys

FIELDS = ["Soru", "Null hipotez", "Değişkenler", "Ölçüt ve eşik", "Suite / blok / örneklem", "Sistemler",
          "Evet / hayır çıkarsa tez ne der"]


def split_rqs(text):
    parts = re.split(r"^## (RQ-[a-z]) — (.+)$", text, flags=re.M)
    rqs = {}
    for i in range(1, len(parts), 3):
        body = parts[i + 2]
        body = re.split(r"^## ", body, flags=re.M)[0]  # sonraki ## başlığına kadar
        rqs[parts[i]] = (parts[i + 1].strip(), body)
    return rqs


def field_text(body, field):
    m = re.search(r"\*\*" + re.escape(field) + r":?\*\*:?\s*(.+?)(?=\n\n\*\*|\n\n---|\Z)", body, flags=re.S)
    return m.group(1).strip() if m else None


def check(text):
    rows, problems = [], []
    rqs = split_rqs(text)
    if not rqs:
        problems.append("hiç `## RQ-x — …` bölümü yok")
    for rq, (title, body) in rqs.items():
        row = {"rq": rq, "title": title}
        for f in FIELDS:
            t = field_text(body, f)
            row[f] = len(t.split()) if t else 0
            if not t:
                problems.append(f"{rq}: '{f}' alanı yok")
        null = field_text(body, "Null hipotez") or ""
        if not re.search(r"H0", null):
            problems.append(f"{rq}: null hipotezde H0 etiketi yok")
        crit = field_text(body, "Ölçüt ve eşik") or ""
        if not re.search(r"(δ|Δ|κ)\s*[=≥<>]|\d+\s*puan|p\s*<\s*0,05", crit):
            problems.append(f"{rq}: ölçütte sayısal eşik yok")
        yn = field_text(body, "Evet / hayır çıkarsa tez ne der") or ""
        if not (re.search(r"\*Evet", yn) or re.search(r"etkisi var|reddi|κ ≥", yn)) or not (
                re.search(r"\*Hayır", yn) or re.search(r"Yok|reddedilemedi|κ <", yn)):
            problems.append(f"{rq}: evet/hayır dallarından biri eksik")
        row["H0_sayisi"] = len(set(re.findall(r"H0-[a-z]\d", null)))  # farklı H0 etiketi sayısı
        rows.append(row)
    if "## Sıralama kuralı" not in text:
        problems.append("sıralama tablosu (`## Sıralama kuralı`) yok")
    if "Önceden yazılan tahminler" not in text:
        problems.append("tahmin bloğu yok (ön kayıt için zorunlu)")
    return rows, problems


def report(rows, problems):
    if rows:
        head = ["RQ"] + [f[:14] for f in FIELDS] + ["H0"]
        print(" | ".join(h.ljust(14) for h in head))
        for r in rows:
            print(" | ".join([r["rq"].ljust(14)] + [str(r[f]).ljust(14) for f in FIELDS] + [str(r["H0_sayisi"]).ljust(14)]))
        print("(hücreler: alanın sözcük sayısı; 0 = eksik)")
    print(f"\nSorun: {len(problems)}")
    for p in problems:
        print("  -", p)
    print("SONUÇ:", "TAM" if not problems else "EKSİK")


SELFTEST_DOC = """# test
## RQ-x — Deneme
**Soru:** Bir şey var mı?
**Null hipotez:** fark yok.
**Değişkenler:** koşul.
**Sistemler:** S-OV.
**Evet / hayır çıkarsa tez ne der:** *Evet:* iyi.
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    if a.selftest:
        rows, problems = check(SELFTEST_DOC)
        report(rows, problems)
        expected = {"RQ-x: 'Ölçüt ve eşik' alanı yok", "RQ-x: 'Suite / blok / örneklem' alanı yok",
                    "RQ-x: null hipotezde H0 etiketi yok", "RQ-x: ölçütte sayısal eşik yok",
                    "RQ-x: evet/hayır dallarından biri eksik", "sıralama tablosu (`## Sıralama kuralı`) yok",
                    "tahmin bloğu yok (ön kayıt için zorunlu)"}
        ok = expected <= set(problems)
        print("selftest:", "OK (7 eksik yakalandı)" if ok else "HATA: beklenen eksikler yakalanmadı")
        sys.exit(0 if ok else 2)
    if not a.path:
        ap.error("dosya yolu ver ya da --selftest")
    text = open(a.path, encoding="utf-8").read()
    rows, problems = check(text)
    report(rows, problems)
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["rq", "title"] + FIELDS + ["H0_sayisi"])
            for r in rows:
                w.writerow([r["rq"], r["title"]] + [r[f] for f in FIELDS] + [r["H0_sayisi"]])
        print("yazıldı:", a.csv)
    sys.exit(0 if not problems else 1)
