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
