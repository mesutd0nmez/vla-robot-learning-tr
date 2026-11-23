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
