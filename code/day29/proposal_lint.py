"""Lecture 29 — "definition of done" check of the thesis proposal draft.

  python proposal_lint.py ../../thesis/proposal-v0.1.md            # report, exit code 0 = COMPLETE
  python proposal_lint.py ../../thesis/proposal-v0.1.md --max-words 700 --csv results/claims.csv
  python proposal_lint.py --selftest

Three checks:
  1. Body word count ≤ limit (the References section, HTML comments and the "Version:" line excluded) — "one page".
  2. Required sections exist: Problem, Aim, Research questions, Method, Contribution, Schedule, Risk, References.
  3. Every sentence/table row that contains a numeric claim carries a source tag:
     [Lecture NN] · [arXiv:XXXX.XXXXX] · [proposal] · [calc…]. Digits in model/label names
     (RQ-a, S-OV, 4-bit, LLaMA-2, Qwen3-VL, T4, OpenVLA-7B, WP1; no week names) do not count.
Both the Turkish original (Ders NN / öneri / hesap, Problem / Amaç / ...) and the English form (Lecture NN / proposal / calc,
Problem / Aim / ...) are accepted, so the linter works on either language version of thesis/proposal-v0.1.md.
Exit code: 0 COMPLETE, 1 INCOMPLETE, 2 selftest failed.
"""
import argparse, csv, re, sys

# canonical key (used in messages) -> heading regex (Turkish original | English)
REQUIRED = {"Problem": r"^##\s+\d*\.?\s*Problem", "Aim": r"^##\s+\d*\.?\s*(Amaç|Aim)",
            "Research questions": r"^##\s+\d*\.?\s*(Araştırma soru|Research question)", "Method": r"^##\s+\d*\.?\s*(Yöntem|Method)",
            "Contribution": r"^##\s+\d*\.?\s*(Beklenen katkı|Expected contribution)", "Schedule": r"^##\s+\d*\.?\s*(Takvim|Schedule)",
            "Risk": r"^##\s+\d*\.?\s*Risk", "References": r"^##\s+(Kaynaklar|References)"}
TAG = re.compile(r"\[(Ders \d+(?:[–-]\d+)?|Lecture \d+(?:[–-]\d+)?|arXiv:\d{4}\.\d{4,5}|öneri|proposal|hesap[^\]]*|calc[^\]]*)\]")
# Names that contain digits but are not numeric claims (removed first)
NAMES = re.compile(r"(RQ-[a-z]|S-OV|S-VN|\d-bit|bf16|LLaMA-2|Qwen3(-VL)?|OpenVLA-7B|OpenVLA-OFT|T4|M2|İP\d|WP\d|H0-\w+|"
                   r"v\d+\.\d+|lecture-\d+|Ders \d+|Lecture \d+(?:[–-]\d+)?|arXiv:\d{4}\.\d{4,5}|EN-B|tr_\w+|\bK\d\b|\bS\d+\b)")
NUM = re.compile(r"\d")


def body_and_refs(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    lines = text.splitlines()
    out, refs, in_refs = [], [], False
    for ln in lines:
        if re.match(r"^##\s+(Kaynaklar|References)", ln):
            in_refs = True
        if ln.startswith(("Sürüm:", "Version:")):
            continue
        (refs if in_refs else out).append(ln)
    return "\n".join(out), "\n".join(refs)


def words(s):
    s = re.sub(r"[`*_#|>\[\]]", " ", s)
    return [w for w in re.split(r"\s+", s) if re.search(r"\w", w)]


def claim_units(body):
    """Units at the level of a sentence (paragraph/bullet) or a table row."""
    units = []
    for ln in body.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("<!--") or re.match(r"^\|[-\s|]+\|$", s) or s.startswith("**Başlık") \
                or s.startswith("**Title") or s.startswith("**Öğrenci") or s.startswith("**Student"):
            continue
        if s.startswith("|"):
            units.append(s); continue
        s = re.sub(r"^(\d+\.|[-*])\s+", "", s)   # drop the list marker
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ\"“*(])", s):
            units.append(sent)
    return units


def lint(path, max_words):
    text = open(path, encoding="utf-8").read()
    body, refs = body_and_refs(text)
    problems, rows = [], []
    n = len(words(body))
    if n > max_words:
        problems.append(f"body {n} words > {max_words} (one-page limit)")
    for name, rx in REQUIRED.items():
        if not re.search(rx, text, flags=re.M):
            problems.append(f"'{name}' section is missing")
    untagged = 0
    for u in claim_units(body):
        stripped = NAMES.sub(" ", TAG.sub(" ", u))
        has_num = bool(NUM.search(stripped))
        tagged = bool(TAG.search(u))
        if has_num:
            rows.append({"tagged": int(tagged), "tags": ";".join(TAG.findall(u)), "text": u[:160]})
            if not tagged:
                untagged += 1
                problems.append(f"untagged numeric claim: {u[:90]}…")
    return n, rows, problems


def selftest():
    bad_en = ("# T\n\n## 1. Problem\n\nSuccess drops from 97 to 49.\n\n## 2. Aim\n\nx\n\n## 4. Method\n\nn = 50 [Lecture 27].\n\n"
              "## References\n\nA 2024\n")
    bad_tr = ("# T\n\n## 1. Problem\n\nBaşarı 97'den 49'a iniyor.\n\n## 2. Amaç\n\nx\n\n## 4. Yöntem\n\nn = 50 [Ders 27].\n\n"
              "## Kaynaklar\n\nA 2024\n")
    import tempfile, os
    want = {"'Research questions' section is missing", "'Contribution' section is missing", "'Schedule' section is missing",
            "'Risk' section is missing"}
    ok = True
    for label, bad, num in (("English", bad_en, "97"), ("Turkish", bad_tr, "97")):
        p = os.path.join(tempfile.mkdtemp(), "bad.md"); open(p, "w", encoding="utf-8").write(bad)
        n, rows, probs = lint(p, 650)
        good = want <= set(probs) and any("untagged" in x and num in x for x in probs) and not any("n = 50" in x for x in probs)
        ok = ok and good
        print(f"  {label} headings/tags: {len(probs)} problems caught", "OK" if good else "FAILED")
    print("selftest", "OK" if ok else "FAILED", "(expected per file: 4 missing sections + 1 untagged claim; English and Turkish forms)")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?"); ap.add_argument("--max-words", type=int, default=700)  # 650 for the Turkish original; the English text runs longer (680)
    ap.add_argument("--csv"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 2)
    n, rows, probs = lint(a.path, a.max_words)
    print(f"body words: {n} / {a.max_words}")
    print(f"numeric-claim units: {len(rows)}, tagged: {sum(r['tagged'] for r in rows)}")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["tagged", "tags", "text"]); w.writeheader(); w.writerows(rows)
    for p in probs:
        print("  -", p)
    print("RESULT:", "COMPLETE" if not probs else f"INCOMPLETE ({len(probs)})")
    sys.exit(0 if not probs else 1)
