"""Lecture 31 — Project design template checker (the automatic half of the rubric).

  python design_lint.py templates/P0-ornek.md          # COMPLETE expected
  python design_lint.py templates/design-template.md   # INCOMPLETE expected (blank template)
  python design_lint.py --selftest

A structural check for each of the eight fields + budget consistency: the MDE for the pooled n you wrote must agree within
±2 points with the power function of code/day27/budget.py (McNemar exact, r = 0.05).
It does not check the quality of the content (is the question any good?) — that is the human half of the rubric.
Both the English field names (Question, Null hypothesis, ...) and the Turkish ones (Soru, Null hipotez, ...) are accepted.
Exit code: 0 COMPLETE, 1 INCOMPLETE, 2 selftest failed.
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "day27"))

FIELDS = ["Question", "Null hypothesis", "Variables", "Cells", "Budget", "Success criterion", "Risks", "If yes / if no"]
ALIAS = {"Soru": "Question", "Null hipotez": "Null hypothesis", "Değişkenler": "Variables", "Hücreler": "Cells", "Bütçe": "Budget",
         "Başarı ölçütü": "Success criterion", "Riskler": "Risks", "Evet / hayır çıkarsa": "If yes / if no"}   # Turkish -> canonical
BACK = {"Question": "Lecture 26 §1", "Null hypothesis": "Lecture 26 §2", "Variables": "Lecture 26 §3", "Cells": "Lecture 27 §1",
        "Budget": "Lecture 27 §4–5", "Success criterion": "Lecture 26 §2, Lecture 19", "Risks": "Lecture 27 §6, Lecture 28",
        "If yes / if no": "Lecture 26 §1 (sixth field)"}


def sections(text):
    out, pat = {}, re.compile(r"^\*\*(.+?):\*\*", re.M)
    ms = list(pat.finditer(text))
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        key = m.group(1).strip()
        out[ALIAS.get(key, key)] = text[m.end():end].strip()
    return out


def num(pattern, s):
    m = re.search(pattern, s)
    return float(m.group(1).replace(",", ".")) if m else None


def lint(text, check_mde=True):
    S, problems = sections(text), []
    for f in FIELDS:
        body = S.get(f)
        if body is None:
            problems.append((f, "field missing")); continue
        if not body or "___" in body:
            problems.append((f, "not filled in ('___' still there)")); continue
        if f == "Null hypothesis" and not (re.search(r"H0", body) and re.search(r"δ\s*=\s*\d", body)):
            problems.append((f, "no H0 label or no δ = number"))
        if f == "Variables" and not (re.search(r"Independent|Bağımsız", body) and re.search(r"Dependent|Bağımlı", body)):
            problems.append((f, "no Independent / Dependent distinction"))
        if f == "Cells" and not (re.search(r"S-(VN|OV)", body) and re.search(r"libero_\w+", body) and re.search(r"n_init\s*\d+", body)):
            problems.append((f, "system (S-VN/S-OV), suite (libero_*) or the n_init number is missing"))
        if f == "Budget":
            h, n, m = num(r"([\d,\.]+)\s*T4 (?:sa|h)", body), num(r"n\s*=\s*(\d+)", body), num(r"MDE\s*=\s*(\d+)", body)
            if None in (h, n, m):
                problems.append((f, "T4 hours, pooled n or the MDE number is missing"))
            elif check_mde:
                from budget import mde
                true = round(100 * mde(int(n)))
                if abs(true - m) > 2:
                    problems.append((f, f"for n = {int(n)} the MDE is {true} points (r = 0.05), you wrote {int(m)}"))
        if f == "Success criterion" and not re.search(r"(p\s*<\s*0[,.]\d+|[Δκδ][^\n]{0,12}?[≥=]\s*[\d,\.]+|\d+\s*(puan|points?))", body):
            problems.append((f, "no numeric threshold (p < 0.05, |Δ| ≥ …, κ ≥ …, δ = …)"))
        if f == "Risks" and len(re.findall(r"^\s*-\s+\S", body, re.M)) < 2:
            problems.append((f, "at least two bullets (- …) are needed"))
        if f == "If yes / if no" and not (re.search(r"Yes|Evet", body) and re.search(r"No\b|Hayır", body)):
            problems.append((f, "both the Yes and the No branch must be written"))
    return problems


def report(path):
    probs = lint(open(path, encoding="utf-8").read())
    ok = not probs
    print(f"{os.path.basename(path)}: {'COMPLETE' if ok else 'INCOMPLETE'} ({len(FIELDS) - len({p[0] for p in probs})}/{len(FIELDS)} fields)")
    for f, why in probs:
        print(f"  - {f}: {why}  → go back to: {BACK[f]}")
    return ok


def selftest():
    good = open(os.path.join(HERE, "templates", "P0-ornek.md"), encoding="utf-8").read()
    assert not lint(good), lint(good)
    blank = open(os.path.join(HERE, "templates", "design-template.md"), encoding="utf-8").read()
    assert len(lint(blank, check_mde=False)) == len(FIELDS)
    bad = good.replace("MDE = 23", "MDE = 10")
    assert any("MDE is 23" in w for _, w in lint(bad)), lint(bad)
    print("selftest OK: the example is COMPLETE, the blank template is 8/8 incomplete, a wrong MDE (10 instead of 23) was caught")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        try:
            selftest(); sys.exit(0)
        except AssertionError as e:
            print("selftest FAILED:", e); sys.exit(2)
    sys.exit(0 if all(report(p) for p in sys.argv[1:]) else 1)
