"""Lecture 26 — "definition of done" check of the RQ draft (a preregistration linter).

  python rq_lint.py ../../thesis/rq-candidates.md          # table + exit code (0 = complete, 1 = incomplete)
  python rq_lint.py --selftest                              # shows that it catches the gaps in a broken draft
  python rq_lint.py <file> --csv out.csv                   # matrix CSV (for figure 03 of plot_day26.py)

Rule: every `## RQ-<letter>` section must contain the fields — **Question**, **Null hypothesis**, **Variables**,
**Measure/Criterion and threshold**, **Suite / block / sample**, **Systems**, **If yes / if no, what the thesis says**
(Question + 6 = 7 lines).
Extra checks: at least one "H0" label in the null hypothesis section; a numeric threshold (δ/Δ/points) in the criterion section;
both a yes branch and a no branch in the "yes / no" section; a ranking table (`## Ordering rule`) exists; a predictions block exists.
No model/GPU needed.

The headings are matched in BOTH the Turkish form (the original thesis files: Soru, Null hipotez, ...) and the English form
(Question, Null hypothesis, ...), so the linter works on either language version of thesis/rq-candidates.md. The keys of the output
(CSV column names such as `Soru`, `H0_sayisi`) are kept as they were, because other scripts read them.
"""
import argparse, csv, re, sys

# canonical key (kept as the CSV column name) -> accepted heading texts (Turkish original + English variants)
FIELD_ALIASES = {
    "Soru": ["Soru", "Question"],
    "Null hipotez": ["Null hipotez", "Null hypothesis"],
    "Değişkenler": ["Değişkenler", "Variables"],
    "Ölçüt ve eşik": ["Ölçüt ve eşik", "Criterion and threshold", "Measure and threshold", "Measure and criterion"],
    "Suite / blok / örneklem": ["Suite / blok / örneklem", "Suite / block / sample"],
    "Sistemler": ["Sistemler", "Systems"],
    "Evet / hayır çıkarsa tez ne der": ["Evet / hayır çıkarsa tez ne der", "If yes / if no, what the thesis says",
                                        "What the thesis says if the answer is yes / no",
                                        "What the thesis says if yes / no"],
}
FIELDS = list(FIELD_ALIASES)
# English labels used in messages / the printed table
EN = {"Soru": "Question", "Null hipotez": "Null hypothesis", "Değişkenler": "Variables", "Ölçüt ve eşik": "Criterion and threshold",
      "Suite / blok / örneklem": "Suite / block / sample", "Sistemler": "Systems",
      "Evet / hayır çıkarsa tez ne der": "If yes / if no, what the thesis says"}

RANK_RE = r"^## (Sıralama kuralı|(Ordering|Ranking|Priority|Sorting|Selection) rule)"
PRED_RE = r"Önceden yazılan tahminler|[Pp]redictions"


def split_rqs(text):
    parts = re.split(r"^## (RQ-[a-z]) — (.+)$", text, flags=re.M)
    rqs = {}
    for i in range(1, len(parts), 3):
        body = parts[i + 2]
        body = re.split(r"^## ", body, flags=re.M)[0]  # up to the next ## heading
        rqs[parts[i]] = (parts[i + 1].strip(), body)
    return rqs


def field_text(body, field):
    names = "|".join(re.escape(n) for n in FIELD_ALIASES.get(field, [field]))
    m = re.search(r"\*\*(?:" + names + r"):?\*\*:?\s*(.+?)(?=\n\n\*\*|\n\n---|\Z)", body, flags=re.S)
    return m.group(1).strip() if m else None


def check(text):
    rows, problems = [], []
    rqs = split_rqs(text)
    if not rqs:
        problems.append("no `## RQ-x — …` section at all")
    for rq, (title, body) in rqs.items():
        row = {"rq": rq, "title": title}
        for f in FIELDS:
            t = field_text(body, f)
            row[f] = len(t.split()) if t else 0
            if not t:
                problems.append(f"{rq}: field '{EN[f]}' is missing")
        null = field_text(body, "Null hipotez") or ""
        if not re.search(r"H0", null):
            problems.append(f"{rq}: no H0 label in the null hypothesis")
        crit = field_text(body, "Ölçüt ve eşik") or ""
        if not re.search(r"(δ|Δ|κ)\s*[=≥<>]|\d+\s*(puan|points?)|p\s*<\s*0[.,]05", crit):
            problems.append(f"{rq}: no numeric threshold in the criterion")
        yn = field_text(body, "Evet / hayır çıkarsa tez ne der") or ""
        if not (re.search(r"\*(Evet|Yes)", yn) or re.search(r"etkisi var|reddi|rejected|effect exists|κ ≥", yn)) or not (
                re.search(r"\*(Hayır|No)\b", yn) or re.search(r"Yok|reddedilemedi|could not be rejected|None|κ <", yn)):
            problems.append(f"{rq}: one of the yes/no branches is missing")
        row["H0_sayisi"] = len(set(re.findall(r"H0-[a-z]\d", null)))  # number of distinct H0 labels
        rows.append(row)
    if not re.search(RANK_RE, text, flags=re.M):
        problems.append("ranking table (`## Ordering rule`) is missing")
    if not re.search(PRED_RE, text):
        problems.append("predictions block is missing (mandatory for preregistration)")
    return rows, problems


def report(rows, problems):
    if rows:
        head = ["RQ"] + [EN[f][:14] for f in FIELDS] + ["H0"]
        print(" | ".join(h.ljust(14) for h in head))
        for r in rows:
            print(" | ".join([r["rq"].ljust(14)] + [str(r[f]).ljust(14) for f in FIELDS] + [str(r["H0_sayisi"]).ljust(14)]))
        print("(cells: word count of the field; 0 = missing)")
    print(f"\nProblems: {len(problems)}")
    for p in problems:
        print("  -", p)
    print("RESULT:", "COMPLETE" if not problems else "INCOMPLETE")


SELFTEST_DOC_EN = """# test
## RQ-x — Trial
**Question:** Is there something?
**Null hypothesis:** no difference.
**Variables:** condition.
**Systems:** S-OV.
**If yes / if no, what the thesis says:** *Yes:* good.
"""

SELFTEST_DOC_TR = """# test
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
        expected = {"RQ-x: field 'Criterion and threshold' is missing", "RQ-x: field 'Suite / block / sample' is missing",
                    "RQ-x: no H0 label in the null hypothesis", "RQ-x: no numeric threshold in the criterion",
                    "RQ-x: one of the yes/no branches is missing", "ranking table (`## Ordering rule`) is missing",
                    "predictions block is missing (mandatory for preregistration)"}
        ok = True
        for label, doc in (("English headings", SELFTEST_DOC_EN), ("Turkish headings", SELFTEST_DOC_TR)):
            print(f"--- {label}")
            rows, problems = check(doc)
            report(rows, problems)
            ok = ok and expected <= set(problems)
        print("selftest:", "OK (7 gaps caught, English and Turkish headings)" if ok else "FAILED: the expected gaps were not caught")
        sys.exit(0 if ok else 2)
    if not a.path:
        ap.error("give a file path or --selftest")
    text = open(a.path, encoding="utf-8").read()
    rows, problems = check(text)
    report(rows, problems)
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["rq", "title"] + FIELDS + ["H0_sayisi"])
            for r in rows:
                w.writerow([r["rq"], r["title"]] + [r[f] for f in FIELDS] + [r["H0_sayisi"]])
        print("written:", a.csv)
    sys.exit(0 if not problems else 1)
