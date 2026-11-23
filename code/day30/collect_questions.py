"""Lecture 30 — Collect everything that goes to the advisor from the repo: a CSV with file:line sources.

Scans the .md files in the repo with keywords for lines that await a decision
("advisor", "S7", "S9", "S10", "decision" …), writes every match with its source file and line number,
and merges the line into one row if several keys catch it.
Purpose: every question on the meeting list should have a trace in the repo (a question with no trace = an invented question).
Both the Turkish keywords (danışman, karar, iddia birimi) and the English ones (advisor, decision, claim unit) are matched,
so the scan works on either language version of the repo. The key names (CSV values such as `danisman`, `iddia_birimi`) are kept.

  python collect_questions.py                    # repo root ../.. ; results/questions_raw.csv
  python collect_questions.py --root . --out x.csv
"""
import argparse, csv, os, re
from collections import Counter

KEYS = {
    "danisman": re.compile(r"danışman|advis[eo]r", re.I),
    "S7": re.compile(r"\bS7\b"),
    "S8": re.compile(r"\bS8\b"),
    "S9": re.compile(r"\bS9\b"),
    "S10": re.compile(r"\bS10\b"),
    "karar": re.compile(r"\bkarar|\bdecision|\bdecide", re.I),
    "iddia_birimi": re.compile(r"iddia birimi|claim unit", re.I),
    "4bit": re.compile(r"4-bit", re.I),
}
SKIP_DIRS = {".git", "node_modules", "images", "slides"}  # slides: the output of this lecture, so it must not count itself


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
    print(f"{len(rows)} lines, {len({r['file'] for r in rows})} files → {a.out}")
    print("per key:", dict(sorted(kc.items(), key=lambda x: -x[1])))
    print("per folder:", dict(top.most_common()))
    # Lines addressed explicitly to the advisor (advisor + decision/question) — raw candidates for the meeting list
    cand = [r for r in rows if "danisman" in r["keys"] and re.search(r"karar|sor|konuş|decision|decid|question|ask|discuss|\?", r["text"], re.I)]
    print(f"candidate lines containing 'advisor' + decision/question: {len(cand)}")
    for r in cand[:12]:
        print(f"  {r['file']}:{r['line']}  {r['text'][:110]}")


if __name__ == "__main__":
    main()
