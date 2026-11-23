"""Lecture 16 — When a Turkish command passes through LIBERO-Plus's BDDL parser, what reaches the model?

For `_language_` tasks, LIBERO-Plus reads the command from the `(:language ...)` field of the BDDL file
(libero/libero/benchmark/__init__.py → grab_language_from_filename → bddl_utils.get_problem_info).
That path uses `bddl.parsing.scan_tokens`: (1) treats everything from ';' to the end of the line as a COMMENT and deletes it,
(2) lowercases the whole file with Python `str.lower()`, (3) reads the '(' and ')' characters as structure,
(4) rejoins the words with single spaces.

This script uses the real `bddl` package (pip install bddl==1.0.1 future); if it is missing it imitates the same 3 steps.
Usage:  python bddl_lang_probe.py [--csv output.csv]
"""
import argparse, csv, re, unicodedata

try:
    from bddl.parsing import scan_tokens          # the function LIBERO / LIBERO-Plus use
    SRC = "bddl.parsing.scan_tokens (real package)"
except Exception:                                  # pragma: no cover
    SRC = "imitation (bddl is not installed)"

    def scan_tokens(string):
        s = re.sub(r";.*$", "", string, flags=re.MULTILINE).lower()
        stack, tokens = [], []
        for t in re.findall(r"[()]|[^\s()]+", s):
            if t == "(":
                stack.append(tokens); tokens = []
            elif t == ")":
                if not stack:
                    raise Exception("Missing open parenthesis")
                toks, tokens = tokens, stack.pop(); tokens.append(toks)
            else:
                tokens.append(t)
        if stack:
            raise Exception("Missing close parenthesis")
        return tokens[0]

TEMPLATE = "(define (problem LIBERO_Tabletop_Manipulation)\n  (:domain robosuite)\n  (:language {cmd})\n)\n"

CASES = [
    ("EN reference", "pick up the black bowl between the plate and the ramekin and place it on the plate"),
    ("TR lowercase", "tabak ile ramekinin arasındaki siyah kaseyi al ve tabağın üstüne koy"),
    ("TR uppercase I", "Işığın altındaki kâseyi al"),
    ("TR uppercase İ", "İki tabağın arasındaki kâseyi al"),
    ("TR ALL CAPS", "SIRADAKİ KASEYİ AL"),
    ("semicolon", "kâseyi al; tabağa koy"),
    ("parentheses", "kâseyi (siyah olanı) tabağa koy"),
    ("apostrophe", "ramekin'in yanındaki kâseyi al"),
    ("empty", ""),
]


def parse_language(cmd):
    tokens = scan_tokens(string=TEMPLATE.format(cmd=cmd))
    # libero/libero/envs/bddl_utils.py get_problem_info: tokens.pop(0)=="define", groups from the end, the ":language" group
    tokens = list(tokens)
    assert tokens.pop(0) == "define"
    lang = ""
    while tokens:
        g = tokens.pop()
        if isinstance(g, list) and g and g[0] == ":language":
            lang = " ".join(g[1:])   # same as bddl_utils.py:83; TypeError if there is a nested list
    return lang


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv"); a = ap.parse_args()
    print("parser:", SRC)
    rows = []
    for label, cmd in CASES:
        try:
            out = parse_language(cmd); err = ""
        except Exception as e:
            out, err = "", f"{type(e).__name__}: {e}"
        same = out == cmd
        extra = [f"U+{ord(c):04X}" for c in out if unicodedata.combining(c)]
        note = err or ("same" if same else "CHANGED") + (f" (combining mark {','.join(extra)})" if extra else "")
        rows.append({"durum": label, "girdi": cmd, "modele_giden": out, "len_in": len(cmd), "len_out": len(out), "not": note})
        print(f"- {label:16s} | {cmd!r}\n  {'':16s} → {out!r}  [{note}]")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print("written:", a.csv)


if __name__ == "__main__":
    main()
