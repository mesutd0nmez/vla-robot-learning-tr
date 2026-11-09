"""Ders 16 — Türkçe bir komut LIBERO-Plus'ın BDDL ayrıştırıcısından geçince modele ne olarak ulaşır?

LIBERO-Plus, `_language_` görevlerinde komutu BDDL dosyasının `(:language ...)` alanından okur
(libero/libero/benchmark/__init__.py → grab_language_from_filename → bddl_utils.get_problem_info).
O yol `bddl.parsing.scan_tokens` kullanır: (1) ';' ile satır sonuna kadar her şeyi YORUM sayıp siler,
(2) tüm dosyayı Python `str.lower()` ile küçültür, (3) '(' ve ')' karakterlerini yapı olarak okur,
(4) kelimeleri tek boşlukla yeniden birleştirir.

Bu script gerçek `bddl` paketini (pip install bddl==1.0.1 future) kullanır; yoksa aynı 3 satırı taklit eder.
Kullanım:  python bddl_lang_probe.py [--csv cikti.csv]
"""
import argparse, csv, re, unicodedata

try:
    from bddl.parsing import scan_tokens          # LIBERO / LIBERO-Plus'ın kullandığı fonksiyon
    SRC = "bddl.parsing.scan_tokens (gerçek paket)"
except Exception:                                  # pragma: no cover
    SRC = "taklit (bddl kurulu değil)"

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
    ("EN referans", "pick up the black bowl between the plate and the ramekin and place it on the plate"),
    ("TR küçük harf", "tabak ile ramekinin arasındaki siyah kaseyi al ve tabağın üstüne koy"),
    ("TR büyük I", "Işığın altındaki kâseyi al"),
    ("TR büyük İ", "İki tabağın arasındaki kâseyi al"),
    ("TR TAMAMI BÜYÜK", "SIRADAKİ KASEYİ AL"),
    ("noktalı virgül", "kâseyi al; tabağa koy"),
    ("parantez", "kâseyi (siyah olanı) tabağa koy"),
    ("kesme işareti", "ramekin'in yanındaki kâseyi al"),
    ("boş", ""),
]


def parse_language(cmd):
    tokens = scan_tokens(string=TEMPLATE.format(cmd=cmd))
    # libero/libero/envs/bddl_utils.py get_problem_info: tokens.pop(0)=="define", sondan gruplar, ":language" grubu
    tokens = list(tokens)
    assert tokens.pop(0) == "define"
    lang = ""
    while tokens:
        g = tokens.pop()
        if isinstance(g, list) and g and g[0] == ":language":
            lang = " ".join(g[1:])   # bddl_utils.py:83 ile aynı; iç içe liste varsa TypeError
    return lang


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv"); a = ap.parse_args()
    print("ayrıştırıcı:", SRC)
    rows = []
    for label, cmd in CASES:
        try:
            out = parse_language(cmd); err = ""
        except Exception as e:
            out, err = "", f"{type(e).__name__}: {e}"
        same = out == cmd
        extra = [f"U+{ord(c):04X}" for c in out if unicodedata.combining(c)]
        note = err or ("aynı" if same else "DEĞİŞTİ") + (f" (birleşik işaret {','.join(extra)})" if extra else "")
        rows.append({"durum": label, "girdi": cmd, "modele_giden": out, "len_in": len(cmd), "len_out": len(out), "not": note})
        print(f"- {label:16s} | {cmd!r}\n  {'':16s} → {out!r}  [{note}]")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print("yazıldı:", a.csv)


if __name__ == "__main__":
    main()
