"""_sidebar.md üretir: klasörler ve içindeki markdown dosyaları.
Her commit'ten önce:  python tools/gen_sidebar.py
"""
import re, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
def title(p):
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "): return line[2:].strip()
    return p.stem
lines = ["- [Ana sayfa](/README.md)", "", "- **Dersler**"]
for p in sorted((root/"lectures").glob("lecture-*.md")):
    n = int(re.search(r"(\d+)", p.stem).group(1))
    t = title(p).replace(f"Ders {n} — ", "")
    lines.append(f"  - [{n}. {t}](/lectures/{p.name})")
for folder, label in [("plan","Plan"),("resources","Kaynaklar"),("notes","Notlar"),("experiments","Deneyler"),("thesis","Tez")]:
    d = root/folder
    if not d.exists(): continue
    files = sorted(d.glob("*.md"))
    if not files: continue
    lines += ["", f"- **{label}**"]
    for p in files:
        lines.append(f"  - [{title(p) if p.name!='README.md' else 'Genel bakış'}](/{folder}/{p.name})")
code = root/"code"
if code.exists():
    days = sorted(x for x in code.iterdir() if x.is_dir() and not x.name.startswith("."))
    if days:
        lines += ["", "- **Kod**"]
        for d in days:
            lines.append(f"  - [{d.name}](/code/{d.name}/README.md)")
(root/"_sidebar.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
print("\n".join(lines))
