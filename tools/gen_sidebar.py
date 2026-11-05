"""_sidebar.md üretir: klasörler ve içindeki markdown dosyaları.
Her commit'ten önce:  python tools/gen_sidebar.py
"""
import re, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
def title(p):
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "): return line[2:].strip()
    return p.stem
lines = ["- [Ana sayfa](/)", "", "- **Dersler**"]
for p in sorted((root/"lectures").glob("lecture-*.md")):
    n = int(re.search(r"(\d+)", p.stem).group(1))
    t = title(p).replace(f"Ders {n} — ", "")
    t = t.split(":")[0].strip()          # kısa başlık: iki noktadan öncesi
    lines.append(f'  - [<span class="nav-num">{n:02d}</span>{t}](/lectures/{p.name})')
for folder, label in [("plan","Plan"),("resources","Kaynaklar"),("notes","Notlar"),("experiments","Deneyler"),("thesis","Tez")]:
    d = root/folder
    if not d.exists(): continue
    files = sorted(d.glob("*.md"))
    if not files: continue
    lines += ["", f"- **{label}**"]
    for p in files:
        lines.append(f'  - [<span class="nav-num">·</span>{title(p) if p.name!="README.md" else "Genel bakış"}](/{folder}/{p.name})')
proj = root/"projects"
if proj.exists():
    lines += ["", "- **Projeler**", f'  - [<span class="nav-num">·</span>Genel bakış](/projects/README.md)']
    for d in sorted(x for x in proj.iterdir() if x.is_dir() and not x.name.startswith(("_", "."))):
        r = d/"README.md"
        if r.exists():
            lines.append(f'  - [<span class="nav-num">{d.name[:2]}</span>{title(r)}](/projects/{d.name}/README.md)')
ds = root/"datasets"
if ds.exists():
    lines += ["", "- **Veri setleri**", f'  - [<span class="nav-num">·</span>Genel bakış](/datasets/README.md)']
    for d in sorted(x for x in ds.iterdir() if x.is_dir() and not x.name.startswith(("_", "."))):
        r = d/"dataset-card.md"
        if r.exists():
            lines.append(f'  - [<span class="nav-num">{d.name[:2]}</span>{title(r)}](/datasets/{d.name}/dataset-card.md)')
code = root/"code"
if code.exists():
    days = sorted(x for x in code.iterdir() if x.is_dir() and not x.name.startswith("."))
    if days:
        lines += ["", "- **Kod**"]
        for d in days:
            lines.append(f'  - [<span class="nav-num">{d.name[-2:]}</span>Gün {int(d.name[-2:])} kodu](/code/{d.name}/README.md)')
(root/"_sidebar.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
print("\n".join(lines))
