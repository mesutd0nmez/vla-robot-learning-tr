"""Generates _sidebar.md: folders and the markdown files inside them.
Run before every commit:  python tools/gen_sidebar.py
"""
import re, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
def title(p):
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "): return line[2:].strip()
    return p.stem
lines = ["- [Home](/)", "", "- **Lectures**"]
# Phase names come from the plan/lecture-plan.md headings ("## Phase 1 — Concepts (Lectures 1–8)")
phase_names = {}
plan = root/"plan"/"lecture-plan.md"
if plan.exists():
    for m in re.finditer(r"^## Phase (\d+) — (.+?)\s*\(", plan.read_text(encoding="utf-8"), re.M):
        phase_names[int(m.group(1))] = m.group(2).strip()
def phase(p):
    """Phase of a lecture: '**Phase X / Day N**' in the meta line (first 8 lines)."""
    m = re.search(r"Phase\s+(\d+)\s*/", "\n".join(p.read_text(encoding="utf-8").splitlines()[:8]))
    return int(m.group(1)) if m else None
by_phase, loose = {}, []
for p in sorted((root/"lectures").glob("lecture-*.md")):
    n = int(re.search(r"(\d+)", p.stem).group(1))
    t = re.sub(rf"^Lecture {n}\s*[—–-]\s*", "", title(p))
    t = t.split(":")[0].strip()          # short title: the part before the colon
    item = f'[<span class="nav-num">{n:02d}</span>{t}](/lectures/{p.name})'
    ph = phase(p)
    (by_phase.setdefault(ph, []) if ph else loose).append(item)
for ph in sorted(by_phase):              # becomes a collapsible sub-group in the menu (index.html)
    name = f" · {phase_names[ph]}" if ph in phase_names else ""
    lines.append(f"  - **Phase {ph}{name}**")
    lines += [f"    - {item}" for item in by_phase[ph]]
lines += [f"  - {item}" for item in loose]
for folder, label in [("plan","Plan"),("resources","Resources"),("literature","VLA and Multilingual VLA Literature"),("notes","Notes"),("experiments","Experiments"),("thesis","Thesis"),("slides","Slides")]:
    d = root/folder
    if not d.exists(): continue
    files = sorted(d.glob("*.md"))
    if not files: continue
    lines += ["", f"- **{label}**"]
    for p in files:
        lines.append(f'  - [<span class="nav-num">·</span>{title(p) if p.name!="README.md" else "Overview"}](/{folder}/{p.name})')
proj = root/"projects"
if proj.exists():
    lines += ["", "- **Projects**", f'  - [<span class="nav-num">·</span>Overview](/projects/README.md)']
    if (proj/"ROADMAP.md").exists():
        lines.append('  - [<span class="nav-num">·</span>Ladder (ROADMAP)](/projects/ROADMAP.md)')
    for d in sorted(x for x in proj.iterdir() if x.is_dir() and not x.name.startswith(("_", "."))):
        r = d/"README.md"
        if r.exists():
            num = ''.join(ch for ch in d.name if ch.isdigit())[:2].zfill(2) or '·'
            lines.append(f'  - [<span class="nav-num">{num}</span>{title(r).split(" — ")[-1]}](/projects/{d.name}/README.md)')
ds = root/"datasets"
if ds.exists():
    lines += ["", "- **Datasets**", f'  - [<span class="nav-num">·</span>Overview](/datasets/README.md)']
    for d in sorted(x for x in ds.iterdir() if x.is_dir() and not x.name.startswith(("_", "."))):
        r = d/"dataset-card.md"
        if r.exists():
            lines.append(f'  - [<span class="nav-num">{d.name[:2]}</span>{title(r)}](/datasets/{d.name}/dataset-card.md)')
code = root/"code"
if code.exists():
    days = sorted(x for x in code.iterdir() if x.is_dir() and not x.name.startswith("."))
    if days:
        lines += ["", "- **Code**"]
        for d in days:
            lines.append(f'  - [<span class="nav-num">{d.name[-2:]}</span>Day {int(d.name[-2:])} code](/code/{d.name}/README.md)')
(root/"_sidebar.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
print("\n".join(lines))
