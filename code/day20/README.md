# day20

Lecture 20 — Designing the Turkish spatial command set: 20 EN–TR pairs × 5 surface forms = 100 rows (`datasets/04-tr-spatial-commands/`). The commands are not written by hand, they are **generated**: the suffixes come from `tr_morph.py`, the "which bowl?" label from LIBERO's 50 init states (`scene_frames.py`), the counterfactual success predicate from `make_cf_bddl.py`; `validate_set.py` checks the set against eight rules.

| File | What | Status |
|---|---|---|
| `tr_morph.py` | Small Turkish suffix generator: GEN/POSS3/ACC/DAT/LOC/ABL/-ki, vowel harmony, buffer consonants, consonant softening, pronominal n; surface form + segmentation (`tabağ-ın üst-ü-ne`) | ran (`--selftest` 42/42) |
| `scene_frames.py` | 10 Spatial scenes × 9 descriptions: the intended bowl in three frames (LIBERO naming / the image the model sees / robot) + the smallest gap over the 50 init states; `--conventions` verifies the convention from the left/right/front/back regions of LIBERO-90/10 | ran (real `.pruned_init`; 5/5 left = −y, 2/2 front = +x) |
| `make_tr_set.py` | Generates the 20 pairs: `pairs.csv` (segmentation, relation word, referent, tier, EN in training?) + `commands.csv` (en / tr_a / tr_b / tr_ascii / tr_bare) | ran |
| `validate_set.py` | BDDL-safe text, suffix consistency, continuity with `datasets/01`, referent definiteness, balance, minimal pairs (11), allomorph table, information position | ran (CLEAN) |
| `make_cf_bddl.py` | If the command points to `bowl_2`, a BDDL copy whose goal is `(On akita_black_bowl_2 plate_1)`; read back with the `bddl` package | ran (5 files; language identical, scene unchanged) — **not loaded in the LIBERO env** (no MuJoCo/LIBERO in this environment) |
| `plot_day20.py` | Figure 02 (three frames; on top of your project-2 render), 03 (design matrix), 04 (referent table), 05 (allomorphs + information position) | ran |

On the Mac (no model needed; `vla` env, numpy + matplotlib; `bddl` optional):
```bash
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO ~/src/LIBERO    # skip if you cloned it in Lecture 19
cd code/day20
python tr_morph.py --selftest
python scene_frames.py --libero ~/src/LIBERO --conventions
python scene_frames.py --libero ~/src/LIBERO --csv ../../datasets/04-tr-spatial-commands/data/referents.csv
python make_tr_set.py --out ../../datasets/04-tr-spatial-commands/data --libero ~/src/LIBERO
python validate_set.py ../../datasets/04-tr-spatial-commands/data --d01 ../../datasets/01-libero-spatial-tr-mini/data/commands.csv
python make_cf_bddl.py --libero ~/src/LIBERO --pairs ../../datasets/04-tr-spatial-commands/data/pairs.csv --out ../../datasets/04-tr-spatial-commands/data/bddl_cf
python plot_day20.py --libero ~/src/LIBERO --data ../../datasets/04-tr-spatial-commands/data \
       --render ../../projects/project-2/libero_first_episode-task-1.png --out ../../lectures/images/lecture-20
```

## tr_morph.py

```python
"""Lecture 20 — a small Turkish suffix generator for spatial commands (rule-based; lexicon + vowel harmony + buffer consonants).

The goal is NOT a general morphological analyzer: instead of writing the suffixes by hand for the ~20 nouns and ~12 relational
nouns of the command set, we generate them, so that we can (1) count how many surface forms (allomorphs) the same morpheme has,
(2) avoid errors like "tabakın / dolapın" when adding a new object to the set, (3) hand the suffix segmentation of every command
(tabağ-ın üst-ü-ne) to Lecture 21 as a column.

Rules (general rules from Göksel & Kerslake 2005, Chapters 3 and 6; only as much as this set needs):
  I ∈ {ı, i, u, ü}: last vowel back/front × unrounded/rounded        A ∈ {a, e}: last vowel back/front
  GEN  -(n)In     POSS3 -(s)I     ACC -(y)I     DAT -(y)A     LOC -DA     ABL -DAn     -ki
  After POSS3 the "pronominal n": -nI, -nA, -ndA, -ndAn, -nIn   (üst-ü-n-e, çekmece-si-n-de-ki)
  Consonant softening (marked in the lexicon): tabak → tabağ-ı, dolap → dolab-ı, ocak → ocağ-ı   (robot → robot-a: does not soften)

Usage:
  python tr_morph.py --selftest
  python tr_morph.py tabak gen        → tabağın   (tabağ-ın)
  python tr_morph.py üst dat --rel    → üstüne    (üst-ü-ne)
"""
import sys

VOWELS, BACK, ROUND = "aeıioöuüâîû", "aıouâû", "oöuüû"
SOFTEN = {"k": "ğ", "p": "b", "ç": "c", "t": "d"}
VOICELESS = "fstkçşhp"
# lexicon: bare form -> does the final consonant soften?  (only the nouns in this set)
LEXICON = {
    "tabak": True, "ocak": True, "dolap": True, "kase": False, "kâse": False, "ramekin": False, "masa": False,
    "kutu": False, "çekmece": False, "raf": False, "şişe": False, "peynir": False, "robot": False, "taraf": False,
    # relational nouns: they take a possessive suffix and are inflected with a case suffix
    "üst": False, "alt": False, "yan": False, "ön": False, "arka": False, "iç": False, "ara": False, "orta": False,
    "sağ": False, "sol": False, "üzer": False, "karşı": False,
}


def _last_vowel(s):
    for ch in reversed(s):
        if ch in VOWELS:
            return ch
    raise ValueError(f"no vowel: {s}")


def _I(s):
    v = _last_vowel(s)
    return {(True, False): "ı", (False, False): "i", (True, True): "u", (False, True): "ü"}[(v in BACK, v in ROUND)]


def _A(s):
    return "a" if _last_vowel(s) in BACK else "e"


def _ends_vowel(s):
    return s[-1] in VOWELS


def _soft(word):
    """Stem before a vowel-initial suffix: if marked in the lexicon, the final consonant softens."""
    head = word.split()[-1]
    if LEXICON.get(head, False) and word[-1] in SOFTEN:
        return word[:-1] + SOFTEN[word[-1]]
    return word


def poss3(word):
    """3rd person singular possessive: üst → üst-ü, arka → arka-sı, çekmece → çekmece-si, tabak → tabağ-ı."""
    if _ends_vowel(word):
        return word + "s" + _I(word), word + "-s" + _I(word)
    g = _soft(word)
    return g + _I(word), g + "-" + _I(word)


def case(word, c, poss=False, seg=None):
    """Adds a case suffix to word. c ∈ {nom, gen, acc, dat, loc, abl}. poss=True: word already carries the 3rd person possessive (the pronominal n appears).
    Returns (surface, segmentation); if seg is given it is used as the beginning of the segmentation."""
    seg = seg or word
    if c == "nom":
        return word, seg
    if poss:
        suf = {"gen": "n" + _I(word) + "n", "acc": "n" + _I(word), "dat": "n" + _A(word),
               "loc": "nd" + _A(word), "abl": "nd" + _A(word) + "n"}[c]
        return word + suf, seg + "-" + suf
    if c in ("loc", "abl"):
        d = "t" if word[-1] in VOICELESS else "d"
        suf = d + _A(word) + ("n" if c == "abl" else "")
        return word + suf, seg + "-" + suf
    if _ends_vowel(word):
        suf = {"gen": "n" + _I(word) + "n", "acc": "y" + _I(word), "dat": "y" + _A(word)}[c]
        return word + suf, seg + "-" + suf
    g = _soft(word)
    suf = {"gen": _I(word) + "n", "acc": _I(word), "dat": _A(word)}[c]
    return g + suf, (g if seg == word else seg[:-1] + g[-1]) + "-" + suf


def ki(surface, seg):
    """-ki: turns a locative expression into an adjective (üstünde → üstündeki, "the one on top"). Does not take part in vowel harmony."""
    return surface + "ki", seg + "-ki"


def rel(noun, c, with_ki=False):
    """Relational noun + POSS3 + case (+ki): rel('üst','dat') → üstüne; rel('ara','loc',True) → arasındaki."""
    p, pseg = poss3(noun)
    s, sseg = case(p, c, poss=True, seg=pseg)
    return ki(s, sseg) if with_ki else (s, sseg)


def bare(noun, c, with_ki=False):
    """Direction noun without possessive: sol-da-ki, sağ-da-ki, ön-de-ki (deictic use; no reference object)."""
    s, sseg = case(noun, c)
    return ki(s, sseg) if with_ki else (s, sseg)


SELFTEST = [
    (lambda: case("tabak", "gen"), "tabağın", "tabağ-ın"), (lambda: case("tabak", "dat"), "tabağa", "tabağ-a"),
    (lambda: case("tabak", "acc"), "tabağı", "tabağ-ı"), (lambda: case("tabak", "abl"), "tabaktan", "tabak-tan"),
    (lambda: case("ocak", "gen"), "ocağın", "ocağ-ın"), (lambda: case("dolap", "gen"), "dolabın", "dolab-ın"),
    (lambda: case("ramekin", "gen"), "ramekinin", "ramekin-in"), (lambda: case("masa", "gen"), "masanın", "masa-nın"),
    (lambda: case("kase", "acc"), "kaseyi", "kase-yi"), (lambda: case("kase", "gen"), "kasenin", "kase-nin"),
    (lambda: case("kâse", "acc"), "kâseyi", "kâse-yi"), (lambda: case("raf", "dat"), "rafa", "raf-a"),
    (lambda: case("raf", "gen"), "rafın", "raf-ın"), (lambda: case("robot", "dat"), "robota", "robot-a"),
    (lambda: case("kurabiye kutusu", "gen", poss=True), "kurabiye kutusunun", "kurabiye kutusu-nun"),
    (lambda: case("şarap şişesi", "acc", poss=True), "şarap şişesini", "şarap şişesi-ni"),
    (lambda: case("krem peynir", "acc"), "krem peyniri", "krem peynir-i"),
    (lambda: case(*[poss3("üst çekmece")[0]], "loc", poss=True), "üst çekmecesinde", "üst çekmecesi-nde"),
    (lambda: rel("üst", "dat"), "üstüne", "üst-ü-ne"), (lambda: rel("üst", "loc", True), "üstündeki", "üst-ü-nde-ki"),
    (lambda: rel("üst", "abl"), "üstünden", "üst-ü-nden"), (lambda: rel("üzer", "dat"), "üzerine", "üzer-i-ne"),
    (lambda: rel("yan", "loc", True), "yanındaki", "yan-ı-nda-ki"), (lambda: rel("yan", "dat"), "yanına", "yan-ı-na"),
    (lambda: rel("ara", "loc", True), "arasındaki", "ara-sı-nda-ki"), (lambda: rel("ara", "dat"), "arasına", "ara-sı-na"),
    (lambda: rel("orta", "loc", True), "ortasındaki", "orta-sı-nda-ki"), (lambda: rel("orta", "abl"), "ortasından", "orta-sı-ndan"),
    (lambda: rel("iç", "dat"), "içine", "iç-i-ne"), (lambda: rel("iç", "loc", True), "içindeki", "iç-i-nde-ki"),
    (lambda: rel("ön", "dat"), "önüne", "ön-ü-ne"), (lambda: rel("arka", "dat"), "arkasına", "arka-sı-na"),
    (lambda: rel("sağ", "dat"), "sağına", "sağ-ı-na"), (lambda: rel("sol", "dat"), "soluna", "sol-u-na"),
    (lambda: rel("alt", "dat"), "altına", "alt-ı-na"), (lambda: rel("karşı", "loc"), "karşısında", "karşı-sı-nda"),
    (lambda: bare("sol", "loc", True), "soldaki", "sol-da-ki"), (lambda: bare("sağ", "loc", True), "sağdaki", "sağ-da-ki"),
    (lambda: bare("ön", "loc", True), "öndeki", "ön-de-ki"), (lambda: bare("arka", "loc", True), "arkadaki", "arka-da-ki"),
    (lambda: bare("üst", "loc", True), "üstteki", "üst-te-ki"), (lambda: bare("ocak", "loc", True), "ocaktaki", "ocak-ta-ki"),
]


def selftest():
    bad = 0
    for fn, surf, seg in SELFTEST:
        got = fn()
        ok = got == (surf, seg)
        bad += not ok
        if not ok:
            print(f"  ERROR expected {surf} ({seg}) → produced {got[0]} ({got[1]})")
    print(f"tr_morph selftest: {len(SELFTEST) - bad}/{len(SELFTEST)} correct")
    return bad == 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    word, c = args[0], args[1]
    s, seg = rel(word, c, "--ki" in sys.argv) if "--rel" in sys.argv else case(word, c)
    print(f"{s}   ({seg})")
```

## scene_frames.py

```python
"""Lecture 20 — which one is "the bowl on the left"? In the 10 scenes of LIBERO-Spatial, computes the descriptions that
separate the two black bowls (left/right, front/back, near/far from the plate, near/far from the robot, higher) from the
REAL coordinates of the 50 init states, and writes which bowl is meant in THREE frames for each description:

  libero : LIBERO's own BDDL naming (an observer standing where the agentview camera is; no mirror)
           right = +y, left = -y, front = +x (near the observer), back = -x      [verified from the LIBERO-90/10 region names]
  image  : the pixels the model sees (obs[::-1, ::-1] = horizontal MIRROR of the real image): left = small u, front = large v
  robot  : the robot's own body (base x = -0.66, facing +x): left = +y, front/near = near the base

torch / LIBERO / MuJoCo are NOT needed: .pruned_init is opened with a restricted numpy unpickler (Lecture 19), the camera pose
was taken by hand from libero_tabletop_manipulation.py:188-197 (fovy = MuJoCo default 45 degrees).

Usage:
  python scene_frames.py --libero ~/src/LIBERO                    # 10 scenes × description table
  python scene_frames.py --libero ~/src/LIBERO --csv referents.csv
  python scene_frames.py --libero ~/src/LIBERO --conventions      # left/right/front/back regions in LIBERO-90/10
"""
import argparse, codecs, csv, glob, io, pathlib, pickle, re, zipfile
import numpy as np

SUITE = "libero_spatial"
ORDER_FILE = "libero/libero/benchmark/libero_suite_task_map.py"
CAM_POS = np.array([0.6586131746834771, 0.0, 1.6103500240372423])           # agentview (tabletop)
CAM_QUAT = (0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349)  # w x y z
ROBOT_BASE = np.array([-0.66, 0.0])   # mounted_panda.py:45 → -0.16 - table_length/2, table_length = 1.0
MIN_M, MIN_PX = 0.05, 20.0            # "unambiguous" threshold: at least 5 cm / 20 pixels of difference in all 50 scenes


class _NumpyOnly(pickle.Unpickler):
    def find_class(self, mod, name):
        if name == "_reconstruct" and mod in ("numpy.core.multiarray", "numpy._core.multiarray"):
            try:
                import numpy._core.multiarray as m
            except ImportError:
                import numpy.core.multiarray as m
            return m._reconstruct
        if (mod, name) == ("numpy", "ndarray"): return np.ndarray
        if (mod, name) == ("numpy", "dtype"): return np.dtype
        if (mod, name) == ("_codecs", "encode"): return codecs.encode
        raise pickle.UnpicklingError(f"disallowed object: {mod}.{name}")


def load_init(path):
    z = zipfile.ZipFile(path)
    pkl = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    return np.asarray(_NumpyOnly(io.BytesIO(z.read(pkl))).load())


def official_order(root, suite=SUITE):
    t = (root / ORDER_FILE).read_text(encoding="utf-8")
    return re.findall(r'"([^"]+)"', re.search(r'"%s":\s*\[(.*?)\]' % suite, t, re.S).group(1))


def parse_bddl(path):
    t = pathlib.Path(path).read_text(encoding="utf-8")
    regions = {}
    for m in re.finditer(r"\(\s*(\w+)\s*\n?\s*\(:target\s+(\w+)\)\s*(?:\(:ranges\s*\(\s*\(([^)]*)\)\s*\)\s*\))?", t):
        name, target, rng = m.groups()
        regions[f"{target}_{name}"] = tuple(map(float, rng.split())) if rng else None
    objs = []
    for line in re.search(r"\(:objects(.*?)\)\s*\(:obj_of_interest", t, re.S).group(1).strip().splitlines():
        if " - " in line:
            objs += line.strip().rsplit(" - ", 1)[0].split()
    init_txt = re.search(r"\(:init(.*?)\)\s*\(:goal", t, re.S).group(1)
    where = {o: reg for _, o, reg in re.findall(r"\((On|In)\s+(\S+)\s+(\S+)\)", init_txt)}
    goal = re.sub(r"\s+", " ", t.split("(:goal")[1]).strip()
    return dict(objects=objs, regions=regions, where=where, goal=goal)


def cam_rotation(q=CAM_QUAT):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def project(p, W=256, fovy=45.0, mirror=True):
    """World point → pixel (u, v). The MuJoCo camera looks along -z, +x is right, +y is up.
    mirror=True: the image the model sees (the raw image comes vertically flipped; [::-1, ::-1] turns it into a horizontal mirror)."""
    pc = (np.asarray(p) - CAM_POS) @ cam_rotation()
    f = 0.5 * W / np.tan(np.radians(fovy) / 2)
    u = W / 2 + f * pc[..., 0] / (-pc[..., 2])
    v = W / 2 - f * pc[..., 1] / (-pc[..., 2])
    return np.stack([W - u if mirror else u, v], -1)


def positions(states, objects):
    """{object: (50, 3)} — qpos = 1 time + 7 arm + 2 gripper + 7 per object (x y z qw qx qy qz), BDDL :objects order."""
    return {o: states[:, 10 + 7 * k: 13 + 7 * k] for k, o in enumerate(objects)}


def _pick(score1, score2, thr, smaller_wins):
    """From the scores of the two bowls (50,), return the winner and the SMALLEST gap over the 50 scenes; 'ambiguous' if the sign changes."""
    d = score2 - score1 if smaller_wins else score1 - score2     # > 0 → bowl_1 kazanır
    if (d > 0).all():  who = "bowl_1"
    elif (d < 0).all(): who = "bowl_2"
    else: return "ambiguous", 0.0, False
    margin = float(np.abs(d).min())
    return who, margin, margin >= thr


def descriptors(pos):
    b1, b2, plate = pos["akita_black_bowl_1"], pos["akita_black_bowl_2"], pos["plate_1"]
    uv1, uv2 = project(b1), project(b2)
    dp1, dp2 = np.linalg.norm(b1[:, :2] - plate[:, :2], axis=1), np.linalg.norm(b2[:, :2] - plate[:, :2], axis=1)
    dr1, dr2 = np.linalg.norm(b1[:, :2] - ROBOT_BASE, axis=1), np.linalg.norm(b2[:, :2] - ROBOT_BASE, axis=1)
    rows = []
    def add(desc, frame, s1, s2, thr, smaller, unit):
        who, m, ok = _pick(s1, s2, thr, smaller)
        rows.append(dict(descriptor=desc, frame=frame, referent=who, min_margin=round(m * (100 if unit == "cm" else 1), 1), unit=unit, unambiguous=ok))
    # left / right
    add("left", "libero", b1[:, 1], b2[:, 1], MIN_M, True, "cm")        # left = small y
    add("left", "image", uv1[:, 0], uv2[:, 0], MIN_PX, True, "px")     # left = small u (mirrored image)
    add("left", "robot", b1[:, 1], b2[:, 1], MIN_M, False, "cm")       # the robot's left = large y
    # front (back = the other bowl)
    add("front", "libero", b1[:, 0], b2[:, 0], MIN_M, False, "cm")     # front = large x (near the observer)
    add("front", "image", uv1[:, 1], uv2[:, 1], MIN_PX, False, "px")   # front = lower in the image (large v)
    add("front", "robot", dr1, dr2, MIN_M, True, "cm")                 # in front of / near the robot = near the base
    # frame-independent (reference object is explicit)
    add("near_plate", "none", dp1, dp2, MIN_M, True, "cm")
    add("near_robot", "none", dr1, dr2, MIN_M, True, "cm")
    add("higher", "none", b1[:, 2], b2[:, 2], MIN_M, False, "cm")
    return rows


def analyze(root):
    out = []
    for i, name in enumerate(official_order(root)):
        info = parse_bddl(root / "libero/libero/bddl_files" / SUITE / (name + ".bddl"))
        st = load_init(root / "libero/libero/init_files" / SUITE / (name + ".pruned_init"))
        pos = positions(st, info["objects"])
        # order check: are objects with a table region within ±1.5 cm of their own region (the verification from Lecture 19)
        for o, xyz in pos.items():
            r = info["regions"].get(info["where"].get(o))
            if r:
                x0, y0, x1, y1 = r
                ok = ((xyz[:, 0] > x0 - .015) & (xyz[:, 0] < x1 + .015) & (xyz[:, 1] > y0 - .015) & (xyz[:, 1] < y1 + .015)).all()
                assert ok, f"task {i}: {o} is not in its region — the object-order assumption is broken"
        for d in descriptors(pos):
            out.append(dict(task_id=i, bowl_1_at=info["where"]["akita_black_bowl_1"].replace("main_table_", ""),
                            bowl_2_at=info["where"]["akita_black_bowl_2"].replace("main_table_", ""), **d))
    return out


def conventions(root):
    """Centers of the regions whose names contain left/right/front/back in the LIBERO-90/10 BDDLs: evidence for the frame convention."""
    seen = {}
    for p in sorted(glob.glob(str(root / "libero/libero/bddl_files/libero_90/*.bddl")) + glob.glob(str(root / "libero/libero/bddl_files/libero_10/*.bddl"))):
        scene = re.match(r"([A-Z_]+SCENE\d+)", pathlib.Path(p).name).group(1)
        for k, r in parse_bddl(p)["regions"].items():
            m = re.search(r"(left|right|front|back)", k)
            if r and m and "contain" not in k:
                seen[(scene, k)] = (m.group(1), (r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
    # compare the left/right and front/back pairs in the same scene
    pairs = []
    for (scene, k), (word, x, y) in seen.items():
        other = {"left": "right", "front": "back"}.get(word)
        k2 = k.replace(word, other) if other else None
        if k2 and (scene, k2) in seen:
            _, x2, y2 = seen[(scene, k2)]
            pairs.append((scene, k, k2, word, x, y, x2, y2))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero", required=True); ap.add_argument("--csv"); ap.add_argument("--conventions", action="store_true")
    a = ap.parse_args()
    root = pathlib.Path(a.libero).expanduser()
    if a.conventions:
        pr = conventions(root)
        print(f"{'scene':24s} {'pair':46s} Δx (cm)  Δy (cm)   reading")
        ok_lr = ok_fb = n_lr = n_fb = 0
        for scene, k, k2, word, x, y, x2, y2 in pr:
            dx, dy = (x - x2) * 100, (y - y2) * 100
            if word == "left":
                n_lr += 1; ok_lr += dy < 0; note = "left = smaller y" if dy < 0 else "REVERSED"
            else:
                n_fb += 1; ok_fb += dx > 0; note = "front = larger x" if dx > 0 else "REVERSED"
            label = k.split("table_")[-1] + " | " + k2.split("table_")[-1]
            print(f"{scene:24s} {label[:46]:46s} {dx:7.1f}  {dy:7.1f}   {note}")
        print(f"\nleft/right pairs: {ok_lr}/{n_lr} 'left = -y';  front/back pairs: {ok_fb}/{n_fb} 'front = +x'")
        return
    rows = analyze(root)
    cur = None
    for r in rows:
        if r["task_id"] != cur:
            cur = r["task_id"]; print(f"\ntask {cur}: bowl_1 @ {r['bowl_1_at']} | bowl_2 @ {r['bowl_2_at']}")
        flag = "" if r["unambiguous"] else "   (ambiguous: below threshold)"
        print(f"   {r['descriptor']:11s} {r['frame']:7s} → {r['referent']:8s} min gap {r['min_margin']:6.1f} {r['unit']}{flag}")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"\nwritten: {a.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
```

## make_tr_set.py

```python
"""Lecture 20 — generates the 20-pair EN–TR spatial command set: datasets/04-tr-spatial-commands/data/{pairs,commands}.csv

Design (Section 5 of the lecture):
  A (6)  the TRAINED commands of LIBERO-Spatial — selection: "X'in REL-POSS-LOC-ki siyah kaseyi al ..." (take the black bowl that is REL of X ...)
  B (6)  descriptions in the same Spatial scenes that match NO training sentence (left/right/front, near/far from the robot/plate);
         which bowl is meant is read from referents.csv, which scene_frames.py computes from the 50 init states
  C (6)  the trained commands of LIBERO-Goal — placement: "... X'in REL-POSS-DAT koy" (put ... on REL of X)
  D (2)  untrained placement (to the right / to the left): no predicate -> today only the language side (Lecture 21 tokenization)

5 surface forms per pair: en | tr_a (GENERATED with the suffix generator, with its segmentation) | tr_b (a second, hand-written form) |
  tr_ascii (tr_a without Turkish characters) | tr_bare (bag of words with the suffixes stripped)   -> 20 × 5 = 100 rows.

Note: the cell values written to pairs.csv / commands.csv (relation, evet/hayır, note, ...) are DATA and deliberately stay Turkish.

Usage:
  python make_tr_set.py --out ../../datasets/04-tr-spatial-commands/data [--referents .../referents.csv] [--libero ~/src/LIBERO]
  If --libero is given, en_seen_* is filled in: whether the EN command occurs among LIBERO's 130 task names.
"""
import argparse, csv, pathlib, re
import tr_morph as M

TR2ASCII = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def N(lemma, c="nom", poss=False):
    """Noun + case → (surface, segmentation, bare). poss=True: lemma is already a possessive compound (kurabiye kutusu, şarap şişesi)."""
    s, seg = M.case(lemma, c, poss=poss)
    return (s, seg, lemma)


def R(noun, c, ki=False):
    s, seg = M.rel(noun, c, ki)
    return (s, seg, noun)


def B(noun, c, ki=False):
    s, seg = M.bare(noun, c, ki)
    return (s, seg, noun)


def W(*words):
    return [(w, w, w) for w in words]


def POSSN(owner_gen, head, c, ki=False):
    """Possessed head of a definite compound: (dolab-ın) üst çekmece-si-nde-ki."""
    p, pseg = M.poss3(head)
    s, seg = M.case(p, c, poss=True, seg=pseg)
    if ki:
        s, seg = M.ki(s, seg)
    return (s, seg, head)


TABAK_USTUNE = [N("tabak", "gen"), R("üst", "dat")]
SEL_TAIL = W("siyah") + [N("kase", "acc")] + W("al", "ve") + TABAK_USTUNE + W("koy")   # siyah kaseyi al ve tabağın üstüne koy
BASE = {0: "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        1: "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        2: "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        4: "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        7: "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        8: "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        9: "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"}
PICK = "pick up the black bowl {} and place it on the plate"

# (id, block, tier, suite, scene task, relation, relational noun, case, frame-dependent?, EN, tr_a parts, part index of the relation word,
#  part index of the reference noun (None if absent), tr_b, descriptor (referents.csv key or None), goal predicate, note)
P = [
 ("A01", "A", 1, "libero_spatial", 0, "2 referans", "ara", "LOC+ki", "hayır", PICK.format("between the plate and the ramekin"),
  [N("tabak")] + W("ile") + [N("ramekin", "gen"), R("ara", "loc", True)] + SEL_TAIL, 3, 2,
  "tabakla ramekin arasında duran siyah kâseyi alıp tabağa koy", None, "(On akita_black_bowl_1 plate_1)", "eğitim komutu; = datasets/01 sp0_tr"),
 ("A02", "A", 1, "libero_spatial", 1, "mesafe", "yan", "LOC+ki", "hayır", PICK.format("next to the ramekin"),
  [N("ramekin", "gen"), R("yan", "loc", True)] + SEL_TAIL, 1, 0,
  "ramekinin yanında duran siyah kâseyi alıp tabağa koy", None, "(On akita_black_bowl_1 plate_1)", "eğitim komutu; = sp1_tr"),
 ("A03", "A", 1, "libero_spatial", 2, "bölge", "orta", "LOC+ki", "hayır", PICK.format("from table center"),
  [N("masa", "gen"), R("orta", "loc", True)] + SEL_TAIL, 1, 0,
  "siyah kâseyi masanın ortasından alıp tabağa koy", None, "(On akita_black_bowl_1 plate_1)", "EN 'from' = kaynak; tr_b ayrılma hâli (ABL) kullanır; = sp2_tr"),
 ("A04", "A", 1, "libero_spatial", 4, "NTPP (içinde)", "—", "LOC+ki", "hayır", PICK.format("in the top drawer of the wooden cabinet"),
  W("ahşap") + [N("dolap", "gen"), POSSN(None, "üst çekmece", "loc", True)] + SEL_TAIL, 2, 1,
  "ahşap dolabın üst çekmecesinin içindeki siyah kâseyi alıp tabağa koy", None, "(On akita_black_bowl_1 plate_1)", "EN 'in' → tr_a'da ilişki adı YOK (yalın bulunma hâli); tr_b'de iç-i-nde-ki; = sp4_tr"),
 ("A05", "A", 1, "libero_spatial", 7, "EC (üstünde)", "üst", "LOC+ki", "hayır", PICK.format("on the stove"),
  [N("ocak", "gen"), R("üst", "loc", True)] + SEL_TAIL, 1, 0,
  "ocağın üzerinde duran siyah kâseyi alıp tabağa koy", None, "(On akita_black_bowl_1 plate_1)", "görev 9 ile AYNI sahne: yalnızca komut ayırır (Ders 19)"),
 ("A06", "A", 1, "libero_spatial", 9, "EC (üstünde)", "üst", "LOC+ki", "hayır", PICK.format("on the wooden cabinet"),
  W("ahşap") + [N("dolap", "gen"), R("üst", "loc", True)] + SEL_TAIL, 2, 1,
  "ahşap dolabın üzerinde duran siyah kâseyi alıp tabağa koy", None, "(On akita_black_bowl_1 plate_1)", "görev 7 ile AYNI sahne"),
 ("B07", "B", 2, "libero_spatial", 1, "yön (sol)", "sol", "LOC+ki (iyeliksiz)", "evet", PICK.format("on the left"),
  [B("sol", "loc", True)] + SEL_TAIL, 0, None,
  "sol taraftaki siyah kâseyi alıp tabağa koy", "left", "çerçeveye göre bowl_1 ya da bowl_2", "deiktik: referans nesne yok, çerçeve örtük"),
 ("B08", "B", 2, "libero_spatial", 1, "yön (sağ)", "sağ", "LOC+ki (iyeliksiz)", "evet", PICK.format("on the right"),
  [B("sağ", "loc", True)] + SEL_TAIL, 0, None,
  "sağ taraftaki siyah kâseyi alıp tabağa koy", "right", "çerçeveye göre bowl_1 ya da bowl_2", "B07'nin minimal çifti (tek sözcük)"),
 ("B09", "B", 2, "libero_spatial", 8, "yön (ön)", "ön", "LOC+ki (iyeliksiz)", "evet", PICK.format("in the front"),
  [B("ön", "loc", True)] + SEL_TAIL, 0, None,
  "ön taraftaki siyah kâseyi alıp tabağa koy", "front", "çerçeveye göre bowl_1 ya da bowl_2", "ön/arka: görüntü ile robot çerçevesi ZIT"),
 ("B10", "B", 2, "libero_spatial", 8, "mesafe (açık çerçeve)", "yakın", "DAT + sıfat", "hayır", PICK.format("closer to the robot"),
  [N("robot", "dat")] + W("daha", "yakın", "olan") + SEL_TAIL, 2, 0,
  "robota yakın duran siyah kâseyi alıp tabağa koy", "near_robot", "(On akita_black_bowl_2 plate_1)", "B09'un çerçevesi açık yazılmış hâli; karşı-olgusal hedef"),
 ("B11", "B", 2, "libero_spatial", 0, "mesafe", "uzak", "DAT + sıfat", "hayır", PICK.format("farther from the plate"),
  [N("tabak", "dat")] + W("daha", "uzak", "olan") + SEL_TAIL, 2, 0,
  "tabaktan uzakta duran siyah kâseyi alıp tabağa koy", "far_plate", "(On akita_black_bowl_2 plate_1)", "Ders 19'un vaadi: eğitim cümlesi olmayan karşı-olgusal tarif"),
 ("B12", "B", 1, "libero_spatial", 0, "mesafe", "yakın", "DAT + sıfat", "hayır", PICK.format("closer to the plate"),
  [N("tabak", "dat")] + W("daha", "yakın", "olan") + SEL_TAIL, 2, 0,
  "tabağın yakınındaki siyah kâseyi alıp tabağa koy", "near_plate", "(On akita_black_bowl_1 plate_1)", "B11'in minimal çifti; hedef eğitimdekiyle aynı kâse"),
 ("C13", "C", 1, "libero_goal", 8, "EC (üstüne)", "üst", "DAT", "hayır", "put the bowl on the plate",
  [N("kase", "acc")] + TABAK_USTUNE + W("koy"), 2, 1,
  "tabağın üzerine kâseyi koy", None, "(On akita_black_bowl_1 plate_1)", "Goal: tek sahne; C14/C15 ile minimal üçlü"),
 ("C14", "C", 1, "libero_goal", 1, "EC (üstüne)", "üst", "DAT", "hayır", "put the bowl on the stove",
  [N("kase", "acc"), N("ocak", "gen"), R("üst", "dat")] + W("koy"), 2, 1,
  "ocağın üzerine kâseyi koy", None, "(On akita_black_bowl_1 flat_stove_1_cook_region)", "Ders 19 conditions_goal8.json'daki wrong komutu"),
 ("C15", "C", 1, "libero_goal", 4, "EC (üstüne)", "üst", "DAT", "hayır", "put the bowl on top of the cabinet",
  [N("kase", "acc"), N("dolap", "gen"), R("üst", "dat")] + W("koy"), 2, 1,
  "dolabın üzerine kâseyi koy", None, "(On akita_black_bowl_1 wooden_cabinet_1_top_side)", "BDDL :language 'on the top of the drawer' der; modele dosya adındaki metin gider"),
 ("C16", "C", 1, "libero_goal", 6, "NTPP (içine)", "iç", "DAT", "hayır", "put the cream cheese in the bowl",
  [N("krem peynir", "acc"), N("kase", "gen"), R("iç", "dat")] + W("koy"), 2, 1,
  "kâsenin içine krem peyniri koy", None, "(On cream_cheese_1 akita_black_bowl_1)", "komut 'in' der, predicate On'dur: TR çevirisi iç-i-ne mi üst-ü-ne mi? (dil ≠ dünya)"),
 ("C17", "C", 1, "libero_goal", 9, "EC (üstüne)", "—", "DAT (yalın)", "hayır", "put the wine bottle on the rack",
  [N("şarap şişesi", "acc", poss=True), N("raf", "dat")] + W("koy"), 1, 1,
  "şarap şişesini rafın üstüne yerleştir", None, "(On wine_bottle_1 wine_rack_1_top_region)", "EN 'on' → doğal TR'de ilişki adı yok (raf-a); tr_b açık biçim"),
 ("C18", "C", 1, "libero_goal", 5, "yön (ön)", "ön", "DAT", "evet", "push the plate to the front of the stove",
  [N("tabak", "acc"), N("ocak", "gen"), R("ön", "dat")] + W("it"), 2, 1,
  "ocağın önüne doğru tabağı it", None, "(On plate_1 main_table_stove_front_region)", "LIBERO'da ocağın 'önü' = +x (kameraya bakan yüz; robota göre ARKA)"),
 ("D19", "D", 3, "libero_goal", 8, "yön (sağ)", "sağ", "DAT", "evet", "put the bowl to the right of the plate",
  [N("kase", "acc"), N("tabak", "gen"), R("sağ", "dat")] + W("koy"), 2, 1,
  "tabağın sağ tarafına kâseyi koy", None, "yok — yeni bölge predicate'i gerekir (Ders 7: sağ = yarı düzlem)", "eğitilmemiş; LIBERO-90'da benzeri var (white bowl / right of the plate)"),
 ("D20", "D", 3, "libero_goal", 8, "yön (sol)", "sol", "DAT", "evet", "put the bowl to the left of the plate",
  [N("kase", "acc"), N("tabak", "gen"), R("sol", "dat")] + W("koy"), 2, 1,
  "tabağın sol tarafına kâseyi koy", None, "yok — yeni bölge predicate'i gerekir", "D19'un minimal çifti: sağ-ı-na / sol-u-na (I ünlüsü ı/u)"),
]
COMPLEMENT = {"right": ("left", True), "far_plate": ("near_plate", True)}   # the other bowl


def load_referents(path):
    d = {}
    if path and pathlib.Path(path).exists():
        for r in csv.DictReader(open(path, encoding="utf-8")):
            d[(int(r["task_id"]), r["descriptor"], r["frame"])] = (r["referent"], float(r["min_margin"]), r["unit"], r["unambiguous"] == "True")
    return d


def referent(refs, task, desc, frame):
    flip = False
    if desc in COMPLEMENT:
        desc, flip = COMPLEMENT[desc]
    if (task, desc, frame) not in refs:
        return "", ""
    who, m, unit, ok = refs[(task, desc, frame)]
    if flip and who.startswith("bowl_"):
        who = "bowl_2" if who == "bowl_1" else "bowl_1"
    return (who if ok else who + "?"), f"{m:g} {unit}"


def libero_commands(root):
    out = {}
    if not root:
        return out
    t = (pathlib.Path(root).expanduser() / "libero/libero/benchmark/libero_suite_task_map.py").read_text(encoding="utf-8")
    for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]:
        names = re.findall(r'"([^"]+)"', re.search(r'"%s":\s*\[(.*?)\]' % suite, t, re.S).group(1))
        out[suite] = {re.sub(r"^[A-Z_]+SCENE\d+_", "", n).replace("_", " ") for n in names}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="."); ap.add_argument("--referents"); ap.add_argument("--libero")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    refs = load_referents(a.referents or out / "referents.csv")
    seen = libero_commands(a.libero)
    pairs, cmds = [], []
    for (pid, block, tier, suite, task, relation, relnoun, case, framedep, en, parts, i_rel, i_ref, tr_b, desc, goal, note) in P:
        tr_a = " ".join(p[0] for p in parts)
        tr_seg = " ".join(p[1] for p in parts)
        tr_bare = " ".join(p[2] for p in parts)
        tr_ascii = tr_a.translate(TR2ASCII)
        rl = {f: referent(refs, task, desc, f) for f in ("libero", "image", "robot", "none")} if desc else {}
        if desc and rl["none"][0]:                       # frame-independent description: the same value in all three columns
            rl = {f: rl["none"] for f in rl}
        row = dict(pair_id=pid, block=block, tier=tier, suite=suite, scene_task_id=task, relation=relation, rel_noun=relnoun,
                   case=case, frame_dependent=framedep, en=en, tr_a=tr_a, tr_a_seg=tr_seg, tr_b=tr_b, tr_ascii=tr_ascii, tr_bare=tr_bare,
                   rel_span=parts[i_rel][0], rel_seg=parts[i_rel][1], ref_span=parts[i_ref][0] if i_ref is not None else "",
                   ref_libero=rl.get("libero", ("bowl_1" if block == "A" else "", ""))[0] if desc else ("bowl_1" if block == "A" else "n/a"),
                   ref_image=rl["image"][0] if desc else ("bowl_1" if block == "A" else "n/a"),
                   ref_robot=rl["robot"][0] if desc else ("bowl_1" if block == "A" else "n/a"),
                   ref_min_margin=(rl["libero"][1] + " | " + rl["image"][1] + " | " + rl["robot"][1]) if desc and framedep == "evet" else (rl["libero"][1] if desc else ""),
                   en_seen_suite=("evet" if en in seen.get(suite, ()) else "hayır") if seen else "",
                   en_seen_libero=("evet" if any(en in s for s in seen.values()) else "hayır") if seen else "",
                   goal_predicate=goal, note=note)
        pairs.append(row)
        for variant, lang, text in [("en", "en", en), ("tr_a", "tr", tr_a), ("tr_b", "tr", tr_b), ("tr_ascii", "tr", tr_ascii), ("tr_bare", "tr", tr_bare)]:
            cmds.append(dict(id=f"{pid}_{variant}", pair_id=pid, suite=suite, task_id=task, lang=lang, variant=variant, instruction=text,
                             relation=relation, frame="çerçeveye bağlı" if framedep == "evet" else "n/a",
                             suffixes=row["rel_seg"] if variant == "tr_a" else "", tier=tier))
    for name, rows in [("pairs.csv", pairs), ("commands.csv", cmds)]:
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"written: {out/'pairs.csv'} ({len(pairs)} pairs), {out/'commands.csv'} ({len(cmds)} rows)\n")
    for r in pairs:
        ref = "" if r["ref_libero"] in ("n/a",) else f"  [libero {r['ref_libero']} | image {r['ref_image']} | robot {r['ref_robot']}]"
        print(f"{r['pair_id']} T{r['tier']} {r['en']}\n      {r['tr_a']}\n      {r['tr_a_seg']}{ref}")


if __name__ == "__main__":
    main()
```

## validate_set.py

```python
"""Lecture 20 — checks that the command set (pairs.csv + commands.csv) conforms to the design rules and prints summary tables.

Checks:
  1. Format: 20 pairs × 5 surface forms = 100 rows; unique ids; BDDL-safe text (Lecture 16: lowercase, no ';' or parentheses, NFC,
     no U+0307, no double spaces) — every row must be able to go into a BDDL (:language …) line tomorrow.
  2. Suffix consistency: rel_span / rel_seg are regenerated with tr_morph and compared; they occur inside tr_a.
  3. Continuity: tr_a of A01–A04 is identical to the sp0/1/2/4 `tr` rows of datasets/01 (comparability with Lecture 15).
  4. Referent definiteness: every description in block B points to the same bowl in all 50 init states (referents.csv, above threshold).
  5. Balance: block × tier, relation type, case distribution; number of frame-dependent rows.
  6. Minimal pairs: pairs whose tr_a differs in exactly ONE word (comparisons that isolate the effect of the suffix / relational noun).
  7. Allomorph table: how many distinct surface forms each morpheme has in the tr_a segmentations (input of Lecture 21).
  8. Common tail: position of the distinguishing information in the sentence for Spatial commands in TR and EN (index of the first differing word).

Usage:  python validate_set.py ../../datasets/04-tr-spatial-commands/data [--d01 ../../datasets/01-libero-spatial-tr-mini/data/commands.csv]
"""
import argparse, csv, itertools, pathlib, re, sys, unicodedata
from collections import Counter, defaultdict
import tr_morph as M


def bddl_safe(t):
    p = []
    if ";" in t: p.append("';'")
    if "(" in t or ")" in t: p.append("parentheses")
    if t != t.lower(): p.append("uppercase")
    if "̇" in t: p.append("U+0307")
    if unicodedata.normalize("NFC", t) != t: p.append("not NFC")
    if "  " in t or t != t.strip(): p.append("whitespace")
    return p


def morpheme_of(seg_token):
    """'tabağ-ın' → [('GEN','ın')], 'üst-ü-nde-ki' → POSS3, LOC, ki … (only the suffixes of this set)."""
    parts = seg_token.split("-")
    out = []
    for i, s in enumerate(parts[1:], 1):
        after_poss = any(m == "POSS3" for m, _ in out) or parts[0].endswith(("kutusu", "şişesi"))
        if s == "ki": m = "ki"
        elif re.fullmatch(r"n?[ıiuü]n", s): m = "GEN"
        elif re.fullmatch(r"s?[ıiuü]", s) and i == 1 and len(parts) > 2: m = "POSS3"
        elif re.fullmatch(r"nd[ae]n|[dt][ae]n", s): m = "ABL"
        elif re.fullmatch(r"nd[ae]|[dt][ae]", s): m = "LOC"
        elif re.fullmatch(r"n[ae]", s) and after_poss: m = "DAT"
        elif re.fullmatch(r"y?[ae]", s): m = "DAT"
        elif re.fullmatch(r"[ny]?[ıiuü]", s): m = "ACC"
        else: m = "?"
        out.append((m, s))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("--d01")
    a = ap.parse_args(); d = pathlib.Path(a.data)
    pairs = list(csv.DictReader(open(d / "pairs.csv", encoding="utf-8")))
    cmds = list(csv.DictReader(open(d / "commands.csv", encoding="utf-8")))
    errors = []
    # 1
    if len(pairs) != 20 or len(cmds) != 100: errors.append(f"size: {len(pairs)} pairs, {len(cmds)} rows")
    if len({c["id"] for c in cmds}) != len(cmds): errors.append("duplicate id")
    for c in cmds:
        p = bddl_safe(c["instruction"])
        if p: errors.append(f"{c['id']}: not BDDL-safe ({', '.join(p)})")
    by_pair = defaultdict(dict)
    for c in cmds: by_pair[c["pair_id"]][c["variant"]] = c["instruction"]
    for pid, v in by_pair.items():
        if len(set(v.values())) != 5: errors.append(f"{pid}: the 5 surface forms are not all different")
    print(f"1. format: {len(pairs)} pairs, {len(cmds)} rows, BDDL-safe: {'yes' if not errors else 'NO'}")
    # 2
    n_ok = 0
    for r in pairs:
        seg_tokens = r["tr_a_seg"].split()
        if r["rel_span"] not in r["tr_a"]: errors.append(f"{r['pair_id']}: rel_span is not in tr_a")
        if r["rel_seg"].replace("-", "") != r["rel_span"]: errors.append(f"{r['pair_id']}: rel_seg ≠ rel_span")
        if r["tr_a_seg"].replace("-", "") != r["tr_a"]: errors.append(f"{r['pair_id']}: segmentation does not match the surface form")
        if r["rel_noun"] in M.LEXICON and r["case"] in ("LOC+ki", "DAT"):
            c = "loc" if r["case"] == "LOC+ki" else "dat"
            exp = M.rel(r["rel_noun"], c, r["case"] == "LOC+ki")
            if exp != (r["rel_span"], r["rel_seg"]): errors.append(f"{r['pair_id']}: tr_morph {exp} ≠ {r['rel_span']}")
            else: n_ok += 1
    print(f"2. suffix consistency: {n_ok} relational words regenerated with tr_morph, all identical")
    # 3
    if a.d01:
        d01 = {r["id"]: r["instruction"] for r in csv.DictReader(open(a.d01, encoding="utf-8"))}
        same = [(p, k) for p, k in [("A01", "sp0_tr"), ("A02", "sp1_tr"), ("A03", "sp2_tr"), ("A04", "sp4_tr")] if by_pair[p]["tr_a"] == d01[k]]
        if len(same) != 4: errors.append("continuity with datasets/01 is broken")
        print(f"3. continuity: tr_a identical to datasets/01: {len(same)}/4")
    # 4
    refs = {(int(r["task_id"]), r["descriptor"], r["frame"]): r for r in csv.DictReader(open(d / "referents.csv", encoding="utf-8"))}
    print("4. referent definiteness (block B; 50 init states):")
    for r in pairs:
        if r["block"] != "B": continue
        amb = [f for f in ("ref_libero", "ref_image", "ref_robot") if r[f].endswith("?") or r[f] in ("", "belirsiz", "ambiguous")]
        if amb: errors.append(f"{r['pair_id']}: ambiguous referent {amb}")
        agree = len({r["ref_libero"], r["ref_image"], r["ref_robot"]}) == 1
        print(f"   {r['pair_id']} task {r['scene_task_id']} {r['rel_span']:10s} libero={r['ref_libero']} image={r['ref_image']} robot={r['ref_robot']}"
              f"  {'frames AGREE' if agree else 'frames DISAGREE'}  (smallest gap: {r['ref_min_margin']})")
    # 5
    print("5. balance:")
    for key in ("block", "tier", "relation", "case", "frame_dependent", "suite"):
        print(f"   {key:16s}", dict(sorted(Counter(r[key] for r in pairs).items())))
    # 6
    print("6. minimal pairs (tr_a differs in a single word):")
    n_min = 0
    for x, y in itertools.combinations(pairs, 2):
        tx, ty = x["tr_a"].split(), y["tr_a"].split()
        if len(tx) == len(ty):
            diff = [(p, q) for p, q in zip(tx, ty) if p != q]
            if len(diff) == 1:
                n_min += 1
                print(f"   {x['pair_id']} ~ {y['pair_id']}: {diff[0][0]} / {diff[0][1]}")
    print(f"   total {n_min} minimal pairs")
    # 7
    allo = defaultdict(Counter)
    for r in pairs:
        for tok in r["tr_a_seg"].split():
            for m, s in morpheme_of(tok):
                allo[m][s] += 1
    print("7. allomorph table (tr_a, 20 commands):")
    for m in ("GEN", "POSS3", "ACC", "DAT", "LOC", "ABL", "ki", "?"):
        if allo[m]:
            print(f"   {m:6s} {len(allo[m])} forms: " + ", ".join(f"-{s}×{n}" for s, n in allo[m].most_common()))
    if allo["?"]: errors.append("unclassifiable suffix present")
    # 8
    sp = [r for r in pairs if r["suite"] == "libero_spatial"]
    def first_diff_positions(key):
        toks = [r[key].split() for r in sp]
        pos = []
        for x, y in itertools.combinations(toks, 2):
            i = next((k for k, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
            pos.append(i)
        return pos
    def common_tail(key):
        toks = [r[key].split()[::-1] for r in sp]
        n = 0
        while all(len(t) > n for t in toks) and len({t[n] for t in toks}) == 1: n += 1
        return n
    def common_head(key):
        toks = [r[key].split() for r in sp]
        n = 0
        while all(len(t) > n for t in toks) and len({t[n] for t in toks}) == 1: n += 1
        return n
    print(f"8. position of the distinguishing information (12 Spatial commands): EN common head {common_head('en')} words, common tail {common_tail('en')}; "
          f"TR common head {common_head('tr_a')}, common tail {common_tail('tr_a')}")
    print("\nRESULT:", "CLEAN — no errors" if not errors else f"{len(errors)} ERRORS")
    for e in errors: print("  -", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
```

## make_cf_bddl.py

```python
"""Lecture 20 — generates BDDL files with a counterfactual goal: if the command describes the OTHER bowl in the scene (akita_black_bowl_2),
LIBERO's success predicate (On akita_black_bowl_1 plate_1) unfairly penalizes the model (Lecture 19 §5).
For every row of pairs.csv that points to bowl_2 in any frame, this script produces a copy of the original BDDL:
  (:goal (And (On akita_black_bowl_2 plate_1)))   +   (:obj_of_interest akita_black_bowl_2 plate_1)   +   (:language <tr_a>)
The scene (:init) and regions do NOT CHANGE -> the 50 init states of the original task can be used as they are.

Every generated file is read back with the `bddl` package (pip install bddl==1.0.1 future): is :language identical to the text we wrote,
is :goal bowl_2, is :init the same as the original. If the package is missing, the same checks are done with regex.

Usage:
  python make_cf_bddl.py --libero ~/src/LIBERO --pairs ../../datasets/04-tr-spatial-commands/data/pairs.csv --out .../data/bddl_cf
Note: in the original LIBERO the command sent to the model comes from the file name, NOT from the BDDL (Lecture 13); you supply the command
with VLANEXT_INSTR_OVERRIDE, this file only changes the success predicate. File name = <pair_id>_<frame>_bowl2.bddl
"""
import argparse, csv, pathlib, re

ORDER_FILE = "libero/libero/benchmark/libero_suite_task_map.py"


def official_order(root, suite="libero_spatial"):
    t = (root / ORDER_FILE).read_text(encoding="utf-8")
    return re.findall(r'"([^"]+)"', re.search(r'"%s":\s*\[(.*?)\]' % suite, t, re.S).group(1))


def rewrite(src, text):
    t = src
    t, n1 = re.subn(r"\(:language[^)]*\)", f"(:language {text})", t)
    t, n2 = re.subn(r"\(:goal\s*\(And \(On akita_black_bowl_1 plate_1\)\)\s*\)", "(:goal\n    (And (On akita_black_bowl_2 plate_1))\n  )", t)
    t, n3 = re.subn(r"\(:obj_of_interest\s*akita_black_bowl_1", "(:obj_of_interest\n    akita_black_bowl_2", t)
    assert (n1, n2, n3) == (1, 1, 1), f"template did not match: language={n1} goal={n2} obj_of_interest={n3}"
    return t


def block(tree, key):
    return next(g for g in tree if isinstance(g, list) and g and g[0] == key)


def verify(path, text, orig_path):
    try:
        from bddl.parsing import scan_tokens
        new, old = scan_tokens(filename=str(path)), scan_tokens(filename=str(orig_path))
        lang = " ".join(block(new, ":language")[1:])
        goal = str(block(new, ":goal")[1:])
        same_init = block(new, ":init") == block(old, ":init") and block(new, ":regions") == block(old, ":regions")
        src = "bddl.parsing.scan_tokens"
    except ImportError:
        raw, raw0 = path.read_text(encoding="utf-8"), orig_path.read_text(encoding="utf-8")
        lang = re.search(r"\(:language\s+([^)]*)\)", raw).group(1).strip().lower()
        goal = raw.split("(:goal")[1]
        init = lambda s: re.sub(r"\s+", " ", re.search(r"\(:init(.*?)\)\s*\(:goal", s, re.S).group(1))
        same_init = init(raw) == init(raw0); src = "regex (no bddl package)"
    return lang == text, "akita_black_bowl_2" in goal and "akita_black_bowl_1" not in goal, same_init, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero", required=True); ap.add_argument("--pairs", required=True); ap.add_argument("--out", default="bddl_cf")
    a = ap.parse_args()
    root = pathlib.Path(a.libero).expanduser(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names = official_order(root)
    index = []
    for r in csv.DictReader(open(a.pairs, encoding="utf-8")):
        if r["suite"] != "libero_spatial": continue
        frames = [f for f in ("libero", "image", "robot") if r[f"ref_{f}"] == "bowl_2"]
        if not frames: continue
        tag = "all" if len(frames) == 3 else "-".join(frames)
        orig = root / "libero/libero/bddl_files/libero_spatial" / (names[int(r["scene_task_id"])] + ".bddl")
        dst = out / f"{r['pair_id']}_{tag}_bowl2.bddl"
        dst.write_text(rewrite(orig.read_text(encoding="utf-8"), r["tr_a"]), encoding="utf-8")
        ok_lang, ok_goal, ok_init, src = verify(dst, r["tr_a"], orig)
        index.append(dict(pair_id=r["pair_id"], frames_pointing_to_bowl2=tag, base_task_id=r["scene_task_id"], base_bddl=orig.name,
                          cf_bddl=dst.name, instruction=r["tr_a"], language_roundtrip=ok_lang, goal_is_bowl2=ok_goal, init_unchanged=ok_init))
        print(f"{dst.name:28s} task {r['scene_task_id']}  language identical: {ok_lang}  goal bowl_2: {ok_goal}  scene unchanged: {ok_init}  [{src}]")
        assert ok_lang and ok_goal and ok_init
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0])); w.writeheader(); w.writerows(index)
    print(f"\n{len(index)} counterfactual BDDL + index.csv → {out}")


if __name__ == "__main__":
    main()
```

## plot_day20.py

```python
"""Lecture 20 figures (02–05). Real data: LIBERO BDDL + .pruned_init (scene_frames.py), pairs.csv, Mesut's project-2 render.
  python plot_day20.py --libero ~/src/LIBERO --data ../../datasets/04-tr-spatial-commands/data \
      --render ../../projects/project-2/libero_first_episode-task-1.png --out ../../lectures/images/lecture-20
"""
import argparse, csv, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import scene_frames as SF

Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"


def scene(root, task):
    name = SF.official_order(root)[task]
    info = SF.parse_bddl(root / "libero/libero/bddl_files" / SF.SUITE / (name + ".bddl"))
    st = SF.load_init(root / "libero/libero/init_files" / SF.SUITE / (name + ".pruned_init"))
    return SF.positions(st, info["objects"])


LBL = {"akita_black_bowl_1": "bowl_1", "akita_black_bowl_2": "bowl_2", "plate_1": "plate", "cookies_1": "cookie\nbox",
       "glazed_rim_porcelain_ramekin_1": "ramekin"}
COL = {"akita_black_bowl_1": "#1f77b4", "akita_black_bowl_2": "#d98c00", "plate_1": "#c0392b", "cookies_1": "#7f6000",
       "glazed_rim_porcelain_ramekin_1": "#666666"}


def topdown(ax, pos, title, mode):
    """Top-down view; up on screen = -x (robot at the top), right = -y -> same layout as the image the model sees."""
    ax.add_patch(Rectangle((-0.6, -0.5), 1.2, 1.0, fc="#f7f1e3", ec="#999", lw=1))            # table (y horizontal, x vertical)
    for o, xyz in pos.items():
        ax.scatter(-xyz[:, 1], -xyz[:, 0], s=14, c=COL[o], alpha=0.5, lw=0)
        m = xyz.mean(0)
        ax.annotate(LBL[o], (-m[1], -m[0]), xytext=(7, 6), textcoords="offset points", fontsize=9, color=COL[o], weight="bold")
    ax.scatter([0], [0.66], marker="s", s=260, c=Y, ec="k", zorder=3); ax.text(0, 0.735, "robot base (x = −0.66)", ha="center", fontsize=9)
    ax.scatter([0], [-0.66], marker="v", s=220, c=B, ec="k", zorder=3); ax.text(0, -0.76, "agentview camera (x = +0.66)", ha="center", fontsize=9)
    ax.annotate("", xy=(0, 0.48), xytext=(0, 0.62), arrowprops=dict(arrowstyle="<-", color="k"))
    if mode == "lr":
        ax.annotate("", xy=(-0.55, -0.44), xytext=(0.55, -0.44), arrowprops=dict(arrowstyle="<->", color="#2e7d32", lw=2))
        ax.text(-0.57, -0.40, "+y", fontsize=9, color="#2e7d32"); ax.text(0.50, -0.40, "−y", fontsize=9, color="#2e7d32")
        ax.text(-0.58, 0.60, "image: LEFT\nrobot: LEFT\nLIBERO: right", fontsize=9, va="top", bbox=dict(fc=G, ec="none"))
        ax.text(0.58, 0.60, "image: RIGHT\nrobot: RIGHT\nLIBERO: left", fontsize=9, va="top", ha="right", bbox=dict(fc=R, ec="none"))
    else:
        ax.annotate("", xy=(0.52, 0.42), xytext=(0.52, -0.42), arrowprops=dict(arrowstyle="<->", color="#2e7d32", lw=2))
        ax.text(0.55, 0.36, "−x", fontsize=9, color="#2e7d32"); ax.text(0.55, -0.40, "+x", fontsize=9, color="#2e7d32")
        ax.text(-0.58, 0.44, "robot: FRONT (near)\nimage: BACK (top)\nLIBERO: back", fontsize=9, va="top", bbox=dict(fc=R, ec="none"))
        ax.text(-0.58, -0.44, "robot: far\nimage: FRONT (bottom)\nLIBERO: front", fontsize=9, va="bottom", bbox=dict(fc=G, ec="none"))
    ax.set_xlim(-0.68, 0.68); ax.set_ylim(-0.82, 0.80); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10.5)


def fig_frames(root, render, out):
    fig, axs = plt.subplots(1, 3, figsize=(17, 6.8))
    p1, p8 = scene(root, 1), scene(root, 8)
    ax = axs[0]
    if render and pathlib.Path(render).exists():
        img = plt.imread(render)[:256, :256]
        ax.imshow(img, extent=(0, 256, 256, 0))
        src = "Mesut's project-2 render (task 1, init 0, [::-1, ::-1] applied)"
    else:
        ax.set_facecolor("#eee"); src = "render not found — projection only"
    for o, xyz in p1.items():
        uv = SF.project(xyz[0])
        ax.scatter(*uv, s=70, fc="none", ec=COL[o], lw=2)
        ax.annotate(LBL[o].replace("\n", " "), uv, xytext=(8, -8), textcoords="offset points", fontsize=9, color="white",
                    bbox=dict(fc=COL[o], ec="none", pad=1.5))
    ax.set_xlim(0, 256); ax.set_ylim(256, 0); ax.set_xticks([0, 128, 256]); ax.set_yticks([0, 128, 256])
    ax.set_xlabel("u (pixels) — small u = left of the image = world +y")
    ax.set_title("(a) The frame the model sees + positions\nCOMPUTED from the init state\n" + src, fontsize=10.5)
    topdown(axs[1], p1, "(b) Task 1 scene, 50 init states (top-down, as in the image)\n«soldaki siyah kase» (\"the black bowl on the left\"):\nimage/robot → bowl_1, LIBERO convention → bowl_2", "lr")
    topdown(axs[2], p8, "(c) Task 8 scene, 50 init states\n«öndeki siyah kase» (\"the black bowl in front\"):\nimage/LIBERO → bowl_1, robot → bowl_2", "fb")
    fig.suptitle("Three frames, one scene: for left/right the LIBERO naming diverges, for front/back the robot frame diverges "
                 "(source: LIBERO .pruned_init + camera pose from libero_tabletop_manipulation.py; fovy 45°)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94)); fig.savefig(out / "02-three-frames.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def D(v):
    """Display translation of Turkish data-cell values (evet/hayır, case notes); the CSV itself stays Turkish."""
    v = str(v)
    for a, b in (("evet", "yes"), ("hayır", "no"), ("iyeliksiz", "no possessive"), ("sıfat", "adjective"), ("yalın", "bare")):
        v = v.replace(a, b)
    return v


def fig_matrix(pairs, out):
    cols = [("pair_id", "pair", 0.6), ("tier", "tier", 0.5), ("scene", "scene", 1.1), ("en_short", "EN (distinguishing part)", 3.6), ("rel_seg", "TR relation word (segmentation)", 2.4),
            ("case", "case", 1.9), ("frame_dependent", "frame?", 0.9), ("en_seen_suite", "EN in training?", 1.1), ("refs", "intended bowl\nlibero | image | robot", 2.6)]
    fig, ax = plt.subplots(figsize=(17, 9.2)); ax.axis("off")
    xs = np.cumsum([0] + [c[2] for c in cols]); H = 1.0
    for j, (_, name, w) in enumerate(cols):
        ax.add_patch(Rectangle((xs[j], 0), w, H * 1.3, fc="#444", ec="white")); ax.text(xs[j] + w / 2, H * 0.65, name, color="white", ha="center", va="center", fontsize=9.5, weight="bold")
    blockc = {"A": B, "B": Y, "C": G, "D": R}
    for i, r in enumerate(pairs):
        y = -(i + 1) * H + 0.0
        en = r["en"].replace("pick up the black bowl ", "… ").replace(" and place it on the plate", " …")
        r = dict(r, scene=("Spatial " if r["suite"] == "libero_spatial" else "Goal ") + r["scene_task_id"], en_short=en,
                 refs="—" if r["ref_libero"] == "n/a" else f"{r['ref_libero'][-1]} | {r['ref_image'][-1]} | {r['ref_robot'][-1]}")
        for j, (k, _, w) in enumerate(cols):
            fc = blockc[r["block"]] if j < 3 else "white"
            if k == "refs" and r["ref_libero"] != "n/a":
                s = {r["ref_libero"], r["ref_image"], r["ref_robot"]}
                fc = R if len(s) > 1 else (Y if s == {"bowl_2"} else B)
            if k == "frame_dependent" and r[k] == "evet": fc = R
            if k == "en_seen_suite": fc = G if r[k] == "evet" else Y
            ax.add_patch(Rectangle((xs[j], y), w, H, fc=fc, ec="#bbb", lw=0.6))
            ax.text(xs[j] + (0.08 if k in ("en_short", "rel_seg", "case") else w / 2), y + H / 2, D(r[k]), ha="left" if k in ("en_short", "rel_seg", "case") else "center", va="center", fontsize=9)
    ax.set_xlim(0, xs[-1]); ax.set_ylim(-len(pairs) * H - 1.6, H * 1.4)
    ax.text(0, -len(pairs) * H - 0.55, "Blocks: A (blue) Spatial training commands · B (yellow) descriptions with NO training sentence in the same scenes · C (green) Goal training commands · D (red) untrained placement (no predicate).\n"
            "Tier 1: current checkpoint + current predicate · Tier 2: current checkpoint + counterfactual BDDL (bowl_2) · Tier 3: new predicate + fine-tune.  Bowl column: 1 = bowl_1 (the training target), 2 = bowl_2; red = the frames disagree.",
            fontsize=9.5, va="top")
    ax.set_title("datasets/04-tr-spatial-commands — design matrix of the 20 EN–TR pairs (pairs.csv; the bowl column was computed from the 50 init states)", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "03-design-matrix.png", dpi=130); plt.close(fig)


def fig_referents(data, out):
    rows = list(csv.DictReader(open(data / "referents.csv", encoding="utf-8")))
    keys = [("left", "libero"), ("left", "image"), ("left", "robot"), ("front", "libero"), ("front", "image"), ("front", "robot"),
            ("near_plate", "none"), ("near_robot", "none"), ("higher", "none")]
    names = ["left\nLIBERO", "left\nimage", "left\nrobot", "front\nLIBERO", "front\nimage", "front\nrobot", "near\nplate", "near\nrobot", "higher\nup"]
    d = {(int(r["task_id"]), r["descriptor"], r["frame"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    for t in range(10):
        for j, k in enumerate(keys):
            r = d[(t,) + k]
            ok = r["unambiguous"] == "True"
            fc = R if not ok else (B if r["referent"] == "bowl_1" else Y)
            ax.add_patch(Rectangle((j, 9 - t), 1, 1, fc=fc, ec="white", lw=2))
            txt = "ambiguous" if r["referent"] in ("belirsiz", "ambiguous") else f"{r['referent']}\n{float(r['min_margin']):.0f} {r['unit']}"
            ax.text(j + .5, 9 - t + .5, txt, ha="center", va="center", fontsize=8.5, color="#333" if ok else "#a00")
        ax.text(-0.1, 9 - t + .5, f"task {t}\n1: {d[(t,'left','libero')]['bowl_1_at'].replace('_region','').replace('wooden_cabinet_1_','cabinet ').replace('flat_stove_1_','stove ')}\n2: {d[(t,'left','libero')]['bowl_2_at'].replace('_region','').replace('wooden_cabinet_1_','cabinet ').replace('flat_stove_1_','stove ')}",
                ha="right", va="center", fontsize=7.6)
    for j, n in enumerate(names): ax.text(j + .5, 10.15, n, ha="center", va="bottom", fontsize=9.5)
    for x in (3, 6): ax.plot([x, x], [0, 10], color="k", lw=1.5)
    ax.set_xlim(-2.6, 9); ax.set_ylim(0, 11.2); ax.axis("off")
    ax.set_title("\"Which bowl?\" — 10 Spatial scenes × 9 descriptions; cell: the intended bowl + the SMALLEST gap over the 50 init states\n"
                 "blue = bowl_1 (the training target), yellow = bowl_2, red = ambiguous (< 5 cm / 20 px or the sign changes) — source: LIBERO .pruned_init", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "04-referents.png", dpi=130); plt.close(fig)


def fig_allomorph(pairs, out):
    import validate_set as V
    from collections import Counter, defaultdict
    allo = defaultdict(Counter)
    for r in pairs:
        for tok in r["tr_a_seg"].split():
            for m, s in V.morpheme_of(tok): allo[m][s] += 1
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]; x = 0; ticks = []; shades = ["#e09f00", "#ffd166", "#ffe6a8", "#fff3d6", "#f1c40f", "#fbe9a7"]
    for m in ("GEN", "POSS3", "ACC", "DAT", "LOC", "ki"):
        bottom = 0
        for i, (s, n) in enumerate(allo[m].most_common()):
            ax.bar(x, n, bottom=bottom, color=shades[i % 6], ec="#555", width=0.7)
            if n >= 2: ax.text(x, bottom + n / 2, f"-{s} ×{n}", ha="center", va="center", fontsize=9)
            bottom += n
        small = [f"-{s}" for s, n in allo[m].most_common() if n < 2]
        if small: ax.text(x, bottom + 0.6, " ".join(small) + "\n(once each)", ha="center", va="bottom", fontsize=8.5)
        ticks.append(f"{m}\n{len(allo[m])} forms"); x += 1
    ax.set_xticks(range(len(ticks))); ax.set_xticklabels(ticks); ax.set_ylabel("occurrences in tr_a (20 commands)"); ax.set_ylim(0, 31)
    ax.set_title("(a) One morpheme, several surface forms: suffix allomorphs in 20 commands\n(counted from the tr_morph segmentations — real)", fontsize=10.5)
    ax = axs[1]
    sp = [r for r in pairs if r["suite"] == "libero_spatial"]
    NH, NT = 5, 7
    for row, (key, lab) in enumerate([("en", "EN"), ("tr_a", "TR (tr_a)")]):
        toks = [r[key].split() for r in sp]
        cells = [(k, {t[k] for t in toks}) for k in range(NH)] + [(NH + 1 + j, {t[-(NT - j)] for t in toks}) for j in range(NT)]
        for x, vals in cells:
            ax.add_patch(Rectangle((x, 1 - row), 1, 0.8, fc=G if len(vals) == 1 else Y, ec="white", lw=1.5))
            ax.text(x + .5, 1 - row + .52, str(len(vals)), ha="center", va="center", fontsize=10, weight="bold")
            ax.text(x + .5, 1 - row + .2, next(iter(vals)) if len(vals) == 1 else "", ha="center", va="center", fontsize=6.5)
        ax.text(NH + .5, 1 - row + .4, "…", ha="center", va="center", fontsize=14)
        ax.text(-0.3, 1 - row + .4, lab, ha="right", va="center", fontsize=10)
    ax.text(NH / 2, 2.0, "first 5 words (aligned from the start)", ha="center", fontsize=9); ax.text(NH + 1 + NT / 2, 2.0, "last 7 words (aligned from the end)", ha="center", fontsize=9)
    ax.set_xlim(-2.2, NH + NT + 1.2); ax.set_ylim(-1.3, 2.4); ax.axis("off")
    ax.text(0, -0.25, "cell = number of DISTINCT words at that position across the 12 Spatial commands\ngreen = always the same (no distinguishing information) · yellow = distinguishing\n"
            "EN: the description is in the MIDDLE — «pick up the black bowl» + description + «and place it on the plate»\nTR: the description comes FIRST — description + «siyah kaseyi al ve tabağın üstüne koy»", fontsize=9, va="top")
    ax.set_title("(b) Where in the sentence is the distinguishing information? (word position, 12 Spatial commands)", fontsize=10.5)
    fig.tight_layout(); fig.savefig(out / "05-allomorphs-position.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero", required=True); ap.add_argument("--data", required=True); ap.add_argument("--render"); ap.add_argument("--out", default=".")
    a = ap.parse_args()
    root, data, out = pathlib.Path(a.libero).expanduser(), pathlib.Path(a.data), pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pairs = list(csv.DictReader(open(data / "pairs.csv", encoding="utf-8")))
    fig_frames(root, a.render, out); fig_matrix(pairs, out); fig_referents(data, out); fig_allomorph(pairs, out)
    print("written:", *sorted(p.name for p in out.glob("0[2-5]-*.png")))


if __name__ == "__main__":
    main()
```
