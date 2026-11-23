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
