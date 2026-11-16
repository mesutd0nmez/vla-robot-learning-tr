# day20

Ders 20 — Türkçe mekânsal komut seti tasarımı: 20 EN–TR çift × 5 yüzey = 100 satır (`datasets/04-tr-spatial-commands/`). Komutlar elle yazılmaz, **üretilir**: ekler `tr_morph.py`'den, "hangi kâse?" etiketi LIBERO'nun 50 init state'inden (`scene_frames.py`), karşı-olgusal success predicate'i `make_cf_bddl.py`'den gelir; `validate_set.py` seti sekiz kurala göre denetler.

| Dosya | Ne | Durum |
|---|---|---|
| `tr_morph.py` | Küçük Türkçe ek üretici: GEN/POSS3/ACC/DAT/LOC/ABL/-ki, ünlü uyumu, kaynaştırma, yumuşama, zamir n'si; yüzey + ayrım (`tabağ-ın üst-ü-ne`) | çalıştırıldı (`--selftest` 42/42) |
| `scene_frames.py` | 10 Spatial sahnesi × 9 tarif: üç çerçevede (LIBERO adlandırması / modelin gördüğü görüntü / robot) kastedilen kâse + 50 init state'teki en küçük fark; `--conventions` LIBERO-90/10'un left/right/front/back bölgelerinden sözleşmeyi doğrular | çalıştırıldı (gerçek `.pruned_init`; 5/5 left = −y, 2/2 front = +x) |
| `make_tr_set.py` | 20 çifti üretir: `pairs.csv` (ayrım, ilişki sözcüğü, referans, tier, EN eğitimde mi) + `commands.csv` (en / tr_a / tr_b / tr_ascii / tr_bare) | çalıştırıldı |
| `validate_set.py` | BDDL-güvenli metin, ek tutarlılığı, `datasets/01` ile süreklilik, referans belirliliği, denge, minimal çiftler (11), allomorf tablosu, bilgi konumu | çalıştırıldı (TEMİZ) |
| `make_cf_bddl.py` | Komut `bowl_2`'yi gösteriyorsa hedefi `(On akita_black_bowl_2 plate_1)` olan BDDL kopyası; `bddl` paketiyle geri okuma | çalıştırıldı (5 dosya; dil birebir, sahne aynı) — **LIBERO env'inde yüklenmedi** (bu ortamda MuJoCo/LIBERO yok) |
| `plot_day20.py` | Görsel 02 (üç çerçeve; senin project-2 render'ın üstünde), 03 (tasarım matrisi), 04 (referans tablosu), 05 (allomorf + bilgi konumu) | çalıştırıldı |

Mac'te (model gerekmez; `vla` env, numpy + matplotlib; `bddl` isteğe bağlı):
```bash
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO ~/src/LIBERO    # Ders 19'da klonladıysan atla
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
"""Ders 20 — Mekânsal komutlar için küçük bir Türkçe ek üretici (kural tabanlı; sözlük + ünlü uyumu + kaynaştırma).

Amaç genel bir morfolojik çözümleyici DEĞİL: komut setindeki ~20 ad ve ~12 ilişki adı için ekleri elle yazmak yerine
üretmek, böylece (1) aynı biçimbirimin kaç yüzey biçimi (allomorf) olduğunu saymak, (2) sete yeni nesne eklerken
"tabakın / dolapın" gibi hataları önlemek, (3) her komutun ek ayrımını (tabağ-ın üst-ü-ne) Ders 21'e sütun olarak vermek.

Kurallar (Göksel & Kerslake 2005, Bölüm 3 ve 6'daki genel kurallar; yalnızca bu setin ihtiyacı kadar):
  I ∈ {ı, i, u, ü}: son ünlü kalın/ince × düz/yuvarlak        A ∈ {a, e}: son ünlü kalın/ince
  GEN  -(n)In     POSS3 -(s)I     ACC -(y)I     DAT -(y)A     LOC -DA     ABL -DAn     -ki
  POSS3'ten sonra "zamir n'si": -nI, -nA, -ndA, -ndAn, -nIn   (üst-ü-n-e, çekmece-si-n-de-ki)
  Yumuşama (sözlükte işaretli): tabak → tabağ-ı, dolap → dolab-ı, ocak → ocağ-ı   (robot → robot-a: yumuşamaz)

Kullanım:
  python tr_morph.py --selftest
  python tr_morph.py tabak gen        → tabağın   (tabağ-ın)
  python tr_morph.py üst dat --rel    → üstüne    (üst-ü-ne)
"""
import sys

VOWELS, BACK, ROUND = "aeıioöuüâîû", "aıouâû", "oöuüû"
SOFTEN = {"k": "ğ", "p": "b", "ç": "c", "t": "d"}
VOICELESS = "fstkçşhp"
# sözlük: yalın biçim → son sesi yumuşar mı?  (yalnızca bu setteki adlar)
LEXICON = {
    "tabak": True, "ocak": True, "dolap": True, "kase": False, "kâse": False, "ramekin": False, "masa": False,
    "kutu": False, "çekmece": False, "raf": False, "şişe": False, "peynir": False, "robot": False, "taraf": False,
    # ilişki adları (relational nouns): iyelik eki alıp hâl ekiyle çekimlenirler
    "üst": False, "alt": False, "yan": False, "ön": False, "arka": False, "iç": False, "ara": False, "orta": False,
    "sağ": False, "sol": False, "üzer": False, "karşı": False,
}


def _last_vowel(s):
    for ch in reversed(s):
        if ch in VOWELS:
            return ch
    raise ValueError(f"ünlü yok: {s}")


def _I(s):
    v = _last_vowel(s)
    return {(True, False): "ı", (False, False): "i", (True, True): "u", (False, True): "ü"}[(v in BACK, v in ROUND)]


def _A(s):
    return "a" if _last_vowel(s) in BACK else "e"


def _ends_vowel(s):
    return s[-1] in VOWELS


def _soft(word):
    """Ünlüyle başlayan ek öncesi gövde: sözlükte işaretliyse son ünsüz yumuşar."""
    head = word.split()[-1]
    if LEXICON.get(head, False) and word[-1] in SOFTEN:
        return word[:-1] + SOFTEN[word[-1]]
    return word


def poss3(word):
    """3. tekil iyelik: üst → üst-ü, arka → arka-sı, çekmece → çekmece-si, tabak → tabağ-ı."""
    if _ends_vowel(word):
        return word + "s" + _I(word), word + "-s" + _I(word)
    g = _soft(word)
    return g + _I(word), g + "-" + _I(word)


def case(word, c, poss=False, seg=None):
    """word'e hâl eki ekler. c ∈ {nom, gen, acc, dat, loc, abl}. poss=True: word zaten 3. tekil iyelikli (zamir n'si gelir).
    (yüzey, ayrım) döndürür; seg verilirse ayrımın başı olarak kullanılır."""
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
    """-ki: yer bildiren ifadeyi sıfata çevirir (üstünde → üstündeki 'üstünde olan'). Ünlü uyumuna girmez."""
    return surface + "ki", seg + "-ki"


def rel(noun, c, with_ki=False):
    """İlişki adı + POSS3 + hâl (+ki): rel('üst','dat') → üstüne; rel('ara','loc',True) → arasındaki."""
    p, pseg = poss3(noun)
    s, sseg = case(p, c, poss=True, seg=pseg)
    return ki(s, sseg) if with_ki else (s, sseg)


def bare(noun, c, with_ki=False):
    """İyeliksiz yön adı: sol-da-ki, sağ-da-ki, ön-de-ki (deiktik kullanım; referans nesne yok)."""
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
            print(f"  HATA beklenen {surf} ({seg}) → üretilen {got[0]} ({got[1]})")
    print(f"tr_morph selftest: {len(SELFTEST) - bad}/{len(SELFTEST)} doğru")
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
"""Ders 20 — "soldaki kâse" hangisi? LIBERO-Spatial'ın 10 sahnesinde iki siyah kâseyi ayıran tarifleri
(sol/sağ, ön/arka, tabağa yakın/uzak, robota yakın/uzak, yüksekte) 50 init state'in GERÇEK koordinatlarından hesaplar
ve her tarif için ÜÇ çerçevede hangi kâsenin kastedildiğini yazar:

  libero : LIBERO'nun kendi BDDL adlandırması (agentview kamerasının yerinde duran gözlemci; ayna yok)
           sağ = +y, sol = -y, ön = +x (gözlemciye yakın), arka = -x      [LIBERO-90/10 bölge adlarından doğrulandı]
  image  : modelin gördüğü piksel (obs[::-1, ::-1] = gerçek görüntünün yatay AYNASI): sol = küçük u, ön = büyük v
  robot  : robotun kendi gövdesi (taban x = -0.66, +x'e bakıyor): sol = +y, ön/yakın = tabana yakın

torch / LIBERO / MuJoCo GEREKMEZ: .pruned_init kısıtlı numpy unpickler ile açılır (Ders 19), kamera pozu
libero_tabletop_manipulation.py:188-197'den elle alınmıştır (fovy = MuJoCo varsayılanı 45 derece).

Kullanım:
  python scene_frames.py --libero ~/src/LIBERO                    # 10 sahne × tarif tablosu
  python scene_frames.py --libero ~/src/LIBERO --csv referents.csv
  python scene_frames.py --libero ~/src/LIBERO --conventions      # LIBERO-90/10'daki left/right/front/back bölgeleri
"""
import argparse, codecs, csv, glob, io, pathlib, pickle, re, zipfile
import numpy as np

SUITE = "libero_spatial"
ORDER_FILE = "libero/libero/benchmark/libero_suite_task_map.py"
CAM_POS = np.array([0.6586131746834771, 0.0, 1.6103500240372423])           # agentview (tabletop)
CAM_QUAT = (0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349)  # w x y z
ROBOT_BASE = np.array([-0.66, 0.0])   # mounted_panda.py:45 → -0.16 - table_length/2, table_length = 1.0
MIN_M, MIN_PX = 0.05, 20.0            # "belirsiz değil" eşiği: 50 sahnenin hepsinde en az 5 cm / 20 piksel fark


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
        raise pickle.UnpicklingError(f"izin verilmeyen nesne: {mod}.{name}")


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
    """Dünya noktası → piksel (u, v). MuJoCo kamerası -z'ye bakar, +x sağ, +y yukarı.
    mirror=True: modelin gördüğü görüntü (ham görüntü dikey ters gelir; [::-1, ::-1] onu yatay aynaya çevirir)."""
    pc = (np.asarray(p) - CAM_POS) @ cam_rotation()
    f = 0.5 * W / np.tan(np.radians(fovy) / 2)
    u = W / 2 + f * pc[..., 0] / (-pc[..., 2])
    v = W / 2 - f * pc[..., 1] / (-pc[..., 2])
    return np.stack([W - u if mirror else u, v], -1)


def positions(states, objects):
    """{nesne: (50, 3)} — qpos = 1 zaman + 7 kol + 2 gripper + nesne başına 7 (x y z qw qx qy qz), BDDL :objects sırası."""
    return {o: states[:, 10 + 7 * k: 13 + 7 * k] for k, o in enumerate(objects)}


def _pick(score1, score2, thr, smaller_wins):
    """İki kâsenin skorlarından (50,) kazananı ve 50 sahnedeki EN KÜÇÜK farkı döndür; işaret değişiyorsa 'belirsiz'."""
    d = score2 - score1 if smaller_wins else score1 - score2     # > 0 → bowl_1 kazanır
    if (d > 0).all():  who = "bowl_1"
    elif (d < 0).all(): who = "bowl_2"
    else: return "belirsiz", 0.0, False
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
    # sol / sağ
    add("left", "libero", b1[:, 1], b2[:, 1], MIN_M, True, "cm")        # sol = küçük y
    add("left", "image", uv1[:, 0], uv2[:, 0], MIN_PX, True, "px")     # sol = küçük u (aynalı görüntü)
    add("left", "robot", b1[:, 1], b2[:, 1], MIN_M, False, "cm")       # robotun solu = büyük y
    # ön (arka = öteki kâse)
    add("front", "libero", b1[:, 0], b2[:, 0], MIN_M, False, "cm")     # ön = büyük x (gözlemciye yakın)
    add("front", "image", uv1[:, 1], uv2[:, 1], MIN_PX, False, "px")   # ön = görüntüde aşağıda (büyük v)
    add("front", "robot", dr1, dr2, MIN_M, True, "cm")                 # robotun önündeki/yakınındaki = tabana yakın
    # çerçeveden bağımsız (referans nesne açık)
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
        # sıra kontrolü: masa bölgeli nesneler kendi bölgesinin ±1.5 cm içinde mi (Ders 19'daki doğrulama)
        for o, xyz in pos.items():
            r = info["regions"].get(info["where"].get(o))
            if r:
                x0, y0, x1, y1 = r
                ok = ((xyz[:, 0] > x0 - .015) & (xyz[:, 0] < x1 + .015) & (xyz[:, 1] > y0 - .015) & (xyz[:, 1] < y1 + .015)).all()
                assert ok, f"görev {i}: {o} bölgesinde değil — nesne sırası varsayımı bozuk"
        for d in descriptors(pos):
            out.append(dict(task_id=i, bowl_1_at=info["where"]["akita_black_bowl_1"].replace("main_table_", ""),
                            bowl_2_at=info["where"]["akita_black_bowl_2"].replace("main_table_", ""), **d))
    return out


def conventions(root):
    """LIBERO-90/10 BDDL'lerinde adı left/right/front/back içeren bölgelerin merkezleri: çerçeve sözleşmesinin kanıtı."""
    seen = {}
    for p in sorted(glob.glob(str(root / "libero/libero/bddl_files/libero_90/*.bddl")) + glob.glob(str(root / "libero/libero/bddl_files/libero_10/*.bddl"))):
        scene = re.match(r"([A-Z_]+SCENE\d+)", pathlib.Path(p).name).group(1)
        for k, r in parse_bddl(p)["regions"].items():
            m = re.search(r"(left|right|front|back)", k)
            if r and m and "contain" not in k:
                seen[(scene, k)] = (m.group(1), (r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
    # aynı sahnede left/right ve front/back çiftlerini karşılaştır
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
        print(f"{'sahne':24s} {'çift':46s} Δx (cm)  Δy (cm)   okuma")
        ok_lr = ok_fb = n_lr = n_fb = 0
        for scene, k, k2, word, x, y, x2, y2 in pr:
            dx, dy = (x - x2) * 100, (y - y2) * 100
            if word == "left":
                n_lr += 1; ok_lr += dy < 0; note = "left = daha küçük y" if dy < 0 else "TERS"
            else:
                n_fb += 1; ok_fb += dx > 0; note = "front = daha büyük x" if dx > 0 else "TERS"
            label = k.split("table_")[-1] + " | " + k2.split("table_")[-1]
            print(f"{scene:24s} {label[:46]:46s} {dx:7.1f}  {dy:7.1f}   {note}")
        print(f"\nleft/right çiftleri: {ok_lr}/{n_lr} 'left = -y';  front/back çiftleri: {ok_fb}/{n_fb} 'front = +x'")
        return
    rows = analyze(root)
    cur = None
    for r in rows:
        if r["task_id"] != cur:
            cur = r["task_id"]; print(f"\ngörev {cur}: bowl_1 @ {r['bowl_1_at']} | bowl_2 @ {r['bowl_2_at']}")
        flag = "" if r["unambiguous"] else "   (belirsiz: eşik altı)"
        print(f"   {r['descriptor']:11s} {r['frame']:7s} → {r['referent']:8s} min fark {r['min_margin']:6.1f} {r['unit']}{flag}")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"\nyazıldı: {a.csv} ({len(rows)} satır)")


if __name__ == "__main__":
    main()
```

## make_tr_set.py

```python
"""Ders 20 — 20 çiftlik EN–TR mekânsal komut setini üretir: datasets/04-tr-spatial-commands/data/{pairs,commands}.csv

Tasarım (derste Bölüm 5):
  A (6)  LIBERO-Spatial'ın EĞİTİLMİŞ komutları — seçim: "X'in REL-POSS-LOC-ki siyah kaseyi al ..."
  B (6)  aynı Spatial sahnelerinde HİÇBİR eğitim cümlesine denk düşmeyen tarifler (sol/sağ/ön, robota/tabağa yakın-uzak);
         hangi kâsenin kastedildiği scene_frames.py'nin 50 init state'ten hesapladığı referents.csv'den okunur
  C (6)  LIBERO-Goal'ın eğitilmiş komutları — yerleştirme: "... X'in REL-POSS-DAT koy"
  D (2)  eğitilmemiş yerleştirme (sağına / soluna): predicate yok → bugün yalnızca dil tarafı (Ders 21 tokenizasyon)

Her çift için 5 yüzey: en | tr_a (ek üreticiyle ÜRETİLİR, ayrımıyla) | tr_b (elle yazılmış ikinci biçim) |
  tr_ascii (tr_a'nın Türkçe karaktersiz hâli) | tr_bare (ekleri atılmış sözcük torbası)   → 20 × 5 = 100 satır.

Kullanım:
  python make_tr_set.py --out ../../datasets/04-tr-spatial-commands/data [--referents .../referents.csv] [--libero ~/src/LIBERO]
  --libero verilirse EN komutun LIBERO'nun 130 görev adında geçip geçmediği (en_seen_*) doldurulur.
"""
import argparse, csv, pathlib, re
import tr_morph as M

TR2ASCII = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def N(lemma, c="nom", poss=False):
    """Ad + hâl → (yüzey, ayrım, yalın). poss=True: lemma zaten iyelikli tamlama (kurabiye kutusu, şarap şişesi)."""
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
    """Belirtili tamlamanın iyelikli başı: (dolab-ın) üst çekmece-si-nde-ki."""
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

# (id, blok, tier, suite, sahne görevi, ilişki, ilişki adı, hâl, çerçeveye bağlı mı, EN, tr_a parçaları, ilişki sözcüğünün parça indeksi,
#  referans adın parça indeksi (yoksa None), tr_b, descriptor (referents.csv anahtarı ya da None), goal predicate, not)
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
COMPLEMENT = {"right": ("left", True), "far_plate": ("near_plate", True)}   # öteki kâse


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
        if desc and rl["none"][0]:                       # çerçeveden bağımsız tarif: üç sütuna aynı değer
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
    print(f"yazıldı: {out/'pairs.csv'} ({len(pairs)} çift), {out/'commands.csv'} ({len(cmds)} satır)\n")
    for r in pairs:
        ref = "" if r["ref_libero"] in ("n/a",) else f"  [libero {r['ref_libero']} | image {r['ref_image']} | robot {r['ref_robot']}]"
        print(f"{r['pair_id']} T{r['tier']} {r['en']}\n      {r['tr_a']}\n      {r['tr_a_seg']}{ref}")


if __name__ == "__main__":
    main()
```

## validate_set.py

```python
"""Ders 20 — Komut setinin (pairs.csv + commands.csv) tasarım kurallarına uyduğunu denetler ve özet tabloları basar.

Denetimler:
  1. Biçim: 20 çift × 5 yüzey = 100 satır; kimlikler benzersiz; BDDL-güvenli metin (Ders 16: küçük harf, ';' ve parantez yok, NFC,
     U+0307 yok, çift boşluk yok) — her satır yarın bir BDDL (:language …) satırına girebilmeli.
  2. Ek tutarlılığı: rel_span / rel_seg tr_morph ile yeniden üretilir ve karşılaştırılır; tr_a içinde geçer.
  3. Süreklilik: A01–A04'ün tr_a'sı datasets/01'in sp0/1/2/4 `tr` satırlarıyla birebir aynı (Ders 15 ile karşılaştırılabilirlik).
  4. Referans belirliliği: B bloğundaki her tarif 50 init state'in hepsinde aynı kâseyi gösteriyor (referents.csv, eşik üstü).
  5. Denge: blok × tier, ilişki türü, hâl dağılımı; çerçeveye bağlı satır sayısı.
  6. Minimal çiftler: tr_a'sı tam olarak BİR sözcükte ayrılan çiftler (ek/ilişki adı etkisini yalıtan karşılaştırmalar).
  7. Allomorf tablosu: tr_a ayrımlarında her biçimbirimin kaç farklı yüzey biçimi var (Ders 21'in girdisi).
  8. Ortak kuyruk: Spatial komutlarında TR ve EN'de ayırt edici bilginin cümledeki konumu (ilk farklı sözcük indeksi).

Kullanım:  python validate_set.py ../../datasets/04-tr-spatial-commands/data [--d01 ../../datasets/01-libero-spatial-tr-mini/data/commands.csv]
"""
import argparse, csv, itertools, pathlib, re, sys, unicodedata
from collections import Counter, defaultdict
import tr_morph as M


def bddl_safe(t):
    p = []
    if ";" in t: p.append("';'")
    if "(" in t or ")" in t: p.append("parantez")
    if t != t.lower(): p.append("büyük harf")
    if "̇" in t: p.append("U+0307")
    if unicodedata.normalize("NFC", t) != t: p.append("NFC değil")
    if "  " in t or t != t.strip(): p.append("boşluk")
    return p


def morpheme_of(seg_token):
    """'tabağ-ın' → [('GEN','ın')], 'üst-ü-nde-ki' → POSS3, LOC, ki … (yalnızca bu setin ekleri)."""
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
    if len(pairs) != 20 or len(cmds) != 100: errors.append(f"boyut: {len(pairs)} çift, {len(cmds)} satır")
    if len({c["id"] for c in cmds}) != len(cmds): errors.append("yinelenen id")
    for c in cmds:
        p = bddl_safe(c["instruction"])
        if p: errors.append(f"{c['id']}: BDDL-güvensiz ({', '.join(p)})")
    by_pair = defaultdict(dict)
    for c in cmds: by_pair[c["pair_id"]][c["variant"]] = c["instruction"]
    for pid, v in by_pair.items():
        if len(set(v.values())) != 5: errors.append(f"{pid}: 5 yüzey birbirinden farklı değil")
    print(f"1. biçim: {len(pairs)} çift, {len(cmds)} satır, BDDL-güvenli: {'evet' if not errors else 'HAYIR'}")
    # 2
    n_ok = 0
    for r in pairs:
        seg_tokens = r["tr_a_seg"].split()
        if r["rel_span"] not in r["tr_a"]: errors.append(f"{r['pair_id']}: rel_span tr_a'da yok")
        if r["rel_seg"].replace("-", "") != r["rel_span"]: errors.append(f"{r['pair_id']}: rel_seg ≠ rel_span")
        if r["tr_a_seg"].replace("-", "") != r["tr_a"]: errors.append(f"{r['pair_id']}: ayrım yüzeyle uyuşmuyor")
        if r["rel_noun"] in M.LEXICON and r["case"] in ("LOC+ki", "DAT"):
            c = "loc" if r["case"] == "LOC+ki" else "dat"
            exp = M.rel(r["rel_noun"], c, r["case"] == "LOC+ki")
            if exp != (r["rel_span"], r["rel_seg"]): errors.append(f"{r['pair_id']}: tr_morph {exp} ≠ {r['rel_span']}")
            else: n_ok += 1
    print(f"2. ek tutarlılığı: {n_ok} ilişki sözcüğü tr_morph ile yeniden üretildi, hepsi aynı")
    # 3
    if a.d01:
        d01 = {r["id"]: r["instruction"] for r in csv.DictReader(open(a.d01, encoding="utf-8"))}
        same = [(p, k) for p, k in [("A01", "sp0_tr"), ("A02", "sp1_tr"), ("A03", "sp2_tr"), ("A04", "sp4_tr")] if by_pair[p]["tr_a"] == d01[k]]
        if len(same) != 4: errors.append("datasets/01 ile süreklilik bozuk")
        print(f"3. süreklilik: datasets/01 ile birebir aynı tr_a: {len(same)}/4")
    # 4
    refs = {(int(r["task_id"]), r["descriptor"], r["frame"]): r for r in csv.DictReader(open(d / "referents.csv", encoding="utf-8"))}
    print("4. referans belirliliği (B bloğu; 50 init state):")
    for r in pairs:
        if r["block"] != "B": continue
        amb = [f for f in ("ref_libero", "ref_image", "ref_robot") if r[f].endswith("?") or r[f] in ("", "belirsiz")]
        if amb: errors.append(f"{r['pair_id']}: belirsiz referans {amb}")
        agree = len({r["ref_libero"], r["ref_image"], r["ref_robot"]}) == 1
        print(f"   {r['pair_id']} görev {r['scene_task_id']} {r['rel_span']:10s} libero={r['ref_libero']} image={r['ref_image']} robot={r['ref_robot']}"
              f"  {'çerçeveler UYUŞUYOR' if agree else 'çerçeveler AYRIŞIYOR'}  (en küçük fark: {r['ref_min_margin']})")
    # 5
    print("5. denge:")
    for key in ("block", "tier", "relation", "case", "frame_dependent", "suite"):
        print(f"   {key:16s}", dict(sorted(Counter(r[key] for r in pairs).items())))
    # 6
    print("6. minimal çiftler (tr_a tek sözcükte ayrılıyor):")
    n_min = 0
    for x, y in itertools.combinations(pairs, 2):
        tx, ty = x["tr_a"].split(), y["tr_a"].split()
        if len(tx) == len(ty):
            diff = [(p, q) for p, q in zip(tx, ty) if p != q]
            if len(diff) == 1:
                n_min += 1
                print(f"   {x['pair_id']} ~ {y['pair_id']}: {diff[0][0]} / {diff[0][1]}")
    print(f"   toplam {n_min} minimal çift")
    # 7
    allo = defaultdict(Counter)
    for r in pairs:
        for tok in r["tr_a_seg"].split():
            for m, s in morpheme_of(tok):
                allo[m][s] += 1
    print("7. allomorf tablosu (tr_a, 20 komut):")
    for m in ("GEN", "POSS3", "ACC", "DAT", "LOC", "ABL", "ki", "?"):
        if allo[m]:
            print(f"   {m:6s} {len(allo[m])} biçim: " + ", ".join(f"-{s}×{n}" for s, n in allo[m].most_common()))
    if allo["?"]: errors.append("sınıflanamayan ek var")
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
    print(f"8. ayırt edici bilginin konumu (12 Spatial komutu): EN ortak baş {common_head('en')} sözcük, ortak kuyruk {common_tail('en')}; "
          f"TR ortak baş {common_head('tr_a')}, ortak kuyruk {common_tail('tr_a')}")
    print("\nSONUÇ:", "TEMİZ — hata yok" if not errors else f"{len(errors)} HATA")
    for e in errors: print("  -", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
```

## make_cf_bddl.py

```python
"""Ders 20 — Karşı-olgusal (counterfactual) hedefli BDDL üretir: komut sahnedeki ÖTEKİ kâseyi (akita_black_bowl_2) tarif
ediyorsa, LIBERO'nun success predicate'i (On akita_black_bowl_1 plate_1) modeli haksız yere cezalandırır (Ders 19 §5).
Bu script pairs.csv'de herhangi bir çerçevede bowl_2'yi gösteren her satır için orijinal BDDL'in kopyasını üretir:
  (:goal (And (On akita_black_bowl_2 plate_1)))   +   (:obj_of_interest akita_black_bowl_2 plate_1)   +   (:language <tr_a>)
Sahne (:init) ve bölgeler DEĞİŞMEZ → orijinal görevin 50 init state'i aynen kullanılabilir.

Üretilen her dosya `bddl` paketiyle (pip install bddl==1.0.1 future) geri okunur: :language yazdığımız metinle birebir mi,
:goal bowl_2 mi, :init orijinalle aynı mı. Paket yoksa aynı kontroller regex ile yapılır.

Kullanım:
  python make_cf_bddl.py --libero ~/src/LIBERO --pairs ../../datasets/04-tr-spatial-commands/data/pairs.csv --out .../data/bddl_cf
Not: orijinal LIBERO'da modele giden komut BDDL'den DEĞİL dosya adından gelir (Ders 13); komutu VLANEXT_INSTR_OVERRIDE ile
verirsin, bu dosya yalnızca success predicate'ini değiştirir. Dosya adı = <pair_id>_<çerçeve>_bowl2.bddl
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
    assert (n1, n2, n3) == (1, 1, 1), f"şablon tutmadı: language={n1} goal={n2} obj_of_interest={n3}"
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
        same_init = init(raw) == init(raw0); src = "regex (bddl paketi yok)"
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
        print(f"{dst.name:28s} görev {r['scene_task_id']}  dil birebir: {ok_lang}  hedef bowl_2: {ok_goal}  sahne aynı: {ok_init}  [{src}]")
        assert ok_lang and ok_goal and ok_init
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0])); w.writeheader(); w.writerows(index)
    print(f"\n{len(index)} karşı-olgusal BDDL + index.csv → {out}")


if __name__ == "__main__":
    main()
```

## plot_day20.py

```python
"""Ders 20 görselleri (02–05). Gerçek veri: LIBERO BDDL + .pruned_init (scene_frames.py), pairs.csv, Mesut'un project-2 render'ı.
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


LBL = {"akita_black_bowl_1": "bowl_1", "akita_black_bowl_2": "bowl_2", "plate_1": "tabak", "cookies_1": "kurabiye\nkutusu",
       "glazed_rim_porcelain_ramekin_1": "ramekin"}
COL = {"akita_black_bowl_1": "#1f77b4", "akita_black_bowl_2": "#d98c00", "plate_1": "#c0392b", "cookies_1": "#7f6000",
       "glazed_rim_porcelain_ramekin_1": "#666666"}


def topdown(ax, pos, title, mode):
    """Kuş bakışı; ekranda yukarı = -x (robot üstte), sağ = -y → modelin gördüğü görüntüyle aynı yerleşim."""
    ax.add_patch(Rectangle((-0.6, -0.5), 1.2, 1.0, fc="#f7f1e3", ec="#999", lw=1))            # masa (y yatay, x dikey)
    for o, xyz in pos.items():
        ax.scatter(-xyz[:, 1], -xyz[:, 0], s=14, c=COL[o], alpha=0.5, lw=0)
        m = xyz.mean(0)
        ax.annotate(LBL[o], (-m[1], -m[0]), xytext=(7, 6), textcoords="offset points", fontsize=9, color=COL[o], weight="bold")
    ax.scatter([0], [0.66], marker="s", s=260, c=Y, ec="k", zorder=3); ax.text(0, 0.735, "robot tabanı (x = −0.66)", ha="center", fontsize=9)
    ax.scatter([0], [-0.66], marker="v", s=220, c=B, ec="k", zorder=3); ax.text(0, -0.76, "agentview kamerası (x = +0.66)", ha="center", fontsize=9)
    ax.annotate("", xy=(0, 0.48), xytext=(0, 0.62), arrowprops=dict(arrowstyle="<-", color="k"))
    if mode == "lr":
        ax.annotate("", xy=(-0.55, -0.44), xytext=(0.55, -0.44), arrowprops=dict(arrowstyle="<->", color="#2e7d32", lw=2))
        ax.text(-0.57, -0.40, "+y", fontsize=9, color="#2e7d32"); ax.text(0.50, -0.40, "−y", fontsize=9, color="#2e7d32")
        ax.text(-0.58, 0.60, "görüntü: SOL\nrobot: SOL\nLIBERO: right", fontsize=9, va="top", bbox=dict(fc=G, ec="none"))
        ax.text(0.58, 0.60, "görüntü: SAĞ\nrobot: SAĞ\nLIBERO: left", fontsize=9, va="top", ha="right", bbox=dict(fc=R, ec="none"))
    else:
        ax.annotate("", xy=(0.52, 0.42), xytext=(0.52, -0.42), arrowprops=dict(arrowstyle="<->", color="#2e7d32", lw=2))
        ax.text(0.55, 0.36, "−x", fontsize=9, color="#2e7d32"); ax.text(0.55, -0.40, "+x", fontsize=9, color="#2e7d32")
        ax.text(-0.58, 0.44, "robot: ÖN (yakın)\ngörüntü: ARKA (üstte)\nLIBERO: back", fontsize=9, va="top", bbox=dict(fc=R, ec="none"))
        ax.text(-0.58, -0.44, "robot: uzak\ngörüntü: ÖN (altta)\nLIBERO: front", fontsize=9, va="bottom", bbox=dict(fc=G, ec="none"))
    ax.set_xlim(-0.68, 0.68); ax.set_ylim(-0.82, 0.80); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10.5)


def fig_frames(root, render, out):
    fig, axs = plt.subplots(1, 3, figsize=(17, 6.8))
    p1, p8 = scene(root, 1), scene(root, 8)
    ax = axs[0]
    if render and pathlib.Path(render).exists():
        img = plt.imread(render)[:256, :256]
        ax.imshow(img, extent=(0, 256, 256, 0))
        src = "Mesut'un project-2 render'ı (görev 1, init 0, [::-1, ::-1] uygulanmış)"
    else:
        ax.set_facecolor("#eee"); src = "render bulunamadı — yalnızca projeksiyon"
    for o, xyz in p1.items():
        uv = SF.project(xyz[0])
        ax.scatter(*uv, s=70, fc="none", ec=COL[o], lw=2)
        ax.annotate(LBL[o].replace("\n", " "), uv, xytext=(8, -8), textcoords="offset points", fontsize=9, color="white",
                    bbox=dict(fc=COL[o], ec="none", pad=1.5))
    ax.set_xlim(0, 256); ax.set_ylim(256, 0); ax.set_xticks([0, 128, 256]); ax.set_yticks([0, 128, 256])
    ax.set_xlabel("u (piksel) — küçük u = görüntünün solu = dünya +y")
    ax.set_title("(a) Modelin gördüğü kare + init state'ten HESAPLANAN konumlar\n" + src, fontsize=10.5)
    topdown(axs[1], p1, "(b) Görev 1 sahnesi, 50 init state (kuş bakışı, görüntüyle aynı yön)\n«soldaki siyah kase»: görüntü/robot → bowl_1, LIBERO sözleşmesi → bowl_2", "lr")
    topdown(axs[2], p8, "(c) Görev 8 sahnesi, 50 init state\n«öndeki siyah kase»: görüntü/LIBERO → bowl_1, robot → bowl_2", "fb")
    fig.suptitle("Üç çerçeve, bir sahne: sol/sağ'da LIBERO adlandırması, ön/arka'da robot çerçevesi ayrışıyor "
                 "(kaynak: LIBERO .pruned_init + kamera pozu libero_tabletop_manipulation.py; fovy 45°)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig(out / "02-three-frames.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_matrix(pairs, out):
    cols = [("pair_id", "çift", 0.6), ("tier", "tier", 0.5), ("scene", "sahne", 1.1), ("en_short", "EN (ayırt edici kısım)", 3.6), ("rel_seg", "TR ilişki sözcüğü (ayrım)", 2.4),
            ("case", "hâl", 1.9), ("frame_dependent", "çerçeve?", 0.9), ("en_seen_suite", "EN eğitimde?", 1.1), ("refs", "kastedilen kâse\nlibero | görüntü | robot", 2.6)]
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
            ax.text(xs[j] + (0.08 if k in ("en_short", "rel_seg", "case") else w / 2), y + H / 2, str(r[k]), ha="left" if k in ("en_short", "rel_seg", "case") else "center", va="center", fontsize=9)
    ax.set_xlim(0, xs[-1]); ax.set_ylim(-len(pairs) * H - 1.6, H * 1.4)
    ax.text(0, -len(pairs) * H - 0.55, "Bloklar: A (mavi) Spatial eğitim komutları · B (sarı) aynı sahnelerde eğitim cümlesi OLMAYAN tarifler · C (yeşil) Goal eğitim komutları · D (kırmızı) eğitilmemiş yerleştirme (predicate yok).\n"
            "Tier 1: mevcut checkpoint + mevcut predicate · Tier 2: mevcut checkpoint + karşı-olgusal BDDL (bowl_2) · Tier 3: yeni predicate + fine-tune.  Kâse sütunu: 1 = bowl_1 (eğitimdeki hedef), 2 = bowl_2; kırmızı = çerçeveler ayrışıyor.",
            fontsize=9.5, va="top")
    ax.set_title("datasets/04-tr-spatial-commands — 20 EN–TR çiftinin tasarım matrisi (pairs.csv; kâse sütunu 50 init state'ten hesaplandı)", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "03-design-matrix.png", dpi=130); plt.close(fig)


def fig_referents(data, out):
    rows = list(csv.DictReader(open(data / "referents.csv", encoding="utf-8")))
    keys = [("left", "libero"), ("left", "image"), ("left", "robot"), ("front", "libero"), ("front", "image"), ("front", "robot"),
            ("near_plate", "none"), ("near_robot", "none"), ("higher", "none")]
    names = ["sol\nLIBERO", "sol\ngörüntü", "sol\nrobot", "ön\nLIBERO", "ön\ngörüntü", "ön\nrobot", "tabağa\nyakın", "robota\nyakın", "daha\nyüksekte"]
    d = {(int(r["task_id"]), r["descriptor"], r["frame"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    for t in range(10):
        for j, k in enumerate(keys):
            r = d[(t,) + k]
            ok = r["unambiguous"] == "True"
            fc = R if not ok else (B if r["referent"] == "bowl_1" else Y)
            ax.add_patch(Rectangle((j, 9 - t), 1, 1, fc=fc, ec="white", lw=2))
            txt = "belirsiz" if r["referent"] == "belirsiz" else f"{r['referent']}\n{float(r['min_margin']):.0f} {r['unit']}"
            ax.text(j + .5, 9 - t + .5, txt, ha="center", va="center", fontsize=8.5, color="#333" if ok else "#a00")
        ax.text(-0.1, 9 - t + .5, f"görev {t}\n1: {d[(t,'left','libero')]['bowl_1_at'].replace('_region','').replace('wooden_cabinet_1_','dolap ').replace('flat_stove_1_','ocak ')}\n2: {d[(t,'left','libero')]['bowl_2_at'].replace('_region','').replace('wooden_cabinet_1_','dolap ').replace('flat_stove_1_','ocak ')}",
                ha="right", va="center", fontsize=7.6)
    for j, n in enumerate(names): ax.text(j + .5, 10.15, n, ha="center", va="bottom", fontsize=9.5)
    for x in (3, 6): ax.plot([x, x], [0, 10], color="k", lw=1.5)
    ax.set_xlim(-2.6, 9); ax.set_ylim(0, 11.2); ax.axis("off")
    ax.set_title("«Hangi kâse?» — 10 Spatial sahnesi × 9 tarif; hücre: kastedilen kâse + 50 init state'teki EN KÜÇÜK fark\n"
                 "mavi = bowl_1 (eğitimdeki hedef), sarı = bowl_2, kırmızı = belirsiz (< 5 cm / 20 px ya da işaret değişiyor) — kaynak: LIBERO .pruned_init", fontsize=11)
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
        if small: ax.text(x, bottom + 0.6, " ".join(small) + "\n(birer kez)", ha="center", va="bottom", fontsize=8.5)
        ticks.append(f"{m}\n{len(allo[m])} biçim"); x += 1
    ax.set_xticks(range(len(ticks))); ax.set_xticklabels(ticks); ax.set_ylabel("tr_a'da geçiş sayısı (20 komut)"); ax.set_ylim(0, 31)
    ax.set_title("(a) Bir biçimbirim, birden çok yüzey: 20 komutta ek allomorfları\n(tr_morph ayrımlarından sayıldı — gerçek)", fontsize=10.5)
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
    ax.text(NH / 2, 2.0, "ilk 5 sözcük (baştan hizalı)", ha="center", fontsize=9); ax.text(NH + 1 + NT / 2, 2.0, "son 7 sözcük (sondan hizalı)", ha="center", fontsize=9)
    ax.set_xlim(-2.2, NH + NT + 1.2); ax.set_ylim(-1.3, 2.4); ax.axis("off")
    ax.text(0, -0.25, "hücre = o konumda 12 Spatial komutunda kaç FARKLI sözcük var\nyeşil = hep aynı (ayırt edici bilgi yok) · sarı = ayırt edici\n"
            "EN: tarif ORTADA — «pick up the black bowl» + tarif + «and place it on the plate»\nTR: tarif EN BAŞTA — tarif + «siyah kaseyi al ve tabağın üstüne koy»", fontsize=9, va="top")
    ax.set_title("(b) Ayırt edici bilgi cümlenin neresinde? (sözcük konumu, 12 Spatial komutu)", fontsize=10.5)
    fig.tight_layout(); fig.savefig(out / "05-allomorphs-position.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero", required=True); ap.add_argument("--data", required=True); ap.add_argument("--render"); ap.add_argument("--out", default=".")
    a = ap.parse_args()
    root, data, out = pathlib.Path(a.libero).expanduser(), pathlib.Path(a.data), pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pairs = list(csv.DictReader(open(data / "pairs.csv", encoding="utf-8")))
    fig_frames(root, a.render, out); fig_matrix(pairs, out); fig_referents(data, out); fig_allomorph(pairs, out)
    print("yazıldı:", *sorted(p.name for p in out.glob("0[2-5]-*.png")))


if __name__ == "__main__":
    main()
```
