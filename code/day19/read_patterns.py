"""Ders 19 — 4 koşullu tabloyu (Ders 18 çıktısı) dört örüntüye göre oku ve "1 paragraf yorum" taslağı üret.

Girdi: Ders 18'in episodes.csv'si (run_id, suite, task_id, condition, init_state, success, ...).
İsteğe bağlı: --features init_features_task0.csv (init_state_geometry.py) → uyumsuz çiftlerin sahne özellikleri;
              --labels grasped.csv (init_state, condition, grasped_bowl ∈ {bowl_1, bowl_2, none}) → faithful/biased sayımı.

Kullanım:
  python read_patterns.py results/episodes.csv [--features init_features_task0.csv] [--labels grasped.csv]
  python read_patterns.py --demo          # Ders 18 görsel 07'nin dört temsili örüntüsünü sınıflandırır
  python read_patterns.py --selftest      # Ders 18 selftest tablosu (9/5/7/4) → beklenen çıktı

Kurallar (pilot için sezgisel eşikler; parametre, dogma değil):
  near  = fark ≤ NEAR (varsayılan 15 puan)   → "yakın"
  far   = fark ≥ FAR  (varsayılan 25 puan)   → "uzak"
  A  dil görmezden geliniyor : empty ve wrong clean'e yakın
  C  yanlış komut 'biased'   : wrong clean'e yakın AMA empty uzak (komut 'bir kâse al' diyor, sahne hangisini seçiyor)
  B  Türkçe okunmuyor        : dil okunuyor (empty uzak) ve tr ≈ empty (tr − empty ≤ NEAR/1.5)
  D  çok dilli omurga        : dil okunuyor ve tr − empty ≥ FAR
  aradakiler "belirsiz" → daha çok episode. C, tr eksenine eklenir: "C+B", "C+D" (görsel 07'nin C paneli tr=empty olduğu için C+B'dir).
Wilson aralıkları çakışıyorsa (n=10'da hemen hep çakışır) sonuç "yön", "bulgu" değil; script bunu her satıra yazar.
"""
import argparse, csv, math, pathlib, sys
from collections import OrderedDict, defaultdict

NEAR, FAR = 15, 25


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load_episodes(path):
    eps = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            eps[r["condition"]][int(r["init_state"])] = int(r["success"]) == 1
    return eps


def classify(sr, near=NEAR, far=FAR):
    """sr: {koşul: yüzde}. Dönüş: (harf, gerekçe listesi)."""
    c, e, w, t = sr["clean"], sr["empty"], sr.get("wrong"), sr.get("tr")
    why = []
    gap_e = c - e
    lang_read = gap_e >= far
    if gap_e <= near and (w is None or c - w <= near):
        why.append(f"empty clean'e yakın (Δ={gap_e:+.0f}); wrong da yakın" if w is not None else f"empty clean'e yakın (Δ={gap_e:+.0f})")
        return "A", why
    if w is not None and c - w <= near and lang_read:
        why.append(f"wrong clean'e yakın (Δ={c - w:+.0f}) ama empty uzak (Δ={gap_e:+.0f}) → komutun 'bir kâse al' kısmı okunuyor, hedef sahneden")
        letter_w = "C"
    else:
        letter_w = None
    if t is not None and lang_read:
        if t - e <= near / 1.5:
            why.append(f"tr ≈ empty (tr−empty={t - e:+.0f}) → Türkçe'den sahnenin üstüne bir şey çıkmıyor")
            return (letter_w or "B") if letter_w != "C" else "C+B", why
        if t - e >= far:
            why.append(f"tr empty'nin çok üstünde (tr−empty={t - e:+.0f}) → Türkçe'den en azından nesne/eylem çıkıyor")
            return (letter_w or "D") if letter_w != "C" else "C+D", why
        why.append(f"tr empty ile clean arasında (tr−empty={t - e:+.0f}): B ile D arasında belirsiz")
        return (letter_w or "belirsiz"), why
    if letter_w:
        return letter_w, why
    why.append(f"empty düşüşü {gap_e:+.0f} puan: {near} ile {far} arasında — ne 'görmezden geliyor' ne 'okuyor' denebilir")
    return "belirsiz", why


def paragraph(letter, rows, n):
    c = rows["clean"]; e = rows["empty"]; w = rows.get("wrong"); t = rows.get("tr")
    head = {"A": "Bu pilotta komutun içeriği sonucu değiştirmedi",
            "B": "Model komutu okuyor ama Türkçe komut sahne tabanının üstüne çıkmıyor",
            "C": "Model komutun eylem kısmını okuyor, hedef nesneyi sahneden seçiyor",
            "D": "Türkçe komut sahne tabanının belirgin üstünde",
            "C+B": "Yanlış komut 'biased', Türkçe komut tabanda", "C+D": "Yanlış komut 'biased' ama Türkçe'den bir şey çıkıyor",
            "belirsiz": "Pilot yön vermiyor"}[letter]
    s = (f"{head}: LIBERO-Spatial görev 0'da {n} eşli init state'te success rate clean {c['k']}/{n}, empty {e['k']}/{n}"
         + (f", wrong {w['k']}/{n}" if w else "") + (f", tr {t['k']}/{n}" if t else "") + ". ")
    s += f"Uyumsuz çiftler (clean✓x✗ / clean✗x✓): empty {e['b']}/{e['c']}" + (f", wrong {w['b']}/{w['c']}" if w else "") + (f", tr {t['b']}/{t['c']}" if t else "") + ". "
    s += ("Wilson %95 aralıkları çakıştığı için bu bir yön bulgusudur; 50 init state'lik koşu (Ders 18 §5) aynı oranları verirse "
          "anlamlılık ve etki büyüklüğü birlikte raporlanır. ")
    if letter in ("A",):
        s += "Spatial görev 0'da sahne görevi tek başına belirliyor (suite_structure.py: 10 görevin 9'u sahneden ayrışır); Türkçe etkisi bu görevde ölçülemez — Goal suite'e geç."
    elif letter.startswith("C"):
        s += "`wrong` videolarındaki grasped_bowl etiketi biased (bowl_1) ağırlıklıysa LIBERO-CF'in 'vision overrides language' örüntüsü; faithful (bowl_2) ağırlıklıysa model komutu okuyor ve success metriği onu cezalandırıyor."
    elif letter in ("B", "C+B"):
        s += "Chen vd.'nin 'yabancı dil ≈ komut yok' örüntüsü; tokenizasyon (Ders 21) ve çeviri katmanı / Türkçe LoRA (Ders 23) karşılaştırmasının gerekçesi."
    elif letter in ("D", "C+D"):
        s += "Qwen3-VL'nin çok dilli ön-eğitimi Türkçe'den nesne/eylem çıkarıyor olabilir (hipotez); mekânsal ilişkinin (arasındaki) okunup okunmadığı distractor koşuluyla ayrıca test edilmeli (Ders 15)."
    return s


def report(eps, features=None, labels=None, near=NEAR, far=FAR, title=""):
    order = [c for c in ["clean", "empty", "wrong", "tr"] if c in eps] + [c for c in eps if c not in ("clean", "empty", "wrong", "tr")]
    base = eps["clean"]; n = len(base)
    rows = OrderedDict()
    for cnd in order:
        d = eps[cnd]; k = sum(d.values()); lo, hi = wilson(k, len(d))
        common = sorted(set(base) & set(d))
        b = sum(base[i] and not d[i] for i in common); c = sum((not base[i]) and d[i] for i in common)
        rows[cnd] = dict(k=k, n=len(d), sr=100 * k / len(d), lo=100 * lo, hi=100 * hi, b=b, c=c, p=mcnemar(b, c),
                         disc=[i for i in common if base[i] != d[i]])
    if title:
        print(f"\n=== {title}")
    print(f"| koşul | k/n | SR | Wilson | Δ | uyumsuz b/c | McNemar | clean ile aralık |")
    print("|---|---|---|---|---|---|---|---|")
    cl = rows["clean"]
    for cnd, r in rows.items():
        overlap = "—" if cnd == "clean" else ("çakışıyor → yön" if r["hi"] >= cl["lo"] and cl["hi"] >= r["lo"] else "ayrışıyor")
        d = "—" if cnd == "clean" else f"{r['sr'] - cl['sr']:+.0f}"
        pm = "—" if cnd == "clean" else ("—" if r["p"] != r["p"] else f"{r['p']:.3f}")
        print(f"| {cnd} | {r['k']}/{r['n']} | {r['sr']:.0f}% | {r['lo']:.0f}–{r['hi']:.0f} | {d} | {'—' if cnd == 'clean' else f'{r['b']}/{r['c']}'} | {pm} | {overlap} |")
    sr = {c_: r["sr"] for c_, r in rows.items()}
    letter, why = classify(sr, near, far)
    print(f"\nÖrüntü: **{letter}**  (eşikler: yakın ≤ {near}, uzak ≥ {far} puan)")
    for w_ in why:
        print(f"  - {w_}")
    # uyumsuz çiftler sahne bazında
    disc_all = sorted({i for c_, r in rows.items() if c_ != "clean" for i in r["disc"]})
    if disc_all:
        print("\nUyumsuz init state'ler (clean ile farklı sonuç veren sahneler):")
        for i in disc_all:
            marks = " ".join(f"{c_}={'✓' if eps[c_][i] else '✗'}" for c_ in order)
            extra = ""
            if features and i in features:
                f_ = features[i]
                extra = f"  | kâse–kâse {100 * f_['d_bowl1_bowl2']:.1f} cm, hedef–tabak {100 * f_['d_bowl1_plate']:.1f} cm, robot sapma {f_['robot_q_dev']:.3f} rad"
            print(f"  init {i:2d}: {marks}{extra}")
        if features:
            fd = [features[i]["robot_q_dev"] for i in disc_all if i in features]
            fa = [features[i]["robot_q_dev"] for i in base if i in features]
            if fd and fa:
                print(f"  robot başlangıç sapması: uyumsuz sahnelerde medyan {sorted(fd)[len(fd) // 2]:.3f} rad, tüm sahnelerde {sorted(fa)[len(fa) // 2]:.3f} rad "
                      f"(fark büyükse kırılganlık robotun başlangıç pozundan; nesneler ±1.5 cm içinde, Ders 19 §3)")
    if labels:
        cnt = defaultdict(lambda: defaultdict(int))
        for (cnd, i), g in labels.items():
            cnt[cnd][g] += 1
        print("\ngrasped_bowl etiketleri (videodan, elle):")
        for cnd, d in cnt.items():
            tot = sum(d.values())
            s_ = ", ".join(f"{g}: {v}/{tot}" for g, v in sorted(d.items()))
            if cnd == "wrong":
                s_ += f"  → faithful (bowl_2) {100 * d.get('bowl_2', 0) / tot:.0f}%, biased (bowl_1) {100 * d.get('bowl_1', 0) / tot:.0f}% (LIBERO-CF grounding ölçüsünün elle sayılmış hâli)"
            print(f"  {cnd}: {s_}")
    print("\nParagraf taslağı:\n" + paragraph(letter, rows, n))
    return letter


def demo():
    pats = {"A": (9, 8, 8, 8), "B": (9, 4, 4, 4), "C": (9, 4, 8, 4), "D": (9, 4, 4, 7)}
    for name, (c, e, w, t) in pats.items():
        eps = {"clean": {i: i < c for i in range(10)}, "empty": {i: i < e for i in range(10)}, "wrong": {i: i < w for i in range(10)}, "tr": {i: i < t for i in range(10)}}
        got = report(eps, title=f"temsili örüntü {name} (Ders 18 görsel 07)")
        assert got == {"A": "A", "B": "B", "C": "C+B", "D": "D"}[name], (name, got)
    print("\ndemo OK: dört temsili örüntü doğru sınıflandı")


def selftest():
    pattern = {"clean": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0], "empty": [1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
               "wrong": [1, 1, 1, 0, 1, 1, 1, 0, 1, 0], "tr": [1, 0, 0, 1, 0, 0, 1, 0, 1, 0]}
    eps = {c: {i: bool(v) for i, v in enumerate(vs)} for c, vs in pattern.items()}
    got = report(eps, title="Ders 18 selftest tablosu (TEMSİLİ)")
    assert got == "B", got   # empty −40 (uzak), wrong −20 (ne yakın ne uzak → C değil), tr−empty = −10 ≤ 10 → tr ≈ empty → B
    print("\nselftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="?")
    ap.add_argument("--features"); ap.add_argument("--labels")
    ap.add_argument("--near", type=float, default=NEAR); ap.add_argument("--far", type=float, default=FAR)
    ap.add_argument("--demo", action="store_true"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.demo:
        return demo()
    if a.selftest:
        return selftest()
    if not a.episodes:
        ap.error("episodes.csv ver (ya da --demo / --selftest)")
    eps = load_episodes(a.episodes)
    if "clean" not in eps or "empty" not in eps:
        sys.exit("episodes.csv'de clean ve empty koşulları gerekli")
    feats = None
    if a.features:
        with open(a.features, encoding="utf-8") as f:
            feats = {int(r["init_state"]): {k: float(v) for k, v in r.items() if k != "init_state"} for r in csv.DictReader(f)}
    labels = None
    if a.labels:
        with open(a.labels, encoding="utf-8") as f:
            labels = {(r["condition"], int(r["init_state"])): r["grasped_bowl"] for r in csv.DictReader(f)}
    report(eps, feats, labels, a.near, a.far, title=pathlib.Path(a.episodes).name)


if __name__ == "__main__":
    main()
