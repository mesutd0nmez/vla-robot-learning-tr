# day24

Ders 24 — Çözüm yaklaşımları (çeviri katmanı / çok dilli omurga / Türkçe LoRA / parser + sembolik / hidden-state hizalama). Okuma-karar günü; model yok. Üç küçük script + bir çizim script'i.

| Dosya | Ne | Durum |
|---|---|---|
| `tr_parse.py` | Çözüm 4'ün en ucuz biçimi: ~25 kelimelik sözlük + ek soyma + 3 şablon ile Türkçe komut → (fiil, hedef, ilişki, referans, varış) → LIBERO İngilizce şablonu; `datasets/04` 80 Türkçe satır → `parse_roundtrip.csv`; `heldout_tr.csv` (12 yeni cümle) ile set dışı davranış | çalıştırıldı: set içi 79/80, held-out 5/12; `--selftest` 4/4 |
| `heldout_tr.csv` | Parser dondurulduktan sonra yazılmış 12 Türkçe cümle + altın EN + not | — |
| `parse_roundtrip.csv` | 80 satırlık çıktı (kopyası `datasets/04-tr-spatial-commands/data/`) | üretildi |
| `mt_layer_mac.py` | Çözüm 1: Opus-MT ya da NLLB-200 ile aynı 80 satırı çevir, birebir eşleşme + chrF + gecikme | py_compile; **bu oturumda çalıştırılamadı** (HF erişimi yok) → Mesut |
| `qlora_budget.py` | Çözüm 3: OpenVLA-7B + LoRA r=32 + 4-bit omurga için bellek ve T4 saati TAHMİNİ (formül açık) | çalıştırıldı (7,7 GB; T4 65–98 sa) |
| `plot_day24.py` | Görsel 02 (Dong Tablo 2), 03 (Chen Tablo 5 + Dong Tablo 4), 04 (Chen Tablo 1), 05 (parser), 06 (bellek haritası) | çalıştırıldı |

Mac'te (`vla` env; ~1 dk, model yok):
```bash
cd code/day24
python tr_parse.py --selftest
python tr_parse.py                                   # 80 satır + held-out özeti
python tr_parse.py --cmd "kaseyi tabağın hemen yanına koy"
python qlora_budget.py
python plot_day24.py
# çeviri katmanı (model indirir; ilk koşu birkaç dk):
pip install "transformers>=4.40" sentencepiece sacrebleu
python mt_layer_mac.py --model Helsinki-NLP/opus-mt-tr-en
```

## tr_parse.py

```python
"""Ders 24 — Çözüm adayı 4'ün en ucuz biçimi: kural tabanlı Türkçe komut → yapısal çerçeve → İngilizce LIBERO şablonu.

Fikir (Ders 6 / Ders 23: "LLM parser + sembolik çözücü"): Türkçe komutu modele hiç göstermeden önce
  (fiil, hedef nesne, ilişki, referans(lar), varış) çerçevesine ayır, sonra çerçeveden LIBERO'nun eğitimde gördüğü
İngilizce şablonu üret. Bu script LLM bile kullanmaz: ~25 kelimelik sözlük + ek soyma + 3 şablon.
Amaç bir çeviri sistemi yazmak DEĞİL; "çeviri katmanı ne kadar ucuz olabilir ve nerede kırılır" sorusuna
datasets/04'ün 80 Türkçe satırı üzerinde GERÇEK bir sayı vermek (başarı = üretilen İngilizce, `en` sütunuyla birebir aynı).

Kullanım:
  python tr_parse.py                                   # datasets/04 commands.csv → özet tablo + parse_roundtrip.csv
  python tr_parse.py --cmd "soldaki siyah kaseyi al ve tabağın üstüne koy"
  python tr_parse.py --selftest
"""
import argparse, csv, json, re, sys, unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / ".." / ".." / "datasets" / "04-tr-spatial-commands" / "data" / "commands.csv"

# ---------------------------------------------------------------- sözlük (ASCII'ye katlanmış kökler)
ASCII = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def fold(w):
    """Türkçe harfleri ASCII'ye katla, NFC; `tr_ascii` yüzeyi ile `tr_a` aynı anahtara düşsün diye."""
    return unicodedata.normalize("NFC", w).lower().translate(ASCII)


OBJECTS = {  # kök → İngilizce ad
    "kase": "bowl", "tabak": "plate", "ramekin": "ramekin", "masa": "table", "dolap": "cabinet", "cekmece": "drawer",
    "ocak": "stove", "raf": "rack", "sise": "bottle", "peynir": "cheese", "robot": "robot",
}
SOFT = {"tabag": "tabak", "dolab": "dolap", "ocag": "ocak"}            # yumuşama geri alımı
ADJ = {"siyah": "black", "ahsap": "wooden", "ust": "top", "sarap": "wine", "krem": "cream"}
RELATIONS = {  # ilişki adı kökü → (İngilizce seçim ifadesi, İngilizce varış ifadesi)
    "ara": ("between {refs}", None), "yan": ("next to {ref}", None), "orta": ("from {ref} center", None),
    "ust": ("on {ref}", "on {ref}"), "uzer": ("on {ref}", "on {ref}"), "ic": ("in {ref}", "in {ref}"),
    "sol": ("on the left", "to the left of {ref}"), "sag": ("on the right", "to the right of {ref}"),
    "on": ("in the front", "to the front of {ref}"),
    "yakin": ("closer to {ref}", None), "uzak": ("farther from {ref}", None),
}
VERBS = {"al": "pick", "alip": "pick", "koy": "put", "yerlestir": "put", "it": "push"}
STOP = {"ve", "ile", "daha", "olan", "duran", "dogru", "taraftaki", "tarafina", "tarafindaki", "-"}
# ek soyma: en uzun eşleşen kök + izin verilen ek dizisi
SUFFIX = re.compile(r"^(l[ae]|n?[iu]n|[sy]?[iu]|[yn]?[ae]|n?d[ae]n?|t[ae]n?|ki|n)*$")  # izin verilen ek dizisi (ASCII)
STEMS = sorted(set(OBJECTS) | set(SOFT) | set(ADJ) | set(RELATIONS) | set(VERBS), key=len, reverse=True)


def analyze(word):
    """Bir sözcüğü (tür, kök, hâl) üçlüsüne indir. Hâl ∈ {acc, dat, loc, abl, ins, gen, ki, bare}."""
    w = fold(word)
    if w in STOP:
        return ("STOP", w, "bare")
    for stem in STEMS:  # en uzun kök önce (ust > us; sarap > sag...)
        if w.startswith(stem) and SUFFIX.match(w[len(stem):]):
            suf = w[len(stem):]
            stem = SOFT.get(stem, stem)
            if stem in VERBS and suf == "":
                return ("VERB", stem, "bare")
            if stem in OBJECTS:
                return ("OBJ", stem, case_of(suf))
            if stem in RELATIONS and (suf != "" or stem not in ADJ):
                return ("REL", stem, case_of(suf))
            if stem in ADJ:
                return ("ADJ", stem, "bare")
    return ("UNK", w, "bare")


def case_of(suf):
    if suf.endswith("ki"):
        return "ki"
    if re.search(r"(d|t)[ae]n$", suf):
        return "abl"
    if re.search(r"n?d[ae]$", suf) or suf.endswith("nda") or suf.endswith("nde"):
        return "loc"
    if re.search(r"(y|n)?[ae]$", suf) and suf not in ("",):
        return "dat"
    if re.search(r"n?[iıu]n$", suf):
        return "gen"
    if re.search(r"(y|n|s)?[iıu]$", suf) and suf:
        return "acc"
    if suf in ("le", "la"):
        return "ins"
    return "bare"


def noun_phrase(tokens):
    """[ADJ...] OBJ → İngilizce 'the wooden cabinet' / 'the top drawer'."""
    adjs = [ADJ[s] for k, s, _ in tokens if k == "ADJ"]
    objs = [OBJECTS[s] for k, s, _ in tokens if k == "OBJ"]
    if not objs:
        return None
    if objs == ["cheese"] and "cream" in adjs:
        return "the cream cheese"
    if objs == ["bottle"] and "wine" in adjs:
        return "the wine bottle"
    return "the " + " ".join(adjs + [objs[-1]])


def parse(cmd):
    toks = [analyze(w) for w in re.findall(r"[^\s]+", cmd.replace("'", " "))]
    # yalın "üst": ardından "çekmece" geliyorsa sıfat (top drawer), yoksa ilişki adı (tr_bare: "tabak üst koy")
    for i, t in enumerate(toks):
        if t == ("ADJ", "ust", "bare") and not (i + 1 < len(toks) and toks[i + 1][:2] == ("OBJ", "cekmece")):
            toks[i] = ("REL", "ust", "bare")
    frame = {"verb": None, "target": None, "relation": None, "refs": [], "dest": None, "dest_rel": None, "tokens": toks}
    verbs = [s for k, s, _ in toks if k == "VERB"]
    if not verbs:
        return frame
    frame["verb"] = VERBS[verbs[0]] if verbs[0] != "koy" or len(verbs) == 1 else "pick"
    if "it" in verbs and len(verbs) == 1:
        frame["verb"] = "push"
    # --- seçim ilişkisi: ilk REL (varış ilişkisinden önce gelen) ---
    rel_idx = [i for i, (k, s, c) in enumerate(toks) if k == "REL"]
    obj_idx = [i for i, (k, s, c) in enumerate(toks) if k == "OBJ"]
    if frame["verb"] == "pick":  # Spatial şablonu: ... siyah kase ... al ve tabak üst koy
        first_verb = next(i for i, (k, s, c) in enumerate(toks) if k == "VERB")
        before = toks[:first_verb]
        after = toks[first_verb + 1:]
        # hedef: "al"dan önceki son OBJ öbeği (kase)
        # hedef: fiilden önce belirtme hâli (-yi) almış OBJ; yoksa (tr_bare) fiilden önceki son OBJ
        acc_before = [i for i in obj_idx if i < first_verb and toks[i][2] == "acc"]
        tgt_pos = acc_before[-1] if acc_before else max((i for i in obj_idx if i < first_verb), default=None)
        if tgt_pos is None:
            return frame
        tgt_start = tgt_pos
        while tgt_start - 1 >= 0 and toks[tgt_start - 1][0] == "ADJ" and toks[tgt_start - 1][1] == "siyah":
            tgt_start -= 1
        frame["target"] = noun_phrase(toks[tgt_start:tgt_pos + 1])
        rels_sel = [i for i in rel_idx if i < first_verb]          # fiilden önceki ilişki adları (seçim)
        # referans öbekleri: fiilden önce, hedef öbeğin dışında kalan [ADJ...] OBJ dizileri
        groups, cur = [], []
        for i, t in enumerate(toks[:first_verb]):
            if tgt_start <= i <= tgt_pos or t[0] not in ("OBJ", "ADJ"):
                continue
            cur.append(t)
            if t[0] == "OBJ":
                groups.append(cur); cur = []
        frame["refs"] = [noun_phrase(g) for g in groups]
        if rels_sel:
            frame["relation"] = toks[rels_sel[-1]][1]
        elif groups:  # ilişki adı yok, yalın bulunma: "dolabın üst çekmecesindeki" → in the top drawer of the cabinet
            frame["relation"] = "ic"
        dest_objs = [i for i in obj_idx if i > first_verb]
        dest_rels = [i for i in rel_idx if i > first_verb]
        if dest_objs:
            frame["dest"] = noun_phrase([toks[dest_objs[0]]])
            frame["dest_rel"] = toks[dest_rels[0]][1] if dest_rels else "ust"
    else:  # Goal şablonu: X-i Y-nin üst-ü-ne koy / Y-nin üst-ü-ne X-i koy
        acc = [i for i in obj_idx if toks[i][2] == "acc"]
        rel = rel_idx[0] if rel_idx else None
        if acc:
            tpos = acc[0]
        else:  # tr_bare: ilk OBJ hedef
            tpos = obj_idx[0] if obj_idx else None
        if tpos is None:
            return frame
        tstart = tpos
        while tstart - 1 >= 0 and toks[tstart - 1][0] == "ADJ":
            tstart -= 1
        frame["target"] = noun_phrase(toks[tstart:tpos + 1])
        others = [i for i in obj_idx if not (tstart <= i <= tpos)]
        if others:
            frame["dest"] = noun_phrase([toks[others[0]]])
        frame["dest_rel"] = toks[rel][1] if rel is not None else None
        if frame["dest_rel"] is None and frame["dest"]:
            # "şarap şişesini rafa koy": ilişki adı yok → DAT = on
            frame["dest_rel"] = "ust"
    return frame


def render(frame):
    """Çerçeveden LIBERO İngilizce şablonu."""
    if not frame["verb"] or not frame["target"]:
        return None
    if frame["verb"] == "pick":
        rel = frame["relation"]
        if rel is None:
            return None
        sel, _ = RELATIONS[rel]
        refs = frame["refs"]
        if rel == "ara":
            if len(refs) != 2:
                return None
            sel = sel.format(refs=f"{refs[0]} and {refs[1]}")
        elif rel == "ic":
            if len(refs) == 2:   # top drawer of the wooden cabinet
                sel = f"in {refs[1]} of {refs[0]}"
            elif len(refs) == 1:
                sel = f"in {refs[0]}"
            else:
                return None
        elif rel == "orta":
            sel = "from table center" if refs and "table" in refs[0] else None
            if sel is None:
                return None
        elif "{ref}" in sel:
            if not refs:
                return None
            sel = sel.format(ref=refs[-1])
        dest = frame["dest"] or "the plate"
        return f"pick up {frame['target']} {sel} and place it on {dest}"
    if frame["verb"] == "put":
        if not frame["dest"] or not frame["dest_rel"]:
            return None
        _, d = RELATIONS[frame["dest_rel"]]
        if d is None:
            return None
        d = d.format(ref=frame["dest"])
        if frame["dest_rel"] == "ust" and frame["dest"] == "the cabinet":
            d = "on top of the cabinet"  # LIBERO-Goal görev 4'ün adı böyle
        return f"put {frame['target']} {d}"
    if frame["verb"] == "push":
        if not frame["dest"] or not frame["dest_rel"]:
            return None
        _, d = RELATIONS[frame["dest_rel"]]
        return f"push {frame['target']} {d.format(ref=frame['dest'])}"
    return None


def run_set(csv_path, out_path):
    rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8")) if r["lang"] == "tr"]
    en = {r["pair_id"]: r["instruction"] for r in csv.DictReader(open(csv_path, encoding="utf-8")) if r["lang"] == "en"}
    out, stats = [], {}
    for r in rows:
        fr = parse(r["instruction"])
        pred = render(fr)
        gold = en[r["pair_id"]]
        ok = pred == gold
        kind = "ok" if ok else ("parse_fail" if pred is None else "wrong_en")
        stats.setdefault(r["variant"], {"ok": 0, "parse_fail": 0, "wrong_en": 0})[kind] += 1
        out.append({"id": r["id"], "pair_id": r["pair_id"], "variant": r["variant"], "tr": r["instruction"],
                    "frame": json.dumps({k: v for k, v in fr.items() if k != "tokens"}, ensure_ascii=False),
                    "en_pred": pred or "", "en_gold": gold, "result": kind})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"{'yüzey':<9}{'n':>4}{'birebir':>9}{'yanlış EN':>11}{'parse yok':>11}")
    for v in ("tr_a", "tr_ascii", "tr_bare", "tr_b"):
        s = stats.get(v, {})
        n = sum(s.values())
        print(f"{v:<9}{n:>4}{s.get('ok',0):>9}{s.get('wrong_en',0):>11}{s.get('parse_fail',0):>11}")
    tot = sum(sum(s.values()) for s in stats.values()); ok = sum(s["ok"] for s in stats.values())
    print(f"{'toplam':<9}{tot:>4}{ok:>9}   ({100*ok/tot:.0f}%)   → {out_path}")
    fails = [o for o in out if o["result"] != "ok"]
    if fails:
        print("\nBaşarısızlar:")
        for o in fails:
            print(f"  {o['id']:<14} {o['tr']}\n      → {o['en_pred'] or '∅'}\n      ≠ {o['en_gold']}")
    return stats


def run_heldout(path):
    """Parser dondurulduktan SONRA yazılmış 12 yeni Türkçe yüzey: sözlük/şablon dışına çıkınca ne oluyor?"""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    ok = 0
    print(f"\nHeld-out ({path.name}):")
    for r in rows:
        pred = render(parse(r["tr"]))
        hit = pred == r["en_gold"]
        ok += hit
        print(f"  [{'OK ' if hit else 'X  '}] {r['id']} {r['tr']}\n        → {pred or '∅'}" + ("" if hit else f"\n        ≠ {r['en_gold']}   ({r['note']})"))
    print(f"  held-out: {ok}/{len(rows)}")
    return ok, len(rows)


def selftest():
    cases = {
        "soldaki siyah kaseyi al ve tabağın üstüne koy": "pick up the black bowl on the left and place it on the plate",
        "kaseyi tabağın sağına koy": "put the bowl to the right of the plate",
        "tabağı ocağın önüne it": "push the plate to the front of the stove",
        "ramekinin yanindaki siyah kaseyi al ve tabagin ustune koy": "pick up the black bowl next to the ramekin and place it on the plate",
    }
    bad = 0
    for tr, en in cases.items():
        got = render(parse(tr))
        flag = "OK " if got == en else "FAIL"
        bad += got != en
        print(f"[{flag}] {tr}\n       {got}")
    print(f"selftest: {len(cases)-bad}/{len(cases)}")
    return bad == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--out", default=str(HERE / "parse_roundtrip.csv"))
    ap.add_argument("--cmd")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.cmd:
        fr = parse(a.cmd)
        print(json.dumps({k: v for k, v in fr.items()}, ensure_ascii=False, indent=1))
        print("→", render(fr))
        sys.exit(0)
    run_set(a.csv, a.out)
    if (HERE / "heldout_tr.csv").exists():
        run_heldout(HERE / "heldout_tr.csv")
```

## mt_layer_mac.py

```python
"""Ders 24 — Çözüm adayı 1 (çeviri katmanı, MT) Mac'te: datasets/04'ün 80 Türkçe komutunu açık bir MT modeliyle
İngilizce'ye çevir, `en` sütunuyla karşılaştır, komut başına gecikmeyi ölç.

Bu oturumda ÇALIŞTIRILAMADI (Hugging Face erişimi yok) — Mac'te `vla` env'inde koş:
    pip install "transformers>=4.40" sentencepiece sacrebleu
    python mt_layer_mac.py --model Helsinki-NLP/opus-mt-tr-en            # ~300 MB, hızlı
    python mt_layer_mac.py --model facebook/nllb-200-distilled-600M      # ~2,4 GB, 200 dil
Çıktı: mt_roundtrip_<model>.csv (tr, en_pred, en_gold, exact, chrf, ms) + özet satırı.

Ölçtüğümüz üç şey: (1) birebir eşleşme (LIBERO şablonuna düşüyor mu?), (2) chrF (anlam yakınlığı için kaba ölçü),
(3) gecikme — kapalı döngüde her episode'da BİR kez çevrilir, adım başına değil; yani 200 ms bile sorun değil.
Asıl soru 'yanlış ama geçerli' çeviriler: `wrong` koşulunu (Ders 18) çeviri katmanının kendisi üretebilir.
"""
import argparse, csv, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / ".." / ".." / "datasets" / "04-tr-spatial-commands" / "data" / "commands.csv"


def load(model_name, device):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, src_lang="tur_Latn" if "nllb" in model_name else None)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()
    forced = tok.convert_tokens_to_ids("eng_Latn") if "nllb" in model_name else None
    return tok, model, forced


def translate(tok, model, forced, text, device):
    import torch
    enc = tok(text, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, forced_bos_token_id=forced, num_beams=4, max_new_tokens=48)
    ms = (time.perf_counter() - t0) * 1000
    return tok.batch_decode(out, skip_special_tokens=True)[0].strip().lower().rstrip("."), ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Helsinki-NLP/opus-mt-tr-en")
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--device", default=None, help="mps / cpu (varsayılan: mps varsa mps)")
    a = ap.parse_args()
    import torch
    device = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    try:
        from sacrebleu.metrics import CHRF
        chrf = CHRF()
    except ImportError:
        chrf = None
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    gold = {r["pair_id"]: r["instruction"] for r in rows if r["lang"] == "en"}
    tok, model, forced = load(a.model, device)
    out_path = HERE / f"mt_roundtrip_{a.model.split('/')[-1]}.csv"
    exact, n, lat = {}, {}, []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "variant", "tr", "en_pred", "en_gold", "exact", "chrf", "ms"])
        for r in rows:
            if r["lang"] != "tr":
                continue
            pred, ms = translate(tok, model, forced, r["instruction"], device)
            g = gold[r["pair_id"]]
            ex = int(pred == g)
            score = chrf.sentence_score(pred, [g]).score if chrf else float("nan")
            w.writerow([r["id"], r["variant"], r["instruction"], pred, g, ex, f"{score:.1f}", f"{ms:.0f}"])
            exact[r["variant"]] = exact.get(r["variant"], 0) + ex; n[r["variant"]] = n.get(r["variant"], 0) + 1; lat.append(ms)
    print(f"model={a.model} device={device}")
    for v in ("tr_a", "tr_ascii", "tr_bare", "tr_b"):
        print(f"  {v:<9} birebir {exact.get(v,0)}/{n.get(v,0)}")
    lat.sort(); print(f"  gecikme medyan {lat[len(lat)//2]:.0f} ms (ilk çağrı {lat[0]:.0f}…{lat[-1]:.0f} ms aralığı) → {out_path.name}")


if __name__ == "__main__":
    main()
```

## qlora_budget.py

```python
"""Ders 24 — Çözüm adayı 3 (Türkçe LoRA) için kaba bellek / süre hesabı. TAHMİNDİR; ölçüm değil.

Girdiler (gerçek): OpenVLA-7B ≈ 7,19 milyar parametre (makale Tablo 1: tam FT 7.188,1 M), LoRA r=32 → 97,6 M eğitilebilir
parametre (%1,4); openvla repo finetune.py: batch 16 ≈ 72 GB, '≥ ~27 GB' alt sınır; `use_quantization` bayrağı
bitsandbytes 4-bit (NF4) ile LoRA'ya izin verir. T4: 16 GB, bf16 yok (fp16), ~65 TFLOPS fp16 tensor (NVIDIA veri sayfası).

    python qlora_budget.py
"""
GB = 1024 ** 3


def mem(params=7.19e9, lora=97.6e6, bits_base=4, act_batch1_gb=1.5, vision_bf16_gb=1.2):
    base = params * bits_base / 8 / GB                   # dondurulmuş omurga (NF4 ≈ 0,5 bayt/param + ölçekler ≈ +%6)
    base *= 1.06
    lora_w = lora * 4 / GB                              # LoRA ağırlıkları fp32
    lora_grad = lora * 4 / GB
    adam = lora * 8 / GB                                # m ve v
    return {"omurga 4-bit": base, "görüntü omurgası (bf16/fp16)": vision_bf16_gb, "LoRA ağırlık": lora_w,
            "LoRA grad": lora_grad, "Adam durumları": adam, "aktivasyon (batch 1, 291 token, checkpointing)": act_batch1_gb}


def hours(a100_hours=(10, 15), t4_slowdown=5.0, quant_penalty=1.3):
    """OpenVLA makalesi: LoRA ile 'tek A100'de 10–15 saat' (yeni görev). T4 ≈ A100'ün 1/5'i (fp16 tensor 65 vs 312 TFLOPS)
    + 4-bit dequant maliyeti ≈ ×1,3 → saat aralığı."""
    return tuple(h * t4_slowdown * quant_penalty for h in a100_hours)


if __name__ == "__main__":
    m = mem()
    print("Bellek (tahmin), OpenVLA-7B + LoRA r=32 + 4-bit omurga, batch 1:")
    for k, v in m.items():
        print(f"  {k:<48} {v:5.2f} GB")
    tot = sum(m.values())
    print(f"  {'TOPLAM':<48} {tot:5.2f} GB   (T4 16 GB: {'sığar' if tot < 14 else 'riskli'}; pay ≈ {16 - tot:.1f} GB)")
    lo, hi = hours()
    print(f"\nSüre (tahmin): A100 10–15 sa → T4 {lo:.0f}–{hi:.0f} sa; Colab ücretsiz oturum ≤ ~12 sa → {lo/12:.0f}–{hi/12:.0f} oturum, "
          f"checkpoint/resume şart.\nKaggle: haftada 30 GPU saati → {lo/30:.1f}–{hi/30:.1f} hafta.")
    print("\nUyarı: bu bir hesap; gerçek sayı Mesut'un T4'te `finetune.py --use_quantization True --batch_size 1` ile ölçeceği değerdir.")
```

## plot_day24.py

```python
"""Ders 24 — görseller 02–06. Bütün sayılar makalelerin tablolarından (başlıklarda kaynak) ya da bu klasördeki gerçek çıktılardan;
tek tahmin görsel 06'daki QLoRA bellek çubuğu (açıkça 'tahmin' yazılı).

  python plot_day24.py            → ../../lectures/images/lecture-24/0X-*.png
"""
import csv, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / ".." / ".." / "lectures" / "images" / "lecture-24"
OUT.mkdir(parents=True, exist_ok=True)
Y, B, G, R, GR = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6", "#bdbdbd"
YD, BD, GD, RD = "#c9961a", "#4a86c8", "#3f9d5a", "#d45c5c"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------- 02: Dong vd. 2606.11906 Tablo 2 — çıkarım zamanı müdahaleler
LANGS = ["ZH", "JA", "KO", "FR", "ES", "AR", "TH", "PT", "VI"]
DONG = {
    "OpenVLA-OFT": {"EN": 97.1,
                    "baseline": [51.7, 59.4, 52.3, 65.3, 64.1, 50.8, 53.2, 63.6, 59.6],
                    "EN-CoT prompt": [56.8, 54.8, 52.0, 62.4, 63.4, 51.1, 52.8, 62.3, 55.1],
                    "average shift": [60.8, 58.9, 54.7, 65.5, 64.5, 53.8, 54.5, 64.2, 57.1],
                    "step-wise shift": [68.8, 68.8, 66.4, 70.9, 69.8, 64.5, 62.6, 70.7, 68.0]},
    "π0.5": {"EN": 96.9,
             "baseline": [56.7, 56.5, 55.4, 61.2, 59.1, 55.7, 56.0, 59.4, 57.3],
             "EN-CoT prompt": [55.2, 55.7, 55.7, 59.9, 57.9, 56.4, 56.5, 57.2, 56.3],
             "average shift": [54.1, 53.9, 52.9, 57.2, 56.2, 53.0, 52.6, 55.6, 54.2],
             "step-wise shift": [80.3, 80.9, 79.8, 82.0, 81.1, 81.8, 79.9, 82.1, 81.9]},
}


def fig02():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    cols = {"baseline": GR, "EN-CoT prompt": B, "average shift": Y, "step-wise shift": G}
    x = np.arange(len(LANGS)); w = 0.2
    for ax, (model, d) in zip(axes, DONG.items()):
        for k, (name, c) in enumerate(cols.items()):
            ax.bar(x + (k - 1.5) * w, d[name], w, color=c, edgecolor="#555", linewidth=0.5, label=name)
        ax.axhline(d["EN"], color=RD, ls="--", lw=1.2)
        ax.text(len(LANGS) - 0.5, d["EN"] + 1, f"EN {d['EN']}", color=RD, ha="right", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(LANGS); ax.set_ylim(0, 105)
        ax.set_title(f"{model} — LIBERO 4 suite ortalaması (%)", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("success rate (%)")
    axes[1].legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.95)
    fig.suptitle("Eğitimsiz müdahaleler: 'İngilizce düşün' prompt'u işe yaramıyor, hidden-state kaydırması yarıyor\n"
                 "Dong vd. 2026 (arXiv:2606.11906) Tablo 2 — Google Translate ile 9 dil, Türkçe yok", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "02-inference-time.png", dpi=170); plt.close(fig)


# ---------------------------------------------------------------- 03: eğitim tarafı — Chen Tablo 5 + Dong Tablo 4
CHEN5 = {  # Qwen3-VL + GR00T başı, LIBERO 4 suite ortalaması, en/zh/fr/ru/ar
    "GR00T (ek eğitim yok)": [95.3, 89.1, 62.3, 56.5, 59.3],
    "E-FT (yalnız EN VLM FT)": [96.3, 84.3, 55.0, 48.5, 52.8],
    "M-FT (çok dilli VLM FT)": [95.4, 87.6, 65.1, 57.1, 62.8],
    "E-CT (EN birlikte eğitim)": [94.0, 86.9, 68.3, 54.5, 57.4],
    "M-CT (çok dilli birlikte)": [92.6, 88.3, 87.4, 83.1, 89.4],
    "MPCA (M-CT + hizalama)": [95.3, 90.6, 89.5, 85.5, 90.8],
}
DONG4 = {"baseline (ZH)": [55.8, 76.0, 9.8, 65.0], "tam fine-tune": [94.0, 97.8, 76.4, 74.0],
         "rastgele %50 adım": [88.8, 90.8, 0.0, 0.0], "kritik %50 adım": [92.6, 96.0, 72.0, 81.2]}


def fig03():
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5), gridspec_kw={"width_ratios": [1.25, 1]})
    M = np.array(list(CHEN5.values()))
    im = a.imshow(M, cmap="YlGn", vmin=40, vmax=100, aspect="auto")
    a.set_xticks(range(5)); a.set_xticklabels(["en", "zh", "fr", "ru", "ar"])
    a.set_yticks(range(len(CHEN5))); a.set_yticklabels(list(CHEN5.keys()), fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            a.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=9, color="black" if M[i, j] > 60 else "#444")
    a.set_title("Çok dilli VLM verisiyle eğitim (50 bin çevrilmiş VQA, 4 dil)\nChen vd. 2026 (arXiv:2606.15714) Tablo 5 — Qwen3-VL + GR00T başı, LIBERO ort. (%)", fontsize=10)
    a.spines[:].set_visible(False)
    suites = ["Spatial", "Object", "Goal", "Long"]; x = np.arange(4); w = 0.2
    for k, (name, v) in enumerate(DONG4.items()):
        b.bar(x + (k - 1.5) * w, v, w, color=[GR, Y, R, G][k], edgecolor="#555", linewidth=0.5, label=name)
    b.set_xticks(x); b.set_xticklabels(suites); b.set_ylim(0, 125); b.grid(axis="y", alpha=0.3)
    b.set_title("Hedef dilde (ZH) komutla fine-tune — OpenVLA-OFT\nDong vd. 2026 Tablo 4; 'kritik adım' = dil gradyanı yüksek adım", fontsize=10)
    b.legend(fontsize=8, loc="upper center", ncol=2)
    b.set_ylabel("success rate (%)")
    fig.suptitle("Eğitimli çözümler: veri hedef dilde olunca açık kapanıyor; yalnız İngilizce ek eğitim (E-FT) açığı BÜYÜTÜYOR", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "03-training-side.png", dpi=170); plt.close(fig)


# ---------------------------------------------------------------- 04: omurga karşılaştırması — Chen Tablo 1 (EN'e göre fark, puan)
CHEN1 = {
    "OpenVLA-OFT (LLaMA-2 / Prismatic)": [-28.6, -28.6, -27.8, -29.8, -27.5],
    "π0.5 (PaliGemma)": [-39.6, -38.0, -41.8, -41.4, -43.5],
    "Qwen2.5-VL + OFT başı": [-17.8, -32.1, -45.3, -59.1, -43.3],
    "Qwen3-VL + OFT başı": [-7.3, -39.1, -48.8, -47.1, -63.1],
    "Qwen3-VL + FAST başı": [-29.6, -70.8, -76.4, -73.6, -70.5],
    "Qwen3-VL + GR00T başı": [-6.1, -33.0, -38.8, -36.0, -48.0],
    "Qwen3-VL + π başı": [-12.1, -40.9, -44.6, -46.8, -55.4],
}


def fig04():
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    langs = ["zh", "fr", "ru", "ar", "∅ (boş komut)"]
    x = np.arange(len(CHEN1)); w = 0.15
    cols = [YD, BD, GD, RD, "#777"]
    for j, (lg, c) in enumerate(zip(langs, cols)):
        vals = [v[j] for v in CHEN1.values()]
        ax.bar(x + (j - 2) * w, vals, w, color=c, label=lg, edgecolor="none")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(list(CHEN1.keys()), rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("success rate farkı, EN'e göre (puan)")
    ax.set_ylim(-85, 5); ax.grid(axis="y", alpha=0.3)
    ax.legend(title="komut dili", fontsize=9, loc="lower left", ncol=5)
    ax.set_title("'Çok dilli omurga' tek başına çözüm değil: Qwen3-VL yalnızca Çince'de iyi; Fransızca/Rusça/Arapça'da LLaMA-2 tabanlı OFT'den KÖTÜ\n"
                 "Chen vd. 2026 (arXiv:2606.15714) Tablo 1 — LIBERO 4 suite ortalaması; omurga ve action başı birlikte değişiyor (temiz karşılaştırma değil)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "04-backbones.png", dpi=170); plt.close(fig)


# ---------------------------------------------------------------- 05: kural tabanlı parser — gerçek çıktı
def fig05():
    here = Path(__file__).resolve().parent
    rows = list(csv.DictReader(open(here / "parse_roundtrip.csv", encoding="utf-8")))
    surf = ["tr_a", "tr_ascii", "tr_bare", "tr_b"]
    ok = [sum(1 for r in rows if r["variant"] == s and r["result"] == "ok") for s in surf]
    n = [sum(1 for r in rows if r["variant"] == s) for s in surf]
    # held-out: tr_parse.run_heldout çıktısından (parse_out.txt) değil, yeniden hesapla
    sys.path.insert(0, str(here))
    import tr_parse
    hrows = list(csv.DictReader(open(here / "heldout_tr.csv", encoding="utf-8")))
    hok = sum(1 for r in hrows if tr_parse.render(tr_parse.parse(r["tr"])) == r["en_gold"])
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    labels = surf + ["held-out\n(12 yeni cümle)"]
    vals = [100 * o / m for o, m in zip(ok, n)] + [100 * hok / len(hrows)]
    cols = [G, G, G, Y, R]
    bars = ax.bar(labels, vals, color=cols, edgecolor="#555", linewidth=0.6)
    for b_, o, m in zip(bars, ok + [hok], n + [len(hrows)]):
        ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height() + 2, f"{o}/{m}", ha="center", fontsize=10)
    ax.set_ylim(0, 122); ax.set_ylabel("EN şablonla birebir eşleşme (%)")
    ax.axvline(3.5, color="#999", ls=":", lw=1)
    ax.text(1.5, 114, "parser bu 80 satıra BAKILARAK yazıldı (üst sınır)", ha="center", fontsize=9, color="#555")
    ax.text(4, 114, "yazıldıktan sonra", ha="center", fontsize=9, color="#555")
    ax.set_title("Çözüm 4'ün en ucuz biçimi: 25 kelimelik sözlük + ek soyma + 3 şablon (tr_parse.py, gerçek çıktı)\n"
                 "Set içinde %99, set dışında %42 — kural tabanlı çeviri katmanının gerçek yüzü", fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "05-parser.png", dpi=170); plt.close(fig)


# ---------------------------------------------------------------- 06: bellek bütçesi — gerçek değerler + bir tahmin
def fig06():
    items = [
        ("OpenVLA inference\nbf16", 16.8, B, "OpenVLA Tablo 2"),
        ("OpenVLA inference\nint8", 10.2, B, "OpenVLA Tablo 2"),
        ("OpenVLA inference\nint4", 7.0, B, "OpenVLA Tablo 2"),
        ("LoRA r=32 eğitim\nbatch 16", 72, Y, "finetune.py / README"),
        ("LoRA r=32 eğitim\nen küçük batch", 27, Y, "README: '≥ ~27 GB'"),
        ("LoRA + 4-bit\nbatch 1 (TAHMİN)", 7.7, R, "qlora_budget.py"),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(items))
    ax.bar(x, [v for _, v, _, _ in items], color=[c for _, _, c, _ in items], edgecolor="#555", linewidth=0.6)
    for i, (_, v, _, src) in enumerate(items):
        ax.text(i, v + 1.5, f"{v:g} GB", ha="center", fontsize=10)
        ax.text(i, -9, src, ha="center", fontsize=7.5, color="#666")
    for yv, lbl, c in [(16, "Colab/Kaggle T4 — 16 GB", RD), (32, "MacBook Pro M2 — 32 GB birleşik (MPS; bitsandbytes yok)", BD), (80, "A100 — 80 GB", GD)]:
        ax.axhline(yv, color=c, ls="--", lw=1.1); ax.text(len(items) - 0.5, yv + 1, lbl, ha="right", color=c, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([n for n, _, _, _ in items], fontsize=9)
    ax.set_ylim(-14, 88); ax.set_ylabel("GPU belleği (GB)")
    ax.set_title("Türkçe LoRA fine-tune Mesut'un donanımına sığar mı? — OpenVLA-7B bellek haritası\n"
                 "Mavi: inference (OpenVLA makalesi Tablo 2); sarı: LoRA eğitimi (openvla repo c8f03f4); kırmızı: 4-bit QLoRA tahmini (ölçülmedi)", fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "06-memory.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    fig02(); fig03(); fig04(); fig05(); fig06()
    print("ok →", OUT.resolve())
```
