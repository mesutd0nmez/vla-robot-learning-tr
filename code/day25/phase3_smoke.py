"""Ders 25 — Faz 3 araç envanteri: code/day17–24 script'lerini kuru koşuda (model yok, GPU yok) çalıştır,
hangisi hangi dış girdiyle çalışıyor tabloya dök.

Kullanım (repo kökünden ya da code/day25 içinden):
    python phase3_smoke.py                              # dış klon yok: yalnızca "yok" bağımlılıklılar
    python phase3_smoke.py --libero ~/src/LIBERO --plus ~/src/LIBERO-plus \
                           --llama code/day22/tokenizer.model --qwen path/to/qwen3/tokenizer.json
    python phase3_smoke.py --only 19,20                 # yalnızca Ders 19 ve 20 script'leri
    python phase3_smoke.py --out results                # smoke.csv + smoke.md buraya

Her satır: ders, script, komut, bağımlılık (yok / LIBERO klonu / LIBERO-plus klonu / bddl paketi /
sentencepiece+tokenizer / Webots / HF+GPU), sonuç (OK / HATA / ATLANDI), süre, çıktının son satırı.
"ATLANDI" = bağımlılık verilmedi; "HATA" = çalıştırıldı ama çıkış kodu 0 değil. Hiçbir script değiştirilmez;
çıktı dosyaları geçici bir klasöre yazılır.
"""
import argparse, csv, os, pathlib, shutil, subprocess, sys, tempfile, time

HERE = pathlib.Path(__file__).resolve().parent
CODE = HERE.parent
ROOT = CODE.parent
DS = ROOT / "datasets"
PY = sys.executable


def manifest(libero, plus, tmp, llama=None, qwen=None):
    """(ders, script, argv, cwd, bağımlılık, kısa açıklama). argv None → atla (bağımlılık yok)."""
    L = str(libero) if libero else None
    P = str(plus) if plus else None
    T = (str(llama), str(qwen)) if (llama and qwen) else None
    try:
        import bddl.parsing  # noqa: F401  (pip install "bddl==1.0.1" future)
        B = True
    except Exception:
        B = False
    t = lambda name: str(tmp / name)
    d = lambda n: CODE / f"day{n}"
    M = []
    # Ders 17 — LIBERO-plus klonundan bozulma sayımları
    M += [(17, "perturbation_table.py", [P, "--json", t("pt.json")] if P else None, d(17), "LIBERO-plus klonu", "7 bozulma × Spatial sayımı"),
          (17, "camera_geometry.py", [P, t("cam")] if P else None, d(17), "LIBERO-plus klonu", "376 kamera pozu (V/H/S/ER/EV)"),
          (17, "robot_init_spread.py", [P, t("rob")] if P else None, d(17), "LIBERO-plus klonu", "500 Panda init_qpos, ‖Δ‖"),
          (17, "layout_shift.py", [P, t("lay")] if P else None, d(17), "LIBERO-plus klonu", "_add_ / _level kaymaları"),
          (17, "classify_rewrites.py", [P, t("cls")] if P else None, d(17), "LIBERO-plus klonu", "görev 0'ın 50 yeniden yazımı R0/R1/R2"),
          (17, "make_tr_variant.py", [P, "--commands", str(d(17) / "tr_commands.csv"), "--out", t("trv")] if (P and B) else None, d(17), "LIBERO-plus klonu + bddl", "Türkçe _language_trK BDDL üretimi (bddl 1.0.1 gerek)")]
    # Ders 18 — koşul dosyası, istatistik, log analizi (selftest)
    M += [(18, "make_conditions.py", ["--task", "0", "--out", t("cond.json")], d(18), "yok", "clean/empty/wrong/tr koşul JSON'u"),
          (18, "spec_order_demo.py", [str(d(18) / "conditions_task0.json")], d(18), "yok", "init × koşul sırası"),
          (18, "analyze_4cond.py", ["--selftest"], d(18), "yok", "eval.log → tablo (Wilson)"),
          (18, "power_day18.py", [t("power")], d(18), "yok", "Fisher / McNemar güç eğrileri")]
    # Ders 19 — okuma protokolü, BDDL yapısı, init state geometrisi
    M += [(19, "read_patterns.py", ["--selftest"], d(19), "yok", "A/B/C/D karar ağacı"),
          (19, "read_patterns.py --demo", ["--demo"], d(19), "yok", "örnek tabloyla paragraf"),
          (19, "suite_structure.py", ["--libero", L] if L else None, d(19), "LIBERO klonu", "komutsuz üst sınır 9/10/1/9"),
          (19, "init_state_geometry.py", ["--libero", L, "--task", "0"] if L else None, d(19), "LIBERO klonu", "50 init state'in geometrisi")]
    # Ders 20 — Türkçe set
    M += [(20, "tr_morph.py", ["--selftest"], d(20), "yok", "ek üretici 42 kontrol"),
          (20, "make_tr_set.py", ["--out", t("set"), "--referents", str(DS / "04-tr-spatial-commands/data/referents.csv")], d(20), "yok", "pairs.csv + commands.csv üretimi"),
          (20, "validate_set.py", [str(DS / "04-tr-spatial-commands/data"), "--d01", str(DS / "01-libero-spatial-tr-mini/data/commands.csv")], d(20), "yok", "set tutarlılığı"),
          (20, "scene_frames.py", ["--libero", L] if L else None, d(20), "LIBERO klonu", "üç çerçevede referans tablosu"),
          (20, "make_cf_bddl.py", ["--libero", L, "--pairs", str(DS / "04-tr-spatial-commands/data/pairs.csv"), "--out", t("cf")] if (L and B) else None, d(20), "LIBERO klonu + bddl", "karşı-olgusal BDDL (bddl 1.0.1 gerek)")]
    # Ders 21 — Webots
    M += [(21, "check_webots_env.py", None, d(21), "Webots", "WEBOTS_HOME + import controller"),
          (21, "ur10e_fk.py", [], d(21), "yok", "UR10e FK (scipy IK --solve ayrı)")]
    # Ders 22 — tokenizasyon
    M += [(22, "tokenize_set.py", ["--llama", T[0], "--qwen", T[1], "--out", t("tok")] if T else None, d(22), "sentencepiece+tokenizer", "100 komut × 2 tokenizer (tokenizer.model / tokenizer.json gerek)"),
          (22, "tokens.csv == datasets/05?", ["--diff", t("tok/tokens.csv"), str(DS / "05-tokenization-tr-commands/data/tokens.csv")] if T else None, HERE, "sentencepiece+tokenizer", "yeniden üretim: ilk 7 sütun birebir mi"),
          (22, "probe_edge_cases.py", ["--llama", T[0], "--qwen", T[1]] if T else None, d(22), "sentencepiece+tokenizer", "İ.lower, yanlış arkadaşlar")]
    # Ders 23 — tahmin aralığı
    M += [(23, "prediction_interval.py", ["--selftest"], d(23), "yok", "PI formülü 5 kontrol"),
          (23, "prediction_interval.py --scores", ["--scores", "90", "80", "90"], d(23), "yok", "3 tekrardan aralık")]
    # Ders 24 — parser, bütçe, MT
    M += [(24, "tr_parse.py --selftest", ["--selftest"], d(24), "yok", "parser 4 kontrol"),
          (24, "tr_parse.py", ["--csv", str(DS / "04-tr-spatial-commands/data/commands.csv"), "--out", t("parse.csv")], d(24), "yok", "80 satır + held-out 12"),
          (24, "qlora_budget.py", [], d(24), "yok", "4-bit LoRA bellek/süre tahmini"),
          (24, "mt_layer_mac.py", None, d(24), "HF+GPU", "Opus-MT / NLLB round-trip (model indirir)")]
    return M


def diff_first_cols(a, b, k=7):
    """İki CSV'nin ilk k sütununu karşılaştır (Ders 22 yeniden üretim kontrolü)."""
    ra = [row[:k] for row in csv.reader(open(a, encoding="utf-8"))]
    rb = [row[:k] for row in csv.reader(open(b, encoding="utf-8"))]
    bad = sum(1 for x, y in zip(ra, rb) if x != y) + abs(len(ra) - len(rb))
    print(f"{len(ra)} vs {len(rb)} satır, ilk {k} sütunda farklı satır: {bad} → {'BİREBİR AYNI' if bad == 0 else 'FARKLI'}")
    return 0 if bad == 0 else 1


def run_one(script, argv, cwd, timeout):
    name = script.split()[0]
    if argv and argv[0] == "--diff":                   # yerleşik kontrol, ayrı script değil
        cmd = [PY, str(HERE / "phase3_smoke.py")] + argv
    else:
        cmd = [PY, str(cwd / name)] + argv
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"})
        rc, out = p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired:
        rc, out = 124, "zaman aşımı"
    dt = time.time() - t0
    lines = [l for l in out.strip().splitlines() if l.strip()]
    last = lines[-1][:110] if lines else ""
    return rc, dt, last, cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--libero", help="LIBERO klon kökü (Ders 19/20 script'leri)")
    ap.add_argument("--plus", help="LIBERO-plus klon kökü (Ders 17 script'leri)")
    ap.add_argument("--llama", help="Meta LLaMA-2 tokenizer.model (Ders 22)")
    ap.add_argument("--qwen", help="Qwen3 tokenizer.json (Ders 22)")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"), help="(iç kullanım) iki CSV'nin ilk 7 sütununu karşılaştır")
    ap.add_argument("--only", help="virgülle ders numaraları, ör. 19,20")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    if a.diff:
        return diff_first_cols(*a.diff)
    only = {int(x) for x in a.only.split(",")} if a.only else None
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="faz3smoke_"))
    rows = []
    for day, script, argv, cwd, need, desc in manifest(a.libero and pathlib.Path(a.libero).expanduser(),
                                                      a.plus and pathlib.Path(a.plus).expanduser(), tmp,
                                                      a.llama and pathlib.Path(a.llama).expanduser(), a.qwen and pathlib.Path(a.qwen).expanduser()):
        if only and day not in only:
            continue
        if argv is None:
            rows.append(dict(ders=day, script=script, bagimlilik=need, sonuc="ATLANDI", sure_s="", son_satir="", aciklama=desc))
            print(f"[--] Ders {day:2d} {script:34s} ATLANDI ({need})")
            continue
        rc, dt, last, cmd = run_one(script, argv, cwd, a.timeout)
        res = "OK" if rc == 0 else "HATA"
        rows.append(dict(ders=day, script=script, bagimlilik=need, sonuc=res, sure_s=f"{dt:.1f}", son_satir=last, aciklama=desc))
        print(f"[{'OK' if rc == 0 else '!!'}] Ders {day:2d} {script:34s} {res:7s} {dt:6.1f}s  {last}")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "smoke.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    ok = sum(r["sonuc"] == "OK" for r in rows); err = sum(r["sonuc"] == "HATA" for r in rows); sk = sum(r["sonuc"] == "ATLANDI" for r in rows)
    with open(out / "smoke.md", "w", encoding="utf-8") as f:
        f.write(f"# Faz 3 kuru koşu — {ok} OK / {err} HATA / {sk} ATLANDI (toplam {len(rows)})\n\n")
        f.write("| Ders | Script | Bağımlılık | Sonuç | Süre | Son satır |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['ders']} | `{r['script']}` | {r['bagimlilik']} | {r['sonuc']} | {r['sure_s']} | {r['son_satir'].replace('|', '/')} |\n")
    print(f"\n{ok} OK / {err} HATA / {sk} ATLANDI → {out / 'smoke.csv'}, {out / 'smoke.md'}")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
