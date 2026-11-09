"""Ders 16 — VLANeXt LIBERO-Plus eval.log → episode tablosu + runs.csv satırları.

Kullanım:
  python parse_plus_log.py <eval.log> --run-id vlanext_plus_t4_01 --condition rewritten --lang en --csv results/runs.csv
  python parse_plus_log.py --selftest      # script'in kendi format string'iyle üretilmiş log'la sınar

Log biçimi scripts/libero_plus_bench_eval.py'dendir (Ders 12'deki libero_bench_eval'den FARKLI: trial/episode_time yok,
category var):
  Episode result: task=%d/%d category=%s success=%s rollout_steps=%d total_episodes=%d/%d total_successes=%d total_success_rate=%.1f%% video=%s
runs.csv sözleşmesi (CLAUDE.md): run_id, model, checkpoint, suite, task_id, lang, condition, n_episodes, success_rate
task_id = LIBERO-Plus indeksi (0 tabanlı, VLANEXT_TASK_IDS ile aynı).
"""
import argparse, csv, pathlib, re, sys, tempfile
from collections import defaultdict

EP = re.compile(r"Episode result: task=(\d+)/(\d+) category=(\S+) success=(True|False) rollout_steps=(\d+) "
                r"total_episodes=(\d+)/(\d+) total_successes=(\d+) total_success_rate=([\d.]+)% video=(\S+)")
OVR = re.compile(r"Instruction override: (.*)")
SUITE = re.compile(r"Task suite: (\S+)")
CKPT = re.compile(r"checkpoint=(\S+)")
HEADER = ["run_id", "model", "checkpoint", "suite", "task_id", "lang", "condition", "n_episodes", "success_rate"]


def parse(text):
    eps = [m.groups() for m in EP.finditer(text)]
    s = SUITE.search(text); c = CKPT.search(text); o = OVR.search(text)
    return {"suite": s.group(1) if s else "?", "checkpoint": pathlib.Path(c.group(1)).name if c else "?",
            "override": o.group(1).strip() if o else "(satır yok — yama 4 uygulanmamış olabilir)", "episodes": eps}


def report(info, run_id, lang, condition, csv_path=None):
    eps = info["episodes"]
    if not eps:
        sys.exit("Log'da 'Episode result' satırı yok — eval başlamadan düşmüş olabilir; log'un sonuna bak.")
    print(f"suite={info['suite']}  checkpoint={info['checkpoint']}  override={info['override']}  episode={len(eps)}")
    per = defaultdict(lambda: [0, 0]); cats = {}
    print(f"{'indeks':>6} {'kategori':>10} {'success':>7} {'adım':>5}  video")
    for task, n, cat, ok, steps, *_rest, video in eps:
        i = int(task) - 1; per[i][0] += 1; per[i][1] += ok == "True"; cats[i] = cat
        print(f"{i:6d} {cat:>10} {ok:>7} {int(steps):5d}  {pathlib.Path(video).name[:60]}")
    tn = sum(v[0] for v in per.values()); ts = sum(v[1] for v in per.values())
    print(f"Toplam: {ts}/{tn} = {100 * ts / tn:.1f}%")
    if csv_path:
        out = pathlib.Path(csv_path); out.parent.mkdir(parents=True, exist_ok=True)
        new = not out.exists()
        with out.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(HEADER)
            for i, (n, s) in sorted(per.items()):
                w.writerow([run_id, "VLANeXt-B", info["checkpoint"], info["suite"] + "_plus", i, lang, condition, n,
                            round(s / n, 4)])
        print(f"{len(per)} satır eklendi → {out}")
    return ts, tn


def selftest():
    fmt = ("Episode result: task=%d/%d category=%s success=%s rollout_steps=%d total_episodes=%d/%d "
           "total_successes=%d total_success_rate=%.1f%% video=%s")
    lines = ["2026-11-09 10:00:00 INFO Task suite: libero_spatial",
             "2026-11-09 10:00:00 INFO Instruction override: ''",
             "2026-11-09 10:00:01 INFO Eval setup: checkpoint=checkpoints/VLANeXt_libero_spatial.pt tasks=2402"]
    demo = [(989, True, 131), (995, False, 220), (996, True, 152)]
    for k, (t, ok, st) in enumerate(demo, 1):
        succ = sum(d[1] for d in demo[:k])
        lines.append("2026-11-09 10:0%d:00 INFO " % k + fmt % (t, 2402, "Language", ok, st, k, 3, succ, 100 * succ / k,
                                                              f"/x/episode={k}--success={ok}--task=.mp4"))
    with tempfile.TemporaryDirectory() as d:
        csvp = pathlib.Path(d) / "runs.csv"
        ts, tn = report(parse("\n".join(lines)), "selftest", "none", "empty", csvp)
        rows = list(csv.DictReader(csvp.open(encoding="utf-8")))
    assert (ts, tn) == (2, 3) and len(rows) == 3 and rows[0]["task_id"] == "988", rows
    print("SELFTEST OK (sentetik log; gerçek sonuç değildir)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?")
    ap.add_argument("--run-id", default="vlanext_plus_t4_01")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--condition", default="rewritten", help="clean | rewritten | empty | tr")
    ap.add_argument("--csv")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.log:
        selftest()
    else:
        report(parse(pathlib.Path(a.log).read_text(encoding="utf-8", errors="replace")), a.run_id, a.lang, a.condition, a.csv)
