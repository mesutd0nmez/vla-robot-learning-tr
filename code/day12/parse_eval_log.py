"""Gün 12 — VLANeXt eval.log dosyasını oku; episode tablosu + runs.csv satırı üret.
Kullanım:  python parse_eval_log.py <eval.log> [--run-id vlanext_spatial_t4_01] [--lang en] [--condition correct] [--csv results/runs.csv]
Log biçimi scripts/libero_bench_eval.py'deki "Episode result: ..." ve "Eval finished: ..." satırlarıdır.
runs.csv sütunları repo CLAUDE.md'deki sözleşmeyle aynıdır:
  run_id, model, checkpoint, suite, task_id, lang, condition, n_episodes, success_rate
"""
import re, sys, csv, argparse, pathlib
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument("log")
p.add_argument("--run-id", default="vlanext_t4_01")
p.add_argument("--lang", default="en")
p.add_argument("--condition", default="correct", help="correct | empty | wrong | paraphrase | tr")
p.add_argument("--csv", default=None, help="varsa satırları bu CSV'ye ekler (başlık yoksa yazar)")
args = p.parse_args()

text = pathlib.Path(args.log).read_text(encoding="utf-8", errors="replace")
ep_re = re.compile(r"Episode result: task=(\d+)/(\d+) trial=(\d+) success=(True|False) rollout_steps=(\d+) episode_time=(\S+)")
suite = re.search(r"Task suite: (\S+)", text)
ckpt  = re.search(r"checkpoint=(\S+)", text)
suite = suite.group(1) if suite else "?"
ckpt  = pathlib.Path(ckpt.group(1)).name if ckpt else "?"

rows, per_task = [], defaultdict(lambda: [0, 0])
for m in ep_re.finditer(text):
    task, n_tasks, trial, ok, steps, t = m.groups()
    task_id = int(task) - 1
    ok = ok == "True"
    per_task[task_id][0] += 1; per_task[task_id][1] += int(ok)
    rows.append((task_id, int(trial), ok, int(steps), t))

if not rows:
    sys.exit("Log'da 'Episode result' satırı yok — eval başlamadan düşmüş olabilir; dosyanın sonuna bak.")

print(f"suite={suite}  checkpoint={ckpt}  episode={len(rows)}")
print(f"{'task':>4} {'trial':>5} {'success':>7} {'steps':>5} {'süre':>8}")
for task_id, trial, ok, steps, t in rows:
    print(f"{task_id:>4} {trial:>5} {str(ok):>7} {steps:>5} {t:>8}")
tot_n = sum(v[0] for v in per_task.values()); tot_s = sum(v[1] for v in per_task.values())
print(f"\nToplam: {tot_s}/{tot_n} = {100*tot_s/tot_n:.1f}%")
fin = re.search(r"Eval finished: .*final_success_rate=([\d.]+)%.*elapsed=(\S+)", text)
if fin:
    print(f"Script'in kendi özeti: SR={fin.group(1)}%  süre={fin.group(2)}")

if args.csv:
    out = pathlib.Path(args.csv); out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run_id", "model", "checkpoint", "suite", "task_id", "lang", "condition", "n_episodes", "success_rate"])
        for task_id in sorted(per_task):
            n, s = per_task[task_id]
            w.writerow([args.run_id, "VLANeXt-B", ckpt, suite, task_id, args.lang, args.condition, n, f"{s/n:.3f}"])
    print(f"{len(per_task)} satır eklendi → {out}")
