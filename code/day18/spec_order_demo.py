"""Lecture 18 — Show the order produced by the patched build_episode_specs without a model (same loop, fake task_suite).

  python spec_order_demo.py conditions_task0.json [--trials 10] [--parallel 2]

It answers the questions "with num_parallel_envs=2, which episodes are in the same batch?" and "which condition does global_episode_id correspond to?"
(The patch's loop: for episode_idx, (name, text) in ((e, c) for e in range(trials) for c in conds) → init state is the outer loop, condition the inner one.)
No model / GPU / LIBERO needed.
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("conditions")
ap.add_argument("--trials", type=int, default=10)
ap.add_argument("--parallel", type=int, default=2)
ap.add_argument("--task-language", default="pick up the black bowl between the plate and the ramekin and place it on the plate")
a = ap.parse_args()

conds = [(k, a.task_language if v is None else v) for k, v in json.load(open(a.conditions, encoding="utf-8")).items()]
specs = []
seen = 0
for episode_idx, (cond_name, cond_text) in ((e, c) for e in range(a.trials) for c in conds):   # identical to the line in the patch
    seen += 1
    specs.append((seen, episode_idx, cond_name, cond_text))

print(f"{len(conds)} conditions × {a.trials} init states = {len(specs)} episodes; num_parallel_envs={a.parallel}\n")
print(f"{'batch':>5} {'episode':>7} {'init':>4} {'cond':6}  command")
for i in range(0, len(specs), a.parallel):
    for g, init, name, text in specs[i:i + a.parallel]:
        print(f"{i // a.parallel + 1:5d} {g:7d} {init:4d} {name:6s}  {text[:60]!r}")
    print()
same = all(len({init for _, init, _, _ in specs[i:i + a.parallel]}) == 1 for i in range(0, len(specs), a.parallel))
print("Does every batch contain a single init state?", "YES" if same else "NO (the number of conditions does not divide parallel)")
print(f"In the log: trial=<init>, condition=<name>; video name '<name>__<command>' — no need to infer the condition from the episode number.")
