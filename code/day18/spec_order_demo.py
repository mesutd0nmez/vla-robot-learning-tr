"""Ders 18 — yamalı build_episode_specs'in ürettiği sırayı modelsiz göster (aynı döngü, sahte task_suite).

  python spec_order_demo.py conditions_task0.json [--trials 10] [--parallel 2]

Soruya cevap: "num_parallel_envs=2 iken hangi episode'lar aynı batch'te?" ve "global_episode_id hangi koşula denk geliyor?"
(Yamanın döngüsü: for episode_idx, (ad, metin) in ((e, c) for e in range(trials) for c in conds) → init state dış, koşul iç döngü.)
Model / GPU / LIBERO gerekmez.
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
for episode_idx, (cond_name, cond_text) in ((e, c) for e in range(a.trials) for c in conds):   # yamadaki satırın aynısı
    seen += 1
    specs.append((seen, episode_idx, cond_name, cond_text))

print(f"{len(conds)} koşul × {a.trials} init state = {len(specs)} episode; num_parallel_envs={a.parallel}\n")
print(f"{'batch':>5} {'episode':>7} {'init':>4} {'koşul':6}  komut")
for i in range(0, len(specs), a.parallel):
    for g, init, name, text in specs[i:i + a.parallel]:
        print(f"{i // a.parallel + 1:5d} {g:7d} {init:4d} {name:6s}  {text[:60]!r}")
    print()
same = all(len({init for _, init, _, _ in specs[i:i + a.parallel]}) == 1 for i in range(0, len(specs), a.parallel))
print("Her batch tek bir init state mi içeriyor?", "EVET" if same else "HAYIR (koşul sayısı parallel'e bölünmüyor)")
print(f"Log'da trial=<init>, condition=<ad>; video adı '<ad>__<komut>' — episode numarasından koşulu çıkarmak gerekmez.")
