"""Day 10 — Prints the task list and English commands of the four suites."""
from libero.libero import benchmark
for name in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    s = benchmark.get_benchmark_dict()[name]()
    print(f"\n== {name} ({s.n_tasks} tasks)")
    for i in range(s.n_tasks):
        print(f"{i:2d}. {s.get_task(i).language}")
