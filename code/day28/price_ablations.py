"""Lecture 28 — Price the ablation cells with the Lecture 27 budget model (the v0.1 → v0.2 difference).

  python price_ablations.py      # ../../thesis/experiment-matrix.csv (v0.2) and results/experiment-matrix_v0.1.csv

`budget.py` from Lecture 27 is imported (same episode model, same scenarios); its results/ folder is NOT
touched — outputs are written under code/day28/results/.
Re-reading ablations (which need no new episodes) are not rows in the matrix; they are listed in ablation-plan.md.
"""
import csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "day27"))
import budget as B  # noqa: E402

B.P_SUCC.update({"tr_bare_ascii": 0.4, "tr_cs": 0.6})   # cost mix only (hypothesis), not a result prediction


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def totals(res, lean=None):
    t = {}
    for tier in ("pilot", "core", "ext"):
        t[tier] = {k: sum(r["h_" + k] for r in res if r["tier"] == tier and r["where"] == "T4") for k in B.SCEN}
    return t


def main():
    v1 = load(os.path.join(HERE, "results", "experiment-matrix_v0.1.csv"))
    v2 = load(os.path.join(HERE, "..", "..", "thesis", "experiment-matrix.csv"))
    out = []
    for name, rows in (("v0.1", v1), ("v0.2", v2)):
        for lean in (None, 20):
            res = B.budget([dict(r) for r in rows], n_init_core=lean)
            t = totals(res)
            pc = {k: t["pilot"][k] + t["core"][k] for k in B.SCEN}
            out.append(dict(matrix=name, n_init_core=lean or 50, **{f"pilot_core_{k}": round(v, 1) for k, v in pc.items()},
                            ext_orta=round(t["ext"]["orta"], 1), kaggle_weeks_orta=round(pc["orta"] / B.KAGGLE_WEEK_H, 1)))
            if name == "v0.2" and lean is None:
                cells = [r for r in res if r["cell"].startswith("AB") or r["cell"] == "B6"]
    with open(os.path.join(HERE, "results", "budget_v01_v02.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    with open(os.path.join(HERE, "results", "ablation_cells.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0])); w.writeheader(); w.writerows(cells)
    print("Ablation cells (v0.2, n_init 50):")
    for r in cells:
        print(f"  {r['cell']:4s} {r['tier']:5s} episode {r['episodes']:5d}  T4 h optimistic/mid/pessimistic {r['h_iyimser']:5.1f} / {r['h_orta']:5.1f} / {r['h_kötümser']:5.1f}  {r['status']}")
    print("\nTotal (pilot + core, T4):")
    for o in out:
        print(f"  {o['matrix']}  n_init {o['n_init_core']:2d}: optimistic {o['pilot_core_iyimser']:6.1f}  mid {o['pilot_core_orta']:6.1f}  "
              f"pessimistic {o['pilot_core_kötümser']:6.1f}  | extension (mid) {o['ext_orta']:6.1f} | Kaggle weeks {o['kaggle_weeks_orta']}")


if __name__ == "__main__":
    main()
