# day27

Ders 27 — deney matrisi ve bütçe. Model yok (GPU yok); LIBERO simülasyonu gerçekten koştu. Matris dosyası `thesis/experiment-matrix.csv`.

| Dosya | Ne | Durum |
|---|---|---|
| `sim_step_bench.py` | LIBERO'da görev başına init state sayısı, reset süresi, `env.step` süresi (render kapalı / 128 / 256 px) | çalıştırıldı (Linux 2 çekirdek, GPU yok) → `results/sim_step.csv` |
| `budget.py` | Matrisi okur, episode süresi modeli (reset + adım × t_sim + çağrı × t_call) ile hücre başına T4 saati (iyimser/orta/kötümser), yalın matris (`--lean`), tersine güç (bütçe → n → MDE); `--calls-sv/--calls-ov` ile ölçülen çağrı süresi; `--png` görsel 02–04 | çalıştırıldı; selftest OK → `results/budget*.csv`, `inverse.csv`, `budget_stdout.txt` |
| `plot_day27.py` | Görsel 06 (`sim_step.csv`) | çalıştırıldı |

Mac'te (`vla` env, Ders 9–10):
```bash
cd code/day27
python sim_step_bench.py --libero ~/LIBERO --tasks libero_spatial:0 --steps 30
python budget.py                       # ~10 s
python budget.py --lean 20 --calls-sv 2.5 --calls-ov 0.8
python plot_day27.py
```

## sim_step_bench.py

```python
"""Ders 27 — Bütçe modelinin ölçülebilen yarısı: LIBERO simülasyon adımı kaç ms?

Model çağrısı (GPU) ile simülasyon adımı (CPU: MuJoCo fizik + offscreen render) bir episode'un
süresini birlikte belirler. GPU yarısını burada ölçemeyiz; CPU yarısını ölçeriz.
Her görev için: init state sayısı (LIBERO'nun eşli n tavanı), reset süresi, adım süresi
(sıfır action; render dahil), 128 ve 256 px kamera.

  python sim_step_bench.py --libero <LIBERO klonu> [--steps 60] [--out results/sim_step.csv]

Mac'te: `conda activate vla` (Ders 9–10 ortamı). Linux başsız sunucuda MUJOCO_GL=egl ya da osmesa.
"""
import argparse, csv, os, platform, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--libero", required=True, help="LIBERO deposunun kökü (içinde libero/libero)")
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--tasks", default="libero_spatial:0,libero_spatial:5,libero_goal:0,libero_goal:8")
ap.add_argument("--res", default="128,256")
ap.add_argument("--out", default="results/sim_step.csv")
a = ap.parse_args()

sys.path.insert(0, a.libero)
cfg_dir = os.path.abspath(os.path.join(os.path.dirname(a.out) or ".", ".libero_cfg"))
os.makedirs(cfg_dir, exist_ok=True)
root = os.path.join(os.path.abspath(a.libero), "libero", "libero")
with open(os.path.join(cfg_dir, "config.yaml"), "w") as f:  # LIBERO'nun etkileşimli sorusunu atla
    f.write(f"benchmark_root: {root}\nbddl_files: {root}/bddl_files\ninit_states: {root}/init_files\n"
            f"datasets: {root}/../datasets\nassets: {root}/assets\n")
os.environ["LIBERO_CONFIG_PATH"] = cfg_dir

import torch  # init state dosyaları torch pickle
_load = torch.load
torch.load = lambda *x, **k: _load(*x, **{**k, "weights_only": False})  # torch>=2.6 varsayılanı
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.env_wrapper import ControlEnv

rows = []
bench = benchmark.get_benchmark_dict()
for spec in a.tasks.split(","):
    suite, tid = spec.split(":"); tid = int(tid)
    ts = bench[suite](); task = ts.get_task(tid)
    inits = ts.get_task_init_states(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    for res in [0] + list(map(int, a.res.split(","))):   # 0 = render kapalı: yalnız fizik
        if res == 0:
            env = ControlEnv(bddl_file_name=bddl, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        else:
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
        if res:
            env.seed(0)
        t0 = time.perf_counter(); env.reset(); env.set_init_state(inits[0]); t_reset = time.perf_counter() - t0
        dummy = np.zeros(7); dummy[-1] = -1.0  # LIBERO eval'lerinin "bekle" action'ı
        for _ in range(5):
            env.step(dummy)  # ısınma
        ts_ = []
        for _ in range(a.steps):
            t0 = time.perf_counter(); obs, r, done, info = env.step(dummy); ts_.append(time.perf_counter() - t0)
        env.close()
        ms = 1000 * np.array(ts_)
        row = dict(suite=suite, task_id=tid, n_init=len(inits), res=res, reset_s=round(t_reset, 2),
                   step_ms_median=round(float(np.median(ms)), 1), step_ms_p90=round(float(np.percentile(ms, 90)), 1),
                   img_shape="x".join(map(str, obs["agentview_image"].shape)) if res else "render yok", language=task.language,
                   host=f"{platform.system()}-{platform.machine()}-{os.cpu_count()}cpu", gl=os.environ.get("MUJOCO_GL", "default"))
        rows.append(row); print(row, flush=True)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("yazıldı:", a.out)
```

## budget.py

```python
"""Ders 27 — Deney matrisinin bütçesi ve tersine güç hesabı.

  python budget.py                         # ../../thesis/experiment-matrix.csv → tablo + results/budget.csv
  python budget.py --png ../../lectures/images/lecture-27
  python budget.py --calls-sv 2.5 --calls-ov 0.8   # kendi T4 ölçümünle (s/çağrı) yeniden hesapla

Episode süresi modeli (her terim ayrı yazılır ki ölçtükçe yerine koyasın):
  T_ep = t_reset + (W + L) · t_sim + ceil(L / chunk) · t_call
    W = 10 bekleme adımı (num_steps_wait; model çağrılmaz)
    L = başarılıysa ortalama başarılı uzunluk L_s, değilse max_steps (Spatial 220, Goal 300;
        openvla c8f03f4 run_libero_eval.py:173-180)
    chunk = S-VN 8 (Ders 12: num_steps_execute 8), S-OV 1 (OpenVLA her adımda bir action)
    t_sim = fizik + render; fizik 22 ms ÖLÇÜLDÜ (sim_step_bench.py, 2 çekirdek, render kapalı)
  Beklenen T_ep = p · T_ep(başarı) + (1 − p) · T_ep(başarısız); p = koşulun beklenen success'i (TAHMİN).
Ölçülmeyen her sayı SCEN içinde ve "hipotez" diye işaretli. Güç: rq_power.py ile aynı model
(McNemar exact, ters gürültü r), ama b+c ~ Binom(n, Δ+2r), b | m ~ Binom(m, (Δ+r)/(Δ+2r)) ayrışımıyla
vektörleştirilmiş (n=300'de saniyeler). n=50, r=0,05 → 23 puan (Ders 26 ile aynı; selftest kontrol eder).
"""
import argparse, csv, math, os
import numpy as np
from scipy.stats import binom

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(HERE, "..", "..", "thesis", "experiment-matrix.csv")
MAX_STEPS = {"libero_spatial": 220, "libero_goal": 300}   # gerçek kod
L_SUCC = {"libero_spatial": 100, "libero_goal": 120}       # hipotez (Ders 18: Spatial ~96; Goal tahmin)
WAIT = 10
T_RESET = 2.5                                              # ölçüldü: 2,4–3,2 s (sim_step_bench.py; ilk yükleme 5,3)
T_PHYS = 0.022                                             # ölçüldü: fizik, render kapalı
# expected success per condition — cost mix only (hipotez); sonuç DEĞİL
P_SUCC = {"clean": 0.8, "empty": 0.3, "wrong": 0.3, "tr_a": 0.5, "tr_ascii": 0.5, "tr_b": 0.5, "tr_bare": 0.4,
          "en_B": 0.3, "tr_B": 0.25, "tr_mt": 0.7, "tr_parse": 0.75}
# scenarios: (render s/adım, S-VN s/çağrı T4, S-OV 4-bit s/çağrı T4, S-OV bf16 MPS s/çağrı)
SCEN = {"iyimser": (0.005, 1.0, 0.5, 1.5), "orta": (0.010, 2.5, 0.8, 2.0), "kötümser": (0.030, 5.0, 1.6, 3.0)}
CHUNK = {"S-VN": 8, "S-OV": 1, "S-OV-mac": 1, "S-OV-lora": 1}
# the fraction of a cell that really has to run (string cache: if MT/parser output == clean, the episode is the clean episode)
RUN_FRACTION = {"tr_parse": 1 / 80, "tr_mt": 0.6}   # parser: set içi 79/80 birebir (Ders 24, gerçek); MT: hipotez
LEAN = (20, 0.0)
KAGGLE_WEEK_H = 30.0                                   # kullanıcı raporu; hesabındaki kotayı kontrol et


def t_episode(system, suite, cond, scen):
    rend, sv, ov, mac = SCEN[scen]
    t_call = {"S-VN": sv, "S-OV": ov, "S-OV-lora": ov, "S-OV-mac": mac}[system]
    t_sim = T_PHYS + rend
    def one(L):
        return T_RESET + (WAIT + L) * t_sim + math.ceil(L / CHUNK[system]) * t_call
    p = P_SUCC.get(cond, 0.5)
    return p * one(L_SUCC[suite]) + (1 - p) * one(MAX_STEPS[suite])


def expand(row):
    """Bir matris satırını (sistem, suite, koşul, episode) parçalarına açar."""
    systems = row["system"].split(";")
    suites = row["suite"].split("+")
    conds = [c for c in row["conditions"].split(";") if c != "train"]
    n_tasks, n_init, rep = int(row["n_tasks"]), int(row["n_init"]), int(row["repeats"])
    out = []
    for s in systems:
        for su in suites:
            tasks = n_tasks / len(suites)
            for c in conds:
                ep = tasks * n_init * rep
                out.append((s, su, c, ep, ep * RUN_FRACTION.get(c, 1.0)))
    return out


def budget(rows, n_init_core=None):
    res = []
    for r in rows:
        if n_init_core and r["tier"] == "core" and r["rq"] in ("a", "b", "c") and int(r["n_init"]) == 50:
            r = {**r, "n_init": str(n_init_core)}
        parts = expand(r)
        ep = sum(p[3] for p in parts); ep_run = sum(p[4] for p in parts)
        h = {sc: sum(p[4] * t_episode(p[0], p[1], p[2], sc) for p in parts) / 3600 for sc in SCEN}
        where = "Mac" if parts and all(p[0] == "S-OV-mac" for p in parts) else "T4"
        if r["cell_id"] == "C3":
            h = {"iyimser": 65.0, "orta": 80.0, "kötümser": 98.0}  # Ders 24 qlora_budget.py TAHMİNİ (eğitim)
        res.append(dict(cell=r["cell_id"], rq=r["rq"], tier=r["tier"], where=where, status=r["status"],
                        episodes=round(ep), episodes_run=round(ep_run),
                        **{f"h_{k}": round(v, 1) for k, v in h.items()}))
    return res


# ---------- güç (vektörleştirilmiş, rq_power.py ile aynı model) ----------
def power(n, delta, r, alpha=0.05):
    pb, pc = delta + r, r
    if pb + pc > 1:
        return float("nan")
    m = np.arange(n + 1)
    pm = binom.pmf(m, n, pb + pc)
    tot = 0.0
    th = pb / (pb + pc) if pb + pc > 0 else 0.5
    for mi in m[1:]:
        b = np.arange(mi + 1)
        k = np.minimum(b, mi - b)
        p = np.minimum(1.0, 2 * binom.cdf(k, mi, 0.5))
        tot += pm[mi] * binom.pmf(b, mi, th)[p < alpha].sum()
    return tot


def mde(n, r=0.05, target=0.8):
    for d in np.arange(0.01, 0.96, 0.01):
        if power(n, float(d), r) >= target:
            return float(d)
    return float("nan")


def selftest():
    assert abs(mde(50, 0.05) - 0.23) < 1e-9, mde(50, 0.05)   # Ders 26 tablosu
    assert abs(mde(100, 0.05) - 0.15) < 1e-9, mde(100, 0.05)
    assert abs(mde(50, 0.0) - 0.16) < 1e-9
    print("selftest OK: MDE(50; r=0,05)=23, MDE(100)=15, MDE(50; r=0)=16 — rq_power.py ile aynı")


def inverse(hours_per_system, scen="orta", suite="libero_spatial", tasks=6, conds=("clean", "tr_a")):
    """Bütçe (T4 saati, bir sistem, bir blok, iki koşul) → görev başına init → havuz n → MDE."""
    t = sum(t_episode("S-OV", suite, c, scen) for c in conds)  # bir init state'in iki koşulu (S-OV pahalı olan)
    n_init = min(50, int(hours_per_system * 3600 / (t * tasks)))
    return n_init, tasks * n_init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=MATRIX); ap.add_argument("--png", default=None)
    ap.add_argument("--lean", type=int, default=20, help="yalın matris: çekirdek hücrelerde görev başına init")
    ap.add_argument("--calls-sv", type=float); ap.add_argument("--calls-ov", type=float)
    a = ap.parse_args()
    if a.calls_sv or a.calls_ov:
        r, sv, ov, mac = SCEN["orta"]; SCEN["ölçüm"] = (r, a.calls_sv or sv, a.calls_ov or ov, mac)
    selftest()
    rows = list(csv.DictReader(open(a.matrix, encoding="utf-8")))
    res = budget(rows)
    keys = list(res[0])
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "budget.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(res)
    print("\nSaniye/episode (orta senaryo, beklenen):")
    for s in ("S-VN", "S-OV", "S-OV-mac"):
        print(f"  {s:9s} Spatial clean {t_episode(s,'libero_spatial','clean','orta'):6.0f}  empty {t_episode(s,'libero_spatial','empty','orta'):6.0f}"
              f" | Goal clean {t_episode(s,'libero_goal','clean','orta'):6.0f}  empty {t_episode(s,'libero_goal','empty','orta'):6.0f}")
    print(f"\n{'hücre':5s} {'rq':3s} {'kat':5s} {'yer':3s} {'episode':>7s} {'koşan':>6s} " + " ".join(f"{k:>9s}" for k in SCEN) + "  durum")
    for r in res:
        print(f"{r['cell']:5s} {r['rq']:3s} {r['tier']:5s} {r['where']:3s} {r['episodes']:7d} {r['episodes_run']:6d} "
              + " ".join(f"{r['h_'+k]:9.1f}" for k in SCEN) + "  " + r["status"])
    print("\nToplamlar (saat):")
    summ = {}
    for tier in ("pilot", "core", "ext"):
        for where in ("T4", "Mac"):
            sel = [r for r in res if r["tier"] == tier and r["where"] == where]
            if sel:
                tot = {k: sum(r["h_" + k] for r in sel) for k in SCEN}
                summ[(tier, where)] = tot
                print(f"  {tier:5s} {where}: " + "  ".join(f"{k} {v:6.1f}" for k, v in tot.items())
                      + (f"   → Kaggle haftası (orta, tek T4): {tot['orta']/KAGGLE_WEEK_H:4.1f}; T4×2: {tot['orta']/KAGGLE_WEEK_H/2:4.1f}" if where == "T4" else ""))
    for rq in ("cal", "a", "b", "c"):
        sel = [r for r in res if r["rq"] == rq and r["tier"] != "ext"]
        print(f"  RQ-{rq:3s} (pilot+core): orta {sum(r['h_orta'] for r in sel):6.1f} sa, episode {sum(r['episodes_run'] for r in sel)}")
    lean = budget([dict(r) for r in rows], n_init_core=a.lean)
    lt = sum(r["h_orta"] for r in lean if r["tier"] in ("pilot", "core") and r["where"] == "T4")
    ll = {k: sum(r["h_" + k] for r in lean if r["tier"] in ("pilot", "core") and r["where"] == "T4") for k in SCEN}
    print(f"\nYalın matris (çekirdekte görev başına {a.lean} init, blok havuzu n={6*a.lean}, MDE {100*mde(6*a.lean):.0f} puan): "
          + "  ".join(f"{k} {v:6.1f}" for k, v in ll.items()) + f"  → Kaggle haftası (orta): {lt/KAGGLE_WEEK_H:.1f}; T4×2: {lt/KAGGLE_WEEK_H/2:.1f}")
    with open(os.path.join(HERE, "results", "budget_lean.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(lean)
    global LEAN
    LEAN = (a.lean, lt)
    print("\nTersine: RQ-a Spatial, S-OV, iki koşul (clean, tr_a), 6 görev — bütçe → n → MDE (r=0,05):")
    inv = []
    for H in (5, 10, 20, 30, 40, 60):
        ni, npool = inverse(H)
        m1, mp = mde(ni) if ni >= 5 else float("nan"), mde(npool) if npool >= 5 else float("nan")
        inv.append((H, ni, npool, m1, mp))
        print(f"  {H:3d} T4 sa → görev başına {ni:2d} init, havuz n={npool:3d} → MDE görev {100*m1:4.0f} / blok {100*mp:4.0f} puan")
    with open(os.path.join(HERE, "results", "inverse.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t4_hours", "n_init_per_task", "n_pooled", "mde_task", "mde_block"]); w.writerows(inv)
    if a.png:
        plots(a.png, res)


def plots(out, res):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
    YD, BD, GD, RD = "#c79a00", "#3b6ea8", "#2e8b57", "#c0392b"
    os.makedirs(out, exist_ok=True)
    # 02 — episode süresinin bileşenleri
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
    for ax, suite in zip(axes, ("libero_spatial", "libero_goal")):
        labels, sim, call, rst = [], [], [], []
        for s, nm in (("S-VN", "S-VN\nT4"), ("S-OV", "S-OV\n4-bit T4"), ("S-OV-mac", "S-OV\nbf16 Mac")):
            for sc in SCEN:
                if sc == "ölçüm": continue
                rend, sv, ov, mac = SCEN[sc]; tc = {"S-VN": sv, "S-OV": ov, "S-OV-mac": mac}[s]
                L = MAX_STEPS[suite]  # başarısız episode = en pahalı durum
                labels.append(f"{nm}\n{sc[:4]}"); rst.append(T_RESET); sim.append((WAIT + L) * (T_PHYS + rend))
                call.append(math.ceil(L / CHUNK[s]) * tc)
        x = np.arange(len(labels))
        ax.bar(x, rst, color="#ddd", edgecolor="#666", label="reset (ölçüldü)")
        ax.bar(x, sim, bottom=rst, color=B, edgecolor=BD, label="simülasyon: fizik ölçüldü + render (hipotez)")
        ax.bar(x, call, bottom=np.array(rst) + np.array(sim), color=Y, edgecolor=YD, label="model çağrıları (hipotez)")
        for xi, tot in zip(x, np.array(rst) + np.array(sim) + np.array(call)):
            ax.text(xi, tot + 8, f"{tot/60:.1f} dk", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel("saniye / BAŞARISIZ episode"); ax.set_title(f"{suite} — max_steps {MAX_STEPS[suite]} (+10 bekleme)", fontsize=10.5)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Ders 27 — Bir episode neye harcanır? S-OV her adımda çağrılır (chunk 1), S-VN 8 adımda bir. "
                 "Fizik 22 ms ve reset 2,5 s bu oturumda ölçüldü; çağrı süreleri TAHMİN (senaryo)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "02-episode-cost.png"), dpi=160); plt.close(fig)
    # 03 — hücre başına T4 saati
    sel = [r for r in res if r["where"] == "T4"]
    fig, ax = plt.subplots(figsize=(14, 5.6))
    x = np.arange(len(sel))
    col = {"pilot": B, "core": Y, "ext": R}
    edge = {"pilot": BD, "core": YD, "ext": RD}
    mid = np.array([r["h_orta"] for r in sel]); lo = np.array([r["h_iyimser"] for r in sel]); hi = np.array([r["h_kötümser"] for r in sel])
    bars = ax.bar(x, mid, color=[col[r["tier"]] for r in sel], edgecolor=[edge[r["tier"]] for r in sel],
                  hatch=None)
    for b_, r in zip(bars, sel):
        if r["status"].startswith("engelli"):
            b_.set_hatch("//")
    ax.errorbar(x, mid, yerr=[mid - lo, hi - mid], fmt="none", ecolor="#444", capsize=3, lw=1)
    for xi, r in zip(x, sel):
        ax.text(xi, hi[list(x).index(xi)] + 2, f"{r['h_orta']:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"{r['cell']}\nRQ-{r['rq']}" for r in sel], fontsize=8)
    ax.axhline(KAGGLE_WEEK_H, color="#555", ls="--", lw=1); ax.text(len(sel) - 0.5, KAGGLE_WEEK_H + 2, "Kaggle ~30 sa/hafta (bildirilen)", ha="right", fontsize=8.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=B, edgecolor=BD, label="pilot"), Patch(facecolor=Y, edgecolor=YD, label="çekirdek"),
                       Patch(facecolor=R, edgecolor=RD, label="genişletme"), Patch(facecolor="white", edgecolor="#444", hatch="//", label="engelli (ön koşul eksik)")],
              fontsize=8.5, loc="upper left")
    core = sum(r["h_orta"] for r in sel if r["tier"] in ("pilot", "core"))
    ln, lt = LEAN
    ext = sum(r["h_orta"] for r in sel if r["tier"] == "ext")
    ax.set_ylabel("T4 saati (çubuk: orta, çizgi: iyimser–kötümser)")
    ax.set_title(f"Ders 27 — Hücre başına T4 bütçesi (model tahmini): pilot+çekirdek ≈ {core:.0f} sa ≈ {core/KAGGLE_WEEK_H:.1f} Kaggle haftası (T4×2 ile ≈ {core/KAGGLE_WEEK_H/2:.1f}); "
                 f"genişletme +{ext:.0f} sa\nyalın matris (görev başına {ln} init, blok n={6*ln}): ≈ {lt:.0f} sa ≈ {lt/KAGGLE_WEEK_H:.1f} hafta (T4×2 ≈ {lt/KAGGLE_WEEK_H/2:.1f})", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(out, "03-cell-hours.png"), dpi=160); plt.close(fig)
    # 04 — tersine güç
    ns = np.array([10, 20, 30, 50, 60, 90, 120, 150, 180, 240, 300])
    m = np.array([100 * mde(int(n)) for n in ns])
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(ns, m, "-o", color=YD, lw=2, ms=5, label="MDE, %80 güç, r = 0,05 (McNemar exact)")
    ax.axvline(50, color=RD, ls="--", lw=1); ax.text(52, 70, "tek görevin tavanı:\nLIBERO'da 50 init state\n(4 görevde sayıldı)", fontsize=8.5, color=RD)
    ax.axvline(300, color=GD, ls="--", lw=1); ax.text(296, 70, "blok havuzu:\n6 görev × 50", fontsize=8.5, color=GD, ha="right")
    ax.axhline(15, color="#555", ls=":", lw=1); ax.text(305, 15.5, "RQ-a eşiği 15", fontsize=8.5, ha="right")
    sec = ax.secondary_xaxis("top", functions=(lambda n: n * sum(t_episode("S-OV", "libero_spatial", c, "orta") for c in ("clean", "tr_a")) / 3600,
                                                  lambda h: h * 3600 / sum(t_episode("S-OV", "libero_spatial", c, "orta") for c in ("clean", "tr_a"))))
    sec.set_xlabel("S-OV 4-bit T4 saati (clean + tr_a, orta senaryo)")
    for n_, mm in zip(ns, m):
        if n_ in (50, 120, 300): ax.annotate(f"n={n_}: {mm:.0f} puan", (n_, mm), textcoords="offset points", xytext=(6, 8), fontsize=8.5)
    ax.set_xlabel("eşli çift sayısı n (görev × init state)"); ax.set_ylabel("yakalanan en küçük fark (puan)")
    ax.set_ylim(0, 85); ax.grid(alpha=0.25); ax.legend(fontsize=9)
    ax.set_title("Ders 27 — Tersine güç: görev başına 50'den fazla init yok → 15 puan ancak blok havuzunda (n≥~100) görünür", fontsize=10.5)
    fig.tight_layout(); fig.savefig(os.path.join(out, "04-inverse-power.png"), dpi=160); plt.close(fig)
    print("yazıldı: 02, 03, 04 →", out)


if __name__ == "__main__":
    main()
```

## plot_day27.py

```python
"""Ders 27 — görsel 06: sim_step_bench.py'nin gerçek ölçümü (results/sim_step.csv)."""
import csv, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = sys.argv[1] if len(sys.argv) > 1 else "../../lectures/images/lecture-27"
rows = list(csv.DictReader(open("results/sim_step.csv", encoding="utf-8")))
tasks = sorted({(r["suite"], int(r["task_id"])) for r in rows}, key=lambda t: (t[0] != "libero_spatial", t[1]))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(12, 5))
w = 0.26
cols = {"0": ("#cfe8ff", "#3b6ea8", "yalnız fizik (render kapalı)"), "128": ("#ffd6d6", "#c0392b", "fizik + 2 kamera 128 px (yazılımsal EGL)"),
        "256": ("#ffd6d6", "#8e2b20", "fizik + 2 kamera 256 px (yazılımsal EGL)")}
for i, res in enumerate(("0", "128", "256")):
    ys = [float(next(r["step_ms_median"] for r in rows if (r["suite"], int(r["task_id"])) == t and r["res"] == res)) for t in tasks]
    fc, ec, lab = cols[res]
    b = ax.bar(np.arange(len(tasks)) + (i - 1) * w, ys, w, color=fc, edgecolor=ec, label=lab, hatch="//" if res == "256" else None)
    for x, y in zip(np.arange(len(tasks)) + (i - 1) * w, ys):
        ax.text(x, y + 3, f"{y:.0f}", ha="center", fontsize=8)
ax.set_xticks(range(len(tasks)))
ax.set_xticklabels([f"{s.replace('libero_', '')} {t}\n(n_init = {next(r['n_init'] for r in rows if r['suite']==s and int(r['task_id'])==t)})" for s, t in tasks])
ax.set_ylabel("ms / env.step (medyan, 60 adım)")
host = rows[0]["host"]
ax.set_title(f"Ders 27 — LIBERO adım süresi, GERÇEK ölçüm ({host}, GPU yok, mujoco 3.3.0, robosuite 1.4.1): "
             "fizik 12–23 ms; render GPU'suz 150–240 ms ekler", fontsize=10)
ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.25)
fig.tight_layout(); os.makedirs(out, exist_ok=True); fig.savefig(os.path.join(out, "06-sim-measured.png"), dpi=160)
print("yazıldı:", os.path.join(out, "06-sim-measured.png"))
```
