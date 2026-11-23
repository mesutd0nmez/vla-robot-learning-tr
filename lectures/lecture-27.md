# Lecture 27 — Experiment Matrix: Data, Model, Benchmark, Metric, Cost for Each RQ — 19 Cells, the Episode-Time Model, the T4 Budget and Inverse Power

**Phase 4 / Day 27** · **Time:** ~55 min (reading 40 + task 15)
**Prerequisites:** Lecture 26 (RQ-a/b/c, `rq_power.py`), Lectures 12 and 18 (T4 notebook, chunk 8, `num_parallel_envs`), Lecture 14 (OpenVLA single step, Mac MPS), Lecture 24 (QLoRA budget, parser 79/80) · **Code:** `code/day27/` (`sim_step_bench.py` real LIBERO measurement, `budget.py` budget + inverse power, `plot_day27.py` — all three ran in this session) · **Output:** `thesis/experiment-matrix.csv` (19 cells, v0.1; the v0.1 snapshot is kept as `code/day27/experiment-matrix_v0.1.csv`)

> **Today in one sentence:** Unrolling the three RQs into a 19-cell matrix (system × suite × block × condition × init × repeats) gives pilot + core ≈ 8,400 episodes run and ≈ 258 T4 hours in the middle scenario (optimistic 151, pessimistic 504) — 63% of that is S-OV, because OpenVLA is called once at every step (chunk 1) and only 4-bit fits on the T4; but inverse power changes where the budget goes: LIBERO has only 50 init states per task (n = 50 → smallest visible difference 23 points), 15 points is visible only in the block pool, and in the pool n = 120 (20 init per task) already catches 13 points — **if you move the claim unit from the task to the block**, the same matrix drops to ≈ 108 hours, that is ~2 Kaggle weeks with T4×2.

---

## 1. What is the matrix, what is a cell?

Lecture 26's six fields locked "what is being asked"; today is "with what, how many times, for how much". The plan table wants five columns — data, model, benchmark, metric, compute cost — but writing these as one row per RQ does not work: within RQ-a there are two systems, two suites, four conditions, and each has a different cost. The right unit is the **cell**: *one system, one suite, one block, one condition set, n init states per task, one repeat count*. A cell is the smallest runnable work package; like a job in CI.

![Matrix flow](images/lecture-27/01-matrix-flow.png)

`thesis/experiment-matrix.csv` (v0.1; the snapshot used by this lecture's code is `code/day27/experiment-matrix_v0.1.csv`) contains 19 cells (columns: `cell_id, rq, tier, purpose, system, suite, block, n_tasks, conditions, n_init, repeats, data_dep, model_dep, metric, status`). Summary:

| group | cells | what it measures | data | model | metric |
|---|---|---|---|---|---|
| **Calibration** | K1, K2, K3 | do the systems hit the published numbers; how much does quantization shift | LIBERO original commands | S-OV 4-bit (T4), S-VN, S-OV bf16 (Mac) | Wilson; in K3 paired Δ(4-bit − bf16) |
| **Pilot** | P1, P2 | the Lecture 18 run + repeats; S-OV first run | `datasets/03` | S-VN, S-OV 4-bit | Wilson + PI (3 repeats) |
| **RQ-a** | A1–A4 | four conditions × two systems × Spatial A / Goal C | `datasets/04` A, C | two systems, Spatial and Goal checkpoints | McNemar + TOST, **in the block pool** |
| **RQ-b** | B1–B5 core, B6 extension | `tr_ascii` opposite-direction test; block B EN-B / TR-B; B's `empty` | `datasets/04` A+C, B + `bddl_cf/` | two systems | paired Δ; tr_B − empty |
| **RQ-c** | C1, C2 core; C3, C4 extension | MT and parser closure; QLoRA training + eval | MT/parser outputs; EN+TR RLDS | two systems; S-OV-LoRA | κ, H0-c3, H0-c4 (δ = 5) |

I propose one change from Lecture 26 (the `repeats` column in the file): RQ-a said "≥3 repeats"; the matrix keeps repeats **only in the pilot** (P1), 1 in the core. The justification is in §5 — a repeat does not take the place of an independent init state. This is a pre-registration change; it enters `thesis/rq-candidates.md` v0.2 with your approval.

---

## 2. What is an episode spent on? The measured half and the estimated half

Budget = Σ cells (number of episodes × episode duration). Splitting the episode duration into its parts makes clear what we can measure and what we estimate:

`T_ep = t_reset + (W + L) · t_sim + ⌈L / chunk⌉ · t_call`

- **W = 10** wait steps (`num_steps_wait`; the model is not called) and **L** = the length of the episode if successful, `max_steps` if not. `max_steps` is real code: Spatial 220, Goal 300 (OpenVLA `c8f03f4`, `run_libero_eval.py:173-180`; the comment line "longest training demo has 193/270 steps"). The successful length is ~100 for Spatial (the ~96 Lecture 18 took from Lecture 11), 120 for Goal — **hypothesis**.
- **chunk**: S-VN 8 (Lecture 12, `num_steps_execute: 8`), S-OV **1** — OpenVLA produces one 7-DoF action at every step; the paper mentions action chunking as "future work" (§6). In a failed Spatial episode S-VN is called 28 times, S-OV 220 times. This is the real asymmetry of the budget.
- **t_sim** and **t_reset**: I **really measured** these in this session (`sim_step_bench.py`; LIBERO repo, robosuite 1.4.1, mujoco 3.3.0, 2-core Linux, no GPU):

![Real measurement](images/lecture-27/06-sim-measured.png)

With rendering off, one `env.step` is 19–23 ms in physics on Spatial, 12–13 ms on Goal; reset 0.9–1.5 s (with rendering 2.1–2.7 s). On all four of four tasks `get_task_init_states` returned exactly **50** init states — this number determines the whole calculation in §5. Rendering, on the other hand, adds 150–240 ms per step on this machine, because there is no GPU and EGL runs in software. On Colab/Kaggle EGL should run in hardware on the T4 (two small cameras, ms scale — **hypothesis**); if `MUJOCO_GL` is set wrong and rendering falls to software, 30–60 s is added to every episode. That is why this is the first row in the If you get stuck table.

**t_call** is the half we cannot measure and it is more than 90% of the budget. My scenarios were built on these grounds:

| system / place | optimistic | middle | pessimistic | basis |
|---|---|---|---|---|
| S-VN, T4 | 1.0 s | 2.5 s | 5.0 s | the upper end of Lecture 18's "successful episode ~1 min / 12 calls" estimate; bf16 emulation on the T4 (Lecture 12) |
| S-OV 4-bit, T4 | 0.5 s | 0.8 s | 1.6 s | OpenVLA §5.4: int4 **3 Hz** on an A5000; T4 memory bandwidth 320 GB/s, A5000 768 GB/s → 3 × 320/768 ≈ 1.25 Hz ≈ 0.8 s (assuming 7B generation is bandwidth-bound) |
| S-OV bf16, Mac M2 | 1.5 s | 2.0 s | 3.0 s | Lecture 14 estimate (1.5–3 s/action) |

For comparison, the paper's single bf16 speed sentence: "approximately 6Hz" on an RTX 4090 (§3.5). And VLAQuantBench (Xu et al., Sep 2026) measured OpenVLA-OFT on an RTX 4090 with real kernels: full inference bf16 139.9 ms, NF4 152.6 ms, LLM.int8() 379.5 ms (Table 10) — that is, 4-bit cuts memory by more than half but does **not speed up**, and 8-bit slows by 2.7×. That is why we never consider 8-bit.

![Episode cost](images/lecture-27/02-episode-cost.png)

Figure 02 shows the failed (most expensive) episode: S-VN on the T4 in the middle scenario 1.3 min, S-OV 4-bit on the T4 3.1 min, S-OV bf16 on the Mac 7.5 min (Spatial). Yellow (model call) dominates every bar; the blue part we measured is a thin strip. So the matrix's first job is to **measure t_call**: the K1 and K2 calibration cells are also clock measurements; with `budget.py --calls-sv X --calls-ov Y` the measured values are substituted and the budget is recomputed.

---

## 3. How does S-OV fit on the T4 — and what does fitting it change?

OpenVLA with bf16 weights wants 16.8 GB (paper Table 2; §3.5 says "15GB" — the two numbers differ within the paper itself). The T4's usable memory is ~15 GB; once the image + KV cache are added, bf16 does not fit. Options: 8-bit (10.2 GB; 1.2 Hz on an A5000, and in Table 2 BridgeData success 71.3 → **58.1**), 4-bit (7.0 GB; 3 Hz; success **71.9** — the same as bf16). The official LIBERO eval script supports 4-bit with a flag (`--load_in_4bit`, `openvla_utils.py:48`), so no code needs to be written. The choice is clear: **S-OV on the T4 = 4-bit.**

![Where to run](images/lecture-27/05-where-to-run.png)

But "4-bit S-OV" is not the same system as the published one. Table 2's equality is on BridgeData, 8 tasks, 80 rollouts; it was not measured on the LIBERO checkpoint. VLAQuantBench's OpenVLA-OFT row is a warning: simple 4-bit rounding (RTN W4) goes 100 → 96 on Spatial but **96 → 82 on Goal** (Table 1). That OFT is not our OpenVLA, and RTN is not our NF4 (the same paper's NF4 rows are ~98 on Spatial when only the LLM part is quantized; I could not read the relevant appendix table directly, **not verified**). Still, the lesson is clear: quantization may be harmless on one suite and a 14-point shift on another.

This has two consequences for the experiment design:

1. **The paired difference stays valid.** RQ-a's questions are differences between conditions (`tr_a − clean`); if all conditions run on the same 4-bit model, quantization is the same on both sides. What changes is the subject of the interpretation: in the thesis one writes not "OpenVLA" but "**OpenVLA (4-bit NF4)**".
2. **Two calibration cells are mandatory.** K1: 4-bit S-OV, `clean`, 10 Spatial tasks × 10 init (~3 T4 hours) — is it within the Wilson interval of the published 84.7 ± 0.9? K3: bf16 S-OV on the Mac with the **same** 25 init, `clean` and `tr_a` — a paired comparison with the 4-bit one directly measures the quantization effect and closes the objection "is the Turkish drop from quantization". K3 is ~4 hours (middle) on the Mac = one night; it spends no quota. This is the Mac's real role in the matrix: a slow but bf16 and free **reference**.

---

## 4. Total: 19 cells, 258 hours — where does it go?

`budget.py` reads the matrix, breaks each cell (system, suite, condition) into its parts and mixes the expected episode duration with the condition's expected success (a successful episode is short: `clean` 0.8, `empty` 0.3, `tr_a` 0.5 … — these are **not result predictions**, only for the cost mix; moving them ±0.2 changes the total by ~10%).

![T4 hours per cell](images/lecture-27/03-cell-hours.png)

| | episodes (run) | T4 hours optimistic / **middle** / pessimistic |
|---|---|---|
| Calibration K1–K2 (+ K3 Mac 4 h) | 200 | 2 / **4** / 8 |
| RQ-a (P1, P2, A1–A4) | 4,960 | 91 / **156** / 305 |
| RQ-b core (B1–B5) | 2,500 | 46 / **78** / 153 |
| RQ-c core (C1, C2) | 735 | 11 / **19** / 38 |
| **Pilot + core** | **≈ 8,400** | **151 / 258 / 504** |
| Extension (B6, C3 training, C4) | 4,200 + training | 154 / **225** / 382 |

Three readings:

- **S-OV cells are 163 of the 258 hours (63%).** A3 + A4 alone are 106 hours. The chunk difference makes the cost of the system comparison asymmetric: one S-VN episode is ~1 min, one S-OV episode ~2–3 min.
- **≈ 8.6 weeks on Kaggle, ≈ 4.3 with T4×2.** Kaggle's weekly GPU quota is not written as a number in the official documentation, it appears in the account settings; secondary sources say "~30 hours" (one notes separately that this is a user report). A session is limited to 12 hours. Kaggle gives T4×2: two separate eval processes on two GPUs (`CUDA_VISIBLE_DEVICES=0/1`, one cell each) should halve the wall-clock time — **hypothesis**; RAM (Lecture 18: ~29 GB) must be enough for two processes. Whether quota hours burn the same on both or are counted per GPU — check in your account.
- **C2 is almost free: 15 of 1,200 episodes run.** If the parser output is identical to the `clean` sentence, that episode is the `clean` episode itself (S-OV greedy: `do_sample=False`, `openvla_utils.py:169` → same input, same action; in S-VN the distribution is the same). In Lecture 24 the parser translated 79 of the 80 in-set rows exactly → what will run is 1/80. For MT the same cache assumes 60% runs (**hypothesis**; with no HF access the MT output still does not exist → C1 is blocked). In software terms: memoization by input string.

---

## 5. Inverse power: budget → n → smallest visible difference

Lecture 26's table said "with n = 100, 15 points are visible". Today's real measurement added a constraint: **there are no more than 50 init states in a LIBERO task.** You cannot raise n per task above 50; repeating (re-running the same init) is no remedy either — in S-OV the greedy solution gives the same result at the same init (apart from GPU nondeterminism), and in S-VN repeats do not count as independent pairs because they depend on the same init. So on a single task the smallest difference visible at 80% power stays at ~23 points, however much budget you put in.

![Inverse power](images/lecture-27/04-inverse-power.png)

The way out is to change the claim unit: RQ-a's question is already at the block level ("is there a Turkish drop on Spatial A"). A block consists of six tasks; pooling the paired pairs across tasks gives n = 6 × n_init. The pooled McNemar is a valid test under the null "there is no difference in any task" — in every task the discordant pairs are symmetric — but the difference it finds is **the block average**; for an individual task it stays descriptive. `budget.py`'s inverse table (S-OV, Spatial, `clean` + `tr_a`, middle scenario):

| T4 hours | init per task | pool n | MDE task | MDE block |
|---|---|---|---|---|
| 5 | 12 | 72 | 68 | 18 |
| 10 | 24 | 144 | 41 | 12 |
| 20 | 49 | 294 | 24 | 7 |
| 30+ | 50 (ceiling) | 300 | 23 | 7 |

`budget.py`'s power function is the same model as `rq_power.py` but vectorized (b + c ~ Binom, b | m ~ Binom); the selftest gives Lecture 26's three values exactly (23 / 15 / 16), n = 300 takes seconds. Reading: for a 15-point threshold at the block level n ≈ 100 suffices (MDE 15), with n = 120 it is 13. The **lean matrix** uses 20 init per task in the core cells: RQ-a 64, RQ-b 31, RQ-c 8, calibration 4 → **≈ 108 T4 hours, ≈ 3.6 Kaggle weeks, ≈ 1.8 with T4×2** (optimistic 63, pessimistic 211).

Two warnings. (i) The pool calculation assumes the tasks have the same reverse noise r and the same Δ; if Δ varies a lot across tasks (one −40, one 0) the pooled average says 20 but does not say "a drop in every task" — that is why the per-task Δs are still in the table, reported in "direction" mode. (ii) The same logic holds for TOST: δ = 15 equivalence was established at n ≈ 25, δ = 10 at n ≈ 55; with 20 init per task these are possible only at the block. The lean matrix moves the "no difference" claim to the block as well. This is a decision about what the thesis's claim unit will be and **will be discussed with the advisor** (Lecture 30 list): "average over these six tasks of LIBERO Spatial", or "in every task"?

Link to the thesis: the claim of Turkish spatial grounding can never move a drop smaller than 23 points on a single task beyond "unmeasurable"; Dong et al.'s 35–50 point drops on Spatial are visible at the task level too, while a difference like Chen's 7-point Qwen3-VL Chinese cell is visible only in the block pool and at n ≥ ~250. The question "which of S-VN and S-OV drops less" lives with the same limit.

---

## 6. What does not fit the budget and what is blocked — the red and hatched cells of the matrix

In figure 03 the hatched bars are cells with a missing precondition, the red ones are extensions:

| cell | why it waits | what unlocks it |
|---|---|---|
| **B3, B4** (block B, 52 h middle) | 5 tier-2 pairs need a counterfactual BDDL (`datasets/04/data/bddl_cf/`); the files are ready, but loading them in a real env and testing that `check_success` counts `bowl_2` has not been done | one `env.reset()` + one `check_success` try (Mac, 5 min) |
| **C1** (MT, 19 h) | no MT output (no HF access in this environment) | `mt_layer_mac.py` (Lecture 24) once on the Mac → outputs to CSV |
| **C3** (QLoRA training, 65–98 h) | S9: advisor decision; Lecture 24's budget is an estimate | advisor + K1 (does 4-bit S-OV work on LIBERO) |
| **C4** (LoRA eval, 67 h) | depends on C3 | C3 |
| **B6** (other surfaces `tr_b`, `tr_bare`; 78 h) | fits but is worth it only if RQ-b is primary | the S1 pattern (Lecture 26 §6) |
| **P1** (Lecture 18 pilot) | S1 is still not in the repo (`projects/05-spatial0-4kosul/` does not exist, `plan/faz3-open-questions.md` Status is empty) | the Lecture 18 notebook |

There is also something that is not a blocker but determines the order: everything comes after **S1 and K1/K2**. S1 gives the pattern (hence which RQ is primary), K1/K2 give the real t_call (hence the budget itself). The first week of the matrix is these three small cells: P1 (~2 h), K1 (~3 h), K2 (~1 h). Then the budget is recomputed with `budget.py --calls-sv … --calls-ov …`, the core of the primary RQ is run, the others in the lean version.

---

## 7. Bridges for the software developer

- **Cell = CI matrix job.** Like `strategy.matrix: {os: [...], dotnet: [...]}` in GitHub Actions: the RQ × system × suite × condition expansion, each combination a separate job. `experiment-matrix.csv` is that YAML itself; `budget.py` is the "how many runner-minutes does this matrix burn" calculation. You drop unnecessary combinations with `exclude` — B6 and C4 being red is this.
- **The T_ep formula = capacity planning.** Request time = fixed cost (reset) + per-item cost (steps × t_sim) + downstream call (calls × t_call). Before opening the profiler, know which term dominates: here the downstream (model) is 90%+; optimizing physics is wasted effort.
- **chunk = batching / the N+1 problem.** S-OV goes to the model at every step (one query per row in an ORM), S-VN takes 8 steps in a single call (batch loading with `Include`). The same work, 8 times fewer round trips.
- **4-bit = a different build configuration.** Release and Debug come from the same source but you do not do the performance test on Debug and report it for Release; K1/K3 is a smoke test of "is this configuration the same as the published behavior", and in the thesis the label is "OpenVLA (4-bit)".
- **String cache = memoization for idempotent requests.** Same input + deterministic service → do not recompute the answer. C2 dropping from 1,200 episodes to 15 is this; the cache key is command string + init state + checkpoint hash.
- **The 50-init ceiling = an A/B test with a sample ceiling.** If a tenant has only 50 users, you can never see a small effect for that tenant; you pool the tenants and report the result as "average across all tenants". A repeat measurement does not create new users from the same user.

---

## 8. Today's task (15 min)

Open `thesis/experiment-matrix.csv` and decide the **claim unit**: is RQ-a's main claim at the task level or at the block level? According to the decision change the `n_init` value of the core cells (block → 20, task → 50) and run `cd code/day27 && python budget.py --matrix experiment-matrix_v0.1.csv --lean 20` (or without `--matrix` for the current file); write the resulting "pilot + core" middle total and the number of Kaggle weeks as one line at the bottom of the file. Then check two things from your own account: how many hours your weekly GPU quota under Kaggle → Settings shows, and measure the physics + 128 px render time on the Mac with `python sim_step_bench.py --libero <LIBERO clone> --tasks libero_spatial:0 --steps 30` (on the Mac rendering is in hardware; compare the number with the 208 ms in this lecture). Commit: `Lecture 27: claim unit + quota`. If you have time to run K1 (Kaggle, ~3 hours), substitute the first call time with `--calls-ov` — this will be the matrix's first real cost number.

## If you get stuck

| Symptom | Fix |
|---|---|
| On Colab/Kaggle the episode is 3–5× slower than expected | Rendering may have fallen to software: is `MUJOCO_GL=egl` set, also set `PYOPENGL_PLATFORM=egl`; look at the 128 px step time with `sim_step_bench.py --tasks libero_spatial:0 --steps 20` (software ~200 ms, hardware should be much less) |
| `sim_step_bench.py`: `AssertionError` (`get_joint_qpos_addr`) | mujoco is too new (4.x) — incompatible with robosuite 1.4.1. On the Mac pin `mujoco==2.3.7`; on Linux with Python 3.12+ 3.3.0 worked in this session |
| `OSError: libmujoco.so... cannot open` | Leftover of an old version: `pip uninstall mujoco`, delete the `site-packages/mujoco` folder, reinstall |
| `No module named 'gym'` | LIBERO `envs/venv.py` imports old `gym`: `pip install gym` |
| `torch.load` → `Weights only load failed` | torch ≥ 2.6; the script contains the `weights_only=False` patch, do the same in your own code (the init files come from a trusted source: the LIBERO repo) |
| LIBERO interactively asks for the dataset path | The script redirects `LIBERO_CONFIG_PATH` to `results/.libero_cfg` and writes `config.yaml`; use the same environment variable in your own code |
| `budget.py`: `KeyError: 'goal'` | In the `suite` column the two suites must be written in full as `libero_spatial+libero_goal` |
| OpenVLA 4-bit `bitsandbytes` error on the T4 | `pip install bitsandbytes` (in a CUDA runtime); there is no 4-bit on the Mac — on the Mac only bf16 (K3) |

---

## Next lecture (Lecture 28)
**Baseline + ablation design.** Today's matrix said *what will be run*; Lecture 28 writes *what it will be compared with*. For each RQ the baselines (command-free ceiling, `empty`, `wrong`, English `clean`, LangGap's EN-B external reference) and the ablations: for RQ-b a surface ablation (`tr_a → tr_ascii → tr_bare`: letters, suffixes, both), for RQ-c the "oracle" versions of MT and the parser (human translation = MT's ceiling; correct template = the parser's ceiling), the quantization ablation (K3: bf16 vs 4-bit) and that decomposing the system difference is not possible on our budget (Chen's StarVLA setup) — when we say "ablation" and when we say "descriptive comparison". Every ablation is added to the matrix as a new cell or a re-reading of an existing cell and priced with `budget.py`.

---

## References
- Output: `thesis/experiment-matrix.csv` (19 cells, v0.1). Inputs: `thesis/rq-candidates.md` (Lecture 26), `datasets/04/data/pairs.csv` (block A 6 / B 6 / C 6), Lecture 12 (`t4_bench_config.yaml`: chunk 8, `num_parallel_envs`), Lecture 14 (Mac MPS estimate), Lecture 18 (episode duration estimate, ~96 steps), Lecture 24 (`qlora_budget.py` 65–98 h, parser 79/80).
- **[1]** Kim et al. (2024). *OpenVLA: An Open-Source Vision-Language-Action Model.* [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) — §3.5 (bf16 ~15 GB; ~6 Hz on an RTX 4090), §5.4 + Table 2 (bf16 71.3% / 16.8 GB; int8 58.1% / 10.2 GB / A5000 1.2 Hz; int4 71.9% / 7.0 GB / A5000 3 Hz; BridgeData 8 tasks), §6 (action chunking is future work). Code: [openvla/openvla](https://github.com/openvla/openvla) `c8f03f4` — `experiments/robot/libero/run_libero_eval.py:72-73,156,173-180` (`num_steps_wait` 10, 50 trials, 256 px, max_steps 220/280/300/520), `experiments/robot/openvla_utils.py:48,169` (`load_in_4bit`, `do_sample=False`).
- **[2]** Xu, Jin, Chen, Wang, Sui, Shi (2026). *VLAQuantBench: Closed-Loop Evaluation of Post-Training Quantization for Vision-Language-Action Models.* [arXiv:2609.25376](https://arxiv.org/abs/2609.25376) · [code](https://github.com/jiuyixu25/VLAQuantBench) — Table 1 (OpenVLA-OFT RTN W4: Spatial 100 → 96, Goal 96 → 82; W8 close to baseline), Table 10 (RTX 4090, OFT: bf16 139.9 ms / 15.0 GiB, NF4 152.6 ms / 6.3 GiB, LLM.int8() 379.5 ms / 9.0 GiB). The accuracy experiments are simulated quantization (the paper's own note). The NF4 LIBERO rows (appendix table) could not be read directly.
- **[3]** Kaggle quota and session limit (secondary sources; I did not read the Kaggle documentation directly): [gpuperhour.com, Sep 2026](https://gpuperhour.com/blog/free-cloud-gpus-and-credits) (quote from the Kaggle documentation: 12 h session; "~30 hours" is a user report, no number in the documentation), [awesome-free-compute-2026](https://amitpandeytiktok.github.io/awesome-free-compute-2026/compute.html) (30 GPU h/week, T4×2). Colab free-tier limits are not published ([Colab FAQ](https://research.google.com/colaboratory/faq.html)).
- **[4]** Memory bandwidths (manufacturer documents): [NVIDIA T4](https://www.nvidia.com/en-us/data-center/tesla-t4/) 320 GB/s, [RTX A5000](https://www.nvidia.com/en-us/design-visualization/rtx-a5000/) 768 GB/s. The scaling "7B generation is bandwidth-bound" is this lecture's assumption (hypothesis).
- **[5]** LIBERO: Liu et al. (2023), [arXiv:2306.03310](https://arxiv.org/abs/2306.03310) · [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) — `get_task_init_states` (50 counted on 4 tasks). Power and TOST: Lecture 26 sources (McNemar exact; Lakens 2017). Dong et al. [2606.11906](https://arxiv.org/abs/2606.11906), Chen et al. [2606.15714](https://arxiv.org/abs/2606.15714) (the magnitude comparison in §5).
- Code: `code/day27/` — `sim_step_bench.py` (ran in this session: Linux, 2 cores, no GPU, Python 3.13, mujoco 3.3.0, robosuite 1.4.1, torch 2.15; `results/sim_step.csv`), `budget.py` (ran; selftest OK; `results/budget.csv`, `budget_lean.csv`, `inverse.csv`, `budget_stdout.txt`), `plot_day27.py` (ran). Note: the `code/day27/results/*` files and figures 02–04 were produced for the **v0.1 matrix** (`python budget.py --matrix experiment-matrix_v0.1.csv`); the current `thesis/experiment-matrix.csv` is v0.2 (22 cells, AB1–AB3 added in Lecture 28), and its v0.2 numbers are in Lecture 28 (`code/day28/results/budget_v01_v02.csv`).
- Figures: 01 and 05 are Mermaid (`.mmd` alongside); 02 and 03 are the `budget.py` **model estimate** (reset and physics measured; t_call, rendering and successful episode length are scenarios — stated in the titles); 04 exact power calculation + middle-scenario hour axis; 06 **real measurement** (`sim_step.csv`). The model was not run in this session (no GPU); the systems' t_call values will be substituted with Mesut's K1/K2 run.
