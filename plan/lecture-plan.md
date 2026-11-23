# Lecture Plan

Each lecture is ≤ 1 hour. Lectures proceed by sequence number; Stage 0 was completed at Lecture 31 (23 Nov 2026; the daily automation was closed). Goal: by the end of Lecture 30, be able to run a VLA on LIBERO, design a language/spatial robustness experiment, and, when asked "let's do a project", produce a research question + experiment plan. This table covers Stage 0; plans for later stages will be added here.

## Source map (what the advisor sent)
| Source | What is it? | Why was it sent? |
|---|---|---|
| `resources/robot-learning-vla-intro-tr.md` | Turkish concept note (60 headings) | Mental map of the field; the backbone of Phase 1 |
| Tatar et al. 2026 (SIU) — Turkish commands in OpenVLA | 4-page IEEE paper | The only existing example of the Turkish + VLA line; its methodology is open to discussion |
| Li, F. 2025 — Leeds PhD thesis (167 pp.) | Spatial reasoning benchmarks in LLMs: StepGame, RoomSpace | The "language ≠ world" idea; how spatial relations (right/left/front/behind/on/in) are benchmarked |
| Hallac ↔ Cohn e-mails | Idea of evaluating Turkish spatial reasoning + 7 arXiv papers | Likely axis of the thesis: spatial/action understanding in Turkish |
| Slide photo (Pat Grady quote) | "Language is a compression of the world, not the world" | Motivating sentence of the thesis |
| RoomSpace Figure 1 | Top-down + north-facing view, story + question | Benchmark example; relation types that can be carried over to LIBERO |
| `DravenALG/VLANeXt` (ICML 2026) | Code: VLA training + LIBERO / LIBERO-Plus eval | The practical tool of Phases 2–3 |
| `DravenALG/awesome-vla-wam` | Reading list: VLA, WAM, Policy, LAM, benchmarks, hardware | Literature map |
| paperswithcode.co/methods/vla | 2025–26 VLA papers | Current direction: WAM, JEPA, RL-for-VLA |

**Candidate thesis axis (not settled with the advisor):** grounding of Turkish (especially spatial) commands in VLA models — measuring and improving it on LIBERO/LIBERO-Plus.

---

## Phase 1 — Concepts (Lectures 1–8)

| Lecture | Topic | Source | Output |
|---|---|---|---|
| 1 | Source map + the robot learning loop (obs → policy → action) | Note §1–9 | Define each source in one line, in your own words |
| 2 | What is a VLA, VLM–VLA difference, OpenVLA architecture (SigLIP+DINOv2 → projector → LLaMA-2 7B → 256-bin action tokens, 7-DoF) | Note §10–14, Tatar §III.A | Draw the OpenVLA pipeline by hand |
| 3 | Benchmark ≠ simulator: LIBERO (4 suites), LIBERO-Plus, MuJoCo/robosuite | Note §15–17, §22–24 | LIBERO suite table (what it measures, how many tasks) |
| 4 | Language grounding, shortcut learning, OpenVLA-OFT ≠ OpenVLA | Note §18–21, §36, §49–51 | 5 experiment ideas for "does it really use language?" |
| 5 | Deep read of Tatar et al.: experimental setup, L2/cosine, tokenization ("Kal"+"dır"), EBO 72% | Whole paper | 3 strengths, 3 weaknesses of the paper |
| 6 | Foundations of spatial reasoning: qualitative spatial relations (direction, distance, topology), "language is a compression of the world" | Li thesis Ch. 1, §2.2–2.3; slide | 1 paragraph linking the quote to VLAs |
| 7 | RoomSpace benchmark: 3D room, viewpoint, question types, consistency check | Li thesis Ch. 5; Figure 1 | List of 10 Turkish commands, transferable to LIBERO, from RoomSpace relations |
| 8 | Weekly synthesis + awesome-vla-wam survey (VLA / WAM / Policy / LAM; π0, OpenVLA-OFT, VLANeXt, DreamZero) | Repo README | Your own 1-page concept map |

## Phase 2 — Environment and first runs (Lectures 9–16)

**Hardware decision:** MacBook Pro M2 (32 GB, 19-core GPU). No CUDA → two tiers:
- **On the Mac (local):** LIBERO simulation (MuJoCo arm64 works), HDF5 dataset inspection, OpenVLA-7B single-image inference (bf16, MPS; 14 GB RAM; slow but works), tokenizer analysis.
- **On free Colab/Kaggle T4:** LIBERO eval with VLANeXt / OpenVLA-OFT (needs flash-attn + CUDA; does not run on the Mac). With few episodes.
- Setup days come **command by command** with guidance (basic level Python/conda).

| Lecture | Topic | Where | Output |
|---|---|---|---|
| 9 | Miniconda (arm64) + Python 3.10 env; `pip install mujoco`; first MuJoCo scene | Mac | `python -c "import mujoco"` works, one scene rendered |
| 10 | LIBERO setup (with robosuite arm64 notes); env `reset()` / `step()` / save an image | Mac | Frames of 1 episode (png) |
| 11 | Open the LIBERO demo HDF5: obs, 7-DoF action, episode/trajectory; plot the action sequence | Mac | Action plot |
| 12 | Colab T4: VLANeXt setup (TUTORIAL.md), HF checkpoint; `libero_bench_eval` 2–3 episodes | Colab | First success rate number + setup notes |
| 13 | Read the eval code: where does the instruction enter, how is the action decoded, how is success computed | Mac (code reading) | Code flow diagram |
| 14 | Load OpenVLA-7B on the Mac with MPS; single image + English prompt → 7-DoF action | Mac | First action vector |
| 15 | Mini reproduction of the Tatar experiment: 5 commands EN/TR, L2 + cosine | Mac | Mini table |
| 16 | Weekly synthesis; set up LIBERO-Plus on Colab in a separate env (asset download) | Colab | LIBERO-Plus works |

## Phase 3 — Experiment design and robustness (Lectures 17–25)

| Lecture | Topic | Output |
|---|---|---|
| 17 | LIBERO-Plus perturbation types; how the language perturbation is applied in code | Perturbation table |
| 18 | Experiment: same task, instruction = correct / empty / wrong / Turkish → success rate (10 episodes) | 4-row result table |
| 19 | Result analysis: is there shortcut learning? In which suite? | 1 paragraph of interpretation |
| 20 | Design of the Turkish spatial command set (to-the-right/left/front/onto/into; suffix differences) | 20-command paired EN–TR set |
| 21 | **Webots (Cyberbotics)** — the second simulator the advisor sent on 16 Nov: its place in the MuJoCo/robosuite/LIBERO stack (physics: ODE fork; renderer: wren; Python/ROS2 controller), installation on macOS, first scene with UR10e/TIAGo, webots.cloud and robotbenchmark.net; evaluation for the thesis: is it a candidate environment for room-scale RoomSpace-style Turkish spatial tasks and a ROS2 bridge, or do we stay with LIBERO? | Comparison table (MuJoCo vs Webots) + first scene image + decision note |
| 22 | Tokenization analysis: Turkish commands with the VLANeXt/LLaMA tokenizer (token counts, meaningless splits) | Table |
| 23 | Cohn papers (abstract level): 2401.03991 AAAI-24, 2405.15064 IJCAI-24, 2406.16528 COSIT-24, 2410.03492, 2507.12059 QR-25, 2511.02627 TMLR, 2605.18380 | 2 sentences for each |
| 24 | Solution approaches: translation layer vs multilingual VLM backbone vs Turkish LoRA fine-tune — pros/cons/cost | Comparison table |
| 25 | Weekly synthesis | Phase 3 summary |

## Phase 4 — Reasoning about the project (Lectures 26–31)

| Lecture | Topic | Output |
|---|---|---|
| 26 | Formulate 3 candidate research questions (RQ) | RQ list |
| 27 | For each RQ: data, model, benchmark, metric, compute cost | Experiment matrix |
| 28 | Baseline + ablation design | Ablation plan |
| 29 | 1-page thesis proposal draft | Draft (English; Turkish title) |
| 30 | Question/presentation list for the advisor meeting | Question list |
| 31 | Final: "let's do a project" simulation — Claude proposes a project, you design it | Evaluation + plan for the next stage |

---

## Open decisions (to be recorded here once settled)
- GPU: MacBook Pro M2 32 GB (no CUDA) + free Colab/Kaggle T4 (decision: 5 Nov)
- Thesis topic: not settled yet, sources are exploratory → RQs will be produced together in Phase 4 (5 Nov)
- Python/conda: basic level → setup days are guided command by command (5 Nov)
- Daily working hours: ?
