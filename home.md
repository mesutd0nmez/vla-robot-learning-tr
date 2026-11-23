<div class="home-hero">

# Robot Learning and VLA — thesis journal

An open record of my Computer Engineering MSc thesis: Vision-Language-Action models, the LIBERO benchmark, and connecting Turkish spatial commands to robot behavior. One hour a day, one topic per lecture; the notes are in English (Turkish only for the research material) and open for everyone to use.

</div>

## Lectures

<div class="home-grid">
<a href="#/lectures/lecture-01"><b>01</b><span>The robot learning loop and the source map<small>Phase 1 · Concepts</small></span></a>
<a href="#/lectures/lecture-02"><b>02</b><span>Inside OpenVLA: image + language → action<small>Phase 1 · Concepts</small></span></a>
<a href="#/lectures/lecture-03"><b>03</b><span>Benchmark ≠ simulator: LIBERO, MuJoCo, LIBERO-Plus<small>Phase 1 · Concepts</small></span></a>
<a href="#/lectures/lecture-04"><b>04</b><span>Language grounding and shortcut learning<small>Phase 1 · Concepts</small></span></a>
<a href="#/lectures/lecture-05"><b>05</b><span>Deep paper read: Tatar et al. (2026)<small>Phase 1 · Concepts</small></span></a>
<a href="#/lectures/lecture-06"><b>06</b><span>Foundations of spatial reasoning<small>Phase 1 · Concepts</small></span></a>
<a href="#/lectures/lecture-07"><b>07</b><span>RoomSpace → LIBERO: a Turkish command set<small>Phase 1 · Concepts</small></span></a>
<a href="#/lectures/lecture-08"><b>08</b><span>Week 1 synthesis and VLA model families<small>Phase 1 · Concepts</small></span></a>
<a href="#/lectures/lecture-09"><b>09</b><span>conda, MuJoCo and the first scene on the Mac<small>Phase 2 · Hands-on</small></span></a>
<a href="#/lectures/lecture-10"><b>10</b><span>LIBERO setup and the first episode<small>Phase 2 · Hands-on</small></span></a>
<a href="#/lectures/lecture-11"><b>11</b><span>LIBERO demo data: inside the HDF5<small>Phase 2 · Hands-on</small></span></a>
<a href="#/lectures/lecture-12"><b>12</b><span>VLANeXt on a Colab T4: first success rate<small>Phase 2 · Hands-on</small></span></a>
<a href="#/lectures/lecture-13"><b>13</b><span>Code reading: command → action → success<small>Phase 2 · Code reading</small></span></a>
<a href="#/lectures/lecture-14"><b>14</b><span>OpenVLA-7B on the Mac: image + command → 7 tokens → action<small>Phase 2 · Hands-on</small></span></a>
<a href="#/lectures/lecture-15"><b>15</b><span>Mini reproduction of Tatar et al.: 5 frames × 6 conditions, L2 + cosine<small>Phase 2 · Hands-on</small></span></a>
<a href="#/lectures/lecture-16"><b>16</b><span>Phase 2 synthesis + LIBERO-Plus: separate runtime, the command's code path, empty command<small>Phase 2 · Synthesis + Hands-on</small></span></a>
<a href="#/lectures/lecture-17"><b>17</b><span>LIBERO-Plus's seven perturbations in code: three carriers, camera, 500 Pandas, R1/R2, Turkish variant<small>Phase 3 · Code reading + Hands-on</small></span></a>
<a href="#/lectures/lecture-18"><b>18</b><span>Same task, four commands: clean / empty / wrong / tr × 10 init states — experiment design, patch, Wilson and power<small>Phase 3 · Experiment + Hands-on</small></span></a>
<a href="#/lectures/lecture-19"><b>19</b><span>Result analysis: is there a shortcut, in which suite? — reading protocol, structural upper bound from BDDL (Spatial 9/10, Goal 1/10), geometry of the 50 init states<small>Phase 3 · Analysis + Hands-on</small></span></a>
<a href="#/lectures/lecture-20"><b>20</b><span>Turkish spatial command set: 20 EN–TR pairs — relation name + three suffixes, "soldaki" relative to whom (three frames), reference from 50 init states, counterfactual BDDL<small>Phase 3 · Design + Data</small></span></a>
<a href="#/lectures/lecture-21"><b>21</b><span>Webots: the advisor's second simulator — its place in the stack (a simulator, not a benchmark), Mac setup + conda vla, first scene with the UR10e, decision note for the thesis<small>Phase 3 · Tool + Decision</small></span></a>
<a href="#/lectures/lecture-22"><b>22</b><span>Tokenization analysis: the same 100 commands, two tokenizers — LLaMA-2 1.6× / Qwen3 1.4× inflation, unstable relation suffix, "soldaki" = sold+aki, sentence-start trap, clean action range<small>Phase 3 · Analysis + Data</small></span></a>
<a href="#/lectures/lecture-23"><b>23</b><span>The Cohn line: seven papers at abstract level — StepGame correction, RoomSpace, cardinal direction (2024→2025), score uncertainty (PI), DecompSR, QSTRBench; collapse with depth (k=1 in LIBERO), drop when the surface changes, reporting a single score<small>Phase 3 · Literature + Methodology</small></span></a>
<a href="#/lectures/lecture-24"><b>24</b><span>Solution approaches: MT translation / parser + symbolic / multilingual backbone / Turkish LoRA / hidden-state alignment — pros and cons with Dong & Chen (Jun 2026) numbers, Mac–T4 budget, decision tree by the Lecture 18 pattern; the parser actually ran (in-set 79/80, held-out 5/12)<small>Phase 3 · Decision + Hands-on</small></span></a>
<a href="#/lectures/lecture-25"><b>25</b><span>Phase 3 synthesis: eight lectures on one page — balance 21 measurements = 11 real / 6 in Mesut's run / 4 hypotheses; phase3_smoke.py 28 OK of 30 scripts (Lecture 22 tokens.csv reproduced exactly); LangGap (same scene, Spatial 97 → 5.9); open questions S1–S10 → three RQ seeds<small>Phase 3 · Synthesis</small></span></a>
<a href="#/lectures/lecture-26"><b>26</b><span>Three candidate research questions: from seed to testable question — 6 fields (null + δ, variables, numeric threshold, suite+block+n, systems, yes/no); rq_power.py: n=50 only catches ≥23 points, TOST δ=15 → n≈25; RQ-a/b/c, conditional ordering by S1; thesis/rq-candidates.md<small>Phase 4 · RQ + Preregistration</small></span></a>
<a href="#/lectures/lecture-27"><b>27</b><span>Experiment matrix: 19 cells (data/model/benchmark/metric/cost); episode time model — LIBERO physics 12–23 ms measured, S-OV chunk 1 → 63% of the budget; pilot+core ≈ 258 T4 h; 50-init ceiling per task → once the claim moves to the block, the lean matrix ≈ 108 h<small>Phase 4 · Experiment matrix</small></span></a>
<a href="#/lectures/lecture-28"><b>28</b><span>Baseline + ablation: the surface ladder is actually a 2×2 (letter × suffix) — the suffix step also deletes 66 of the 97 Turkish letters; tr_bare_ascii produced, LLaMA-2 letter +2.7 / suffix +5.4 / interaction +4.3 tokens; oracles = the clean cell; S-OV vs S-VN descriptive; matrix v0.2 ≈ 303 h (lean 126)<small>Phase 4 · Ablation plan</small></span></a>
<a href="#/lectures/lecture-29"><b>29</b><span>Thesis proposal draft: on one page problem → gap → aim → RQ-a/b/c → method → 3 contributions; Heilmeier questions; schedule in T4 hours (lean ≈ 108 h, 16-week Gantt), 7 risks + plan B; proposal_lint.py 680/700 words, 12/12 numeric claims labeled; thesis/proposal-v0.1.md<small>Phase 4 · Thesis proposal</small></span></a>
<a href="#/lectures/lecture-30"><b>30</b><span>Advisor meeting: open decisions in the repo scanned (184 lines / 37 files) and reduced to 11 questions; impact × urgency → 17 of the 30 min go to five decisions (suite, claim unit, weak/strong reading, LoRA/GPU, schedule); decision tree, 7 slides, ADR decision log; slides/advisor-meeting-01.md<small>Phase 4 · Advisor meeting</small></span></a>
<a href="#/lectures/lecture-31"><b>31</b><span>Final: the "let's do a project" simulation — Claude makes three proposals (four conditions / tr_ascii opposite direction / parser-MT κ), you cast them into the eight-field template; design_lint.py checks n↔MDE consistency; Stage 0 balance (repo scan) and carried-over items — S1 first real table<small>Phase 4 · Final</small></span></a>
</div>

## VLA and Multilingual VLA Literature

<div class="home-grid">
<a href="#/literature/makaleler"><b>01</b><span>Paper list: 15 peer-reviewed works and 2 preprints<small>Literature · List</small></span></a>
<a href="#/literature/rapor"><b>02</b><span>Report: the state of VLA and multilingual VLA<small>Literature · 2 pages</small></span></a>
</div>

## How I work

Every lecture has the same skeleton: 35 minutes of reading (with figures) and a 15-minute task. The plan and daily progress are in the `Plan` section; setup and experiment scripts are in the `Code` section.

## Stages

| Stage | Scope | Status |
|---|---|---|
| 0. Accelerated introduction (Lectures 1–31) | Concepts → environment setup → first experiments → reasoning about the project | Lectures 1–31 ready — complete |
| 1. Literature and topic selection | Settling the thesis axis with the advisor, literature survey | — |
| 2. Experiment infrastructure | Setting up the benchmark, model, metrics and dataset | — |
| 3. Experiments | Main experiments, ablations, result analysis | — |
| 4. Thesis writing and defense | Thesis text, paper, presentation | — |

Source code and raw markdown: [github.com/mesutd0nmez/vla-robot-learning-tr](https://github.com/mesutd0nmez/vla-robot-learning-tr)
