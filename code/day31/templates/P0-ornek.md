# Project design — P0 (name: does `wrong` shift the target on Goal task 8 — EXAMPLE, filled in by Claude)

**Question:** In the LIBERO Goal task 8 scene (`put the bowl on the plate`), on the same init states does S-VN with the `wrong` command (`put the bowl on the stove`) change the target relative to `clean`, or does it behave like `empty` and ignore the command?

**Null hypothesis:** H0-1: P(success | wrong) = P(success | clean) (no difference; δ = 15 points). H0-2: P(success | wrong) = P(success | empty) (δ = 10 points).

**Variables:** Independent: command condition {clean, empty, wrong}. Dependent: episode success (the task 8 predicate) and the target reached (plate / stove / none, video label). Shared: the no-command ceiling (Lecture 19: Goal 1/10). Fixed: init states 0..49, checkpoint, `num_steps_wait` 10.

**Cells:**
- S-VN, libero_goal, C (task 8), clean;empty;wrong, n_init 50, repeats 1

**Budget:** 3.2 T4 h (mid scenario, budget.py: Goal clean 57 s + empty 88 s + wrong 88 s, × 50 init) · pooled n = 50 · MDE = 23 points

**Success criterion:** "wrong shifts the target": McNemar p < 0.05 and |Δ(wrong − clean)| ≥ 23 and at least 10/50 episodes labelled stove. "Ignores the command": wrong ≈ empty within TOST δ = 10.

**Risks:**
- The Goal ceiling is low (1/10) → `empty` ≈ 0; the wrong − empty distinction is visible only through the target label (video labelling 50 × 3).
- A single task → no generalization; the result is written as "on task 8".

**If yes / if no:** Yes: "S-VN reads the command on Goal and goes to the wrong target" → `wrong` becomes a valid positive control for telling whether the Turkish `tr` condition is read. No: wrong ≈ empty → language is weakly read on Goal too; the Goal C block measurement of RQ-a becomes meaningless with this system.
