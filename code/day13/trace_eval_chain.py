"""Lecture 13 — Finds the critical lines in the VLANeXt + LIBERO eval chain and prints a file:line table.

No GPU needed; it only reads source code. Usage (Mac):
    git clone --depth 1 https://github.com/DravenALG/VLANeXt ~/vla-work/VLANeXt
    python trace_eval_chain.py ~/vla-work/VLANeXt [~/vla-work/LIBERO]

Each row: WHICH QUESTION → which file → which line → the line itself. If upstream changes it
prints "NOT FOUND"; that means the pattern is stale, find the new one with grep.
"""
import re
import sys
from pathlib import Path

# (question, file, regex to search for)
ANCHORS = [
    # --- 1. Where does the command come from ---
    ("Where does the command string come from?", "LIBERO:libero/libero/benchmark/__init__.py",
     r"def grab_language_from_filename"),
    ("Where the command is bound to the episode (the line to change for Turkish)",
     "src/evaluation/libero_bench/parallel_eval_utils.py", r"task_description = task\.language"),
    ("How many times is the command sent to the model? (on every model call)",
     "src/evaluation/libero_bench/parallel_eval_utils.py", r"action_request_labels\.append\(specs\[env_id\]\.task_description\)"),
    # --- 2. Observation ---
    ("180-degree rotation", "src/evaluation/libero_bench/libero_utils.py", r"img\[::-1, ::-1\]"),
    ("Proprio vector (eef_pos + axis-angle + gripper)",
     "src/evaluation/libero_bench/parallel_eval_utils.py", r"quat2axisangle\(obs\[\"robot0_eef_quat\"\]\), \[gripper_state\]"),
    ("Template (Qwen, image branch): messages = [image, image, text]", "src/evaluation/libero_bench/VLANeXt_utils.py",
     r"^ {16}content\.append\(\{\"type\": \"text\", \"text\": label\}\)"),
    ("apply_chat_template(add_generation_prompt=True) (image branch)", "src/evaluation/libero_bench/VLANeXt_utils.py",
     r"^ {16}texts\.append\(effective_processor\.apply_chat_template\(messages, tokenize=False, add_generation_prompt=True\)"),
    # --- 3. Model ---
    ("Proprio tokens are prepended to the sequence", "src/models/VLANeXt.py",
     r"inputs_embeds = torch\.cat\(\[proprio_embeds, inputs_embeds\], dim=1\)"),
    ("16 meta queries are appended to the end of the sequence", "src/models/VLANeXt.py",
     r"inputs_embeds = torch\.cat\(\[inputs_embeds, queries_embeds\], dim=1\)"),
    ("VLM is called with all layer outputs (soft/tight)", "src/models/VLANeXt.py",
     r"\"output_hidden_states\": output_hidden_states_flag"),
    ("Policy block i attends to VLM layer i", "src/models/modules/policies.py",
     r"relevant_vlm_states = vlm_hidden_states\[-len\(self\.blocks\):\]"),
    ("Cross-attention: key/value = [x, vlm_proj(hidden)]", "src/models/modules/policies.py",
     r"kv_list = \[x_norm, v_feat\]"),
    ("Flow-matching: start from noise", "src/models/VLANeXt.py",
     r"action = torch\.randn\(B, self\.num_actions, self\.action_dim"),
    ("Flow-matching: Euler step", "src/models/VLANeXt.py",
     r"action = self\.noise_scheduler\.step\(output, t, action\)\.prev_sample"),
    # --- 4. Action decoding ---
    ("How many steps of the chunk are executed (num_steps_execute)", "src/evaluation/libero_bench/parallel_eval_utils.py",
     r"steps_to_exec = min\(num_steps_execute, len\(chunk\)\)"),
    ("Denormalization: first 6 dims min/max", "src/evaluation/libero_bench/parallel_eval_utils.py",
     r"action\[:6\] = \(action\[:6\] \+ 1\) / 2 \* \(action_max - action_min\) \+ action_min"),
    ("Gripper threshold: > 0 → +1, otherwise −1", "src/evaluation/libero_bench/parallel_eval_utils.py",
     r"action\[6\] = 1\.0 if action\[6\] > 0 else -1\.0"),
    ("Where the min/max statistics come from (legacy/mixed)", "src/datasets/libero_act.py", r"\"legacy\": \{"),
    # --- 5. Success ---
    ("done → success=True (step phase; :237 is the wait phase)", "src/evaluation/libero_bench/parallel_eval_utils.py",
     r"^ {20}states\[env_id\]\[\"success\"\] = True"),
    ("LIBERO: done = _check_success()", "LIBERO:libero/libero/envs/bddl_base_domain.py",
     r"done = self\._check_success\(\)"),
    ("LIBERO: AND of all goal predicates", "LIBERO:libero/libero/envs/problems/libero_tabletop_manipulation.py",
     r"result = self\._eval_predicate\(state\) and result"),
    ("LIBERO: On predicate = contact + xy distance < 0.03 m", "LIBERO:libero/libero/envs/object_states/base_object_states.py",
     r"< 0\.03"),
]


def find(root: Path, rel: str, pattern: str):
    p = root / rel
    if not p.exists():
        return None, f"file missing: {p}"
    rx = re.compile(pattern)
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if rx.search(line):
            return i, line.strip()
    return None, "NOT FOUND (the pattern may be stale)"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    vla = Path(sys.argv[1]).expanduser()
    lib = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else None
    print(f"VLANeXt: {vla}\nLIBERO : {lib or '(not given, LIBERO rows are skipped)'}\n")
    print(f"{'#':>2} | {'Question':62} | {'File':64} | Line")
    print("-" * 150)
    for n, (q, rel, pat) in enumerate(ANCHORS, 1):
        if rel.startswith("LIBERO:"):
            if lib is None:
                continue
            root, rel = lib, rel[len("LIBERO:"):]
        else:
            root = vla
        ln, text = find(root, rel, pat)
        loc = f"{rel}:{ln}" if ln else rel
        print(f"{n:>2} | {q:62} | {loc:64} | {text[:70]}")


if __name__ == "__main__":
    main()
