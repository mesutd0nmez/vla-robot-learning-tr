"""Ders 13 — VLANeXt + LIBERO eval zincirinde kritik satırları bulur ve dosya:satır tablosu basar.

GPU gerekmez; yalnızca kaynak kodu okur. Kullanım (Mac):
    git clone --depth 1 https://github.com/DravenALG/VLANeXt ~/vla-work/VLANeXt
    python trace_eval_chain.py ~/vla-work/VLANeXt [~/vla-work/LIBERO]

Her satır: HANGİ SORU → hangi dosya → hangi satır → satırın kendisi. Upstream değişirse
"BULUNAMADI" yazar; o zaman desen eskimiş demektir, grep ile yenisini bul.
"""
import re
import sys
from pathlib import Path

# (soru, dosya, aranan düzenli ifade)
ANCHORS = [
    # --- 1. Komut nereden geliyor ---
    ("Komut string'i nereden geliyor?", "LIBERO:libero/libero/benchmark/__init__.py",
     r"def grab_language_from_filename"),
    ("Komutun episode'a bağlandığı yer (Türkçe için değişecek satır)",
     "src/evaluation/libero_bench/parallel_eval_utils.py", r"task_description = task\.language"),
    ("Komut modele kaç kez gönderiliyor? (her model çağrısında)",
     "src/evaluation/libero_bench/parallel_eval_utils.py", r"action_request_labels\.append\(specs\[env_id\]\.task_description\)"),
    # --- 2. Gözlem ---
    ("180 derece döndürme", "src/evaluation/libero_bench/libero_utils.py", r"img\[::-1, ::-1\]"),
    ("Proprio vektörü (eef_pos + axis-angle + gripper)",
     "src/evaluation/libero_bench/parallel_eval_utils.py", r"quat2axisangle\(obs\[\"robot0_eef_quat\"\]\), \[gripper_state\]"),
    ("Sablon (Qwen, image dalı): messages = [image, image, text]", "src/evaluation/libero_bench/VLANeXt_utils.py",
     r"^ {16}content\.append\(\{\"type\": \"text\", \"text\": label\}\)"),
    ("apply_chat_template(add_generation_prompt=True) (image dalı)", "src/evaluation/libero_bench/VLANeXt_utils.py",
     r"^ {16}texts\.append\(effective_processor\.apply_chat_template\(messages, tokenize=False, add_generation_prompt=True\)"),
    # --- 3. Model ---
    ("Proprio token'ları dizinin başına ekleniyor", "src/models/VLANeXt.py",
     r"inputs_embeds = torch\.cat\(\[proprio_embeds, inputs_embeds\], dim=1\)"),
    ("16 meta query dizinin sonuna ekleniyor", "src/models/VLANeXt.py",
     r"inputs_embeds = torch\.cat\(\[inputs_embeds, queries_embeds\], dim=1\)"),
    ("VLM tüm katman çıktılarıyla çağrılıyor (soft/tight)", "src/models/VLANeXt.py",
     r"\"output_hidden_states\": output_hidden_states_flag"),
    ("Policy bloğu i, VLM katmanı i'ye bakıyor", "src/models/modules/policies.py",
     r"relevant_vlm_states = vlm_hidden_states\[-len\(self\.blocks\):\]"),
    ("Cross-attention: key/value = [x, vlm_proj(hidden)]", "src/models/modules/policies.py",
     r"kv_list = \[x_norm, v_feat\]"),
    ("Flow-matching: gürültüden başla", "src/models/VLANeXt.py",
     r"action = torch\.randn\(B, self\.num_actions, self\.action_dim"),
    ("Flow-matching: Euler adımı", "src/models/VLANeXt.py",
     r"action = self\.noise_scheduler\.step\(output, t, action\)\.prev_sample"),
    # --- 4. Action decode ---
    ("Chunk'ın kaç adımı uygulanıyor (num_steps_execute)", "src/evaluation/libero_bench/parallel_eval_utils.py",
     r"steps_to_exec = min\(num_steps_execute, len\(chunk\)\)"),
    ("Denormalizasyon: ilk 6 boyut min/max", "src/evaluation/libero_bench/parallel_eval_utils.py",
     r"action\[:6\] = \(action\[:6\] \+ 1\) / 2 \* \(action_max - action_min\) \+ action_min"),
    ("Gripper eşiği: > 0 → +1, değilse −1", "src/evaluation/libero_bench/parallel_eval_utils.py",
     r"action\[6\] = 1\.0 if action\[6\] > 0 else -1\.0"),
    ("Min/max istatistikleri nereden (legacy/mixed)", "src/datasets/libero_act.py", r"\"legacy\": \{"),
    # --- 5. Success ---
    ("done → success=True (step fazı; :237 bekleme fazı)", "src/evaluation/libero_bench/parallel_eval_utils.py",
     r"^ {20}states\[env_id\]\[\"success\"\] = True"),
    ("LIBERO: done = _check_success()", "LIBERO:libero/libero/envs/bddl_base_domain.py",
     r"done = self\._check_success\(\)"),
    ("LIBERO: tüm goal predicate'lerinin VE'si", "LIBERO:libero/libero/envs/problems/libero_tabletop_manipulation.py",
     r"result = self\._eval_predicate\(state\) and result"),
    ("LIBERO: On predicate = temas + xy mesafe < 0.03 m", "LIBERO:libero/libero/envs/object_states/base_object_states.py",
     r"< 0\.03"),
]


def find(root: Path, rel: str, pattern: str):
    p = root / rel
    if not p.exists():
        return None, f"dosya yok: {p}"
    rx = re.compile(pattern)
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if rx.search(line):
            return i, line.strip()
    return None, "BULUNAMADI (desen eskimiş olabilir)"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    vla = Path(sys.argv[1]).expanduser()
    lib = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else None
    print(f"VLANeXt: {vla}\nLIBERO : {lib or '(verilmedi, LIBERO satırları atlanır)'}\n")
    print(f"{'#':>2} | {'Soru':62} | {'Dosya':64} | Satır")
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
