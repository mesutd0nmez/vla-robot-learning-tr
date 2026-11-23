"""Lecture 24 — solution candidate 1 (translation layer, MT) on the Mac: translate the 80 Turkish commands of datasets/04 into
English with an open MT model, compare with the `en` column, measure the latency per command.

NOT RUN in this session (no Hugging Face access) — run it on the Mac in the `vla` env:
    pip install "transformers>=4.40" sentencepiece sacrebleu
    python mt_layer_mac.py --model Helsinki-NLP/opus-mt-tr-en            # ~300 MB, fast
    python mt_layer_mac.py --model facebook/nllb-200-distilled-600M      # ~2.4 GB, 200 languages
Output: mt_roundtrip_<model>.csv (tr, en_pred, en_gold, exact, chrf, ms) + a summary line.

The three things we measure: (1) exact match (does it land on the LIBERO template?), (2) chrF (a rough measure of semantic closeness),
(3) latency — in closed loop it is translated ONCE per episode, not per step; so even 200 ms is not a problem.
The real question is 'wrong but valid' translations: the translation layer itself can produce the `wrong` condition (Lecture 18).
"""
import argparse, csv, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / ".." / ".." / "datasets" / "04-tr-spatial-commands" / "data" / "commands.csv"


def load(model_name, device):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, src_lang="tur_Latn" if "nllb" in model_name else None)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()
    forced = tok.convert_tokens_to_ids("eng_Latn") if "nllb" in model_name else None
    return tok, model, forced


def translate(tok, model, forced, text, device):
    import torch
    enc = tok(text, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, forced_bos_token_id=forced, num_beams=4, max_new_tokens=48)
    ms = (time.perf_counter() - t0) * 1000
    return tok.batch_decode(out, skip_special_tokens=True)[0].strip().lower().rstrip("."), ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Helsinki-NLP/opus-mt-tr-en")
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--device", default=None, help="mps / cpu (default: mps if available)")
    a = ap.parse_args()
    import torch
    device = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    try:
        from sacrebleu.metrics import CHRF
        chrf = CHRF()
    except ImportError:
        chrf = None
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    gold = {r["pair_id"]: r["instruction"] for r in rows if r["lang"] == "en"}
    tok, model, forced = load(a.model, device)
    out_path = HERE / f"mt_roundtrip_{a.model.split('/')[-1]}.csv"
    exact, n, lat = {}, {}, []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "variant", "tr", "en_pred", "en_gold", "exact", "chrf", "ms"])
        for r in rows:
            if r["lang"] != "tr":
                continue
            pred, ms = translate(tok, model, forced, r["instruction"], device)
            g = gold[r["pair_id"]]
            ex = int(pred == g)
            score = chrf.sentence_score(pred, [g]).score if chrf else float("nan")
            w.writerow([r["id"], r["variant"], r["instruction"], pred, g, ex, f"{score:.1f}", f"{ms:.0f}"])
            exact[r["variant"]] = exact.get(r["variant"], 0) + ex; n[r["variant"]] = n.get(r["variant"], 0) + 1; lat.append(ms)
    print(f"model={a.model} device={device}")
    for v in ("tr_a", "tr_ascii", "tr_bare", "tr_b"):
        print(f"  {v:<9} exact {exact.get(v,0)}/{n.get(v,0)}")
    lat.sort(); print(f"  latency median {lat[len(lat)//2]:.0f} ms (range {lat[0]:.0f}…{lat[-1]:.0f} ms) -> {out_path.name}")


if __name__ == "__main__":
    main()
