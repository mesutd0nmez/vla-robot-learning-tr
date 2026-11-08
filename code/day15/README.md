# day15

Ders 15 — Tatar vd. mini-reprodüksiyonu: 5 LIBERO-Spatial karesi × 6 komut koşulu (en / para / tr / tr2 / distractor / empty) → OpenVLA-7B (Mac, MPS) → 7-DoF action; L2 / cosine / token metrikleri.

| Dosya | Ne |
|---|---|
| `make_commands.py` | Komut setini üretir → `datasets/01-libero-spatial-tr-mini/data/commands.csv` (30 satır; çeldirici eşlemesi BDDL `:init`'ten) |
| `count_tokens.mjs` | Her komutun OpenVLA prompt'unu LLaMA-2 tokenizer'ıyla sayar (`llama-tokenizer-js`; model gerekmez) → `token_counts.json` |
| `render_frames.py` | Görev 0–4 için 180° döndürülmüş 256×256 agentview kareleri + `contact_sheet.png` (`vla` env) |
| `run_tatar_mini.py` | Modeli bir kez yükler, 30 çağrı → `results/actions.csv` + `results/json/*.json` (`openvla` env, Ders 14) |
| `analyze_day15.py` | `actions.csv` → `metrics.csv`, `summary.md` (koşul tablosu + Wilcoxon), `metrics.png`; `--demo` sentetik veriyle hattı test eder |
| `plot_day15.py` | Dersteki 02 (metrik geometrisi), 03 (token sayıları), 05 (literatür) görselleri |

Kullanım:
```bash
# 0) kareler — vla env (Ders 10)
conda activate vla && mkdir -p ~/vla-work/day15 && cd ~/vla-work/day15
for f in render_frames.py run_tatar_mini.py analyze_day15.py make_commands.py count_tokens.mjs; do
  curl -O https://raw.githubusercontent.com/mesutd0nmez/vla-robot-learning-tr/main/code/day15/$f; done
curl -O https://raw.githubusercontent.com/mesutd0nmez/vla-robot-learning-tr/main/datasets/01-libero-spatial-tr-mini/data/commands.csv
python render_frames.py --tasks 0,1,2,3,4 --out frames

# 1) 30 çağrı — openvla env (Ders 14)
conda activate openvla
python run_tatar_mini.py --frames frames --commands commands.csv --out results/actions.csv
python run_tatar_mini.py --frames frames --commands commands.csv --only sp0_en,sp0_tr --repeat 2 --out results/determinism.csv

# 2) analiz (numpy, scipy, matplotlib)
python analyze_day15.py --actions results/actions.csv --out results/
python analyze_day15.py --demo --out demo/        # sentetik veriyle hat testi

# (isteğe bağlı) token sayıları modelsiz: npm install llama-tokenizer-js && node count_tokens.mjs commands.csv token_counts.json
```


## make_commands.py

```python
"""Ders 15 — Mini Tatar deneyinin komut setini üretir: datasets/01-libero-spatial-tr-mini/data/commands.csv

5 LIBERO-Spatial karesi (görev 0–4) × 6 koşul = 30 satır.
Koşullar (Ders 4 matrisi + Ders 7'nin iki yüzey formu):
  en          LIBERO'nun kendi komutu (task.language; referans)
  para        aynı anlam, farklı kelimeler, aynı dil  -> tokenizer-dışı gürültü kontrolü
  tr          Türkçe çeviri, LIBERO stili (-daki sıfat-fiil yapısı, "kase", "al ... koy")
  tr2         ikinci Türkçe yüzey formu ("duran", "kâse", "kaldır ... yerleştir") -> TR-içi parafraz kontrolü
  distractor  AYNI sahnedeki ÖTEKİ siyah kâseyi tarif eden komut (BDDL :init'ten okunmuş) -> mekânsal grounding testi
  empty       boş komut -> Ders 4'ün dil-bağımsız tabanı

"distractor" eşlemesi LIBERO BDDL dosyalarındaki (:init ...) bloğundan: her Spatial sahnesinde iki akita_black_bowl var;
bowl_1 hedef, bowl_2 çeldirici. bowl_2'nin bölgesini tarif eden Spatial komutu seçildi.
    python make_commands.py --out ../../datasets/01-libero-spatial-tr-mini/data/commands.csv
"""
import argparse
import csv
import os

# LIBERO-Spatial komutları (task.language), Ders 10 sırası
EN = {
    0: "pick up the black bowl between the plate and the ramekin and place it on the plate",
    1: "pick up the black bowl next to the ramekin and place it on the plate",
    2: "pick up the black bowl from table center and place it on the plate",
    3: "pick up the black bowl on the cookie box and place it on the plate",
    4: "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    5: "pick up the black bowl on the ramekin and place it on the plate",
    6: "pick up the black bowl next to the cookie box and place it on the plate",
    7: "pick up the black bowl on the stove and place it on the plate",
    8: "pick up the black bowl next to the plate and place it on the plate",
    9: "pick up the black bowl on the wooden cabinet and place it on the plate",
}

# BDDL :init -> akita_black_bowl_2'nin bölgesi -> o bölgeyi tarif eden görev
# 0: bowl_2 next_to_ramekin -> 1 | 1: bowl_2 next_to_box -> 6 | 2: bowl_2 next_to_plate -> 8
# 3: bowl_2 wooden_cabinet_1_top_side -> 9 | 4: bowl_2 wooden_cabinet_1_top_side -> 9
DISTRACTOR = {0: 1, 1: 6, 2: 8, 3: 9, 4: 9}

PARA = {
    0: "grab the black bowl located between the plate and the ramekin and put it on the plate",
    1: "grab the black bowl located next to the ramekin and put it on the plate",
    2: "grab the black bowl located at the center of the table and put it on the plate",
    3: "grab the black bowl located on top of the cookie box and put it on the plate",
    4: "grab the black bowl located in the upper drawer of the wooden cabinet and put it on the plate",
}

TR = {
    0: "tabak ile ramekinin arasındaki siyah kaseyi al ve tabağın üstüne koy",
    1: "ramekinin yanındaki siyah kaseyi al ve tabağın üstüne koy",
    2: "masanın ortasındaki siyah kaseyi al ve tabağın üstüne koy",
    3: "kurabiye kutusunun üstündeki siyah kaseyi al ve tabağın üstüne koy",
    4: "ahşap dolabın üst çekmecesindeki siyah kaseyi al ve tabağın üstüne koy",
}

TR2 = {
    0: "tabakla ramekin arasında duran siyah kâseyi kaldır ve tabağa yerleştir",
    1: "ramekinin yanında duran siyah kâseyi kaldır ve tabağa yerleştir",
    2: "masanın ortasında duran siyah kâseyi kaldır ve tabağa yerleştir",
    3: "kurabiye kutusunun üzerinde duran siyah kâseyi kaldır ve tabağa yerleştir",
    4: "ahşap dolabın üst çekmecesinde duran siyah kâseyi kaldır ve tabağa yerleştir",
}

# ilişki türü (Ders 6 kalkülüsleri) ve TR'de ilişkiyi taşıyan ekler
RELATION = {0: "arasında (2 referans)", 1: "yanında (mesafe)", 2: "ortasında (bölge)", 3: "üstünde (EC)", 4: "içinde (NTPP)"}
SUFFIX_TR = {0: "-in;ara-sı-nda-ki;-ın;üst-ü-ne", 1: "-in;yan-ı-nda-ki;-ın;üst-ü-ne", 2: "-nın;orta-sı-nda-ki;-ın;üst-ü-ne",
             3: "-nun;üst-ü-nde-ki;-ın;üst-ü-ne", 4: "-ın;çekmece-si-nde-ki;-ın;üst-ü-ne"}
SUFFIX_TR2 = {0: "-la;ara-sı-nda;-a", 1: "-in;yan-ı-nda;-a", 2: "-nın;orta-sı-nda;-a",
              3: "-nun;üzer-i-nde;-a", 4: "-ın;çekmece-si-nde;-a"}


def rows():
    out = []
    for t in range(5):
        base = dict(suite="libero_spatial", task_id=t, relation=RELATION[t], frame="camera")
        out.append(dict(base, id=f"sp{t}_en", lang="en", condition="en", instruction=EN[t], suffixes="", variant="a",
                        note="task.language (referans)"))
        out.append(dict(base, id=f"sp{t}_para", lang="en", condition="para", instruction=PARA[t], suffixes="", variant="b",
                        note="EN parafraz: grab/located/put"))
        out.append(dict(base, id=f"sp{t}_tr", lang="tr", condition="tr", instruction=TR[t], suffixes=SUFFIX_TR[t], variant="a",
                        note="LIBERO stili, -daki yapısı, 'kase'"))
        out.append(dict(base, id=f"sp{t}_tr2", lang="tr", condition="tr2", instruction=TR2[t], suffixes=SUFFIX_TR2[t], variant="b",
                        note="'duran' yapısı, 'kâse', 'kaldır' (Tatar: Kal+dır)"))
        d = DISTRACTOR[t]
        out.append(dict(base, id=f"sp{t}_distractor", lang="en", condition="distractor", instruction=EN[d], suffixes="",
                        variant="a", note=f"ayni sahnedeki akita_black_bowl_2'yi tarif eder (= görev {d} komutu)"))
        out.append(dict(base, id=f"sp{t}_empty", lang="none", condition="empty", instruction="", suffixes="", variant="a",
                        note="bos komut (prompt: 'In: What action should the robot take to ?')"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "datasets",
                                                  "01-libero-spatial-tr-mini", "data", "commands.csv"))
    args = ap.parse_args()
    cols = ["id", "suite", "task_id", "lang", "condition", "instruction", "relation", "frame", "suffixes", "variant", "note"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows():
            w.writerow(r)
    print(f"yazıldı: {args.out} ({len(rows())} satır)")


if __name__ == "__main__":
    main()
```

## count_tokens.mjs

```js
// Ders 15 — commands.csv'deki her komutun OpenVLA prompt'unu LLaMA-2 tokenizer'ıyla sayar (model GEREKMEZ).
// LLaMA-2 BPE'nin JS uyarlaması: npm install llama-tokenizer-js   (Ders 14'te 29871 = "▁" doğrulandı)
//   node count_tokens.mjs ../../datasets/01-libero-spatial-tr-mini/data/commands.csv token_counts.json
// Çıktı: her satır için prompt token sayısı (BOS dahil), komutun kendi token sayısı, parça listesi;
// ayrıca mekânsal ifadelerin ve "kase/kâse/kaldır" kelimelerinin tek başına parçalanışı.
import fs from "node:fs";
import llamaTokenizer from "llama-tokenizer-js";

const [csvPath, outPath = "token_counts.json"] = process.argv.slice(2);
const text = fs.readFileSync(csvPath, "utf8");

// basit CSV ayrıştırıcı (tırnaklı alanlar dahil)
function parseCSV(s) {
  const rows = []; let row = [], field = "", q = false;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (q) { if (c === '"') { if (s[i + 1] === '"') { field += '"'; i++; } else q = false; } else field += c; }
    else if (c === '"') q = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
    else if (c !== "\r") field += c;
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  return rows;
}
const [header, ...data] = parseCSV(text).filter(r => r.length > 1);
const col = Object.fromEntries(header.map((h, i) => [h, i]));

const prompt = (ins) => `In: What action should the robot take to ${ins.toLowerCase()}?\nOut:`;
const pieces = (s) => llamaTokenizer.decode ? llamaTokenizer.encode(s, false, false).map(id => llamaTokenizer.decode([id], false, false)) : [];

const rows = data.map(r => {
  const ins = r[col.instruction];
  const p = prompt(ins);
  const idsPrompt = llamaTokenizer.encode(p);            // BOS dahil
  const idsIns = llamaTokenizer.encode(ins, false, false); // BOS/EOS yok
  return {
    id: r[col.id], task_id: +r[col.task_id], lang: r[col.lang], condition: r[col.condition], instruction: ins,
    prompt_tokens_with_bos: idsPrompt.length, instruction_tokens: idsIns.length,
    last_prompt_token: idsPrompt[idsPrompt.length - 1],
    instruction_pieces: pieces(ins),
  };
});

// mekânsal ifadeler ve tek kelimeler
const phrases = {
  "between the plate and the ramekin": "en", "tabak ile ramekinin arasındaki": "tr", "tabakla ramekin arasında": "tr2",
  "next to the ramekin": "en", "ramekinin yanındaki": "tr", "ramekinin yanında": "tr2",
  "from table center": "en", "masanın ortasındaki": "tr", "masanın ortasında": "tr2",
  "on the cookie box": "en", "kurabiye kutusunun üstündeki": "tr", "kurabiye kutusunun üzerinde": "tr2",
  "in the top drawer of the wooden cabinet": "en", "ahşap dolabın üst çekmecesindeki": "tr", "ahşap dolabın üst çekmecesinde": "tr2",
  "kase": "w", "kâse": "w", "kaseyi": "w", "kâseyi": "w", "kaldır": "w", "al": "w", "koy": "w", "yerleştir": "w", "duran": "w",
};
const phraseRows = Object.entries(phrases).map(([ph, kind]) => ({ phrase: ph, kind, n: llamaTokenizer.encode(ph, false, false).length, pieces: pieces(ph) }));

fs.writeFileSync(outPath, JSON.stringify({ rows, phrases: phraseRows }, null, 1));
for (const r of rows) console.log(`${r.id.padEnd(16)} ${String(r.prompt_tokens_with_bos).padStart(3)} prompt  ${String(r.instruction_tokens).padStart(3)} komut  son=${r.last_prompt_token}`);
console.log("\n-- ifadeler --");
for (const p of phraseRows) console.log(`${String(p.n).padStart(3)}  ${p.phrase.padEnd(42)} ${p.pieces.join("|")}`);
console.log(`\nyazıldı: ${outPath}`);
```

## render_frames.py

```python
"""Ders 15 — LIBERO-Spatial görev 0–4 için OpenVLA'nın göreceği kareleri üretir (frames/frame_task{t}.png).

Ders 14'ün `render_libero_frame.py` mantığı, 5 görev için döngüde. `vla` env'inde çalışır (mujoco 2.3.7, robosuite 1.4.1, LIBERO).
    conda activate vla
    python render_frames.py --tasks 0,1,2,3,4 --init 0 --out frames
Her kare: init_states[init] → 10 no-op adım (run_libero_eval.py: num_steps_wait) → agentview 256x256 → 180° döndür.
Yanına `frames/contact_sheet.png` (5 kare yan yana, komutlarıyla) yazar — derste/projede kullanmak için.
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl" if os.uname().sysname == "Linux" else "glfw")

import numpy as np
from PIL import Image, ImageDraw
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def render(suite, task_id, init, settle):
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(task_id)[init])
    for _ in range(settle):
        obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
    env.close()
    return obs["agentview_image"][::-1, ::-1], task.language


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="0,1,2,3,4")
    ap.add_argument("--init", type=int, default=0)
    ap.add_argument("--settle", type=int, default=10)
    ap.add_argument("--out", default="frames")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    tiles = []
    for t in [int(x) for x in args.tasks.split(",")]:
        frame, lang = render(suite, t, args.init, args.settle)
        p = os.path.join(args.out, f"frame_task{t}.png")
        Image.fromarray(frame).save(p)
        print(f"görev {t}: {lang}\n  → {p}")
        tile = Image.fromarray(frame).copy()
        ImageDraw.Draw(tile).text((4, 4), f"task {t}", fill=(255, 255, 255))
        tiles.append(tile)
    sheet = Image.new("RGB", (256 * len(tiles), 256))
    for i, tl in enumerate(tiles):
        sheet.paste(tl, (256 * i, 0))
    sheet.save(os.path.join(args.out, "contact_sheet.png"))
    print("yazıldı:", os.path.join(args.out, "contact_sheet.png"))


if __name__ == "__main__":
    main()
```

## run_tatar_mini.py

```python
"""Ders 15 — Tatar vd. mini-reprodüksiyonu: 5 LIBERO-Spatial karesi × 6 komut koşulu → OpenVLA-7B (MPS) → 7-DoF action.

Ders 14'ün `openvla_mps_single_action.py` zinciri, model BİR kez yüklenip 30 çağrı için döngüye alınmış hâli.
Girdi : frames/frame_task{0..4}.png  (render_frames.py; 180° döndürülmüş 256x256 agentview)
        datasets/01-libero-spatial-tr-mini/data/commands.csv  (make_commands.py)
Çıktı : results/actions.csv  — satır başına: id, task_id, condition, lang, instruction, prompt_tokens,
        tok0..tok6 (token id), bin0..bin6, n0..n6 ([-1,1]), a0..a6 (LIBERO birimi), gripper_libero, seconds
        results/json/<id>.json — Ders 14 formatında tam kayıt
Kullanım (openvla env, Ders 14):
    python run_tatar_mini.py --frames frames --commands ../../datasets/01-libero-spatial-tr-mini/data/commands.csv \
        --out results/actions.csv
    python run_tatar_mini.py ... --only sp0_en,sp0_tr      # yalnızca seçili satırlar (hızlı deneme)
    python run_tatar_mini.py ... --repeat 2                # her satırı 2 kez: greedy ile aynı çıkmalı (determinizm kontrolü)
Greedy decode (do_sample=False): Tatar vd.'nin sıcaklık 0.3 örneklemesi yerine deterministik; EN–EN gürültüsü 0 olmalı,
dil-dışı gürültü kontrolünü `para` koşulu üstlenir.
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

CKPT = "openvla/openvla-7b-finetuned-libero-spatial"
UNNORM_KEY = "libero_spatial"
DIMS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def center_crop_like_training(img, crop_scale=0.9):
    w, h = img.size
    s = math.sqrt(crop_scale)
    cw, ch = round(w * s), round(h * s)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)


def load_model(dtype, attn):
    device = pick_device()
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(CKPT, attn_implementation=attn, torch_dtype=dtype,
                                                low_cpu_mem_usage=True, device_map={"": device}, trust_remote_code=True)
    vla.eval()
    print(f"[*] model yüklendi ({time.time()-t0:.0f} s), device={device}")
    return processor, vla, device


def prepare_image(path):
    img = Image.open(path).convert("RGB")
    if img.size != (224, 224):
        img = img.resize((224, 224), Image.LANCZOS)
    return center_crop_like_training(img)


def one_call(processor, vla, device, dtype, img, instruction):
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inputs = processor(prompt, img).to(device, dtype=dtype)
    ids = inputs["input_ids"][0].tolist()
    t1 = time.time()
    with torch.inference_mode():
        action = vla.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
    dt = time.time() - t1
    input_ids = inputs["input_ids"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat((input_ids, torch.tensor([[29871]], device=device)), dim=1)
    with torch.inference_mode():
        gen = vla.generate(input_ids=input_ids, pixel_values=inputs["pixel_values"],
                           attention_mask=torch.ones_like(input_ids), max_new_tokens=7, do_sample=False)
    tok = gen[0, -7:].cpu().numpy()
    disc = np.clip(vla.vocab_size - tok - 1, 0, vla.bin_centers.shape[0] - 1)
    norm = vla.bin_centers[disc]
    g = -np.sign(2 * action[6] - 1)
    return dict(prompt=prompt, prompt_token_ids=ids, action_token_ids=tok.tolist(), bins=disc.tolist(),
                normalized=norm.tolist(), action=[float(x) for x in action], gripper_libero=float(g), seconds=dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="frames")
    ap.add_argument("--commands", required=True)
    ap.add_argument("--out", default="results/actions.csv")
    ap.add_argument("--only", default="", help="virgülle ayrılmış id listesi")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.commands, encoding="utf-8")))
    if args.only:
        keep = set(args.only.split(","))
        rows = [r for r in rows if r["id"] in keep]
    dtype = getattr(torch, args.dtype)
    processor, vla, device = load_model(dtype, args.attn)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    jdir = os.path.join(os.path.dirname(args.out) or ".", "json")
    os.makedirs(jdir, exist_ok=True)
    cols = (["id", "task_id", "condition", "lang", "instruction", "repeat", "prompt_tokens"]
            + [f"tok{i}" for i in range(7)] + [f"bin{i}" for i in range(7)]
            + [f"n{i}" for i in range(7)] + [f"a{i}" for i in range(7)] + ["gripper_libero", "seconds"])
    images = {}
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            t = int(r["task_id"])
            if t not in images:
                images[t] = prepare_image(os.path.join(args.frames, f"frame_task{t}.png"))
            for rep in range(args.repeat):
                res = one_call(processor, vla, device, dtype, images[t], r["instruction"])
                rec = dict(id=r["id"], task_id=t, condition=r["condition"], lang=r["lang"], instruction=r["instruction"],
                           repeat=rep, prompt_tokens=len(res["prompt_token_ids"]), gripper_libero=res["gripper_libero"],
                           seconds=round(res["seconds"], 3))
                for i in range(7):
                    rec[f"tok{i}"] = res["action_token_ids"][i]
                    rec[f"bin{i}"] = res["bins"][i]
                    rec[f"n{i}"] = round(res["normalized"][i], 5)
                    rec[f"a{i}"] = round(res["action"][i], 5)
                w.writerow(rec)
                f.flush()
                json.dump(dict(checkpoint=CKPT, **{k: r[k] for k in r}, **res, device=str(device), dtype=args.dtype),
                          open(os.path.join(jdir, f"{r['id']}_r{rep}.json"), "w"), indent=1, ensure_ascii=False)
                print(f"{r['id']:16s} rep{rep} tokens={res['action_token_ids']} g={res['gripper_libero']:+.0f} "
                      f"{res['seconds']:.2f}s  [{rec['prompt_tokens']} prompt token]")
    print(f"[*] yazıldı: {args.out}  → şimdi: python analyze_day15.py --actions {args.out}")


if __name__ == "__main__":
    main()
```

## analyze_day15.py

```python
"""Ders 15 — actions.csv'den Tatar-tarzı metrikleri hesaplar: L2, cosine (7-D / 6-D, normalize / gerçek birim),
token uyumsuzluğu, bin mesafesi, gripper uyumu; koşul başına özet tablo + Wilcoxon (n=5) + görsel.

    python analyze_day15.py --actions results/actions.csv --out results/        # gerçek koşu
    python analyze_day15.py --demo --out demo/                                 # SENTETİK veriyle hattı test et

Referans: her görevde `en` koşulu. Karşılaştırılan koşullar: para, tr, tr2, distractor, empty; ek çift: tr→tr2 (TR-içi parafraz).
Metrik tanımları (n = [-1,1] bin merkezi vektörü, a = LIBERO birimli action, a[6] ∈ [0,1]):
  l2_n7, cos_n7   : 7 boyut, normalize uzay (gripper ±0.996 → tek flip L2'ye ~2.0 katar!)
  l2_n6, cos_n6   : ilk 6 boyut, normalize uzay (gripper ayrı raporlanır)  ← önerilen ana metrik
  l2_a7, cos_a7   : 7 boyut, gerçek birim (Tatar vd. ile karşılaştırılabilir; OpenVLA gripper çıktısı ±1 → flip ≈ 2.0)
  tok_mismatch    : 7 token'dan kaçı farklı (0..7)
  max_bin         : en büyük |Δbin| (255 bin üzerinden)
  within5         : |Δbin| ≤ 5 olan boyut oranı (Shin 2026'nın "5-bin accuracy"inin çift versiyonu)
  grip_agree      : LIBERO'ya giden gripper işareti aynı mı (1/0)
"""
import argparse
import csv
import os

import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:  # scipy yoksa p-değeri atlanır
    wilcoxon = None

N_BINS = 256
BINS = np.linspace(-1, 1, N_BINS)
BIN_CENTERS = (BINS[:-1] + BINS[1:]) / 2.0
VOCAB = 32000
Q01 = np.array([-0.7454732114076613, -0.6616071462631226, -0.9375, -0.1071428582072258, -0.20678570866584778,
                -0.1842857152223587, 0.0])
Q99 = np.array([0.9375, 0.8758928775787354, 0.9321428537368774, 0.1039285734295845, 0.17678570747375488,
                0.14571428298950195, 1.0])
MASK = np.array([True] * 6 + [False])
CONDS = ["para", "tr", "tr2", "distractor", "empty"]
COLORS = {"en": "#ffd166", "para": "#cfe8ff", "tr": "#ffd166", "tr2": "#f3c86a", "distractor": "#ffd6d6", "empty": "#ffd6d6",
          "tr_vs_tr2": "#d4f5d4"}


def bins_to_action(b):
    n = BIN_CENTERS[np.clip(b, 0, 254)]
    return n, np.where(MASK, 0.5 * (n + 1) * (Q99 - Q01) + Q01, n)


def cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else float("nan")


def pair_metrics(ref, oth):
    """ref/oth: dict(bins=np.array(7), n=..., a=..., tok=..., g=±1)"""
    d = np.abs(ref["bins"] - oth["bins"])
    return dict(
        l2_n7=float(np.linalg.norm(ref["n"] - oth["n"])), cos_n7=cos(ref["n"], oth["n"]),
        l2_n6=float(np.linalg.norm(ref["n"][:6] - oth["n"][:6])), cos_n6=cos(ref["n"][:6], oth["n"][:6]),
        l2_a7=float(np.linalg.norm(ref["a"] - oth["a"])), cos_a7=cos(ref["a"], oth["a"]),
        tok_mismatch=int(np.sum(ref["tok"] != oth["tok"])), max_bin=int(d.max()),
        within5=float(np.mean(d <= 5)), grip_agree=int(ref["g"] == oth["g"]),
    )


def read_actions(path):
    recs = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if int(r.get("repeat", 0)) != 0:
            continue
        b = np.array([int(r[f"bin{i}"]) for i in range(7)])
        n, a = bins_to_action(b)
        recs[(int(r["task_id"]), r["condition"])] = dict(
            bins=b, n=n, a=a, tok=np.array([int(r[f"tok{i}"]) for i in range(7)]),
            g=float(r["gripper_libero"]), instruction=r["instruction"], prompt_tokens=int(r["prompt_tokens"]))
    return recs


def make_demo(seed=5):
    """SENTETİK veri: hattı test etmek için. Gerçek model çıktısı DEĞİLDİR. Bin'ler rastgele; koşullar
    'para küçük sapma, tr/tr2 orta + bazen gripper flip, distractor farklı hedef, empty uzak' varsayımıyla üretilir."""
    rng = np.random.default_rng(seed)
    recs = {}
    for t in range(5):
        base = np.concatenate([rng.integers(60, 200, 6), [rng.choice([0, 254])]])
        def mk(b):
            b = np.clip(b, 0, 254)
            n, a = bins_to_action(b)
            return dict(bins=b, n=n, a=a, tok=VOCAB - b - 1, g=-np.sign(2 * a[6] - 1), instruction="(sentetik)", prompt_tokens=0)
        recs[(t, "en")] = mk(base)
        recs[(t, "para")] = mk(base + np.concatenate([rng.integers(-3, 4, 6), [0]]))
        for c in ("tr", "tr2"):
            b = base + np.concatenate([rng.integers(-25, 26, 6), [0]])
            if rng.random() < 0.4:
                b[6] = 254 - b[6]
            recs[(t, c)] = mk(b)
        recs[(t, "distractor")] = mk(base + np.concatenate([rng.integers(-60, 61, 6), [0]]))
        b = base + np.concatenate([rng.integers(-80, 81, 6), [0]])
        if rng.random() < 0.6:
            b[6] = 254 - b[6]
        recs[(t, "empty")] = mk(b)
    return recs


def summarize(recs):
    rows = []
    tasks = sorted({t for t, _ in recs})
    for t in tasks:
        if (t, "en") not in recs:
            continue
        for c in CONDS:
            if (t, c) in recs:
                rows.append(dict(task_id=t, pair=c, **pair_metrics(recs[(t, "en")], recs[(t, c)])))
        if (t, "tr") in recs and (t, "tr2") in recs:
            rows.append(dict(task_id=t, pair="tr_vs_tr2", **pair_metrics(recs[(t, "tr")], recs[(t, "tr2")])))
    return rows


def table(rows, metrics=("l2_n6", "cos_n6", "l2_a7", "tok_mismatch", "max_bin", "within5", "grip_agree")):
    pairs = [p for p in CONDS + ["tr_vs_tr2"] if any(r["pair"] == p for r in rows)]
    head = "| çift (ref=en) | n | " + " | ".join(metrics) + " |\n|---|---|" + "---|" * len(metrics) + "\n"
    body = ""
    for p in pairs:
        sub = [r for r in rows if r["pair"] == p]
        cells = []
        for m in metrics:
            v = np.array([r[m] for r in sub], dtype=float)
            if m == "grip_agree":
                cells.append(f"{int(v.sum())}/{len(v)}")
            elif m in ("tok_mismatch", "max_bin"):
                cells.append(f"{v.mean():.1f}")
            else:
                cells.append(f"{np.nanmean(v):.3f} ± {np.nanstd(v):.3f}")
        body += f"| {'tr→tr2' if p == 'tr_vs_tr2' else 'en→' + p} | {len(sub)} | " + " | ".join(cells) + " |\n"
    return head + body


def tests(rows, metric="l2_n6"):
    """tr vs para, tr vs empty (görev eşli). n=5'te kesin iki yönlü Wilcoxon'un en küçük p'si 0.0625: anlamlılık İMKÂNSIZ,
    yalnızca etki yönü/büyüklüğü raporlanır."""
    out = []
    by = {(r["task_id"], r["pair"]): r[metric] for r in rows}
    tasks = sorted({t for t, _ in by})
    for a, b in (("tr", "para"), ("tr", "empty"), ("tr", "distractor"), ("tr2", "tr")):
        x = np.array([by[(t, a)] for t in tasks if (t, a) in by and (t, b) in by])
        y = np.array([by[(t, b)] for t in tasks if (t, a) in by and (t, b) in by])
        if len(x) < 2:
            continue
        d = x - y
        line = f"{metric}: {a} − {b}: ortalama fark {d.mean():+.3f} (n={len(d)}, {int((d > 0).sum())}/{len(d)} görevde {a} > {b})"
        if wilcoxon is not None and np.any(d != 0):
            try:
                p = wilcoxon(d, method="exact").pvalue
                line += f", Wilcoxon kesin iki yönlü p = {p:.4f} (n={len(d)} için min 0.0625)"
            except Exception as e:  # noqa
                line += f", Wilcoxon hesaplanamadı ({e})"
        out.append(line)
    return out


def plot(rows, out_png, title_note):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = [p for p in CONDS + ["tr_vs_tr2"] if any(r["pair"] == p for r in rows)]
    labels = {"para": "EN parafraz", "tr": "TR", "tr2": "TR-2", "distractor": "çeldirici (öteki kâse)", "empty": "boş", "tr_vs_tr2": "TR→TR-2"}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    panels = [("l2_n6", "L2, 6 boyut, normalize [-1,1]"), ("cos_n6", "cosine, 6 boyut, normalize"),
              ("tok_mismatch", "farklı token sayısı (0–7)"), ("grip_agree", "gripper uyumu (görev sayısı)")]
    for ax, (m, lab) in zip(axes, panels):
        xs = np.arange(len(pairs))
        for i, p in enumerate(pairs):
            v = np.array([r[m] for r in rows if r["pair"] == p], dtype=float)
            if m == "grip_agree":
                ax.bar(i, v.sum(), color=COLORS[p], edgecolor="#333")
                ax.set_ylim(0, max(5, len(v)))
            else:
                ax.bar(i, np.nanmean(v), color=COLORS[p], edgecolor="#333")
                ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, color="#333", s=18, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([labels[p] for p in pairs], rotation=30, ha="right", fontsize=8)
        ax.set_title(lab, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Ders 15 — koşul başına metrikler (referans: EN; nokta = görev)\n{title_note}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print("görsel:", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", default="results/actions.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--demo", action="store_true", help="SENTETİK veriyle hattı test et (gerçek model çıktısı değildir)")
    ap.add_argument("--png", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.demo:
        recs = make_demo()
        note = "TEMSİLİ (SENTETİK) VERİ — yalnızca analiz hattının testi; gerçek sayılar Mac'te üretilecek"
    else:
        recs = read_actions(args.actions)
        note = f"gerçek koşu: {args.actions}"
    rows = summarize(recs)
    with open(os.path.join(args.out, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md = table(rows)
    tst = tests(rows)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(("**" + note + "**\n\n") + md + "\n" + "\n".join("- " + t for t in tst) + "\n")
    print(note + "\n")
    print(md)
    print("\n".join(tst))
    plot(rows, args.png or os.path.join(args.out, "metrics.png"), note)


if __name__ == "__main__":
    main()
```

## plot_day15.py

```python
"""Ders 15 görselleri 02, 03, 05 (matplotlib).
  02-metric-geometry.png : libero_spatial q01/q99 (GERÇEK, dataset_statistics.json) → bir bin ve bir gripper flip'inin
                           L2'ye katkısı, normalize uzay vs gerçek birim
  03-token-counts.png    : commands.csv'nin LLaMA-2 token sayıları (GERÇEK, count_tokens.mjs çıktısı token_counts.json)
  05-literature-gap.png  : Chen vd. 2026 (arXiv:2606.15714) Tablo 1 LIBERO göreli düşüşler + Dong vd. 2026 (arXiv:2606.11906)
                           Tablo 1 LIBERO-Spatial mutlak başarı (GERÇEK makale değerleri)
    python plot_day15.py --tokens token_counts.json --out ../../lectures/images/lecture-15
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
Q01 = np.array([-0.7454732114076613, -0.6616071462631226, -0.9375, -0.1071428582072258, -0.20678570866584778, -0.1842857152223587])
Q99 = np.array([0.9375, 0.8758928775787354, 0.9321428537368774, 0.1039285734295845, 0.17678570747375488, 0.14571428298950195])
DIMS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def fig02(out):
    one_bin_norm = 2 / 255
    one_bin_real = (Q99 - Q01) / 255
    full_norm = np.full(6, 2.0)                 # bin 0 → 254 : -0.996 → +0.996
    full_real = (Q99 - Q01) * (254 / 255)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    x = np.arange(7)
    ax = axes[0]
    ax.bar(x - 0.2, np.r_[full_norm, 1.992], 0.4, color=Y, edgecolor="#333", label="normalize uzay [-1,1] (bin merkezi)")
    ax.bar(x + 0.2, np.r_[full_real, 1.992], 0.4, color=B, edgecolor="#333", label="gerçek birim (q01/q99 ile açılmış)")
    for i in range(7):
        ax.text(i + 0.2, (np.r_[full_real, 1.992])[i] + 0.03, f"{(np.r_[full_real, 1.992])[i]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(DIMS)
    ax.set_ylabel("tek boyutun L2'ye EN BÜYÜK katkısı")
    ax.set_title("Bir boyut uçtan uca değişirse L2'ye ne katar?\n(libero_spatial q01/q99 — gerçek değerler)", fontsize=10)
    ax.set_ylim(0, 2.6)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    ax.bar(x[:6] - 0.2, np.full(6, one_bin_norm), 0.4, color=Y, edgecolor="#333", label="1 bin, normalize (2/255 = 0.0078)")
    ax.bar(x[:6] + 0.2, one_bin_real, 0.4, color=B, edgecolor="#333", label="1 bin, gerçek birim ((q99−q01)/255)")
    for i in range(6):
        ax.text(i + 0.2, one_bin_real[i] + 0.0002, f"{one_bin_real[i]:.4f}", ha="center", fontsize=8)
    ax.set_xticks(x[:6])
    ax.set_xticklabels(DIMS[:6])
    ax.set_title("Bir bin'lik fark ne kadar?\n(dönüş eksenleri gerçek birimde 5–8× daha ince)", fontsize=10)
    ax.set_ylim(0, 0.0105)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Metrik geometrisi: gripper tek başına L2 ≈ 2.0 katar; 'hangi uzayda ölçtün' yazılmazsa sayılar karşılaştırılamaz",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "02-metric-geometry.png"), dpi=150)


def fig03(tokens_json, out):
    d = json.load(open(tokens_json, encoding="utf-8"))
    rows = d["rows"]
    conds = ["en", "para", "tr", "tr2", "distractor", "empty"]
    labels = {"en": "EN", "para": "EN parafraz", "tr": "TR", "tr2": "TR-2", "distractor": "çeldirici", "empty": "boş"}
    colors = {"en": B, "para": B, "tr": Y, "tr2": Y, "distractor": R, "empty": R}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    w = 0.13
    for j, c in enumerate(conds):
        v = [next(r["prompt_tokens_with_bos"] for r in rows if r["task_id"] == t and r["condition"] == c) for t in range(5)]
        ax.bar(np.arange(5) + (j - 2.5) * w, v, w, color=colors[c], edgecolor="#333", label=labels[c])
        for t, val in enumerate(v):
            ax.text(t + (j - 2.5) * w, val + 0.4, str(val), ha="center", fontsize=6.5)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"görev {t}" for t in range(5)])
    ax.set_ylabel("prompt token sayısı (BOS dahil)")
    ax.set_title("Prompt uzunluğu: 'In: What action … to {komut}?\\nOut:' (LLaMA-2 tokenizer)", fontsize=10)
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 62)
    ax = axes[1]
    ph = [p for p in d["phrases"] if p["kind"] in ("en", "tr", "tr2")]
    en = [p for p in ph if p["kind"] == "en"]
    tr = [p for p in ph if p["kind"] == "tr"]
    tr2 = [p for p in ph if p["kind"] == "tr2"]
    xs = np.arange(5)
    ax.bar(xs - 0.27, [p["n"] for p in en], 0.27, color=B, edgecolor="#333", label="EN ifade")
    ax.bar(xs, [p["n"] for p in tr], 0.27, color=Y, edgecolor="#333", label="TR (-daki)")
    ax.bar(xs + 0.27, [p["n"] for p in tr2], 0.27, color="#f3c86a", edgecolor="#333", label="TR-2 (duran)")
    for i in range(5):
        ax.text(xs[i] - 0.27, en[i]["n"] + 0.2, str(en[i]["n"]), ha="center", fontsize=8)
        ax.text(xs[i], tr[i]["n"] + 0.2, str(tr[i]["n"]), ha="center", fontsize=8)
        ax.text(xs[i] + 0.27, tr2[i]["n"] + 0.2, str(tr2[i]["n"]), ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(["between…\narasındaki", "next to…\nyanındaki", "table center\nortasındaki", "on the box\nüstündeki",
                        "in the drawer\nçekmecesindeki"], fontsize=8)
    ax.set_ylabel("yalnızca mekânsal ifadenin token sayısı")
    ax.set_title("Mekânsal ifade tek başına: TR 1.6–3×\n(ör. 'çekmecesindeki' = ç|ek|m|ec|es|inde|ki)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Gerçek token sayıları — commands.csv × llama-tokenizer-js (Mac'te HF tokenizer ile doğrula)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "03-token-counts.png"), dpi=150)


def fig05(out):
    # Chen vd. 2026, arXiv:2606.15714, Tablo 1 — LIBERO ortalaması, İngilizce'ye göre puan düşüşü
    langs = ["zh", "fr", "ru", "ar", "komut yok"]
    oft = [-28.6, -28.6, -27.8, -29.8, -27.5]
    pi05 = [-39.6, -38.0, -41.8, -41.4, -43.5]
    # Dong vd. 2026, arXiv:2606.11906, Tablo 1 — LIBERO-Spatial mutlak başarı (%)
    langs2 = ["EN", "zh", "ja", "ko", "fr", "es", "ar", "th", "pt", "vi"]
    oft2 = [97.6, 55.8, 56.8, 49.2, 64.0, 63.8, 52.8, 53.0, 62.6, 59.2]
    pi2 = [97.8, 67.8, 70.0, 69.8, 72.8, 73.6, 68.0, 68.6, 75.4, 69.6]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    x = np.arange(len(langs))
    ax.bar(x - 0.2, oft, 0.4, color=[B] * 4 + [R], edgecolor="#333", label="OpenVLA-OFT")
    ax.bar(x + 0.2, pi05, 0.4, color=[Y] * 4 + [R], edgecolor="#333", hatch="//", label="π0.5")
    for i in range(5):
        ax.text(i - 0.2, oft[i] - 1.5, f"{oft[i]}", ha="center", fontsize=8)
        ax.text(i + 0.2, pi05[i] - 1.5, f"{pi05[i]}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.set_ylabel("başarı farkı (puan, EN'ye göre)")
    ax.set_ylim(-50, 0)
    ax.set_title("Chen vd. 2026 (arXiv:2606.15714) Tablo 1, LIBERO ort.\nOpenVLA-OFT: yabancı dil ≈ 'komut yok' tabanı", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    x = np.arange(len(langs2))
    ax.bar(x - 0.2, oft2, 0.4, color=B, edgecolor="#333", label="OpenVLA-OFT")
    ax.bar(x + 0.2, pi2, 0.4, color=Y, edgecolor="#333", label="π0.5")
    for i in range(len(langs2)):
        ax.text(i - 0.2, oft2[i] + 1, f"{oft2[i]:.0f}", ha="center", fontsize=7)
        ax.text(i + 0.2, pi2[i] + 1, f"{pi2[i]:.0f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(langs2)
    ax.set_ylim(0, 110)
    ax.set_ylabel("LIBERO-Spatial başarı (%)")
    ax.set_title("Dong vd. 2026 (arXiv:2606.11906) Tablo 1, LIBERO-Spatial\n9 dilde 22–48 puan düşüş; Türkçe yok", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Haziran 2026: çok dilli VLA boşluğu iki bağımsız çalışmada ölçüldü — ikisinde de metrik success rate, Türkçe yok",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "05-literature-gap.png"), dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="token_counts.json")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig02(args.out)
    if os.path.exists(args.tokens):
        fig03(args.tokens, args.out)
    else:
        print("token_counts.json yok; önce count_tokens.mjs çalıştır")
    fig05(args.out)
    print("yazıldı:", args.out)


if __name__ == "__main__":
    main()
```
