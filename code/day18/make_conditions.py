"""Ders 18 — datasets/01-libero-spatial-tr-mini/data/commands.csv'den bir görevin koşul JSON'unu üret.

  python make_conditions.py --task 0                      # → conditions_task0.json (clean/empty/wrong/tr)
  python make_conditions.py --task 0 --extra para,tr2     # 6 koşul: + EN parafraz + ikinci Türkçe yüzey formu
  python make_conditions.py --task 3 --csv ../../datasets/01-libero-spatial-tr-mini/data/commands.csv

Çıktı VLANEXT_CONDITIONS'ın beklediği biçimdir: {"ad": komut | null}. null = task.language (eval kodu doldurur;
böylece "clean" koşulunun metni dosyadan değil benchmark'tan gelir ve bir yazım hatası clean'i bozamaz).
Eşleme: clean ← en (yalnızca kontrol için yazılır, JSON'da null), empty ← empty (""), wrong ← distractor, tr ← tr.
Model / GPU gerekmez.
"""
import argparse, csv, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--task", type=int, default=0)
ap.add_argument("--csv", default=str(HERE / "../../datasets/01-libero-spatial-tr-mini/data/commands.csv"))
ap.add_argument("--extra", default="", help="virgülle: para,tr2 (commands.csv'deki condition adları)")
ap.add_argument("--out", default=None)
a = ap.parse_args()

rows = {r["condition"]: r for r in csv.DictReader(open(a.csv, encoding="utf-8")) if int(r["task_id"]) == a.task}
if not rows:
    sys.exit(f"HATA: commands.csv'de task_id={a.task} yok (0–4 var)")
need = {"en": "clean", "empty": "empty", "distractor": "wrong", "tr": "tr"}
missing = [k for k in need if k not in rows]
if missing:
    sys.exit(f"HATA: görev {a.task} için eksik koşul(lar): {missing}")

conds = {"clean": None, "empty": "", "wrong": rows["distractor"]["instruction"], "tr": rows["tr"]["instruction"]}
for name in [x for x in a.extra.split(",") if x.strip()]:
    if name not in rows:
        sys.exit(f"HATA: --extra {name} commands.csv'de yok; olanlar: {sorted(rows)}")
    conds[name] = rows[name]["instruction"]

for name, text in conds.items():          # Ders 16/17 kuralları: override yolu büyük harfi korur ama kontrol zararsız
    if text and any(ch in text for ch in ";()"):
        print(f"UYARI: {name} koşulunda ';' ya da parantez var — BDDL yoluna geçersen scan_tokens kırılır (Ders 16)")

out = pathlib.Path(a.out or HERE / f"conditions_task{a.task}.json")
out.write_text(json.dumps(conds, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"görev {a.task} — clean (task.language, JSON'da null): {rows['en']['instruction']}")
for k, v in conds.items():
    print(f"  {k:6s} → {v!r}")
print(f"{len(conds)} koşul → {out}")
print(f'Colab: os.environ["VLANEXT_CONDITIONS"] = open("{out.name}").read()')
