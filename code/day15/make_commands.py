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
