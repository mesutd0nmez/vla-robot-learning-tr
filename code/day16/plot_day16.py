"""Ders 16 görselleri (02, 04, 05, 06).

Kullanım:  python plot_day16.py <çıktı_klasörü/> [--plus <LIBERO-plus kökü>] [--probe probe.csv]
  --plus verilirse 02 ve 04 sayıları task_classification.json'dan YENİDEN hesaplanır; verilmezse aşağıdaki gömülü
  sayılar kullanılır (LIBERO-plus 4976dc3 klonundan bu script'in --plus moduyla alındı).
  --probe: bddl_lang_probe.py --csv çıktısı (05 için).
06'daki sayılar LIBERO-plus README liderlik tablosundandır (elle aktarıldı).
"""
import argparse, csv, json, os, pathlib
from collections import Counter, defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--plus"); ap.add_argument("--probe")
a = ap.parse_args(); out = a.out.rstrip("/") + "/"; os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
CATS = ["Camera Viewpoints", "Robot Initial States", "Language Instructions", "Light Conditions",
        "Background Textures", "Sensor Noise", "Objects Layout"]
SHORT = ["Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout"]
COUNTS = {  # suite -> kategori sırası CATS
    "libero_spatial": [376, 350, 390, 292, 258, 351, 385],
    "libero_object": [396, 398, 354, 297, 248, 422, 403],
    "libero_goal": [408, 409, 410, 279, 281, 379, 425],
    "libero_10": [419, 393, 383, 274, 289, 449, 312],
}
LANG_DIFF = {  # suite -> Language kategorisinde zorluk 1..5 sayıları
    "libero_spatial": [115, 100, 58, 82, 35], "libero_object": [103, 67, 72, 63, 49],
    "libero_goal": [45, 49, 52, 79, 185], "libero_10": [65, 64, 77, 104, 73],
}
if a.plus:
    d = json.loads((pathlib.Path(a.plus) / "libero/libero/benchmark/task_classification.json").read_text(encoding="utf-8"))
    COUNTS = {s: [Counter(x["category"] for x in d[s])[c] for c in CATS] for s in SUITES}
    LANG_DIFF = {}
    for s in SUITES:
        c = Counter(x["difficulty_level"] for x in d[s] if x["category"] == "Language Instructions")
        LANG_DIFF[s] = [c[i] for i in range(1, 6)]
    print("yeniden hesaplandı:", COUNTS, LANG_DIFF)

# --- 02: 10.030 görev, suite × kategori
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(SUITES)); bottom = np.zeros(len(SUITES))
cols = ["#9ecae1", "#c6dbef", "#ffd166", "#fdd0a2", "#d9d9d9", "#e5c3e8", "#c7e9c0"]
for j, (name, c) in enumerate(zip(SHORT, cols)):
    v = np.array([COUNTS[s][j] for s in SUITES])
    ax.bar(x, v, 0.62, bottom=bottom, color=c, edgecolor="white", label=name, zorder=3)
    for xi, (b, val) in enumerate(zip(bottom, v)):
        ax.text(xi, b + val / 2, str(val), ha="center", va="center", fontsize=8.5,
                fontweight="bold" if name == "Language" else None)
    bottom += v
for xi, t in enumerate(bottom):
    ax.text(xi, t + 25, f"{int(t)}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(["Spatial", "Object", "Goal", "Long (10)"])
ax.set_ylabel("görev sayısı"); ax.set_ylim(0, 2800); ax.yaxis.grid(True, color="#eee", zorder=0)
ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=9)
ax.set_title(f"LIBERO-Plus: {int(bottom.sum())} görev = 4 suite × 7 bozulma — kaynak: task_classification.json (LIBERO-plus 4976dc3)",
             fontsize=10)
fig.text(0.5, 0.005, "Her görev tek bir bozulma örneğidir ve 1 episode koşulur (num_trials_per_task: 1). "
         "Bozulmamış orijinal görev bu listede YOK.", ha="center", fontsize=8.5, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out + "02-task-map.png", dpi=170); plt.close(fig)

# --- 04: Language görevlerinde zorluk dağılımı (zorluk = 4 modelden kaçının çözdüğü)
fig, ax = plt.subplots(figsize=(11, 4.6))
lv_cols = ["#d4f5d4", "#e8f5c8", "#fff3c4", "#ffe0c2", "#ffd6d6"]
labels = ["L1: 4 modelin 4'ü çözdü", "L2: 3'ü", "L3: 2'si", "L4: 1'i", "L5: hiçbiri"]
y = np.arange(len(SUITES))[::-1]; left = np.zeros(len(SUITES))
tot = np.array([sum(LANG_DIFF[s]) for s in SUITES], dtype=float)
for k in range(5):
    v = np.array([LANG_DIFF[s][k] for s in SUITES]) / tot * 100
    ax.barh(y, v, 0.6, left=left, color=lv_cols[k], edgecolor="#999", linewidth=0.5, label=labels[k])
    for yi, l, val, n in zip(y, left, v, [LANG_DIFF[s][k] for s in SUITES]):
        if val > 4:
            ax.text(l + val / 2, yi, f"{val:.0f}%\n({n})", ha="center", va="center", fontsize=8)
    left += v
ax.set_yticks(y); ax.set_yticklabels([f"{n} ({int(t)} görev)" for n, t in zip(["Spatial", "Object", "Goal", "Long"], tot)])
ax.set_xlim(0, 100); ax.set_xlabel("Language görevlerinin yüzdesi")
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, fontsize=8.5)
ax.set_title("Language bozulması: zorluk seviyesi dağılımı — kaynak: task_classification.json; seviye tanımı: arXiv:2510.13626 Ek C.3",
             fontsize=9.5)
fig.text(0.5, 0.005, "Zorluk görevin kendisinin değil, 4 modelin (OpenVLA-OFT, π0, π0-fast, UniVLA) sonucudur: "
         "Goal'da Language görevlerinin %45'ini hiçbiri çözemedi.", ha="center", fontsize=8.5, color="#555")
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(out + "04-language-difficulty.png", dpi=170); plt.close(fig)

# --- 05: Türkçe komut BDDL ayrıştırıcısından geçince (gerçek bddl 1.0.1 çıktısı)
if a.probe:
    rows = list(csv.DictReader(open(a.probe, encoding="utf-8")))
    import textwrap
    wr = lambda t, n: "\n".join(textwrap.wrap(t, n)) or t
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(rows) + 1.0)); ax.axis("off")
    cell = [[r["durum"], wr(r["girdi"] or "(boş)", 44), wr(r["modele_giden"] or "—", 44), wr(r["not"], 34)] for r in rows]
    tb = ax.table(cellText=cell, colLabels=["durum", "BDDL'e yazılan", "task.language (modele giden)", "not"],
                  loc="center", cellLoc="left", colWidths=[0.13, 0.33, 0.33, 0.27])
    tb.auto_set_font_size(False); tb.set_fontsize(8.6); tb.scale(1, 2.6)
    for (r, c), cl in tb.get_celld().items():
        cl.set_edgecolor("#bbb")
        if r == 0:
            cl.set_facecolor("#cfe8ff"); cl.set_text_props(fontweight="bold")
        elif c == 3:
            t = rows[r - 1]["not"]
            cl.set_facecolor("#d4f5d4" if t == "aynı" else "#ffd6d6")
        elif c == 2:
            cl.set_facecolor("#fffbea")
    ax.set_title("bddl.parsing.scan_tokens: ';' sonrası yorum, str.lower(), parantez = yapı — gerçek çıktı (bddl 1.0.1, bddl_lang_probe.py)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(out + "05-bddl-turkish-probe.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# --- 06: LIBERO-Plus liderlik tablosu: Language sütunu vs diğer 6 bozulmanın ortalaması (README)
LB = {  # model: Camera, Robot, Language, Light, Background, Noise, Layout, Total
    "OpenVLA": [0.8, 3.5, 23.0, 8.1, 34.8, 15.2, 28.5, 15.6],
    "OpenVLA-OFT": [56.4, 31.9, 79.5, 88.7, 93.3, 75.8, 74.2, 69.6],
    "OpenVLA-OFT_w": [10.4, 38.7, 70.5, 76.8, 93.6, 49.9, 69.9, 55.8],
    "NORA": [2.2, 37.0, 65.1, 45.7, 58.6, 12.8, 62.1, 39.0],
    "WorldVLA": [0.1, 27.9, 41.6, 43.7, 17.1, 10.9, 38.0, 25.0],
    "UniVLA": [1.8, 46.2, 69.6, 69.0, 81.0, 21.2, 31.9, 43.9],
    "π0": [13.8, 6.0, 58.8, 85.0, 81.4, 79.0, 68.9, 53.6],
    "π0-Fast": [65.1, 21.6, 61.0, 73.2, 73.2, 74.4, 68.8, 61.6],
    "RIPT-VLA": [55.2, 31.2, 77.6, 88.4, 91.6, 73.5, 74.2, 68.4],
    "OpenVLA-OFT_m": [55.6, 21.7, 81.0, 92.7, 91.0, 78.6, 68.7, 67.9],
    "OpenVLA-OFT+": [92.8, 30.3, 85.8, 94.9, 93.9, 89.3, 77.6, 79.6],
}
fig, ax = plt.subplots(figsize=(10.5, 4.8))
names = list(LB); lang = np.array([LB[n][2] for n in names]); others = np.array([np.mean(LB[n][:2] + LB[n][3:7]) for n in names])
rank = np.array([1 + sum(v > LB[n][2] for v in LB[n][:2] + LB[n][3:7]) for n in names])
order = np.argsort(lang)
yy = np.arange(len(names))
ax.hlines(yy, others[order], lang[order], color="#bbb", zorder=1)
ax.scatter(others[order], yy, color="#cfe8ff", edgecolor="#3d7fc4", s=60, zorder=3, label="diğer 6 bozulmanın ortalaması")
ax.scatter(lang[order], yy, color="#ffd166", edgecolor="#b07d00", s=70, zorder=3, label="Language bozulması")
for k, i in enumerate(order):
    ax.text(102, k, f"{rank[i]}.", va="center", fontsize=8.5, color="#555")
ax.text(2, -0.9, "sağ sütundaki sayı: Language, 7 bozulma içinde kaçıncı en kolay (1 = en kolay)", fontsize=8, color="#555", va="center")
ax.set_yticks(yy); ax.set_yticklabels([names[i] for i in order]); ax.set_xlim(0, 110)
ax.set_xlabel("success rate (%) — 4 suite ortalaması"); ax.xaxis.grid(True, color="#eee", zorder=0)
ax.set_ylim(-1.6, len(names) - 0.4)
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.set_title("LIBERO-Plus: komut yeniden yazılınca başarı, diğer bozulmalardan daha AZ düşüyor\nkaynak: LIBERO-plus README liderlik tablosu",
             fontsize=10)
fig.tight_layout(); fig.savefig(out + "06-leaderboard-language.png", dpi=170); plt.close(fig)
print("ok")
