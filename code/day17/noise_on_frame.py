"""Ders 17 — LIBERO-Plus 'Sensor Noise': env_wrapper.py'deki beş bozulmayı bir kareye uygular.

Fonksiyonlar LIBERO-plus `libero/libero/envs/env_wrapper.py` (4976dc3) satır 30-174'ten, PARAMETRE TABLOLARI
BİREBİR aktarılarak yeniden yazıldı (dış depodan kod çalıştırmamak için; algoritmalar aynı, Hendrycks & Dietterich
ImageNet-C'nin bilinen uygulamaları). motion_blur için `wand` (ImageMagick) gerekir; yoksa o sütun atlanır.

Kullanım:  python noise_on_frame.py <kare.png> <çıktı.png> [--size 128] [--severity 1,5,10] [--seed 0]
Kare: Ders 10/14'teki gibi gerçek bir agentview karesi (LIBERO-Plus eval'de 128x128 kullanılır; --size ile ölçeklenir).
"""
import argparse, pathlib, sys
from io import BytesIO
import numpy as np
from PIL import Image
from scipy.ndimage import zoom as scizoom
from skimage.filters import gaussian

# --- env_wrapper.py:33-44  motion_blur: (radius, sigma), açı U(-45,45)
MOTION = [(5, 2), (8, 3), (10, 4), (12, 5), (15, 6), (18, 8), (20, 10), (25, 12), (30, 15), (35, 20)]
# --- env_wrapper.py:61  gaussian_blur: sigma
GAUSS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# --- env_wrapper.py:79-88  zoom_blur: zoom faktörleri
ZOOM = [np.arange(1, 1.11, 0.01), np.arange(1, 1.16, 0.01), np.arange(1, 1.21, 0.02), np.arange(1, 1.26, 0.02), np.arange(1, 1.31, 0.03),
        np.arange(1, 1.36, 0.01), np.arange(1, 1.41, 0.01), np.arange(1, 1.46, 0.02), np.arange(1, 1.51, 0.02), np.arange(1, 1.56, 0.03)]
# --- env_wrapper.py:148  fog: (yoğunluk alpha, wibbledecay)
FOG = [(0.5, 3), (1.0, 2.8), (1.5, 2.5), (2.0, 2.2), (2.5, 2.0), (3.0, 1.8), (3.5, 1.6), (4.0, 1.5), (4.5, 1.4), (5.0, 1.3)]
# --- env_wrapper.py:159  glass_blur: (sigma, max_delta, iterations)
GLASS = [(0.5, 1, 3), (0.7, 1, 3), (0.9, 2, 3), (1.0, 2, 2), (1.1, 3, 2), (1.3, 3, 2), (1.5, 4, 2), (1.8, 4, 2), (2.2, 5, 1), (2.5, 5, 1)]


def motion_blur(x, severity=1):
    from wand.image import Image as WandImage  # ImageMagick
    r, s = MOTION[severity - 1]
    buf = BytesIO(); x.save(buf, format="PNG")
    with WandImage(blob=buf.getvalue()) as w:
        w.motion_blur(radius=r, sigma=s, angle=float(np.random.uniform(-45, 45)))
        out = np.array(w)[..., :3]
    return np.clip(out, 0, 255).astype(np.uint8)


def gaussian_blur(x, severity=1):
    return (np.clip(gaussian(np.array(x) / 255.0, sigma=GAUSS[severity - 1], channel_axis=-1), 0, 1) * 255).astype(np.uint8)


def clipped_zoom(img, zf):  # env_wrapper.py:66-76
    h = img.shape[0]; ch = int(np.ceil(h / float(zf))); top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zf, zf, 1), order=1)
    trim = (img.shape[0] - h) // 2
    return img[trim:trim + h, trim:trim + h]


def zoom_blur(x, severity=1):
    c = ZOOM[severity - 1]; x = (np.array(x) / 255.0).astype(np.float32); out = np.zeros_like(x)
    for zf in c: out += clipped_zoom(x, zf)
    return (np.clip((x + out) / (len(c) + 1), 0, 1) * 255).astype(np.uint8)


def plasma_fractal(mapsize=256, wibbledecay=3):  # env_wrapper.py:98-144 (diamond-square)
    maparray = np.empty((mapsize, mapsize), dtype=np.float64); maparray[0, 0] = 0; stepsize = mapsize; wibble = 100
    def wibbledmean(a): return a / 4 + wibble * np.random.uniform(-wibble, wibble, a.shape)
    def fillsquares():
        c = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]; s = c + np.roll(c, -1, 0); s += np.roll(s, -1, 1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(s)
    def filldiamonds():
        m = maparray.shape[0]; dr = maparray[stepsize // 2:m:stepsize, stepsize // 2:m:stepsize]; ul = maparray[0:m:stepsize, 0:m:stepsize]
        maparray[0:m:stepsize, stepsize // 2:m:stepsize] = wibbledmean(dr + np.roll(dr, 1, 0) + ul + np.roll(ul, -1, 1))
        maparray[stepsize // 2:m:stepsize, 0:m:stepsize] = wibbledmean(dr + np.roll(dr, 1, 1) + ul + np.roll(ul, -1, 0))
    while stepsize >= 2:
        fillsquares(); filldiamonds(); stepsize //= 2; wibble /= wibbledecay
    maparray -= maparray.min(); return maparray / maparray.max()


def fog(x, severity=1):
    c = FOG[severity - 1]; x = np.array(x) / 255.0; mx = x.max(); h, w = x.shape[:2]
    x = x + c[0] * plasma_fractal(wibbledecay=c[1])[:h, :w][..., None]
    return (np.clip(x * mx / (mx + c[0]), 0, 1) * 255).astype(np.uint8)


def glass_blur(x, severity=1):
    sg, dmax, iters = GLASS[severity - 1]
    x = np.uint8(gaussian(np.array(x) / 255.0, sigma=sg, channel_axis=-1) * 255); h, w = x.shape[:2]
    for _ in range(iters):
        for hh in range(h - dmax, dmax, -1):
            for ww in range(w - dmax, dmax, -1):
                dx, dy = np.random.randint(-dmax, dmax, size=(2,)); hp, wp = hh + dy, ww + dx
                x[hh, ww], x[hp, wp] = x[hp, wp], x[hh, ww]
    return (np.clip(gaussian(x / 255.0, sigma=sg, channel_axis=-1), 0, 1) * 255).astype(np.uint8)


FAMILIES = [("motion_blur (N=1..10)", 1, motion_blur), ("gaussian_blur (N=11..20)", 11, gaussian_blur),
            ("zoom_blur (N=21..30)", 21, zoom_blur), ("fog (N=31..40)", 31, fog), ("glass_blur (N=41..50)", 41, glass_blur)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame"); ap.add_argument("out")
    ap.add_argument("--size", type=int, default=128); ap.add_argument("--severity", default="1,5,10"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); np.random.seed(a.seed)
    try:
        import wand.image  # noqa
        have_wand = True
    except Exception as e:
        print("UYARI: wand/ImageMagick yok, motion_blur atlanacak:", e, file=sys.stderr); have_wand = False
    img = Image.open(a.frame).convert("RGB")
    if img.size != (a.size, a.size):
        side = min(img.size)
        img = img.crop(((img.width - side) // 2, (img.height - side) // 2, (img.width + side) // 2, (img.height + side) // 2)).resize((a.size, a.size), Image.LANCZOS)
    sev = [int(s) for s in a.severity.split(",")]
    fams = [f for f in FAMILIES if have_wand or f[2] is not motion_blur]

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(sev), len(fams) + 1, figsize=(2.3 * (len(fams) + 1), 2.45 * len(sev)))
    axes = np.atleast_2d(axes)
    for r, s in enumerate(sev):
        axes[r, 0].imshow(img); axes[r, 0].set_ylabel(f"severity {s}", fontsize=10)
        if r == 0: axes[r, 0].set_title("orijinal", fontsize=9)
        for c, (label, base_code, fn) in enumerate(fams, start=1):
            axes[r, c].imshow(fn(img, severity=s))
            if r == 0: axes[r, c].set_title(label, fontsize=8.5)
            axes[r, c].text(0.03, 0.03, f"_noise_{base_code + s - 1}", transform=axes[r, c].transAxes, fontsize=7.5, color="#fff",
                            bbox=dict(facecolor="#000", alpha=0.5, pad=1.5, lw=0))
    for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"LIBERO-Plus 'Sensor Noise': env_wrapper.py'nin 5 bozulması ({a.size}×{a.size} agentview karesi; parametre tabloları koddan) — noise_on_frame.py"
                 + ("" if have_wand else "  [motion_blur: wand yok, atlandı]"), fontsize=9.5)
    fig.text(0.5, 0.005, "yalnızca agentview_image değişir; bilek (eye-in-hand) kamerası ve proprio dokunulmaz (env_wrapper.py:288-308). Kare: " + pathlib.Path(a.frame).name,
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95)); fig.savefig(a.out, dpi=170); plt.close(fig)
    print("yazıldı:", a.out, "| wand:", have_wand, "| aileler:", [f[2].__name__ for f in fams])


if __name__ == "__main__":
    main()
