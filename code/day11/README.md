# day11

## inspect_hdf5.py

```python
"""Gün 11 — LIBERO demo HDF5 dosyasını incele.
Kullanım:  python inspect_hdf5.py <dosya.hdf5> [--demo 0]
Çıktı: yapı dökümü, episode uzunlukları, action istatistikleri, action grafiği (actions_demoN.png),
       ilk/orta/son kare şeridi (frames_demoN.png).
"""
import sys, json, argparse, numpy as np, h5py, imageio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

p = argparse.ArgumentParser(); p.add_argument("path"); p.add_argument("--demo", type=int, default=0)
args = p.parse_args()

with h5py.File(args.path, "r") as f:
    data = f["data"]
    print("== Dosya düzeyi attr'lar")
    for k in data.attrs:
        v = data.attrs[k]; v = v if len(str(v)) < 120 else str(v)[:117] + "..."
        print(f"  {k}: {v}")
    if "problem_info" in data.attrs:
        print("  komut:", json.loads(data.attrs["problem_info"])["language_instruction"])
    demos = sorted(data.keys(), key=lambda s: int(s.split("_")[1]))
    lens = [data[d]["actions"].shape[0] for d in demos]
    print(f"\n== {len(demos)} episode | uzunluk min/ort/max = {min(lens)}/{np.mean(lens):.0f}/{max(lens)}")

    d = data[f"demo_{args.demo}"]
    print(f"\n== demo_{args.demo} yapısı")
    def show(name, obj):
        if isinstance(obj, h5py.Dataset): print(f"  {name:28s} {obj.shape} {obj.dtype}")
    d.visititems(show)

    A = d["actions"][:]                      # (T, 7)
    print("\n== action istatistikleri (T=%d)" % len(A))
    for i, n in enumerate(["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]):
        print(f"  {n:8s} min {A[:,i].min():+.3f} max {A[:,i].max():+.3f} ort {A[:,i].mean():+.3f}")
    print("  gripper benzersiz değerler:", np.unique(A[:, 6]))

    T = len(A); fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    for i, n in enumerate(["Δx", "Δy", "Δz"]): ax[0].plot(A[:, i], label=n)
    ax[0].legend(); ax[0].set_ylabel("delta (normalize)"); ax[0].grid(alpha=.3)
    ax[0].set_title(f"demo_{args.demo}: insan teleoperasyonu — {T} adım")
    ax[1].step(range(T), A[:, 6], where="post", color="tab:red"); ax[1].set_ylabel("gripper"); ax[1].set_xlabel("t"); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"actions_demo{args.demo}.png", dpi=150)

    imgs = d["obs"]["agentview_rgb"]        # (T, H, W, 3)
    idx = [0, T // 2, T - 1]
    strip = np.concatenate([imgs[i][::-1, ::-1] for i in idx], axis=1)   # 180° döndür
    imageio.imwrite(f"frames_demo{args.demo}.png", strip)
    print(f"\nkaydedildi: actions_demo{args.demo}.png, frames_demo{args.demo}.png")
```

