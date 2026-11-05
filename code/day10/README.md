# day10

## libero_first_episode.py

```python
"""Gün 10 — İlk LIBERO episode'u.
Bir LIBERO-Spatial görevini açar, sabit başlangıç durumuna getirir, 60 adım basit action uygular,
agentview + bilek kamerası karelerini PNG olarak kaydeder.
Çalıştır (vla ortamı aktif):  python libero_first_episode.py --task 0
"""
import os, argparse, numpy as np, imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

p = argparse.ArgumentParser()
p.add_argument("--suite", default="libero_spatial")   # libero_spatial | libero_object | libero_goal | libero_10
p.add_argument("--task", type=int, default=0)         # 0..9
p.add_argument("--steps", type=int, default=60)
args = p.parse_args()

suite = benchmark.get_benchmark_dict()[args.suite]()
task = suite.get_task(args.task)
print("Görev     :", task.name)
print("Komut     :", task.language)                    # <- VLA'ya gidecek İngilizce instruction
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0)
env.reset()
init_states = suite.get_task_init_states(args.task)   # 50 sabit başlangıç durumu (benchmark protokolü)
obs = env.set_init_state(init_states[0])
print("obs anahtarları:", [k for k in obs.keys() if "image" in k or "eef" in k or "gripper" in k])

frames = []
for t in range(args.steps):
    a = np.zeros(7)
    a[2] = -0.3 if t < 20 else 0.0      # 20 adım aşağı in
    a[6] = -1.0 if t < 30 else 1.0      # gripper: -1 açık, +1 kapalı
    obs, reward, done, info = env.step(a)
    if t % 20 == 0 or t == args.steps - 1:
        # LIBERO görüntüleri ters gelir; OpenVLA 180° döndürür ([::-1, ::-1])
        top = obs["agentview_image"][::-1, ::-1]
        wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1]
        frames.append(np.concatenate([top, wrist], axis=0))
print("Başarı?   :", env.check_success(), "| son ödül:", reward)
env.close()
imageio.imwrite("libero_first_episode.png", np.concatenate(frames, axis=1))
print("kaydedildi: libero_first_episode.png")
```

## list_tasks.py

```python
"""Gün 10 — Dört suite'in görev listesini ve İngilizce komutlarını yazdırır."""
from libero.libero import benchmark
for name in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    s = benchmark.get_benchmark_dict()[name]()
    print(f"\n== {name} ({s.n_tasks} görev)")
    for i in range(s.n_tasks):
        print(f"{i:2d}. {s.get_task(i).language}")
```

