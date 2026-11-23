"""Day 9 — First MuJoCo scene.
Table + plate + a falling blue ball. Saves four moments side by side as a PNG.
Run:  python first_scene.py
No display needed on the Mac; offscreen rendering is used.
"""
import mujoco, numpy as np, imageio

XML = """
<mujoco>
  <visual><global offwidth="640" offheight="480"/><headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8"/></visual>
  <worldbody>
    <light pos="0.5 -1 2.5" dir="-0.2 0.4 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.85 0.85 0.85 1"/>
    <body name="table" pos="0 0 0.4"><geom type="box" size="0.4 0.3 0.02" rgba="0.6 0.4 0.2 1"/></body>
    <body name="plate" pos="0.1 0 0.425"><geom type="cylinder" size="0.08 0.005" rgba="0.95 0.95 0.95 1"/></body>
    <body name="ball" pos="0.1 0 0.7"><freejoint/><geom type="sphere" size="0.08" rgba="0.2 0.4 0.9 1" mass="0.1"/></body>
    <camera name="agentview" pos="0 0 2" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>"""

model = mujoco.MjModel.from_xml_string(XML)   # scene definition (MJCF)
data = mujoco.MjData(model)                    # instantaneous state: positions, velocities
renderer = mujoco.Renderer(model, 480, 640)    # offscreen camera

frames = []
for t in range(400):                           # 400 steps × 0.002 s = 0.8 s of simulation
    mujoco.mj_step(model, data)                # advance the physics by one step
    if t in (0, 60, 120, 399):
        renderer.update_scene(data, camera="agentview")
        frames.append(renderer.render().copy())

imageio.imwrite("first_scene.png", np.concatenate(frames, axis=1))
print("ball z (height):", round(float(data.qpos[2]), 3), "→ landed on the table" if data.qpos[2] < 0.5 else "")
print("kaydedildi: first_scene.png")
