"""Lecture 21 — first arm motion in Webots + camera recording + spatial relation table.

Moves the UR10e (supervisor TRUE, controller <extern>) through four intermediate poses, at each pose saves
the wrist camera and the main 3D view (agentview-like) as PNG/JPG,
reads the objects' world coordinates with the Supervisor API and writes the
left/right/front/back relations with respect to the robot to CSV.

Run (with ders21_masa.wbt open and playing in Webots, in a separate terminal, vla env active):
  macOS : export WEBOTS_HOME=/Applications/Webots.app
          $WEBOTS_HOME/Contents/MacOS/webots-controller code/day21/controllers/ders21_kol/ders21_kol.py --out out
  Linux : $WEBOTS_HOME/webots-controller code/day21/controllers/ders21_kol/ders21_kol.py --out out
If the launcher does not work (see README "without the launcher"):
  PYTHONPATH=$WEBOTS_HOME/Contents/lib/controller/python python ders21_kol.py --out out
"""
import argparse
import csv
import math
import os
import sys

from controller import Supervisor  # Webots' ctypes-based Python API (needs WEBOTS_HOME)

p = argparse.ArgumentParser()
p.add_argument("--out", default="out", help="folder the PNG/JPG/CSV files are written to")
p.add_argument("--settle", type=int, default=60, help="number of steps to wait at each pose")
p.add_argument("--quit", action="store_true", help="quit Webots when done (for CI / headless runs)")
args, _ = p.parse_known_args()
os.makedirs(args.out, exist_ok=True)

robot = Supervisor()
dt = int(robot.getBasicTimeStep())          # basicTimeStep in the world (16 ms)

# --- actuators: the 6 joints of the UR10e (RotationalMotor names in the PROTO) ---
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
motors = [robot.getDevice(j) for j in JOINTS]
sensors = [robot.getDevice(j + "_sensor") for j in JOINTS]
for s in sensors:
    s.enable(dt)
for m in motors:
    m.setVelocity(1.0)                       # rad/s; maxVelocity is 3.14 / 6.28 in the PROTO

# --- sensor: the wrist camera we put on toolSlot ---
cam = robot.getDevice("wrist_camera")
cam.enable(dt)

# --- Supervisor: objects in the scene (DEF names in the world) ---
DEFS = ["BOWL", "PLATE", "STOVE", "CEREAL", "APPLE", "CAN"]
nodes = {d: robot.getFromDef(d) for d in DEFS}
robot_node = robot.getSelf()

# Four poses (rad): [pan, lift, elbow, w1, w2, w3]. The arm leans over the table.
POSES = {
    "00_home":   [0.000, -1.800, 1.200, -1.570, 0.000, 0.000],
    "01_bowl":   [0.582, -0.542, -1.637, -2.533, -1.571, 0.000],
    "02_center": [0.194, -0.868, -1.055, -2.789, -1.571, 0.000],
    "03_plate":  [-0.068, -0.568, -1.593, -2.552, -1.571, 0.000],
}


def world_to_robot(pw, R, t):
    """World point -> robot base frame (R: 3x3 row-major, t: base position)."""
    d = [pw[i] - t[i] for i in range(3)]
    # R is world<-robot; robot coordinate = R^T d
    return [sum(R[3 * k + i] * d[k] for k in range(3)) for i in range(3)]


rows = []
for name, q in POSES.items():
    for m, qi in zip(motors, q):
        m.setPosition(qi)
    for _ in range(args.settle):          # wait for the arm to reach the target (synchronization TRUE)
        if robot.step(dt) == -1:
            sys.exit(0)
    # camera frames
    cam.saveImage(os.path.join(args.out, f"wrist_{name}.png"), 100)
    robot.exportImage(os.path.join(args.out, f"view_{name}.jpg"), 95)
    # joint readings
    qs = [s.getValue() for s in sensors]
    err = max(abs(a - b) for a, b in zip(qs, q))
    print(f"[{name}] t={robot.getTime():6.2f}s  max|q-q*|={err:.3f} rad  "
          + " ".join(f"{v:+.2f}" for v in qs))
    rows.append({"pose": name, "t": round(robot.getTime(), 3), "max_err_rad": round(err, 4),
                 **{j: round(v, 4) for j, v in zip(JOINTS, qs)}})

with open(os.path.join(args.out, "joints.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# --- spatial relations: the Supervisor reads the object positions ---
# The UR10e's "front" is −x in the base frame (shoulder_lift = −90° extends the arm toward −x); since the base is
# rotated −90° about the z axis in the world, its front is world +y and its left is world −x. Below, forward = −x_r, left = −y_r.
t = robot_node.getPosition()
R = robot_node.getOrientation()           # 9 elements, row-major
rel_rows = []
for d, n in nodes.items():
    pw = n.getPosition()
    pr = world_to_robot(pw, R, t)          # base frame
    fwd, left = -pr[0], -pr[1]             # forward (toward the table) and left
    rel_rows.append({"def": d, "x_w": round(pw[0], 3), "y_w": round(pw[1], 3), "z_w": round(pw[2], 3),
                     "ileri_r": round(fwd, 3), "sol_r": round(left, 3),
                     "sol_sag": "sol" if left > 0.03 else ("sag" if left < -0.03 else "orta"),
                     "on_arka": "yakin" if fwd < 0.75 else "uzak"})
    print(f"{d:7s} world=({pw[0]:+.2f},{pw[1]:+.2f},{pw[2]:+.2f})  robot(forward,left)=({fwd:+.2f},{left:+.2f})"
          f"  -> {rel_rows[-1]['sol_sag']}, {rel_rows[-1]['on_arka']}")

# pairwise relations (in the robot frame): is A to the left of / right of / in front of / behind B?
pairs = []
for a in rel_rows:
    for b in rel_rows:
        if a["def"] == b["def"]:
            continue
        d_sol = a["sol_r"] - b["sol_r"]        # >0: A is to the left of B (relative to the robot)
        d_ileri = a["ileri_r"] - b["ileri_r"]  # >0: A is behind B (farther from the robot)
        pairs.append({"a": a["def"], "b": b["def"],
                      "d_sol": round(d_sol, 3), "d_ileri": round(d_ileri, 3),
                      "iliski_tr": ("solunda" if d_sol > 0.05 else "sağında" if d_sol < -0.05 else "aynı hizada")
                                   + " / " + ("gerisinde" if d_ileri > 0.05 else "önünde" if d_ileri < -0.05 else "aynı derinlikte")})
for fname, data in [("objects.csv", rel_rows), ("pairs.csv", pairs)]:
    with open(os.path.join(args.out, fname), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)
print(f"Written: {args.out}/wrist_*.png, view_*.jpg, joints.csv, objects.csv, pairs.csv")
print(f"basicTimeStep={dt} ms; simulation time={robot.getTime():.2f} s")
if args.quit:
    robot.simulationQuit(0)
robot.step(dt)
