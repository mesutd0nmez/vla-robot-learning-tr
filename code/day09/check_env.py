"""Day 9 — Environment check. Every line should say OK."""
import sys, platform
print("python :", sys.version.split()[0], "| arch:", platform.machine(), "(arm64 expected)")
import numpy; print("numpy  :", numpy.__version__)
import mujoco; print("mujoco :", mujoco.__version__)
import torch; print("torch  :", torch.__version__, "| MPS:", torch.backends.mps.is_available())
print("OK")
