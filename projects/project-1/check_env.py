"""Gün 9 — Ortam kontrolü. Her satır OK demeli."""
import sys, platform
print("python :", sys.version.split()[0], "| mimari:", platform.machine(), "(arm64 bekleniyor)")
import numpy; print("numpy  :", numpy.__version__)
import mujoco; print("mujoco :", mujoco.__version__)
import torch; print("torch  :", torch.__version__, "| MPS:", torch.backends.mps.is_available())
print("OK")
