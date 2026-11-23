"""Lecture 21 — Webots + conda `vla` connection check (runs without opening Webots).

Checks: is WEBOTS_HOME correct, does the Python API folder exist, can libController be loaded,
where is the launcher (webots-controller), which environment python3 comes from.
Run (vla active):  WEBOTS_HOME=/Applications/Webots.app python check_webots_env.py
"""
import os
import platform
import sys

home = os.environ.get("WEBOTS_HOME")
print("python      :", sys.executable, sys.version.split()[0], "| arch:", platform.machine())
print("WEBOTS_HOME :", home)
if not home:
    print("  -> export WEBOTS_HOME=/Applications/Webots.app   (macOS)  |  .../webots (Linux tarball)")
    sys.exit(1)
mac = sys.platform == "darwin"
pyapi = os.path.join(home, "Contents", "lib", "controller", "python") if mac else os.path.join(home, "lib", "controller", "python")
lib = os.path.join(home, "Contents", "lib", "controller", "libController.dylib") if mac else os.path.join(home, "lib", "controller", "libController.so")
launcher = os.path.join(home, "Contents", "MacOS", "webots-controller") if mac else os.path.join(home, "webots-controller")
for label, path in [("Python API", pyapi), ("libController", lib), ("launcher", launcher)]:
    print(f"{label:12s}: {'OK  ' if os.path.exists(path) else 'MISSING '} {path}")
if platform.machine() == "arm64" and mac:
    print("Apple Silicon: in R2025a the launcher has a bug and does not run on arm Macs (PR #6806, fixed in R2025b).")
    print("  If the launcher fails: PYTHONPATH=" + pyapi + " python ders21_kol.py  (with Webots open)")
sys.path.insert(0, pyapi)
try:
    import controller  # noqa: F401
    from controller import Supervisor, Camera, Motor  # noqa: F401
    print("import controller : OK  (loaded via ctypes:", os.path.basename(lib), "; the numpy/torch of your environment can be used as is)")
except Exception as e:  # pragma: no cover
    print("import controller : ERROR ->", repr(e))
    print("  Possible cause: wrong WEBOTS_HOME, Webots opened under Rosetta (x86_64 dylib), or Python < 3.8")
    sys.exit(2)
print("All good. Now open ders21_masa.wbt in Webots, press play, then start ders21_kol.py.")
