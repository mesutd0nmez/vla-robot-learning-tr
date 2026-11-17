"""Ders 21 — Webots + conda `vla` bağlantı kontrolü (Webots'u açmadan çalışır).

Kontroller: WEBOTS_HOME doğru mu, Python API klasörü var mı, libController yüklenebiliyor mu,
launcher (webots-controller) nerede, python3 hangi ortamdan geliyor.
Çalıştır (vla aktif):  WEBOTS_HOME=/Applications/Webots.app python check_webots_env.py
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
    print(f"{label:12s}: {'VAR ' if os.path.exists(path) else 'YOK '} {path}")
if platform.machine() == "arm64" and mac:
    print("Apple Silicon: R2025a'da launcher'ın arm Mac'te çalışmama hatası var (PR #6806, R2025b'de düzeltildi).")
    print("  Launcher hata verirse: PYTHONPATH=" + pyapi + " python ders21_kol.py  (Webots açıkken)")
sys.path.insert(0, pyapi)
try:
    import controller  # noqa: F401
    from controller import Supervisor, Camera, Motor  # noqa: F401
    print("import controller : OK  (ctypes ile", os.path.basename(lib), "yüklendi; numpy/torch ortamın aynen kullanılabilir)")
except Exception as e:  # pragma: no cover
    print("import controller : HATA ->", repr(e))
    print("  Olası sebep: WEBOTS_HOME yanlış, Rosetta altında açılmış Webots (x86_64 dylib), ya da Python < 3.8")
    sys.exit(2)
print("Hepsi tamam. Şimdi Webots'ta ders21_masa.wbt'yi aç, oynat, sonra ders21_kol.py'yi başlat.")
