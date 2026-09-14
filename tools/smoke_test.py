"""Smoke test: extract the PATCHED code objects from dist/Creative Relight.exe,
exec them in a test harness, and run the real ImageProcessor.process() on a
synthetic transparent EXR. Also unit-test the media_preview EXR preview path."""
import struct, zlib, marshal, sys, types, os, io

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.makedirs("test_tmp", exist_ok=True)

EXE = "dist/Creative Relight.exe"
with open(EXE, "rb") as f:
    exe = f.read()
MAGIC = b"MEI\x0c\x0b\x0a\x0b\x0e"
idx = exe.rfind(MAGIC)
magic, pkglen, toc_off, toclen, pyvers, pylib = struct.unpack("!8sIIII64s", exe[idx: idx + 88])
arch = idx + 88 - pkglen
ctoc = exe[arch + toc_off: arch + toc_off + toclen]
pos = 0
pyz = None
while pos < len(ctoc):
    (el,) = struct.unpack("!i", ctoc[pos:pos + 4])
    ep, cs, us, cf, ty = struct.unpack("!IIIBc", ctoc[pos + 4:pos + 18])
    nm = ctoc[pos + 18:pos + el].rstrip(b"\x00").decode()
    if ty.decode() in ("z", "Z"):
        pyz = exe[arch + ep: arch + ep + cs]
    pos += el
(tp,) = struct.unpack("!I", pyz[8:12])
toc = dict(marshal.loads(pyz[tp:]))

def get_code(mod):
    t, p, l = toc[mod]
    return marshal.loads(zlib.decompress(pyz[p:p + l]))

# --- build a fake package tree in sys.modules from the PYZ's own modules ---
import importlib
fake_src = types.ModuleType("src"); fake_src.__path__ = []
fake_utils = types.ModuleType("src.utils"); fake_utils.__path__ = []
fake_proc = types.ModuleType("src.processors"); fake_proc.__path__ = []
fake_ui = types.ModuleType("ui"); fake_ui.__path__ = []
sys.modules.update({"src": fake_src, "src.utils": fake_utils,
                    "src.processors": fake_proc, "ui": fake_ui})

def load(mod, container=None):
    co = get_code(mod)
    m = types.ModuleType(mod)
    m.__file__ = co.co_filename
    sys.modules[mod] = m
    if container:
        setattr(container, mod.split(".")[1], m)
    exec(co, m.__dict__)
    return m

print("== loading patched modules from exe ==")
# stub torch (file_handler imports it at module level; we don't need it here)
torch_stub = types.ModuleType("torch")
torch_stub.save = lambda *a, **k: None
torch_stub.load = lambda *a, **k: None
tqdm_stub = types.ModuleType("tqdm"); tqsub = types.ModuleType("tqdm.std"); \
    sys.modules["tqdm"] = tqdm_stub; sys.modules["tqdm.std"] = tqsub
sys.modules["torch"] = torch_stub
iu = load("src.utils.image_utils", fake_utils)
fh = load("src.utils.file_handler", fake_utils)
# lightweight stand-ins for the heavy model classes
sys.modules["lighting_model"] = types.ModuleType("lighting_model")  # shadowed below
lighting_stub = types.ModuleType("src.models.lighting_model")
class _Stub:
    def __init__(self, device): print(f"  [stub LightingModel(device={device})]")
    def process(self, arr):
        # albedo = copy of input, specular = grayscale copy
        alb = np.clip(arr, 0, 1)
        spec = alb.mean(axis=2)
        return {"albedo": alb, "specular": spec}
lighting_stub.LightingModel = _Stub
sys.modules["lighting_model"].LightingModel = _Stub  # 'from lighting_model import LightingModel'
lighting_stub.LightingModel = _Stub

class _StubG:  # forward declaration for geometry stub module
    pass

geom_stub = types.ModuleType("src.models.geometry_model")
class _StubG:
    def __init__(self, device): print(f"  [stub GeometryModel(device={device})]")
    def process(self, arr):
        h, w = arr.shape[:2]
        depth = np.zeros((h, w), dtype=np.uint16)
        normal = np.zeros((h, w, 3), dtype=np.uint8); normal[..., 2] = 255
        return {"depth": depth, "normal": normal}
geom_stub.GeometryModel = _StubG
sys.modules["geometry_model"] = geom_stub  # direct-import fallback name
sys.modules["src.models.lighting_model"] = lighting_stub
sys.modules["src.models.geometry_model"] = geom_stub
fake_models = types.ModuleType("src.models"); fake_models.__path__ = []
fake_models.LightingModel = _Stub; fake_models.GeometryModel = _StubG
sys.modules["src.models"] = fake_models

sys.modules["image_utils"] = iu  # direct-import name used by processors
ip = load("src.processors.image_processor", fake_proc)

# --- UI module import check (classes defined at import time) ---
print("== loading patched ui.main_window (import-time check) ==")
try:
    mw = load("ui.main_window", fake_ui)
    print("ui.main_window executes OK (PyQt6 import works)")
except ImportError as e:
    print(f"ui.main_window import needs dep: {e} (acceptable in harness)")

mp = load("ui.media_preview", fake_ui)

# --- EXR processing test ---
print("== processing synthetic EXR through patched ImageProcessor ==")
import numpy as np
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
h, w = 96, 128
img = np.zeros((h, w, 4), dtype=np.float32)
yy, xx = np.mgrid[0:h, 0:w]
circle = ((yy - h/2)**2 + (xx - w/2)**2) < (min(h, w)/3)**2
img[..., 2] = 0.8; img[..., 1] = 0.4; img[..., 0] = 0.1
img[..., 3] = np.where(circle, 1.0, 0.0)
assert cv2.imwrite("test_tmp/smoke.exr", img)

proc = ip.ImageProcessor(device="cpu")
res = proc.process("test_tmp/smoke.exr", "test_tmp/out", export_config={
    "albedo": True, "specular": True, "depth": True, "normal": True, "shading": False})

print("saved_files:", res.get("saved_files"))
saved = res.get("saved_files", {})
assert "alpha" in saved, "alpha map missing!"
for k, v in saved.items():
    data = cv2.imread(v, cv2.IMREAD_UNCHANGED)
    print(f"  {k}: {v} shape={None if data is None else data.shape}")
a = cv2.imread(saved["alpha"], cv2.IMREAD_UNCHANGED)
assert a is not None and a.min() == 0 and a.max() == 255
n = cv2.imread(saved["normal"], cv2.IMREAD_UNCHANGED)
print("normal center px (should be ~blue 255):", n[h//2, w//2])
print("\nSMOKE TEST PASSED")
