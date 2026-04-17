# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules


try:
    project_root = Path(__file__).resolve().parent
except NameError:
    # Some PyInstaller invocation contexts do not define __file__ for spec execution.
    project_root = Path.cwd().resolve()
src_path = project_root / "src"
pkg_root = src_path / "face_smoothing"

hidden_imports = [
    "dotenv",
    "yaml",
    "cv2",
    "torch",
    "torchvision",
    "onnxruntime",
    "concurrent.futures",
    "queue",
    "threading",
]
hidden_imports += collect_submodules("face_smoothing")

datas = [
    (str(pkg_root / "configs"), "face_smoothing/configs"),
    (str(pkg_root / "models"), "face_smoothing/models"),
]
binaries = []

for mod in ("cv2", "onnxruntime", "insightface", "torch", "torchvision", "watchdog"):
    d, b, h = collect_all(mod)
    datas += d
    binaries += b
    hidden_imports += h

a = Analysis(
    ["infer.py"],
    pathex=[str(project_root), str(src_path)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="infer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="infer",
)
