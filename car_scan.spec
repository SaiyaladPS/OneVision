# PyInstaller one-folder build. Runtime models and OCR data are copied as
# external resources rather than treated as Python imports. Training sources
# and generated datasets remain available in the repository but are not part
# of the application bundle.
from pathlib import Path

ROOT = Path(SPECPATH)

datas = []


def add_runtime_tree(path: Path) -> None:
    """Copy a source/resource tree while excluding local build artefacts."""

    if not path.exists():
        return
    excluded_parts = {".git", "__pycache__", "output", "train_data"}
    for file_path in path.rglob("*"):
        if not file_path.is_file() or excluded_parts.intersection(file_path.parts):
            continue
        destination = str(file_path.relative_to(ROOT).parent)
        datas.append((str(file_path), destination))


for folder in (
    "model/detect_license",
    "model/type_car_license",
    "model/thai_license_plate",
    "model/lao_license_plate",
    "model/paddleocr_train2",
    "tools/tesseract/tessdata",
):
    add_runtime_tree(ROOT / folder)

pipeline_config = ROOT / "pipeline_config.yaml"
if pipeline_config.exists():
    datas.append((str(pipeline_config), "."))

binaries = []

a = Analysis(
    [str(ROOT / "run_gui.py")],
    pathex=[str(ROOT), str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=["scan", "plate_ocr_torch", "psycopg", "psycopg_binary"],
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
    name="CarScan",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name="CarScan",
)
