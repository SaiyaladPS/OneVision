"""Development entry point for the Car Scan web application."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Bind to all network interfaces by default so the web UI can be opened from
# another device on the same LAN (for example: http://192.168.100.147:8000).
# Users can still restrict the server by setting CAR_SCAN_WEB_HOST explicitly.
os.environ.setdefault("CAR_SCAN_WEB_HOST", "0.0.0.0")

VENV_PYTHON = (
    ROOT / ".venv" / "Scripts" / "python.exe"
    if sys.platform == "win32"
    else ROOT / ".venv" / "bin" / "python"
)


def _in_project_venv() -> bool:
    try:
        return Path(sys.executable).resolve() == VENV_PYTHON.resolve()
    except OSError:
        return False


if VENV_PYTHON.is_file() and not _in_project_venv():
    raise SystemExit(subprocess.call([str(VENV_PYTHON), *sys.argv]))

# Allow `python run_web.py` directly from a source checkout.  Installed users
# use the `car-scan-web` console script instead.
SOURCE_DIR = ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

try:
    from car_scan.web import main
except ModuleNotFoundError as error:
    if error.name == "fastapi":
        print(
            "ยังไม่ได้ติดตั้ง FastAPI ใน environment นี้\n"
            "รัน:  .\\.venv\\Scripts\\Activate.ps1\n"
            "      pip install fastapi uvicorn python-multipart\n"
            "แล้วค่อย python run_web.py",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    raise


if __name__ == "__main__":
    raise SystemExit(main())
