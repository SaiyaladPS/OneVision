"""Development entry point for the Car Scan PyQt6 application."""

import sys
from pathlib import Path

# Allow `python run_gui.py` directly from a source checkout.  Installed users
# use the `car-scan` console script instead.
SOURCE_DIR = Path(__file__).resolve().parent / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from car_scan.gui import main


if __name__ == "__main__":
    raise SystemExit(main())
