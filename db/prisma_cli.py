"""Run Prisma CLI with the venv Scripts directory on PATH.

Usage from the repository root:
    .venv\\Scripts\\python.exe db\\prisma_cli.py generate
    .venv\\Scripts\\python.exe db\\prisma_cli.py db push
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "prisma" / "schema.prisma"


def main() -> int:
    scripts = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = scripts + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.path.insert(0, str(ROOT / "src"))
    from prisma.cli import main as prisma_main

    argv = list(sys.argv[1:])
    if "--schema" not in argv:
        argv.extend(["--schema", str(SCHEMA)])
    sys.argv = ["prisma", *argv]
    return int(prisma_main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
