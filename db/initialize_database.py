"""Initialize PostgreSQL through Prisma and import existing scan JSON files once.

Usage from the repository root:
    .venv\\Scripts\\python.exe db\\initialize_database.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from car_scan.config import Settings  # noqa: E402
from car_scan.database import DatabaseRepository  # noqa: E402
from car_scan.prisma_db import push_schema  # noqa: E402


def main() -> int:
    settings = Settings.from_env()
    if not settings.database_url:
        raise SystemExit("ไม่พบ CAR_SCAN_DATABASE_URL ใน environment หรือไฟล์ .env")

    push_schema(settings.database_url)
    repository = DatabaseRepository(settings.database_url)
    existing = repository.count_scans()
    imported = 0

    # Existing scan outputs are real project data. Import them only when the
    # database is empty so rerunning this command is safe and idempotent.
    if existing == 0:
        for result_path in sorted((settings.root / "runs").rglob("*_result.json")):
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if not isinstance(result, dict) or "input" not in result:
                    continue
                repository.save_scan(result)
                imported += 1
            except (OSError, ValueError, TypeError, KeyError) as error:
                print(f"ข้าม {result_path}: {error}", file=sys.stderr)

    print(f"Prisma schema ready: {settings.database_url.split('@')[-1]}")
    print(f"Existing scans: {existing}")
    print(f"Imported scan results from runs: {imported}")
    print(f"Total scans now: {repository.count_scans()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
