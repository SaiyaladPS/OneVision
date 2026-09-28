"""Shared Prisma Client Python connection for PostgreSQL."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "prisma" / "schema.prisma"

_client: Any = None
_client_url: str | None = None
_schema_ready = False


def configure_database_url(database_url: str) -> str:
    url = database_url.strip()
    if not url:
        raise ValueError("CAR_SCAN_DATABASE_URL is not configured")
    os.environ["CAR_SCAN_DATABASE_URL"] = url
    os.environ.setdefault("DATABASE_URL", url)
    return url


def get_client(database_url: str) -> Any:
    """Return a process-wide sync Prisma client connected to PostgreSQL."""

    global _client, _client_url
    url = configure_database_url(database_url)
    if _client is not None and _client_url == url and getattr(_client, "is_connected", lambda: False)():
        return _client
    if _client is not None:
        try:
            _client.disconnect()
        except Exception:
            pass
        _client = None
        _client_url = None
    try:
        from prisma import Prisma
    except Exception as error:  # pragma: no cover - depends on generate
        raise RuntimeError(
            "ยังไม่ได้ generate Prisma Client — รัน: python -m prisma generate --schema prisma/schema.prisma"
        ) from error
    client = Prisma()
    client.connect()
    _client = client
    _client_url = url
    return client


def disconnect() -> None:
    global _client, _client_url
    if _client is None:
        return
    try:
        _client.disconnect()
    finally:
        _client = None
        _client_url = None


def prisma_command(*args: str) -> list[str]:
    return [sys.executable, "-m", "prisma", *args, "--schema", str(SCHEMA_PATH)]


def _prisma_env() -> dict[str, str]:
    """Put the venv Scripts dir first so Node can spawn prisma-client-py."""

    env = os.environ.copy()
    scripts = str(Path(sys.executable).resolve().parent)
    env["PATH"] = scripts + os.pathsep + env.get("PATH", "")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def push_schema(database_url: str) -> None:
    """Create or update tables from prisma/schema.prisma without a migration history."""

    configure_database_url(database_url)
    if not SCHEMA_PATH.is_file():
        raise RuntimeError(f"ไม่พบ Prisma schema ที่ {SCHEMA_PATH}")
    completed = subprocess.run(
        prisma_command("db", "push", "--skip-generate"),
        check=False,
        capture_output=True,
        text=True,
        cwd=str(SCHEMA_PATH.parents[1]),
        env=_prisma_env(),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip() or f"exit {completed.returncode}"
        raise RuntimeError(f"prisma db push ไม่สำเร็จ: {detail}")


def ensure_schema(database_url: str) -> Any:
    """Push schema once per process, then return a connected client."""

    global _schema_ready
    if not _schema_ready:
        push_schema(database_url)
        _schema_ready = True
    return get_client(database_url)
