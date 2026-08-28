"""Small PostgreSQL repository used by the desktop application."""

from __future__ import annotations

from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS scan_runs (
    id BIGSERIAL PRIMARY KEY,
    source_path TEXT NOT NULL,
    annotated_image TEXT,
    plate_count INTEGER NOT NULL DEFAULT 0,
    scanned_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    raw_result JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS plates (
    id BIGSERIAL PRIMARY KEY,
    scan_id BIGINT NOT NULL REFERENCES scan_runs(id) ON DELETE CASCADE,
    plate_index INTEGER NOT NULL,
    country VARCHAR(16) NOT NULL,
    province TEXT,
    plate_prefix TEXT,
    plate_number TEXT,
    ocr_text TEXT,
    ocr_confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    detection_confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    raw_plate JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_plates_plate_number ON plates (plate_number);
CREATE INDEX IF NOT EXISTS ix_scan_runs_scanned_at ON scan_runs (scanned_at DESC);
"""


class DatabaseRepository:
    """Open short-lived connections so the repository is safe in QThreads."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url.strip()
        if not self.database_url:
            raise ValueError("CAR_SCAN_DATABASE_URL is not configured")

    def _connect(self):
        try:
            import psycopg
        except ImportError as error:  # pragma: no cover - depends on environment
            raise RuntimeError("ติดตั้ง psycopg[binary] ก่อนใช้งาน PostgreSQL") from error
        return psycopg.connect(self.database_url)

    def initialize_schema(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(SCHEMA)

    def ping(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")

    def count_scans(self) -> int:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM scan_runs")
                return int(cursor.fetchone()[0])

    def save_scan(self, result: dict[str, Any]) -> int:
        """Persist one scan and all detected plates in one transaction."""

        try:
            from psycopg.types.json import Jsonb
        except ImportError as error:  # pragma: no cover - guarded by _connect
            raise RuntimeError("ติดตั้ง psycopg[binary] ก่อนใช้งาน PostgreSQL") from error

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO scan_runs (source_path, annotated_image, plate_count, raw_result)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        result["input"],
                        result.get("annotated_image"),
                        result.get("plate_count", 0),
                        Jsonb(result),
                    ),
                )
                scan_id = int(cursor.fetchone()[0])
                for plate in result.get("plates", []):
                    ocr = plate.get("ocr", {})
                    cursor.execute(
                        """
                        INSERT INTO plates (
                            scan_id, plate_index, country, province, plate_prefix,
                            plate_number, ocr_text, ocr_confidence,
                            detection_confidence, raw_plate
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            scan_id,
                            plate.get("id", 0),
                            plate.get("country", "unknown"),
                            plate.get("province") or None,
                            plate.get("plate_prefix") or None,
                            plate.get("plate_number") or None,
                            ocr.get("text") or None,
                            float(ocr.get("confidence", 0.0)),
                            float(plate.get("detection_confidence", 0.0)),
                            Jsonb(plate),
                        ),
                    )
        return scan_id
