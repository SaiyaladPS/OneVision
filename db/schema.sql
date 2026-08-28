-- The application runs this idempotently on first use.  This file is also
-- provided for teams that prefer to manage database migrations separately.
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
