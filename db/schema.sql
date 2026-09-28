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

ALTER TABLE scan_runs ADD COLUMN IF NOT EXISTS operator_id INTEGER;
ALTER TABLE scan_runs ADD COLUMN IF NOT EXISTS operator_name TEXT;
ALTER TABLE scan_runs ADD COLUMN IF NOT EXISTS operator_username TEXT;
ALTER TABLE scan_runs ADD COLUMN IF NOT EXISTS media_type TEXT;
ALTER TABLE scan_runs ADD COLUMN IF NOT EXISTS output_video TEXT;
CREATE INDEX IF NOT EXISTS ix_scan_runs_operator_id ON scan_runs (operator_id);

ALTER TABLE plates ADD COLUMN IF NOT EXISTS province_code TEXT;
ALTER TABLE plates ADD COLUMN IF NOT EXISTS plate_prefix_code TEXT;
ALTER TABLE plates ADD COLUMN IF NOT EXISTS recognition_confidence DOUBLE PRECISION;
ALTER TABLE plates ADD COLUMN IF NOT EXISTS overall_confidence DOUBLE PRECISION;
ALTER TABLE plates ADD COLUMN IF NOT EXISTS confidence_level TEXT;
ALTER TABLE plates ADD COLUMN IF NOT EXISTS vehicle_type TEXT;
ALTER TABLE plates ADD COLUMN IF NOT EXISTS vehicle_type_confidence DOUBLE PRECISION;
ALTER TABLE plates ADD COLUMN IF NOT EXISTS plate_type TEXT;
ALTER TABLE plates ADD COLUMN IF NOT EXISTS vehicle_image TEXT;
ALTER TABLE plates ADD COLUMN IF NOT EXISTS crop_image TEXT;
ALTER TABLE plates ADD COLUMN IF NOT EXISTS ocr_ready_image TEXT;

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_users_role ON users (role);

CREATE TABLE IF NOT EXISTS cameras (
    id SERIAL PRIMARY KEY,
    host TEXT NOT NULL UNIQUE,
    stream_url TEXT,
    label TEXT,
    kind TEXT NOT NULL DEFAULT 'ip',
    device_index INTEGER,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_cameras_enabled ON cameras (enabled);
