"""Application configuration loaded from environment variables.

Secrets are intentionally not stored in source code.  For local development
use a .env file loaded by the shell or set CAR_SCAN_DATABASE_URL directly.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def resource_root() -> Path:
    """Return the directory containing bundled models/resources."""

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    # config.py lives in <root>/src/car_scan during development.
    return Path(__file__).resolve().parents[2]


def _path(value: str | None, default: Path) -> Path:
    return Path(value).expanduser() if value else default


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE settings without requiring python-dotenv."""

    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the GUI and scan service."""

    root: Path
    output_dir: Path
    database_url: str
    # Identifier included in archived image names. Camera scans override
    # this with the actual selected camera index.
    archive_camera_id: str = "0"
    detector_confidence: float = 0.35
    vehicle_type_confidence: float = 0.25
    plate_type_confidence: float = 0.25
    character_confidence: float = 0.20
    padding: float = 0.03
    imgsz: int = 640
    video_frame_stride: int = 1
    video_target_scans_per_second: float = 2.0
    video_min_confirmations: int = 3
    camera_frame_stride: int = 6
    camera_min_confirmations: int = 3
    temporal_min_quality: float = 0.65
    target_fps: float = 24.0
    # Sample video/camera frames by default; enable full accuracy explicitly
    # when every frame must be analysed.
    full_accuracy_mode: bool = False
    pipeline_mode: str = "auto"
    pipeline_config_override: Path | None = None
    plate_type_model_override: Path | None = None
    debug_enabled: bool = False
    scan_roi_override: dict[str, Any] | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        _load_env_file(Path.cwd() / ".env")
        _load_env_file(Path(__file__).resolve().parents[2] / ".env")
        root = _path(os.getenv("CAR_SCAN_ROOT"), resource_root()).resolve()
        output = _path(os.getenv("CAR_SCAN_OUTPUT_DIR"), root / "runs" / "scan").resolve()
        database_url = os.getenv("CAR_SCAN_DATABASE_URL", "").strip()
        return cls(
            root=root,
            output_dir=output,
            database_url=database_url,
            archive_camera_id=os.getenv("CAR_SCAN_ARCHIVE_CAMERA_ID", "0").strip() or "0",
            detector_confidence=float(os.getenv("CAR_SCAN_DETECTOR_CONFIDENCE", "0.35")),
            vehicle_type_confidence=float(os.getenv("CAR_SCAN_VEHICLE_TYPE_CONFIDENCE", "0.25")),
            plate_type_confidence=float(os.getenv("CAR_SCAN_PLATE_TYPE_CONFIDENCE", "0.25")),
            character_confidence=float(os.getenv("CAR_SCAN_CHARACTER_CONFIDENCE", "0.20")),
            padding=float(os.getenv("CAR_SCAN_PADDING", "0.03")),
            imgsz=int(os.getenv("CAR_SCAN_IMGSZ", "640")),
            video_frame_stride=max(1, int(os.getenv("CAR_SCAN_VIDEO_FRAME_STRIDE", "1"))),
            video_target_scans_per_second=max(
                0.5, float(os.getenv("CAR_SCAN_VIDEO_TARGET_SCANS_PER_SECOND", "2.0"))
            ),
            video_min_confirmations=max(
                1, int(os.getenv("CAR_SCAN_VIDEO_MIN_CONFIRMATIONS", "3"))
            ),
            camera_frame_stride=max(1, int(os.getenv("CAR_SCAN_CAMERA_FRAME_STRIDE", "4"))),
            camera_min_confirmations=max(
                2, int(os.getenv("CAR_SCAN_CAMERA_MIN_CONFIRMATIONS", "3"))
            ),
            temporal_min_quality=max(
                0.0, float(os.getenv("CAR_SCAN_TEMPORAL_MIN_QUALITY", "0.65"))
            ),
            target_fps=max(1.0, float(os.getenv("CAR_SCAN_TARGET_FPS", "24.0"))),
            full_accuracy_mode=os.getenv("CAR_SCAN_FULL_ACCURACY_MODE", "0").strip().lower()
            in ("1", "true", "yes", "on"),
            pipeline_mode=os.getenv("CAR_SCAN_PIPELINE_MODE", "auto").strip().lower(),
            pipeline_config_override=_path(
                os.getenv("CAR_SCAN_PIPELINE_CONFIG"), root / "pipeline_config.yaml"
            ).resolve(),
            plate_type_model_override=(
                Path(os.environ["CAR_SCAN_PLATE_TYPE_MODEL"]).expanduser().resolve()
                if os.getenv("CAR_SCAN_PLATE_TYPE_MODEL")
                else None
            ),
            debug_enabled=os.getenv("CAR_SCAN_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on"),
        )

    @property
    def detector_model(self) -> Path:
        return self.root / "model" / "detect_license" / "weights" / "best.pt"

    @property
    def vehicle_type_model(self) -> Path:
        return self.root / "model" / "type_car_license" / "iruvd_run1" / "weights" / "best.pt"

    @property
    def plate_type_model(self) -> Path | None:
        return self.plate_type_model_override

    @property
    def pipeline_config(self) -> Path:
        return self.pipeline_config_override or (self.root / "pipeline_config.yaml")

    @property
    def debug_dir(self) -> Path:
        return self.output_dir / "debug"

    @property
    def thai_model(self) -> Path:
        return self.root / "model" / "thai_license_plate" / "weights" / "best.pt"

    @property
    def lao_model(self) -> Path:
        return self.root / "model" / "lao_license_plate" / "weights" / "best.pt"

    @property
    def ocr_source(self) -> Path:
        return self.root / "model" / "paddleocr_train2"

    @property
    def ocr_config(self) -> Path:
        return self.ocr_source / "configs" / "rec" / "plate_rec_small.yml"

    @property
    def ocr_weights(self) -> Path:
        finetuned = self.root / "runs" / "ocr" / "plate_rec_digits" / "best_accuracy"
        if finetuned.with_suffix(".pdparams").is_file():
            return finetuned
        return self.ocr_source / "output" / "plate_rec_small" / "best_accuracy"

    @property
    def ocr_lao_config(self) -> Path:
        return self.ocr_source / "configs" / "rec" / "plate_rec_lao_full.yml"

    @property
    def ocr_lao_weights(self) -> Path | None:
        weights = self.root / "runs" / "ocr" / "plate_rec_lao_full" / "best_accuracy"
        return weights if weights.with_suffix(".pdparams").is_file() else None

    @property
    def torch_ocr_model(self) -> Path:
        return self.root / "runs" / "ocr" / "plate_crnn.pt"
