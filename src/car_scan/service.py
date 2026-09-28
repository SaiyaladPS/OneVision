"""Application service wrapping the existing licence-plate scan engine."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2

from .config import Settings, redact_stream_url
from .database import jsonable_value

LOGGER = logging.getLogger(__name__)


TEMPORAL_PROVINCE_MIN_CONFIDENCE = 0.12
TEMPORAL_PROVINCE_MIN_OBSERVATIONS = 2
TEMPORAL_PROVINCE_SCORE_MARGIN = 0.10
RESULT_SCHEMA_VERSION = "1.0"
MAX_REJECTED_EVIDENCE_PER_RUN = 100
REJECTED_EVIDENCE_MIN_CONFIDENCE = 0.40


@dataclass
class LiveScanSession:
    """Temporal voting state shared by video, camera, and CCTV scans."""

    tracks: list[dict[str, Any]] = field(default_factory=list)
    rejected_evidence: list[dict[str, Any]] = field(default_factory=list)
    rejected_keys: set[str] = field(default_factory=set)
    last_rejected_evidence_frame: int = 0
    confirmed_keys: set[str] = field(default_factory=set)
    confirmed_plates: list[dict[str, Any]] = field(default_factory=list)
    confirmed_tracks: dict[int, int] = field(default_factory=dict)
    cached_vehicle_types: list[dict[str, Any]] | None = None
    cached_plates: list[dict[str, Any]] = field(default_factory=list)
    cached_until_frame: int = 0
    sampled_frames: int = 0
    last_detect_count: int = 0
    last_published: list[dict[str, Any]] = field(default_factory=list)
    pending_ocr: list[tuple[dict[str, Any], Any]] = field(default_factory=list)
    # Frame index of the previous processed sample and the resulting track
    # gap tolerance. Video scans every ``stride`` frame, but a CCTV lane on a
    # shared CPU/GPU hub may only see one frame every 1-3 seconds, so a fixed
    # 30-frame gap would start a fresh track on nearly every sample.
    last_sampled_frame: int = -1
    track_gap_frames: int = 30


TRACK_GAP_MIN_FRAMES = 30
TRACK_GAP_SAMPLE_MULTIPLIER = 3
TRACK_GAP_MAX_SECONDS = 6.0


class ScanService:
    """Keep model/OCR details out of the Qt window."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._scanner: Any = None
        self._preprocess_plate_crop: Callable[..., Any] | None = None
        self._write_image: Callable[..., Any] | None = None
        self._default_scan_roi: Any = None
        self.compute_plan: Any = None

    def load_runtime(self) -> None:
        """Load detector/OCR once so every camera shares the same GPU models."""

        if self._scanner is not None:
            return
        from scan import CustomPaddleOCR, LicensePlateScanner, preprocess_plate_crop, write_image
        from .compute import resolve_compute

        self._preprocess_plate_crop = preprocess_plate_crop
        self._write_image = write_image
        plan = resolve_compute(self.settings.compute_mode)
        ocr = CustomPaddleOCR(
            self.settings.ocr_source,
            self.settings.ocr_config,
            self.settings.ocr_weights,
            self.settings.torch_ocr_model,
            lao_config=self.settings.ocr_lao_config,
            lao_weights=self.settings.ocr_lao_weights,
            thai_config=self.settings.ocr_thai_config,
            thai_weights=self.settings.ocr_thai_weights,
            paddle_device=plan.ocr_device,
        )
        LOGGER.info(
            "OCR engines generic=%s lao_full=%s thai_full=%s thai_weights=%s yolo=%s ocr_device=%s",
            ocr.available,
            ocr.lao_available,
            ocr.thai_available,
            ocr.thai_weights,
            plan.yolo_device,
            plan.ocr_device,
        )
        self._scanner = LicensePlateScanner(
            self.settings.detector_model,
            self.settings.thai_model,
            self.settings.lao_model,
            ocr,
            self.settings.vehicle_type_model,
            plate_type_path=self.settings.plate_type_model,
            pipeline_config_path=self.settings.pipeline_config,
            pipeline_mode=self.settings.pipeline_mode,
            scan_roi_override=self.settings.scan_roi_override,
            yolo_device=plan.yolo_device,
        )
        self._default_scan_roi = getattr(self._scanner, "scan_roi", None)
        self.compute_plan = plan

    def unload_runtime(self) -> None:
        """Drop loaded models so the next load can pin a different device."""

        self._scanner = None
        self._preprocess_plate_crop = None
        self._write_image = None
        self.compute_plan = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def ocr_engine_status(self) -> dict[str, Any]:
        """Report the Thai and Lao character models used to read each plate."""

        thai_path = self.settings.thai_model
        lao_path = self.settings.lao_model
        return {
            "loaded": self._scanner is not None,
            "engine": "yolo-character",
            "generic": False,
            "lao_full": lao_path.is_file(),
            "thai_full": thai_path.is_file(),
            "tesseract_lao": False,
            "tesseract_thai": False,
            "lao_weights": str(lao_path),
            "thai_weights": str(thai_path),
        }

    def new_live_session(self) -> LiveScanSession:
        return LiveScanSession(
            last_rejected_evidence_frame=-self.settings.rejected_evidence_frame_gap,
        )

    def _apply_scanner_roi(self, scanner: Any, scan_roi: dict[str, Any] | None) -> None:
        if scanner is None:
            return
        if scan_roi is None:
            scanner.scan_roi = getattr(self, "_default_scan_roi", getattr(scanner, "scan_roi", None))
            return
        try:
            scanner.scan_roi = scanner._normalise_scan_roi(scan_roi)
        except Exception:
            scanner.scan_roi = getattr(self, "_default_scan_roi", getattr(scanner, "scan_roi", None))

    def _scan_stream_frame(
        self,
        scanner: Any,
        frame: Any,
        sampled_frames: int,
        cached_vehicle_types: list[dict[str, Any]] | None,
        *,
        run_ocr: bool = False,
    ) -> tuple[list[dict[str, Any]], Any, list[dict[str, Any]] | None]:
        """One detector pass using the video scan settings."""

        refresh_vehicle_types = (
            cached_vehicle_types is None
            or (sampled_frames - 1) % self.settings.stream_vehicle_refresh_scans == 0
        )
        plates, annotated = scanner.scan(
            frame,
            self.settings.detector_confidence,
            self.settings.character_confidence,
            self.settings.padding,
            self.settings.imgsz,
            run_ocr=run_ocr,
            vehicle_type_confidence=self.settings.vehicle_type_confidence,
            plate_type_confidence=self.settings.plate_type_confidence,
            fast_mode=not self.settings.full_accuracy_mode,
            vehicle_types_override=(None if refresh_vehicle_types else cached_vehicle_types),
        )
        next_types = cached_vehicle_types
        if int(getattr(scanner, "last_plate_detection_count", 0)) <= 0:
            next_types = None
        elif refresh_vehicle_types:
            next_types = [dict(item) for item in getattr(scanner, "last_vehicle_types", [])]
        return plates, annotated, next_types

    def _update_stream_overlay(
        self,
        session: LiveScanSession,
        scanner: Any,
        frame: Any,
        plates: list[dict[str, Any]],
        annotated: Any,
        frame_index: int,
        stride: int,
    ) -> Any:
        current = [plate for plate in plates if self._has_complete_registration(plate)]
        if current:
            session.cached_plates = current
            session.cached_until_frame = frame_index + max(2, stride * 2)
            return annotated
        if frame_index > session.cached_until_frame:
            session.cached_plates = []
            return scanner.draw_scan_roi(frame)
        if session.cached_plates:
            return self._draw_cached_plates(scanner.draw_scan_roi(frame), session.cached_plates)
        return annotated

    def _ingest_stream_plates(
        self,
        session: LiveScanSession,
        scanner: Any,
        frame: Any,
        plates: list[dict[str, Any]],
        frame_index: int,
        *,
        write_image: Callable[..., Any] | None,
        preprocess_plate_crop: Callable[..., Any] | None,
        min_confirmations: int,
        output_stem: str,
        archive_camera_id: str,
        media: str,
        source_fps: float,
        plate_callback: Callable[[dict[str, Any]], None] | None,
        persist_archive: bool,
        run_full_ocr: bool = True,
    ) -> None:
        preprocess = preprocess_plate_crop or (lambda crop, parameters=None: crop)
        prefix = media if media in {"video", "camera", "image"} else "video"
        for plate in plates:
            merged = self._merge_video_plate(
                session.tracks,
                plate,
                frame,
                frame_index,
                preprocess,
                max_gap=session.track_gap_frames,
            )
            if merged is None:
                # Unconfirmed or invalid crops are not archived. Only records
                # that pass temporal and registration checks reach the GUI and
                # the scan/data archive.
                continue
            track, candidate = merged
            key = self._video_plate_key(candidate.get("best_plate", plate))
            count = int(candidate.get("count", 0))
            average_quality = float(candidate.get("vote_score", 0.0)) / max(1, count)
            best = candidate.get("best_plate", plate)
            if not self._has_complete_registration(best):
                continue
            if key in session.confirmed_keys:
                self._refresh_published_track(
                    session,
                    track,
                    candidate,
                    frame_index,
                    write_image=write_image,
                    scanner=scanner,
                    output_stem=output_stem,
                    archive_camera_id=archive_camera_id,
                    media_prefix=prefix,
                    run_full_ocr=run_full_ocr,
                    count=count,
                    average_quality=average_quality,
                    plate_callback=plate_callback,
                )
                continue
            if (
                count < min_confirmations
                or average_quality < self.settings.temporal_min_quality
            ):
                continue
            if write_image is None:
                continue
            # Tiny CCTV plates read as several one/two-digit variants of the
            # same registration. Publish only the reading that currently
            # leads its track; a minority variant waits until it wins.
            if not self._is_dominant_candidate(
                track, candidate, key, frame_index, session.track_gap_frames
            ):
                continue
            record = self._materialise_candidate(
                candidate,
                len(session.confirmed_plates) + 1,
                output_stem,
                write_image,
                scanner,
                run_full_ocr=run_full_ocr,
                persist_archive=persist_archive,
                archive_camera_id=archive_camera_id,
                write_files=False,
            )
            if record is None:
                continue
            record[f"{prefix}_first_frame"] = track["first_frame"]
            record[f"{prefix}_confirmed_frame"] = frame_index
            if prefix == "video":
                record["video_last_frame"] = frame_index
            record[f"{prefix}_occurrences"] = count
            record["temporal_average_quality"] = round(average_quality, 4)
            record[f"{prefix}_track_id"] = int(track["track_id"])
            record["best_selection_score"] = float(candidate.get("best_selection_score", -1.0))
            record["crop_visual_quality"] = float(candidate.get("best_visual_quality", 0.0))
            record["image_revision"] = frame_index
            session.confirmed_keys.add(key)
            track_id = int(track["track_id"])
            known_index = session.confirmed_tracks.get(track_id)
            published: dict[str, Any] | None = None
            if known_index is not None and known_index < len(session.confirmed_plates):
                existing = session.confirmed_plates[known_index]
                if not self._should_replace_live_record(existing, record):
                    continue
                record["id"] = existing.get("id", known_index + 1)
                record["live_result_key"] = existing.get(
                    "live_result_key", f"{prefix}:{known_index + 1}"
                )
                session.confirmed_plates[known_index] = record
                published = record
            else:
                existing_index = self._find_duplicate_record(
                    session.confirmed_plates,
                    record,
                    max_fuzzy_frame_gap=max(60, int(round(source_fps * 10.0))),
                )
                if existing_index is None:
                    record["live_result_key"] = f"{prefix}:{len(session.confirmed_plates) + 1}"
                    session.confirmed_plates.append(record)
                    session.confirmed_tracks[track_id] = len(session.confirmed_plates) - 1
                    published = record
                else:
                    existing = session.confirmed_plates[existing_index]
                    record["id"] = existing.get("id", existing_index + 1)
                    record["live_result_key"] = existing.get(
                        "live_result_key", f"{prefix}:{existing_index + 1}"
                    )
                    if self._record_quality(record) > self._record_quality(existing):
                        session.confirmed_plates[existing_index] = record
                        session.confirmed_tracks[track_id] = existing_index
                        published = record
            if published is None:
                continue
            # Archive only readings that actually reach the operator; a
            # variant that lost to an existing record leaves no stray files.
            self._persist_candidate_files(
                published,
                candidate,
                int(published.get("id", len(session.confirmed_plates))),
                output_stem,
                write_image,
                scanner,
                archive_camera_id=archive_camera_id,
                persist_archive=persist_archive,
            )
            public = dict(published)
            session.last_published.append(public)
            if not run_full_ocr:
                crop = candidate.get("best_ready_crop")
                if crop is not None:
                    session.pending_ocr.append(
                        (public, crop.copy() if hasattr(crop, "copy") else crop)
                    )
            if plate_callback is not None:
                plate_callback(dict(published))

    def _refresh_published_track(
        self,
        session: LiveScanSession,
        track: dict[str, Any],
        candidate: dict[str, Any],
        frame_index: int,
        *,
        write_image: Callable[..., Any] | None,
        scanner: Any,
        output_stem: str,
        archive_camera_id: str,
        media_prefix: str,
        run_full_ocr: bool,
        count: int,
        average_quality: float,
        plate_callback: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        """Replace a confirmed track when a later frame reads it more clearly."""

        track_id = int(track.get("track_id") or 0)
        known_index = session.confirmed_tracks.get(track_id)
        if known_index is None or known_index >= len(session.confirmed_plates):
            return
        existing = session.confirmed_plates[known_index]
        score = float(candidate.get("best_selection_score", -1.0))
        visual_quality = float(candidate.get("best_visual_quality", 0.0))
        existing_visual = float(existing.get("crop_visual_quality", 0.0))
        # Avoid rerunning materialisation for an unchanged candidate, but do
        # allow a substantially sharper close-up to pass even if its model
        # confidence is slightly lower than the distant frame.
        if (
            score <= float(existing.get("best_selection_score", -1.0))
            and visual_quality < existing_visual + 0.05
        ):
            return
        if write_image is None:
            return
        record = self._materialise_candidate(
            candidate,
            int(existing.get("id") or known_index + 1),
            output_stem,
            write_image,
            scanner,
            run_full_ocr=run_full_ocr,
            archive_camera_id=archive_camera_id,
            persist_archive=False,
            write_files=False,
        )
        if record is None:
            return
        record["id"] = existing.get("id", known_index + 1)
        record["live_result_key"] = existing.get(
            "live_result_key", f"{media_prefix}:{known_index + 1}"
        )
        for field in (
            "crop_image",
            "full_vehicle_image",
            "ocr_ready_image",
            "plate_json",
            "archive_filename",
            "archive_sequence",
        ):
            if existing.get(field):
                record[field] = existing.get(field)
        record[f"{media_prefix}_first_frame"] = existing.get(
            f"{media_prefix}_first_frame", track.get("first_frame")
        )
        record[f"{media_prefix}_confirmed_frame"] = frame_index
        if media_prefix == "video":
            record["video_last_frame"] = frame_index
        record[f"{media_prefix}_occurrences"] = count
        record["temporal_average_quality"] = round(average_quality, 4)
        record[f"{media_prefix}_track_id"] = track_id
        record["best_selection_score"] = score
        record["crop_visual_quality"] = float(candidate.get("best_visual_quality", 0.0))
        record["image_revision"] = frame_index
        # A sharper crop is not automatically a better recognition. Keep the
        # existing live value unless temporal confidence/recognition evidence
        # also shows that this candidate is stronger.
        if not self._should_replace_live_record(existing, record):
            return
        self._overwrite_published_images(existing, candidate, write_image)
        session.confirmed_keys.discard(self._video_plate_key(existing))
        session.confirmed_keys.add(self._video_plate_key(record))
        session.confirmed_plates[known_index] = record
        session.last_published.append(dict(record))
        self._refresh_plate_manifest(record)
        if plate_callback is not None:
            plate_callback(dict(record))

    @staticmethod
    def _overwrite_published_images(
        existing: dict[str, Any],
        candidate: dict[str, Any],
        write_image: Callable[..., Any],
    ) -> None:
        # Keep exactly two archive image files: the full frame and the
        # preprocessed crop that was actually passed to OCR. The legacy
        # ``ocr_ready_image`` field remains an alias for API compatibility.
        crop_image = candidate.get("best_ready_crop")
        full_image = candidate.get("best_vehicle_frame")
        written: set[str] = set()
        for field, image in (
            ("crop_image", crop_image),
            ("ocr_ready_image", crop_image),
            ("full_vehicle_image", full_image),
        ):
            path_text = existing.get(field)
            if not path_text or image is None or str(path_text) in written:
                continue
            write_image(Path(str(path_text)), image)
            written.add(str(path_text))

    @classmethod
    def _is_dominant_candidate(
        cls,
        track: dict[str, Any],
        candidate: dict[str, Any],
        key: str,
        frame_index: int,
        max_gap: int,
    ) -> bool:
        """Return whether ``candidate`` leads the recent readings of its track.

        Candidates are compared after one-digit aggregation, and only those
        still observed within the track gap count: a car that left the ROI
        must not block the next car stopping at the same spot.
        """

        count = int(candidate.get("count", 0))
        for other in track.get("candidates", {}).values():
            last_frame = other.get("last_frame")
            if last_frame is not None and int(frame_index) - int(last_frame) > max(1, int(max_gap)):
                continue
            aggregated = cls._aggregate_video_candidate(track, other)
            other_plate = aggregated.get("best_plate")
            if not isinstance(other_plate, dict) or not cls._has_complete_registration(other_plate):
                continue
            if cls._video_plate_key(other_plate) == key:
                continue
            if int(aggregated.get("count", 0)) > count:
                return False
        return True

    def process_live_frame(
        self,
        session: LiveScanSession,
        scanner: Any,
        frame: Any,
        frame_index: int,
        *,
        write_image: Callable[..., Any] | None,
        preprocess_plate_crop: Callable[..., Any] | None,
        stride: int,
        min_confirmations: int | None = None,
        output_stem: str,
        archive_camera_id: str,
        media: str = "video",
        source_fps: float = 25.0,
        plate_callback: Callable[[dict[str, Any]], None] | None = None,
        persist_archive: bool = True,
        scan_roi: dict[str, Any] | None = None,
        apply_roi: bool = False,
        run_full_ocr: bool | None = None,
    ) -> Any:
        """Detect, vote, and confirm one sampled frame with the video pipeline."""

        if apply_roi:
            self._apply_scanner_roi(scanner, scan_roi)
        session.sampled_frames += 1
        session.last_published = []
        session.pending_ocr = []
        if run_full_ocr is None:
            run_full_ocr = True if media != "camera" else bool(self.settings.full_accuracy_mode)
        session.track_gap_frames = self._track_gap_frames(
            session, frame_index, max(1, int(stride)), source_fps
        )
        session.last_sampled_frame = int(frame_index)
        plates, annotated, next_types = self._scan_stream_frame(
            scanner,
            frame,
            session.sampled_frames,
            session.cached_vehicle_types,
            run_ocr=False,
        )
        session.cached_vehicle_types = next_types
        session.last_detect_count = int(getattr(scanner, "last_plate_detection_count", len(plates)))
        annotated = self._update_stream_overlay(
            session,
            scanner,
            frame,
            plates,
            annotated,
            frame_index,
            max(1, int(stride)),
        )
        self._ingest_stream_plates(
            session,
            scanner,
            frame,
            plates,
            frame_index,
            write_image=write_image,
            preprocess_plate_crop=preprocess_plate_crop,
            min_confirmations=(
                self.settings.video_min_confirmations
                if min_confirmations is None
                else int(min_confirmations)
            ),
            output_stem=output_stem,
            archive_camera_id=archive_camera_id,
            media=media,
            source_fps=source_fps,
            plate_callback=plate_callback,
            persist_archive=persist_archive,
            run_full_ocr=bool(run_full_ocr),
        )
        return annotated

    @staticmethod
    def _track_gap_frames(
        session: LiveScanSession,
        frame_index: int,
        stride: int,
        source_fps: float,
    ) -> int:
        """Scale the track gap to the interval the pipeline actually samples at.

        A plate must survive a few *missed samples*, not a fixed number of
        source frames. Offline video keeps the historical 30-frame gap; a
        CCTV lane sampled every 60 frames gets roughly three samples of
        tolerance, capped so two cars queuing at the same spot several
        seconds apart are still separate tracks.
        """

        interval = max(1, int(stride))
        if session.last_sampled_frame >= 0:
            observed = int(frame_index) - int(session.last_sampled_frame)
            if observed > 0:
                interval = max(interval, observed)
        fps = float(source_fps) if source_fps and source_fps > 0 else 25.0
        ceiling = max(TRACK_GAP_MIN_FRAMES, int(round(fps * TRACK_GAP_MAX_SECONDS)))
        return max(
            TRACK_GAP_MIN_FRAMES,
            min(ceiling, interval * TRACK_GAP_SAMPLE_MULTIPLIER),
        )

    def infer_frame(
        self,
        frame: Any,
        *,
        sampled_frames: int,
        cached_vehicle_types: list[dict[str, Any]] | None,
        scan_roi: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], Any, list[dict[str, Any]] | None]:
        """Run one GPU pass using the same settings as video scanning."""

        self.load_runtime()
        self._apply_scanner_roi(self._scanner, scan_roi)
        return self._scan_stream_frame(
            self._scanner,
            frame,
            sampled_frames,
            cached_vehicle_types,
            run_ocr=False,
        )

    def _prepare_preview_frame(self, frame: Any) -> Any:
        """Return a bounded, independent preview without affecting scan pixels."""

        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            return frame
        height, width = frame.shape[:2]
        maximum = max(1, int(self.settings.preview_max_dimension))
        largest_side = max(height, width)
        if largest_side <= maximum:
            return frame.copy()
        scale = maximum / float(largest_side)
        return cv2.resize(
            frame,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    def _can_save_rejected_evidence(
        self,
        evidence_count: int,
        frame_index: int,
        last_evidence_frame: int,
    ) -> bool:
        """Rate-limit disk-heavy rejected samples during streaming scans."""

        limit = min(MAX_REJECTED_EVIDENCE_PER_RUN, self.settings.max_rejected_evidence_per_run)
        return (
            evidence_count < limit
            and frame_index - last_evidence_frame >= self.settings.rejected_evidence_frame_gap
        )

    def _write_result_manifest(
        self,
        result: dict[str, Any],
        result_path: Path,
    ) -> dict[str, Any]:
        """Write one portable result contract for image, video, and camera scans."""

        media_type = str(result.get("media_type") or "unknown")
        source_uri = str(result.get("input") or "")
        result["plates"] = [
            plate
            for plate in result.get("plates", [])
            if isinstance(plate, dict) and self._has_complete_registration(plate)
        ]
        result["plate_count"] = len(result["plates"])
        source: dict[str, Any] = {"type": media_type, "uri": source_uri}
        if media_type == "camera":
            source["camera_index"] = int(result.get("camera_index", 0))
            source["filename"] = f"camera_{source['camera_index']}"
        else:
            source["filename"] = Path(source_uri).name
        frame_rate = float(result.get("source_fps") or result.get("fps") or 0.0)
        for plate in result.get("plates", []):
            if not isinstance(plate, dict):
                continue
            plate["source"] = dict(source)
            if media_type in {"video", "camera"}:
                frame_id = plate.get(
                    "video_confirmed_frame",
                    plate.get("camera_confirmed_frame", plate.get("video_last_frame")),
                )
                if frame_id is not None:
                    frame = {"frame_id": int(frame_id), "camera": self.settings.archive_camera_id}
                    if frame_rate > 0:
                        frame["timestamp"] = round(int(frame_id) / frame_rate, 3)
                    plate["frame"] = frame
            dataset = plate.get("dataset")
            if isinstance(dataset, dict):
                dataset.setdefault("exported", False)
                dataset.setdefault("split", "unassigned")
                dataset.setdefault("country", "lao" if plate.get("country") == "lao" else "thai")
                dataset.setdefault("qc", str(plate.get("qc", {}).get("status") or "REVIEW"))
                dataset.setdefault("export_status", "PASS" if dataset["qc"] == "PASS" else "REJECT")
            self._refresh_plate_manifest(plate)
        # Never put rejected crop records into the persistent scan manifest.
        result["rejected_plates"] = []
        result["rejected_plate_count"] = 0
        result["schema_version"] = RESULT_SCHEMA_VERSION
        result["source"] = source
        log_path = result_path.parent.parent / "log" / f"{result_path.stem}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_lines = [
            f"source={source['uri']}",
            f"media_type={media_type}",
            f"plate_count={int(result.get('plate_count', 0))}",
            f"rejected_plate_count={int(result.get('rejected_plate_count', 0))}",
            f"created_at={datetime.now().isoformat(timespec='seconds')}",
        ]
        for plate in result.get("plates", []):
            if not isinstance(plate, dict):
                continue
            registration = f"{plate.get('plate_prefix') or ''}{plate.get('plate_number') or ''}"
            log_lines.append(
                f"PASS\tcountry={plate.get('country', 'unknown')}\tplate={registration or '-'}"
                f"\tconfidence={float(plate.get('recognition_confidence', 0.0)):.4f}"
            )
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        result["artifacts"] = {
            "annotated_image": str(result.get("annotated_image") or ""),
            "annotated_video": str(result.get("output_video") or ""),
            "result_json": str(result_path),
            "detection_log": str(log_path),
        }
        result["result_json"] = str(result_path)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def scan_file(self, image_path: Path) -> dict[str, Any]:
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"ไม่พบไฟล์ภาพ: {image_path}")
        if not 0 <= self.settings.padding < 1:
            raise ValueError("ค่า padding ต้องอยู่ระหว่าง 0 ถึง 1")

        # The legacy engine is imported here so opening the GUI does not load
        # OpenCV/YOLO until the user actually starts a scan.
        from scan import CustomPaddleOCR, LicensePlateScanner, preprocess_plate_crop, read_image, write_image

        image = read_image(image_path)
        ocr = CustomPaddleOCR(
            self.settings.ocr_source,
            self.settings.ocr_config,
            self.settings.ocr_weights,
            self.settings.torch_ocr_model,
            lao_config=self.settings.ocr_lao_config,
            lao_weights=self.settings.ocr_lao_weights,
            thai_config=self.settings.ocr_thai_config,
            thai_weights=self.settings.ocr_thai_weights,
        )
        scanner = LicensePlateScanner(
            self.settings.detector_model,
            self.settings.thai_model,
            self.settings.lao_model,
            ocr,
            self.settings.vehicle_type_model,
            plate_type_path=self.settings.plate_type_model,
            pipeline_config_path=self.settings.pipeline_config,
            pipeline_mode=self.settings.pipeline_mode,
            scan_roi_override=self.settings.scan_roi_override,
        )
        session = self.new_live_session()
        annotated = self.process_live_frame(
            session,
            scanner,
            image,
            1,
            write_image=write_image,
            preprocess_plate_crop=preprocess_plate_crop,
            stride=1,
            min_confirmations=1,
            output_stem=image_path.stem,
            archive_camera_id=self.settings.archive_camera_id,
            media="image",
            source_fps=1.0,
            persist_archive=True,
        )
        plates = session.confirmed_plates
        for index, plate in enumerate(plates, start=1):
            plate["id"] = index

        scan_output_dir = self._dated_output_dir()
        annotated_path = scan_output_dir / "log" / f"{image_path.stem}_annotated.jpg"
        write_image(annotated_path, annotated)
        if self.settings.debug_enabled:
            self._write_image_debug_artifacts(
                image_path.stem,
                image,
                annotated,
                plates,
                [],
                preprocess_plate_crop,
                write_image,
                debug_dir=scan_output_dir / "debug",
            )
        result = {
            "media_type": "image",
            "input": str(image_path),
            "plate_count": len(plates),
            "rejected_plate_count": len(session.rejected_evidence),
            "rejected_plates": session.rejected_evidence,
            "plates": plates,
            "annotated_image": str(annotated_path),
            "output_dir": str(scan_output_dir),
            "sampled_frames": session.sampled_frames,
        }
        return self._write_result_manifest(
            result,
            scan_output_dir / "json" / f"{image_path.stem}_result.json",
        )

    def scan_video(
        self,
        video_path: Path,
        frame_callback: Callable[[Any, int, int], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
        plate_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Scan a video using plate crops only and write an annotated MP4.

        Detection is sampled at the configured scan rate. Repeated sightings
        are tracked by position and resolved with temporal confidence voting;
        expensive full OCR is deferred until a plate is confirmed.
        """

        video_path = video_path.resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"ไม่พบไฟล์วิดีโอ: {video_path}")
        if not 0 <= self.settings.padding < 1:
            raise ValueError("ค่า padding ต้องอยู่ระหว่าง 0 ถึง 1")

        from scan import CustomPaddleOCR, LicensePlateScanner, preprocess_plate_crop, write_image

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"ไม่สามารถเปิดวิดีโอ: {video_path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        # Keep the source playback speed.  ``target_fps`` controls preview
        # callbacks only; using it for the writer makes 30/60 FPS videos play
        # back in slow motion when the UI target is 24 FPS.
        output_fps = source_fps
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            capture.release()
            raise ValueError("วิดีโอไม่มีขนาดเฟรมที่ใช้งานได้")

        scan_output_dir = self._dated_output_dir()
        output_video = scan_output_dir / "video" / f"{video_path.stem}_annotated.mp4"
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("ไม่สามารถสร้างไฟล์วิดีโอผลลัพธ์ MP4")

        ocr = CustomPaddleOCR(
            self.settings.ocr_source,
            self.settings.ocr_config,
            self.settings.ocr_weights,
            self.settings.torch_ocr_model,
            lao_config=self.settings.ocr_lao_config,
            lao_weights=self.settings.ocr_lao_weights,
            thai_config=self.settings.ocr_thai_config,
            thai_weights=self.settings.ocr_thai_weights,
        )
        scanner = LicensePlateScanner(
            self.settings.detector_model,
            self.settings.thai_model,
            self.settings.lao_model,
            ocr,
            self.settings.vehicle_type_model,
            plate_type_path=self.settings.plate_type_model,
            pipeline_config_path=self.settings.pipeline_config,
            pipeline_mode=self.settings.pipeline_mode,
            scan_roi_override=self.settings.scan_roi_override,
        )
        session = self.new_live_session()
        frame_index = 0
        effective_stride = (
            1
            if self.settings.full_accuracy_mode
            else max(
                self.settings.video_frame_stride,
                int(round(source_fps / self.settings.video_target_scans_per_second)),
            )
        )
        last_preview_at = 0.0
        preview_interval = 1.0 / self.settings.preview_fps
        cancelled = False
        last_annotated_frame: Any = None

        try:
            while True:
                if stop_requested is not None and stop_requested():
                    cancelled = True
                    break
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                annotated = frame
                if (frame_index - 1) % effective_stride == 0:
                    annotated = self.process_live_frame(
                        session,
                        scanner,
                        frame,
                        frame_index,
                        write_image=write_image,
                        preprocess_plate_crop=preprocess_plate_crop,
                        stride=effective_stride,
                        min_confirmations=self.settings.video_min_confirmations,
                        output_stem=f"{video_path.stem}_live",
                        archive_camera_id=self.settings.archive_camera_id,
                        media="video",
                        source_fps=source_fps,
                        plate_callback=plate_callback,
                        persist_archive=True,
                    )
                else:
                    annotated = scanner.draw_scan_roi(frame)
                    if session.cached_plates:
                        annotated = self._draw_cached_plates(annotated, session.cached_plates)
                writer.write(annotated)
                # ``annotated`` is already a fresh buffer from draw_scan_roi
                # or scanner.scan.  Avoid a second full-resolution copy for
                # every video frame just to retain the final snapshot.
                last_annotated_frame = annotated
                now = time.monotonic()
                if frame_callback is not None and (frame_index == 1 or now - last_preview_at >= preview_interval):
                    frame_callback(self._prepare_preview_frame(annotated), frame_index, frame_count)
                    last_preview_at = now
        finally:
            capture.release()
            writer.release()

        plates = self._finalise_video_tracks(
            session.tracks,
            video_path.stem,
            write_image,
            scanner,
            min_confirmations=self.settings.video_min_confirmations,
            dedupe_frame_gap=max(60, int(round(source_fps * 10.0))),
        )
        annotated_image = scan_output_dir / "log" / f"{video_path.stem}_annotated.jpg"
        if last_annotated_frame is not None:
            write_image(annotated_image, last_annotated_frame)
        result = {
            "media_type": "video",
            "input": str(video_path),
            "plate_count": len(plates),
            "rejected_plate_count": len(session.rejected_evidence),
            "rejected_plates": session.rejected_evidence,
            "plates": plates,
            "frame_count": frame_index,
            "fps": round(output_fps, 3),
            "source_fps": round(source_fps, 3),
            "target_fps": round(self.settings.target_fps, 3),
            "scan_stride": effective_stride,
            "sampled_frames": session.sampled_frames,
            "annotated_image": str(annotated_image) if annotated_image.is_file() else "",
            "output_video": str(output_video),
            "output_dir": str(scan_output_dir),
            "cancelled": cancelled,
        }
        return self._write_result_manifest(
            result,
            scan_output_dir / "json" / f"{video_path.stem}_video_result.json",
        )

    def scan_camera(
        self,
        camera_index: int = 0,
        frame_callback: Callable[[Any, int], None] | None = None,
        plate_callback: Callable[[dict[str, Any]], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
        stop_after_first: bool = False,
        source: str | None = None,
    ) -> dict[str, Any]:
        """Scan a live camera or IP/RTSP stream until stopped."""

        from scan import CustomPaddleOCR, LicensePlateScanner, preprocess_plate_crop, write_image

        stream = str(source or "").strip()
        if stream:
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            capture = cv2.VideoCapture(stream, cv2.CAP_FFMPEG)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(stream)
            if not capture.isOpened():
                raise ValueError("ไม่สามารถเปิดกล้อง IP/RTSP ได้ กรุณาตรวจ URL และบัญชีกล้อง")
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            source_label = redact_stream_url(stream)
        else:
            backend = cv2.CAP_DSHOW if os.name == "nt" and hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
            capture = cv2.VideoCapture(camera_index, backend)
            if not capture.isOpened() and backend != cv2.CAP_ANY:
                capture.release()
                capture = cv2.VideoCapture(camera_index)
            if not capture.isOpened():
                raise ValueError(f"ไม่สามารถเปิดกล้องหมายเลข {camera_index} ได้ กรุณาตรวจสิทธิ์กล้องหรือเลือกกล้องอื่น")
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            capture.set(cv2.CAP_PROP_FPS, self.settings.target_fps)
            source_label = f"camera://{camera_index}"
        ocr = CustomPaddleOCR(
            self.settings.ocr_source,
            self.settings.ocr_config,
            self.settings.ocr_weights,
            self.settings.torch_ocr_model,
            lao_config=self.settings.ocr_lao_config,
            lao_weights=self.settings.ocr_lao_weights,
            thai_config=self.settings.ocr_thai_config,
            thai_weights=self.settings.ocr_thai_weights,
        )
        scanner = LicensePlateScanner(
            self.settings.detector_model,
            self.settings.thai_model,
            self.settings.lao_model,
            ocr,
            self.settings.vehicle_type_model,
            plate_type_path=self.settings.plate_type_model,
            pipeline_config_path=self.settings.pipeline_config,
            pipeline_mode=self.settings.pipeline_mode,
            scan_roi_override=self.settings.scan_roi_override,
        )
        scan_output_dir = self._dated_output_dir()
        session_stem = f"camera_{'ip' if stream else camera_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = self.new_live_session()
        frame_index = 0
        last_annotated_frame: Any = None
        # In accuracy mode do not intentionally throttle capture: inference remains the
        # limiting factor, but every frame delivered by the camera is considered.
        frame_interval = 0.0 if self.settings.full_accuracy_mode else 1.0 / self.settings.target_fps
        preview_interval = 1.0 / self.settings.preview_fps
        camera_stride = (
            1 if self.settings.full_accuracy_mode else self.settings.camera_frame_stride
        )

        # Capture and inference are decoupled with a one-frame latest-value
        # buffer. Slow model calls can no longer stop the live preview or build
        # an ever-growing queue of stale camera frames.
        camera_lock = threading.Lock()
        frame_available = threading.Event()
        reader_stop = threading.Event()
        camera_state: dict[str, Any] = {
            "latest": None,
            "frame_count": 0,
            "error": "",
            "cached_plates": [],
        }

        def capture_latest_frame() -> None:
            failed_reads = 0
            last_preview_at = 0.0
            next_frame_at = time.monotonic()
            while not reader_stop.is_set() and (
                stop_requested is None or not stop_requested()
            ):
                wait_for_frame = next_frame_at - time.monotonic()
                if wait_for_frame > 0 and reader_stop.wait(wait_for_frame):
                    break
                next_frame_at = max(next_frame_at + frame_interval, time.monotonic())
                ok, captured = capture.read()
                if not ok:
                    failed_reads += 1
                    if failed_reads >= 30:
                        with camera_lock:
                            camera_state["error"] = (
                                "กล้องหยุดส่งภาพ กรุณาปิดโปรแกรมอื่นที่กำลังใช้กล้องแล้วลองใหม่"
                            )
                        frame_available.set()
                        return
                    continue
                failed_reads = 0
                with camera_lock:
                    camera_state["frame_count"] += 1
                    captured_index = int(camera_state["frame_count"])
                    camera_state["latest"] = (captured_index, captured)
                    preview_plates = list(camera_state["cached_plates"])
                frame_available.set()

                now = time.monotonic()
                if frame_callback is None or (
                    captured_index != 1 and now - last_preview_at < preview_interval
                ):
                    continue
                preview = scanner.draw_scan_roi(captured)
                if preview_plates:
                    preview = self._draw_cached_plates(preview, preview_plates)
                frame_callback(self._prepare_preview_frame(preview), captured_index)
                last_preview_at = now

        reader_thread = threading.Thread(
            target=capture_latest_frame,
            name=f"car-scan-camera-{camera_index}",
            daemon=True,
        )
        reader_thread.start()

        try:
            while stop_requested is None or not stop_requested():
                frame_available.wait(0.10)
                frame_available.clear()
                with camera_lock:
                    reader_error = str(camera_state["error"] or "")
                    latest = camera_state["latest"]
                if reader_error:
                    raise RuntimeError(reader_error)
                if latest is None or int(latest[0]) <= frame_index:
                    continue
                latest_index = int(latest[0])
                frame = latest[1].copy()
                frame_index = latest_index
                annotated = frame
                if (frame_index - 1) % camera_stride == 0:
                    annotated = self.process_live_frame(
                        session,
                        scanner,
                        frame,
                        frame_index,
                        write_image=write_image,
                        preprocess_plate_crop=preprocess_plate_crop,
                        stride=camera_stride,
                        min_confirmations=self.settings.camera_min_confirmations,
                        output_stem=session_stem,
                        archive_camera_id=str(camera_index),
                        media="camera",
                        source_fps=self.settings.target_fps,
                        plate_callback=plate_callback,
                        run_full_ocr=self.settings.full_accuracy_mode,
                    )
                    with camera_lock:
                        camera_state["cached_plates"] = list(session.cached_plates)
                    if stop_after_first and session.confirmed_plates:
                        last_annotated_frame = annotated
                        break
                else:
                    annotated = scanner.draw_scan_roi(frame)
                    if session.cached_plates:
                        annotated = self._draw_cached_plates(annotated, session.cached_plates)

                last_annotated_frame = annotated
        finally:
            reader_stop.set()
            frame_available.set()
            reader_thread.join(timeout=1.0)
            capture.release()
            if reader_thread.is_alive():
                reader_thread.join(timeout=1.0)
            with camera_lock:
                frame_index = max(frame_index, int(camera_state["frame_count"]))

        snapshot_path = scan_output_dir / "log" / f"{session_stem}_annotated.jpg"
        if last_annotated_frame is not None:
            write_image(snapshot_path, last_annotated_frame)
        result = {
            "media_type": "camera",
            "input": source_label,
            "camera_index": camera_index,
            "plate_count": len(session.confirmed_plates),
            "rejected_plate_count": len(session.rejected_evidence),
            "rejected_plates": session.rejected_evidence,
            "plates": session.confirmed_plates,
            "frame_count": frame_index,
            "sampled_frames": session.sampled_frames,
            "target_fps": round(self.settings.target_fps, 3),
            "annotated_image": str(snapshot_path) if last_annotated_frame is not None else "",
            "output_dir": str(scan_output_dir),
        }
        return self._write_result_manifest(
            result,
            scan_output_dir / "json" / f"{session_stem}_result.json",
        )

    def _merge_video_plate(
        self,
        tracks: list[dict[str, Any]],
        plate: dict[str, Any],
        frame: Any,
        frame_index: int,
        preprocess_plate_crop: Callable[[Any], Any],
        max_gap: int = TRACK_GAP_MIN_FRAMES,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Add one valid reading to its spatial track and retain its best crop."""

        if not self._has_usable_registration(plate):
            return None
        country = str(plate.get("country", ""))
        box = [int(value) for value in plate["box"]]
        track = self._find_video_track(tracks, country, box, frame_index, max_gap=max_gap)
        if track is None:
            track = {
                "track_id": len(tracks) + 1,
                "country": country,
                "first_frame": frame_index,
                "last_frame": frame_index,
                "last_box": box,
                "observations": 0,
                "candidates": {},
            }
            tracks.append(track)

        track["last_frame"] = frame_index
        track["last_box"] = box
        track["observations"] += 1
        key = self._video_plate_key(plate)
        candidate = track["candidates"].setdefault(
            key,
            {
                "count": 0,
                "vote_score": 0.0,
                "best_quality": -1.0,
                "best_selection_score": -1.0,
                "best_visual_quality": 0.0,
                "best_plate": None,
                "best_crop": None,
                "best_ready_crop": None,
                "best_vehicle_frame": None,
                "province_votes": {},
            },
        )
        quality = self._video_plate_quality(plate)
        candidate["count"] += 1
        candidate["vote_score"] += quality
        candidate["last_frame"] = frame_index

        reading = plate.get("country_readings", {}).get(country, {})
        if isinstance(reading, dict):
            province_votes = candidate.setdefault("province_votes", {})
            raw_candidates = reading.get("province_candidates", [])
            province_candidates = raw_candidates if isinstance(raw_candidates, list) else []
            if not province_candidates and reading.get("province_code"):
                province_candidates = [
                    {
                        "code": reading.get("province_code"),
                        "province": reading.get("province"),
                        "confidence": reading.get("province_confidence", 0.0),
                        "source": reading.get("province_status", "plate"),
                    }
                ]
            # One frame can be too blurred to pass the still-image threshold.
            # Vote only once per code per frame, then require repeated support
            # before a weak candidate can become an archived province.
            seen_codes: set[str] = set()
            for province_candidate in province_candidates:
                if not isinstance(province_candidate, dict):
                    continue
                province_code = str(province_candidate.get("code") or "")
                province_confidence = float(province_candidate.get("confidence", 0.0))
                if (
                    not province_code
                    or province_code in seen_codes
                    or province_confidence < TEMPORAL_PROVINCE_MIN_CONFIDENCE
                ):
                    continue
                seen_codes.add(province_code)
                province_vote = province_votes.setdefault(
                    province_code,
                    {
                        "count": 0,
                        "score": 0.0,
                        "best_confidence": -1.0,
                        "snapshot": {},
                    },
                )
                province_vote["count"] += 1
                province_vote["score"] += province_confidence
                if province_confidence > float(province_vote["best_confidence"]):
                    province_vote["best_confidence"] = province_confidence
                    province_vote["snapshot"] = {
                        "province": str(province_candidate.get("province") or ""),
                        "province_code": province_code,
                        "province_confidence": province_confidence,
                        "province_status": str(province_candidate.get("source") or "temporal_candidate"),
                        "province_margin": float(reading.get("province_margin", 0.0)),
                    }
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        visual_quality = self._crop_visual_quality(crop)
        # Confidence selects a semantically trustworthy frame; sharpness is a
        # small tie-breaker so final OCR receives the clearest crop instead of
        # a blurred frame that happened to score a fraction higher.
        # Model confidence can stay high on a distant, blurred plate. Blend
        # semantic quality with crop sharpness so an approaching vehicle gets
        # a fresh OCR crop even when its raw confidence is not calibrated.
        semantic_quality = max(0.0, min(1.0, quality))
        selection_score = semantic_quality * 0.60 + visual_quality * 0.40

        best_plate = candidate.get("best_plate")
        if isinstance(best_plate, dict):
            previous_vehicle = str(best_plate.get("vehicle_type") or "unknown")
            current_vehicle = str(plate.get("vehicle_type") or "unknown")
            if previous_vehicle == "unknown" and current_vehicle != "unknown":
                for field in (
                    "vehicle_type",
                    "vehicle_type_confidence",
                    "vehicle_type_box",
                    "vehicle_id",
                    "vehicle",
                ):
                    best_plate[field] = plate.get(field)
                candidate["best_vehicle_frame"] = frame.copy()

        if selection_score <= float(candidate.get("best_selection_score", -1.0)):
            return track, self._aggregate_video_candidate(track, candidate)

        ox1, oy1, ox2, oy2 = [int(value) for value in plate["ocr_crop_box"]]
        candidate["best_quality"] = quality
        candidate["best_selection_score"] = selection_score
        candidate["best_visual_quality"] = round(visual_quality, 4)
        candidate["best_plate"] = dict(plate)
        candidate["best_crop"] = crop.copy()
        candidate["best_vehicle_frame"] = frame.copy()
        processing = plate.get("processing", {})
        parameters = (
            processing.get("preprocessing_parameters", {})
            if isinstance(processing, dict)
            else {}
        )
        candidate["best_ready_crop"] = (
            preprocess_plate_crop(frame[oy1:oy2, ox1:ox2], parameters)
            if parameters
            else preprocess_plate_crop(frame[oy1:oy2, ox1:ox2])
        )
        return track, self._aggregate_video_candidate(track, candidate)

    @staticmethod
    def _crop_visual_quality(crop: Any) -> float:
        """Estimate useful plate detail on a stable 0..1 scale."""

        if crop is None or not hasattr(crop, "shape") or crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        height, width = gray.shape[:2]
        if max(height, width) > 320:
            scale = 320.0 / max(height, width)
            gray = cv2.resize(
                gray,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        # Around 1,000 is already a very crisp plate crop. Saturating keeps
        # sensor noise from overpowering recognition confidence.
        return max(0.0, min(1.0, variance / 1000.0))

    @staticmethod
    def _near_registration_variant(first: dict[str, Any], second: dict[str, Any]) -> bool:
        """Return true for one unstable digit within one spatial track."""

        if str(first.get("country") or "") != str(second.get("country") or ""):
            return False
        if str(first.get("plate_prefix") or "") != str(second.get("plate_prefix") or ""):
            return False
        first_number = str(first.get("plate_number") or "")
        second_number = str(second.get("plate_number") or "")
        if not first_number or len(first_number) != len(second_number):
            return False
        return sum(a != b for a, b in zip(first_number, second_number)) == 1

    @classmethod
    def _aggregate_video_candidate(
        cls,
        track: dict[str, Any],
        anchor: dict[str, Any],
    ) -> dict[str, Any]:
        """Combine a dominant reading with one-digit temporal variants.

        The dominant exact text must occur at least twice. This recovers a
        stable result such as 1356, 1356, 1358 without letting three unrelated
        single-frame guesses become a confirmed registration.
        """

        anchor_plate = anchor.get("best_plate")
        if not isinstance(anchor_plate, dict):
            return anchor
        nearby = [
            candidate
            for candidate in track.get("candidates", {}).values()
            if candidate is anchor
            or (
                isinstance(candidate.get("best_plate"), dict)
                and cls._near_registration_variant(anchor_plate, candidate["best_plate"])
            )
        ]
        if len(nearby) <= 1:
            return anchor
        dominant = max(
            nearby,
            key=lambda item: (
                int(item.get("count", 0)),
                float(item.get("vote_score", 0.0)) / max(1, int(item.get("count", 0))),
                float(item.get("best_selection_score", item.get("best_quality", -1.0))),
            ),
        )
        if int(dominant.get("count", 0)) < 2:
            return anchor

        aggregate = dict(dominant)
        aggregate["count"] = sum(int(item.get("count", 0)) for item in nearby)
        aggregate["vote_score"] = sum(float(item.get("vote_score", 0.0)) for item in nearby)
        aggregate["variant_support"] = {
            cls._video_plate_key(item["best_plate"]): int(item.get("count", 0))
            for item in nearby
        }
        province_votes: dict[str, dict[str, Any]] = {}
        for item in nearby:
            for code, vote in item.get("province_votes", {}).items():
                if not isinstance(vote, dict):
                    continue
                combined = province_votes.setdefault(
                    str(code),
                    {"count": 0, "score": 0.0, "best_confidence": -1.0, "snapshot": {}},
                )
                combined["count"] += int(vote.get("count", 0))
                combined["score"] += float(vote.get("score", 0.0))
                if float(vote.get("best_confidence", -1.0)) > float(
                    combined["best_confidence"]
                ):
                    combined["best_confidence"] = float(vote.get("best_confidence", -1.0))
                    combined["snapshot"] = dict(vote.get("snapshot", {}))
        aggregate["province_votes"] = province_votes
        return aggregate

    @staticmethod
    def _apply_temporal_province_vote(winner: dict[str, Any]) -> dict[str, Any] | None:
        """Apply the majority province result to the best temporal plate crop."""

        best = winner.get("best_plate")
        if not isinstance(best, dict):
            return None
        province_votes = winner.get("province_votes", {})
        if not isinstance(province_votes, dict) or not province_votes:
            return dict(best)

        ranked_votes = sorted(
            province_votes.values(),
            key=lambda value: (
                int(value.get("count", 0)),
                float(value.get("score", 0.0)),
                float(value.get("best_confidence", 0.0)),
            ),
            reverse=True,
        )
        selected = ranked_votes[0]
        snapshot = selected.get("snapshot", {})
        if not isinstance(snapshot, dict):
            snapshot = {}
        best_confidence = float(selected.get("best_confidence", 0.0))
        repeated = (
            int(selected.get("count", 0)) >= TEMPORAL_PROVINCE_MIN_OBSERVATIONS
            and float(selected.get("score", 0.0)) / max(1, int(selected.get("count", 0)))
            >= TEMPORAL_PROVINCE_MIN_CONFIDENCE
        )
        runner_up_score = float(ranked_votes[1].get("score", 0.0)) if len(ranked_votes) > 1 else 0.0
        has_margin = len(ranked_votes) == 1 or float(selected.get("score", 0.0)) >= (
            runner_up_score + TEMPORAL_PROVINCE_SCORE_MARGIN
        )
        if best_confidence < 0.45 and not (repeated and has_margin):
            return dict(best)
        record = dict(best)
        country = str(record.get("country", ""))
        record["province"] = str(snapshot.get("province") or "")
        record["province_code"] = str(snapshot.get("province_code") or "")
        readings = record.get("country_readings", {})
        if isinstance(readings, dict):
            readings = dict(readings)
            reading = readings.get(country, {})
            if isinstance(reading, dict):
                reading = dict(reading)
                reading.update(
                    {
                        "province": record["province"],
                        "province_code": str(snapshot.get("province_code") or ""),
                        "province_confidence": round(float(snapshot.get("province_confidence", 0.0)), 4),
                        "province_status": (
                            "temporal_consensus"
                            if best_confidence >= 0.45
                            else "temporal_weak_consensus"
                        )
                        if record["province"]
                        else "temporal_unconfirmed",
                        "province_margin": round(float(snapshot.get("province_margin", 0.0)), 4),
                    }
                )
                readings[country] = reading
            record["country_readings"] = readings
        plate_result = record.get("plate")
        if isinstance(plate_result, dict):
            plate_result = dict(plate_result)
            plate_result["province"] = record["province"]
            record["plate"] = plate_result
        return record

    def _finalise_video_tracks(
        self,
        tracks: list[dict[str, Any]],
        video_stem: str,
        write_image: Callable[[Path, Any], None],
        scanner: Any,
        min_confirmations: int = 1,
        dedupe_frame_gap: int = 300,
    ) -> list[dict[str, Any]]:
        """Return one vote-backed, highest-quality result for each tracked plate."""

        output: list[dict[str, Any]] = []
        for track in tracks:
            consensus: dict[str, dict[str, Any]] = {}
            for candidate in track["candidates"].values():
                aggregated = self._aggregate_video_candidate(track, candidate)
                best_plate = aggregated.get("best_plate", {})
                key = self._video_plate_key(best_plate) if isinstance(best_plate, dict) else ""
                previous = consensus.get(key)
                if previous is None or int(aggregated.get("count", 0)) > int(
                    previous.get("count", 0)
                ):
                    consensus[key] = aggregated
            candidates = [
                candidate
                for candidate in consensus.values()
                if int(candidate.get("count", 0)) >= min_confirmations
                and float(candidate.get("vote_score", 0.0)) / max(1, int(candidate.get("count", 0)))
                >= self.settings.temporal_min_quality
            ]
            if not candidates:
                continue
            # Repeated readings are decisive, but a very clear single-frame
            # plate remains valid for short clips.  Choose the text that was
            # read consistently first; a detector's per-character confidence
            # only breaks a tie.  In particular, a single very confident
            # misread such as 1356 must not beat several readings of 6356.
            winner = max(
                candidates,
                key=lambda item: (
                    int(item["count"]),
                    float(item["vote_score"]) / max(1, int(item["count"])),
                    float(item["best_quality"]),
                ),
            )
            record = self._materialise_candidate(
                winner,
                len(output) + 1,
                video_stem,
                write_image,
                scanner,
                archive_camera_id=self.settings.archive_camera_id,
                write_files=False,
            )
            if record is None:
                continue
            record["video_first_frame"] = track["first_frame"]
            record["video_last_frame"] = track["last_frame"]
            record["video_occurrences"] = int(winner["count"])
            record["video_track_observations"] = int(track["observations"])
            record["video_vote_score"] = round(float(winner["vote_score"]), 4)
            record["_archive_candidate"] = winner
            output.append(record)
        output = self._deduplicate_video_records(output, dedupe_frame_gap)
        for record_id, record in enumerate(output, start=1):
            candidate = record.pop("_archive_candidate", None)
            if not isinstance(candidate, dict):
                continue
            record["id"] = record_id
            self._persist_candidate_files(
                record,
                candidate,
                record_id,
                video_stem,
                write_image,
                scanner,
                archive_camera_id=self.settings.archive_camera_id,
                persist_archive=True,
            )
        return output

    def _apply_reading_and_quality(
        self,
        record: dict[str, Any],
        scanner: Any,
        crop: Any,
        *,
        run_full_ocr: bool,
    ) -> dict[str, Any]:
        """Fuse structured YOLO text with optional OCR and refresh quality."""

        country = str(record.get("country", ""))
        readings = record.get("country_readings", {})
        reading = readings.get(country, {}) if isinstance(readings, dict) else {}
        attach_province = getattr(scanner, "_attach_province_candidates", None)
        if (
            callable(attach_province)
            and country == "thai"
            and run_full_ocr
            and isinstance(reading, dict)
            and crop is not None
            and not str(reading.get("province_code") or "")
        ):
            try:
                reading = attach_province(
                    reading, crop, country, self.settings.character_confidence, self.settings.imgsz, fast_mode=False
                )
                if isinstance(readings, dict):
                    readings[country] = reading
            except Exception:
                LOGGER.exception("Thai province pass failed during confirmation")
        if isinstance(reading, dict):
            finalise_country = getattr(scanner, "finalise_country_reading", None)
            if callable(finalise_country) and isinstance(readings, dict):
                country, ocr, prefix, number = finalise_country(
                    crop, country, readings, run_ocr=run_full_ocr
                )
                record["country"] = country
                reading = readings.get(country, reading)
            else:
                ocr, prefix, number = scanner.finalise_reading(
                    crop,
                    country,
                    reading,
                    run_ocr=run_full_ocr,
                )
            record["ocr"] = ocr
            record["plate_prefix"] = prefix
            record["plate_number"] = number
            province = str(reading.get("province") or ocr.get("province") or record.get("province") or "")
            province_code = str(
                reading.get("province_code") or ocr.get("province_code") or record.get("province_code") or ""
            )
            record["province"] = province
            record["province_code"] = province_code
        vehicle = record.get("vehicle", {})
        vehicle_type = str(
            (
                vehicle.get("normalized_class")
                or vehicle.get("type")
                or record.get("vehicle_type")
                or "unknown"
            )
            if isinstance(vehicle, dict)
            else record.get("vehicle_type") or "unknown"
        )
        plate_type = record.get("plate_type", {})
        normalized_plate_type = str(
            plate_type.get("normalized_class")
            if isinstance(plate_type, dict)
            else plate_type or "unknown"
        )
        validator = getattr(scanner, "context_validator", None)
        if validator is not None:
            record["validation"] = validator.validate(
                country,
                vehicle_type or "unknown",
                normalized_plate_type or "unknown",
                str(record.get("plate_prefix") or ""),
                str(record.get("plate_number") or ""),
                str(record.get("province") or ""),
            )
        from scan import build_plate_quality

        evidence = record.get("cross_model_digit_evidence", {})
        primary_country = (
            str(evidence.get("primary_country"))
            if isinstance(evidence, dict) and evidence.get("primary_country")
            else country
        )
        quality = build_plate_quality(
            country=country,
            primary_country=primary_country,
            readings=record.get("country_readings", {}),
            plate_prefix=str(record.get("plate_prefix") or ""),
            plate_number=str(record.get("plate_number") or ""),
            ocr=record.get("ocr", {}),
            validation=record.get("validation", {}),
            detection_confidence=float(record.get("detection_confidence", 0.0)),
            country_confidence=float(record.get("country_confidence", 0.0)),
            digit_evidence=evidence if isinstance(evidence, dict) else {},
        )
        record.update(quality)
        plate_result = record.get("plate")
        if isinstance(plate_result, dict):
            plate_result.update(
                {
                    "prefix": record.get("plate_prefix", ""),
                    "number": record.get("plate_number", ""),
                    "text": record.get("ocr", {}).get("text", ""),
                    "ocr_confidence": record.get("ocr", {}).get("confidence", 0.0),
                    "ocr": record.get("ocr", {}),
                    "validation": record.get("validation", {}),
                    "province": record.get("province", ""),
                    "province_code": record.get("province_code", ""),
                }
            )
        record["confirmed"] = self._has_complete_registration(record)
        return record

    def refine_deferred_ocr(self, record: dict[str, Any], crop: Any) -> dict[str, Any] | None:
        """Run PaddleOCR after a CCTV plate is already on screen."""

        scanner = self._scanner
        if scanner is None or crop is None:
            return None
        trial = dict(record)
        readings = record.get("country_readings")
        if isinstance(readings, dict):
            trial["country_readings"] = {
                key: dict(value) if isinstance(value, dict) else value
                for key, value in readings.items()
            }
        before = (
            str(record.get("country") or ""),
            str(record.get("plate_prefix") or ""),
            str(record.get("plate_number") or ""),
            str(record.get("province") or ""),
        )
        self._apply_reading_and_quality(trial, scanner, crop, run_full_ocr=True)
        after = (
            str(trial.get("country") or ""),
            str(trial.get("plate_prefix") or ""),
            str(trial.get("plate_number") or ""),
            str(trial.get("province") or ""),
        )
        ocr = trial.get("ocr") if isinstance(trial.get("ocr"), dict) else {}
        improved = after != before or str(ocr.get("method") or "").startswith(
            ("custom-paddleocr", "tesseract-thai", "tesseract-lao")
        )
        if not trial.get("confirmed") or not self._has_complete_registration(trial):
            return None
        record.update(trial)
        return record

    def _materialise_candidate(
        self,
        winner: dict[str, Any],
        record_id: int,
        output_stem: str,
        write_image: Callable[[Path, Any], None],
        scanner: Any,
        run_full_ocr: bool = True,
        archive_camera_id: str = "0",
        persist_archive: bool = True,
        write_files: bool = True,
    ) -> dict[str, Any] | None:
        """Run full OCR once and persist the best temporal crop.

        ``write_files=False`` returns the finished record without touching
        disk so a caller can decide whether the reading is actually going to
        be published before archiving it; call ``_persist_candidate_files``
        afterwards for the records that survive.
        """

        best = self._apply_temporal_province_vote(winner)
        if not isinstance(best, dict) or winner.get("best_crop") is None or winner.get("best_ready_crop") is None:
            return None
        record = dict(best)
        record["id"] = record_id
        self._apply_reading_and_quality(
            record,
            scanner,
            winner["best_ready_crop"],
            run_full_ocr=run_full_ocr,
        )
        if not record["confirmed"]:
            return None
        if not write_files:
            return record
        return self._persist_candidate_files(
            record,
            winner,
            record_id,
            output_stem,
            write_image,
            scanner,
            archive_camera_id=archive_camera_id,
            persist_archive=persist_archive,
        )

    def _persist_candidate_files(
        self,
        record: dict[str, Any],
        winner: dict[str, Any],
        record_id: int,
        output_stem: str,
        write_image: Callable[[Path, Any], None],
        scanner: Any,
        *,
        archive_camera_id: str = "0",
        persist_archive: bool = True,
    ) -> dict[str, Any]:
        """Write the archive/crop files for a record built by ``_materialise_candidate``."""

        if not self._has_complete_registration(record):
            raise ValueError("refusing to archive an unverified plate crop")

        if persist_archive:
            archive_paths = winner.get("archive_paths")
            if isinstance(archive_paths, dict):
                full_path = Path(str(archive_paths["full_vehicle_image"]))
                crop_path = Path(str(archive_paths["crop_image"]))
                ready_path = Path(str(archive_paths["ocr_ready_image"]))
                record["archive_filename"] = str(archive_paths.get("archive_filename") or full_path.name)
                record["archive_sequence"] = int(archive_paths.get("archive_sequence", 0))
                dataset = record.get("dataset", {})
                record["training_status"] = (
                    str(dataset.get("export_status") or "PASS")
                    if isinstance(dataset, dict)
                    else "PASS"
                )
            else:
                full_vehicle = winner.get("best_vehicle_frame")
                if full_vehicle is None:
                    # Compatibility for candidates produced by older callers.
                    full_vehicle = winner["best_crop"]
                full_path, crop_path, ready_path, _ = self._save_archive_images(
                    record,
                    archive_camera_id,
                    full_vehicle,
                    winner["best_crop"],
                    winner["best_ready_crop"],
                    None,
                    scanner,
                    write_image,
                )
                winner["archive_paths"] = {
                    "full_vehicle_image": str(full_path),
                    "crop_image": str(crop_path),
                    "ocr_ready_image": str(ready_path),
                    "archive_filename": str(record.get("archive_filename") or full_path.name),
                    "archive_sequence": int(record.get("archive_sequence", 0)),
                }
            record["full_vehicle_image"] = str(full_path)
        else:
            scan_output_dir = self._dated_output_dir()
            crop_path = scan_output_dir / f"{output_stem}_plate_{record_id}.jpg"
            # Save the exact preprocessed crop consumed by OCR. Keep the
            # legacy ready-path field as an alias to this same file.
            ready_path = crop_path
            write_image(crop_path, winner["best_ready_crop"])
        record["crop_image"] = str(crop_path)
        record["ocr_ready_image"] = str(ready_path)
        record["character_annotated_image"] = ""
        if persist_archive:
            record["plate_json"] = str(self._plate_manifest_path(record))
            self._write_plate_manifest(record)
        return record

    def _plate_manifest_path(self, record: dict[str, Any]) -> Path:
        explicit = str(record.get("plate_json") or "").strip()
        if explicit:
            return Path(explicit)
        image_path = str(record.get("full_vehicle_image") or record.get("crop_image") or "").strip()
        if image_path:
            image = Path(image_path)
            stem = image.stem.removesuffix("-plate_crops").removesuffix("-full_vehicle")
            return image.parent.parent / "json" / f"{stem}_plate.json"
        archive_name = str(record.get("archive_filename") or "").strip()
        if archive_name:
            stem = Path(archive_name).name.removesuffix("-full_vehicle.jpg")
            date_text = stem.split("-")[-2] if len(stem.split("-")) >= 2 else None
            return self._dated_output_dir(date_text) / "json" / f"{stem}_plate.json"
        raise ValueError("verified plate record has no archive image path")

    def _write_plate_manifest(self, record: dict[str, Any]) -> Path:
        """Write the complete verified record beside the daily scan manifest."""

        if not self._has_complete_registration(record):
            raise ValueError("refusing to write an unverified plate manifest")
        path = self._plate_manifest_path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = jsonable_value(record)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def _refresh_plate_manifest(self, record: dict[str, Any]) -> None:
        """Update an existing plate JSON when a later OCR pass improves it."""

        try:
            if not self._has_complete_registration(record):
                return
            path = self._plate_manifest_path(record)
        except ValueError:
            return
        if path.is_file():
            self._write_plate_manifest(record)

    def _dated_output_dir(self, date_text: str | None = None) -> Path:
        """Return the output folder for one calendar day (``YYYYMMDD``)."""

        date_text = date_text or datetime.now().strftime("%Y%m%d")
        output_dir = self.settings.output_dir / self._filename_part(date_text, "unknown_date")
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("thai", "laos", "json", "log", "video"):
            (output_dir / name).mkdir(parents=True, exist_ok=True)
        return output_dir

    def _archive_directories(self, country: str, date_text: str | None = None) -> Path:
        """Return the Thai/Lao evidence directory for one calendar day."""

        dated_output_dir = self._dated_output_dir(date_text)
        directory = dated_output_dir / ("thai" if country == "thai" else "laos")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def _filename_part(value: object, fallback: str = "unknown") -> str:
        """Make OCR/province text safe to use as one Windows filename part."""

        text = str(value or "").strip()
        text = re.sub(r"\s+", "_", text)
        text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
        text = text.strip(" .-_")
        return text or fallback

    def _next_archive_sequence(self, dated_output_dir: Path, date_text: str) -> int:
        """Return the next three-digit sequence for this calendar day."""

        pattern = re.compile(
            rf"-{re.escape(date_text)}-(\d{{3,}})-(?:rejected-)?full_vehicle\.jpg$",
            re.IGNORECASE,
        )
        highest = 0
        for path in dated_output_dir.rglob("*-full_vehicle.jpg"):
            match = pattern.search(path.name)
            if match:
                highest = max(highest, int(match.group(1)))
        return highest + 1

    def _archive_stem(
        self,
        plate: dict[str, Any],
        archive_camera_id: str,
        date_text: str,
        sequence: int,
    ) -> str:
        """Build ``camera-province-prefix-number-date-sequence`` filenames."""

        reading = self._selected_character_reading(plate)
        province = self._filename_part(
            plate.get("province_code") or reading.get("province_code") or plate.get("province"),
            "unknown_province",
        )
        prefix = self._filename_part(
            plate.get("plate_prefix") or reading.get("plate_prefix"),
            "unknown_prefix",
        )
        number = self._filename_part(plate.get("plate_number"), "unknown_number")
        camera = self._filename_part(archive_camera_id, "0")
        parts = [camera, province, prefix, number, date_text, f"{sequence:03d}"]
        return "-".join(parts)

    @staticmethod
    def _rejection_key(plate: dict[str, Any]) -> str:
        """Keep one review sample for each unstable registration hypothesis."""

        return "|".join(
            (
                str(plate.get("country") or "unknown"),
                str(plate.get("plate_prefix") or ""),
                str(plate.get("plate_number") or ""),
                str(plate.get("raw_plate_class") or ""),
            )
        )

    @staticmethod
    def _should_archive_rejection(plate: dict[str, Any]) -> bool:
        """Avoid filling training evidence with weak detector/background noise."""

        return (
            str(plate.get("crop_status") or "full") == "full"
            and str(plate.get("country") or "") in {"thai", "lao"}
            and float(plate.get("recognition_confidence", 0.0)) >= REJECTED_EVIDENCE_MIN_CONFIDENCE
        )

    def _save_rejected_images(
        self,
        plate: dict[str, Any],
        archive_camera_id: str,
        full_vehicle: Any,
        crop: Any,
        ready_crop: Any,
        write_image: Callable[[Path, Any], None],
    ) -> dict[str, Any]:
        """Persist an uncertain scan as a review-only REJECT training sample."""

        record = dict(plate)
        date_text = datetime.now().strftime("%Y%m%d")
        country_dir = self._archive_directories(str(record.get("country") or ""), date_text)
        dated_output_dir = country_dir.parent
        sequence = self._next_archive_sequence(dated_output_dir, date_text)
        stem = self._archive_stem(record, archive_camera_id, date_text, sequence)
        full_path = country_dir / f"{stem}-rejected-full_vehicle.jpg"
        crop_path = country_dir / f"{stem}-rejected-plate_crops.jpg"
        # Keep rejected evidence to the same two images: full frame and the
        # preprocessed crop that would have been used by OCR.
        ready_path = crop_path
        write_image(full_path, full_vehicle)
        write_image(ready_path, ready_crop)
        record.update(
            {
                "confirmed": False,
                "training_status": "REJECT",
                "archive_filename": full_path.name,
                "archive_sequence": sequence,
                "full_vehicle_image": str(full_path),
                "crop_image": str(crop_path),
                "ocr_ready_image": str(ready_path),
            }
        )
        (dated_output_dir / "json" / f"{stem}_rejected_plate.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return record

    @staticmethod
    def _selected_character_reading(plate: dict[str, Any]) -> dict[str, Any]:
        """Choose the country reading whose boxes explain the final plate."""

        readings = plate.get("country_readings", {})
        if not isinstance(readings, dict):
            return {}
        country = str(plate.get("country") or "")
        selected = readings.get(country)
        if isinstance(selected, dict):
            return selected
        candidates = [value for value in readings.values() if isinstance(value, dict)]
        return max(candidates, key=lambda value: float(value.get("score", 0.0)), default={})

    @staticmethod
    def _country_model_names(scanner: Any, country: str, reading: dict[str, Any]) -> list[str]:
        """Read the exact class order from the YOLO model used for this crop."""

        models = getattr(scanner, "country_models", {})
        model = models.get(country) if isinstance(models, dict) else None
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            return [str(names[key]) for key in sorted(names)]
        if isinstance(names, (list, tuple)):
            return [str(value) for value in names]
        tokens = reading.get("tokens", []) if isinstance(reading, dict) else []
        return sorted({str(token.get("label")) for token in tokens if isinstance(token, dict)})

    def _ensure_yolo11_country_layout(self, scanner: Any) -> None:
        """Create the flat Thai/Lao dataset roots before any sample is saved."""

        for country in ("thai", "lao"):
            class_names = self._country_model_names(scanner, country, {})
            if not class_names:
                continue
            dataset_dir = self._dated_output_dir() / "yolo11_ocr_dataset" / country
            (dataset_dir / "images").mkdir(parents=True, exist_ok=True)
            (dataset_dir / "labels").mkdir(parents=True, exist_ok=True)
            names_yaml = "\n".join(
                f"  - {json.dumps(name, ensure_ascii=False)}" for name in class_names
            )
            data_text = (
                "path: .\n"
                "train: images\n"
                "val: images\n"
                f"nc: {len(class_names)}\n"
                "names:\n"
                f"{names_yaml}\n"
            )
            for filename in ("data.yml", "data.yaml"):
                (dataset_dir / filename).write_text(data_text, encoding="utf-8")
            self._write_dataset_exports(dataset_dir, class_names)

    def _write_yolo11_character_sample(
        self,
        plate: dict[str, Any],
        country: str,
        reading: dict[str, Any],
        crop: Any,
        stem: str,
        sequence: int,
        scanner: Any,
        write_image: Callable[[Path, Any], None],
        full_vehicle: Any | None = None,
        archive_root: Path | None = None,
    ) -> None:
        """Add the detected character boxes to a trainable YOLO11 dataset."""

        if not isinstance(reading, dict) or crop is None or not hasattr(crop, "shape"):
            return
        tokens = reading.get("tokens", [])
        if not isinstance(tokens, list) or not tokens:
            return
        class_names = self._country_model_names(scanner, country, reading)
        if not class_names:
            return
        class_ids = {name: index for index, name in enumerate(class_names)}
        height, width = crop.shape[:2]
        if width <= 0 or height <= 0:
            return
        labels: list[str] = []
        for token in tokens:
            if not isinstance(token, dict) or str(token.get("label")) not in class_ids:
                continue
            box = token.get("box", [])
            if not isinstance(box, list) or len(box) != 4:
                continue
            left, top, right, bottom = [float(value) for value in box]
            left = max(0.0, min(float(width), left))
            top = max(0.0, min(float(height), top))
            right = max(left, min(float(width), right))
            bottom = max(top, min(float(height), bottom))
            box_width = right - left
            box_height = bottom - top
            if box_width <= 0 or box_height <= 0:
                continue
            centre_x = (left + right) / 2.0 / width
            centre_y = (top + bottom) / 2.0 / height
            labels.append(
                f"{class_ids[str(token['label'])]} {centre_x:.6f} {centre_y:.6f} "
                f"{box_width / width:.6f} {box_height / height:.6f}"
            )
        if not labels:
            return

        archive_root = archive_root or self._dated_output_dir()
        archive_root.mkdir(parents=True, exist_ok=True)
        dataset_dir = archive_root / "yolo11_ocr_dataset" / country
        image_dir = dataset_dir / "images"
        label_dir = dataset_dir / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{stem}.jpg"
        write_image(image_path, crop)
        (label_dir / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        if full_vehicle is not None:
            full_height, full_width = full_vehicle.shape[:2]
            full_box = plate.get("box", [])
            if len(full_box) == 4 and full_width > 0 and full_height > 0:
                plate_left, plate_top, plate_right, plate_bottom = [float(value) for value in full_box]
                # Older callers may provide the crop itself as the fallback
                # full image. In that case its character coordinates already
                # use the full image origin.
                if full_width <= plate_right or full_height <= plate_bottom:
                    plate_left = 0.0
                    plate_top = 0.0
                full_labels: list[str] = []
                for token in tokens:
                    if not isinstance(token, dict) or str(token.get("label")) not in class_ids:
                        continue
                    box = token.get("box", [])
                    if not isinstance(box, list) or len(box) != 4:
                        continue
                    left, top, right, bottom = [float(value) for value in box]
                    left = max(0.0, min(float(width), left)) + plate_left
                    top = max(0.0, min(float(height), top)) + plate_top
                    right = max(left, min(float(width), right)) + plate_left
                    bottom = max(top, min(float(height), bottom)) + plate_top
                    box_width = max(0.0, min(float(full_width), right) - left)
                    box_height = max(0.0, min(float(full_height), bottom) - top)
                    if box_width <= 0 or box_height <= 0:
                        continue
                    centre_x = (left + right) / 2.0 / full_width
                    centre_y = (top + bottom) / 2.0 / full_height
                    full_labels.append(
                        f"{class_ids[str(token['label'])]} {centre_x:.6f} {centre_y:.6f} "
                        f"{box_width / full_width:.6f} {box_height / full_height:.6f}"
                    )
                if full_labels:
                    # Keep full-vehicle and plate-crop samples in the same
                    # YOLO split. The suffix prevents the two samples from
                    # colliding while preserving the shared archive name.
                    full_image_path = image_dir / f"{stem}_full_vehicle.jpg"
                    full_label_path = label_dir / f"{stem}_full_vehicle.txt"
                    write_image(full_image_path, full_vehicle)
                    full_label_path.write_text(
                        "\n".join(full_labels) + "\n", encoding="utf-8"
                    )

        dataset_dir.mkdir(parents=True, exist_ok=True)
        names_yaml = "\n".join(f"  - {json.dumps(name, ensure_ascii=False)}" for name in class_names)
        data_text = (
            "path: .\n"
            "train: images\n"
            "val: images\n"
            f"nc: {len(class_names)}\n"
            "names:\n"
            f"{names_yaml}\n"
        )
        for filename in ("data.yml", "data.yaml"):
            (dataset_dir / filename).write_text(data_text, encoding="utf-8")
        self._write_dataset_exports(dataset_dir, class_names)

    def _write_dataset_exports(self, dataset_dir: Path, class_names: list[str]) -> None:
        """Create upload-ready ZIP files for Roboflow and CVAT."""

        country = dataset_dir.name
        export_dir = dataset_dir.parent.parent / "dataset_exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        roboflow_zip = export_dir / f"roboflow_yolo11_{country}.zip"
        cvat_zip = export_dir / f"cvat_yolo11_{country}.zip"

        with zipfile.ZipFile(roboflow_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in dataset_dir.rglob("*"):
                if path.is_file() and "source" not in path.relative_to(dataset_dir).parts:
                    archive.write(path, path.relative_to(dataset_dir).as_posix())

        names_text = "\n".join(class_names) + "\n"
        obj_data = (
            f"classes = {len(class_names)}\n"
            "names = obj.names\n"
            "train = train.txt\n"
            "valid = valid.txt\n"
            "backup = backup/\n"
        )
        with zipfile.ZipFile(cvat_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("obj.names", names_text)
            archive.writestr("obj.data", obj_data)
            source_images = dataset_dir / "images"
            source_labels = dataset_dir / "labels"
            list_lines: list[str] = []
            target_dir = "obj_train_data"
            for image_path in sorted(source_images.glob("*")):
                if not image_path.is_file():
                    continue
                relative_image = image_path.relative_to(source_images)
                label_path = source_labels / relative_image.with_suffix(".txt")
                if not label_path.is_file():
                    continue
                image_name = relative_image.as_posix()
                label_name = relative_image.with_suffix(".txt").as_posix()
                list_lines.append(f"{target_dir}/{image_name}")
                archive.write(image_path, f"{target_dir}/{image_name}")
                archive.write(label_path, f"{target_dir}/{label_name}")
            archive.writestr("train.txt", "\n".join(list_lines) + ("\n" if list_lines else ""))
            archive.writestr("valid.txt", "")

        self._last_dataset_exports = {
            "roboflow": str(roboflow_zip),
            "cvat": str(cvat_zip),
        }

    def _save_archive_images(
        self,
        plate: dict[str, Any],
        archive_camera_id: str,
        full_vehicle: Any,
        crop: Any,
        ready_crop: Any,
        character_annotated_crop: Any,
        scanner: Any,
        write_image: Callable[[Path, Any], None],
    ) -> tuple[Path, Path, Path, Path]:
        """Save a confirmed vehicle and its plate crop with one shared name."""

        date_text = datetime.now().strftime("%Y%m%d")
        country = str(plate.get("country") or "")
        country_dir = self._archive_directories(country, date_text)
        dated_output_dir = country_dir.parent
        sequence = self._next_archive_sequence(dated_output_dir, date_text)
        stem = self._archive_stem(plate, archive_camera_id, date_text, sequence)
        full_path = country_dir / f"{stem}-full_vehicle.jpg"
        crop_path = country_dir / f"{stem}-plate_crops.jpg"
        # Save only the full frame and the exact preprocessed crop consumed by
        # OCR. ``ocr_ready_image`` points to the same crop for compatibility.
        ready_path = crop_path
        json_dir = dated_output_dir / "json"
        write_image(full_path, full_vehicle)
        write_image(ready_path, ready_crop)
        plate["archive_filename"] = full_path.name
        plate["archive_sequence"] = sequence
        dataset = plate.get("dataset", {})
        plate["training_status"] = (
            str(dataset.get("export_status") or "PASS")
            if isinstance(dataset, dict)
            else "PASS"
        )
        plate["ocr_folder"] = str(json_dir)
        plate["ocr_ready_archive_image"] = str(ready_path)
        plate["ocr_character_annotated_image"] = ""
        (json_dir / f"{stem}_plate.json").write_text(
            json.dumps(
                {
                    "province": plate.get("province", ""),
                    "province_code": plate.get("province_code", ""),
                    "plate_prefix": plate.get("plate_prefix", ""),
                    "plate_prefix_code": plate.get("plate_prefix_code", ""),
                    "plate_number": plate.get("plate_number", ""),
                    "country": plate.get("country", ""),
                    "ocr": plate.get("ocr", {}),
                    "country_readings": plate.get("country_readings", {}),
                    "cross_model_digit_evidence": plate.get("cross_model_digit_evidence", {}),
                    "overall_confidence": plate.get("overall_confidence", 0.0),
                    "confidence_level": plate.get("confidence_level", "LOW"),
                    "qc": plate.get("qc", {}),
                    "detection": plate.get("detection", {}),
                    "character_validation": plate.get("character_validation", {}),
                    "cfd": plate.get("cfd", {}),
                    "dataset": plate.get("dataset", {}),
                    "validation": plate.get("validation", {}),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return full_path, crop_path, ready_path, None

    def _write_image_debug_artifacts(
        self,
        stem: str,
        image: Any,
        annotated: Any,
        plates: list[dict[str, Any]],
        rejected_plates: list[dict[str, Any]],
        preprocess_plate_crop: Callable[[Any], Any],
        write_image: Callable[[Path, Any], None],
        debug_dir: Path | None = None,
    ) -> None:
        """Persist opt-in, stage-oriented artifacts for pipeline diagnosis."""

        debug_root = debug_dir or self.settings.debug_dir
        folders = {
            name: debug_root / name
            for name in (
                "vehicles",
                "plates",
                "crops",
                "thai",
                "lao",
                "motorcycle",
                "car",
                "preprocessing",
                "ocr",
                "rejected",
            )
        }
        for folder in folders.values():
            folder.mkdir(parents=True, exist_ok=True)
        write_image(folders["plates"] / f"{stem}_annotated.jpg", annotated)
        write_image(folders["vehicles"] / f"{stem}_context.jpg", annotated)

        for state, items in (("accepted", plates), ("rejected", rejected_plates)):
            for index, plate in enumerate(items, start=1):
                box = plate.get("box", [])
                ocr_box = plate.get("ocr_crop_box", box)
                if len(box) != 4 or len(ocr_box) != 4:
                    continue
                x1, y1, x2, y2 = [int(value) for value in box]
                ox1, oy1, ox2, oy2 = [int(value) for value in ocr_box]
                crop = image[y1:y2, x1:x2]
                processing = plate.get("processing", {})
                parameters = (
                    processing.get("preprocessing_parameters", {})
                    if isinstance(processing, dict)
                    else {}
                )
                ready = preprocess_plate_crop(image[oy1:oy2, ox1:ox2], parameters)
                name = f"{stem}_{state}_{index}"
                write_image(folders["crops"] / f"{name}.jpg", crop)
                write_image(folders["preprocessing"] / f"{name}.jpg", ready)

                country = str(plate.get("country", "unknown"))
                if country in ("thai", "lao"):
                    write_image(folders[country] / f"{name}.jpg", crop)
                vehicle = plate.get("vehicle", {})
                vehicle_type = str(
                    vehicle.get("normalized_class", "unknown")
                    if isinstance(vehicle, dict)
                    else plate.get("vehicle_type", "unknown")
                )
                if vehicle_type in ("motorcycle", "car"):
                    write_image(folders[vehicle_type] / f"{name}.jpg", crop)
                if state == "rejected":
                    write_image(folders["rejected"] / f"{name}.jpg", crop)
                (folders["ocr"] / f"{name}.json").write_text(
                    json.dumps(
                        {
                            "country": country,
                            "country_confidence": plate.get("country_confidence", 0.0),
                            "vehicle_type": vehicle_type,
                            "vehicle": plate.get("vehicle", {}),
                            "raw_plate_class": plate.get("raw_plate_class", "unknown"),
                            "plate_type": plate.get("plate_type", "unknown"),
                            "plate_type_raw_class": plate.get(
                                "plate_type_raw_class", "unknown"
                            ),
                            "plate_type_confidence": plate.get(
                                "plate_type_confidence", 0.0
                            ),
                            "ocr": plate.get("ocr", {}),
                            "ocr_confidence": plate.get("ocr", {}).get("confidence", 0.0),
                            "validation": plate.get("validation", {}),
                            "final_validated_result": bool(
                                plate.get("validation", {}).get("valid", False)
                            ),
                            "processing": plate.get("processing", {}),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

    @staticmethod
    def _find_video_track(
        tracks: list[dict[str, Any]],
        country: str,
        box: list[int],
        frame_index: int,
        max_gap: int = TRACK_GAP_MIN_FRAMES,
    ) -> dict[str, Any] | None:
        best_track: dict[str, Any] | None = None
        best_score = 0.0
        for track in tracks:
            if frame_index - int(track["last_frame"]) > max(1, int(max_gap)):
                continue
            previous = track["last_box"]
            iou = ScanService._box_iou(box, previous)
            # Keep one physical ROI track when the country classifier corrects
            # Thai/Lao on a later frame. Require stronger spatial overlap for
            # a cross-country match so nearby plates remain separate.
            if track.get("country") != country and iou < 0.30:
                continue
            first_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            second_center = ((previous[0] + previous[2]) / 2, (previous[1] + previous[3]) / 2)
            distance = ((first_center[0] - second_center[0]) ** 2 + (first_center[1] - second_center[1]) ** 2) ** 0.5
            scale = max(1.0, box[2] - box[0], box[3] - box[1], previous[2] - previous[0], previous[3] - previous[1])
            normalised_distance = distance / scale
            first_area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
            second_area = max(1, (previous[2] - previous[0]) * (previous[3] - previous[1]))
            size_ratio = min(first_area, second_area) / max(first_area, second_area)
            if iou >= 0.12:
                score = 0.60 + iou
            # An approaching vehicle can grow substantially between samples;
            # area ratio must therefore be tolerant enough to keep one track.
            elif normalised_distance <= 0.70 and size_ratio >= 0.30:
                score = 0.30 + (1.0 - normalised_distance) * 0.25 + size_ratio * 0.10
            else:
                continue
            if score > best_score:
                best_track, best_score = track, score
        return best_track

    @staticmethod
    def _box_iou(first: list[int], second: list[int]) -> float:
        left, top = max(first[0], second[0]), max(first[1], second[1])
        right, bottom = min(first[2], second[2]), min(first[3], second[3])
        intersection = max(0, right - left) * max(0, bottom - top)
        area_first = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        area_second = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        union = area_first + area_second - intersection
        return intersection / union if union else 0.0

    @staticmethod
    def _draw_cached_plates(frame: Any, plates: list[dict[str, Any]]) -> Any:
        """Keep boxes visible between sampled frames without rerunning models."""

        from scan import draw_character_boxes

        annotated = frame.copy()
        for plate in plates:
            box = plate.get("box", [])
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = [int(value) for value in box]
            country = str(plate.get("country", ""))
            colour = (0, 180, 0) if country == "thai" else (0, 100, 255)
            readings = plate.get("country_readings", {})
            reading = readings.get(country, {}) if isinstance(readings, dict) else {}
            if isinstance(reading, dict):
                # Character boxes are relative to the crop; repaint them from
                # the last good reading between model inference frames.
                annotated = draw_character_boxes(annotated, reading, (x1, y1))
            prefix = str(plate.get("plate_prefix") or "")
            number = str(plate.get("plate_number") or "")
            separator = "-" if country == "thai" else " "
            label = f"{country.upper()}: {prefix}{separator}{number}".strip()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(
                annotated,
                label[:64],
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                colour,
                2,
                cv2.LINE_AA,
            )
        return annotated

    @staticmethod
    def _video_plate_key(plate: dict[str, Any]) -> str:
        return f"{plate.get('country', 'unknown')}|{plate.get('plate_prefix', '')}{plate.get('plate_number', '')}"

    @staticmethod
    def _record_quality(record: dict[str, Any]) -> tuple[float, float, float, float, float, int]:
        """Rank duplicate readings so the clearest crop/text survives."""

        return (
            float(record.get("temporal_average_quality", 0.0)),
            float(record.get("recognition_confidence", 0.0)),
            float(record.get("crop_visual_quality", 0.0)),
            float(record.get("best_selection_score", 0.0)),
            float(record.get("detection_confidence", 0.0)),
            int(record.get("video_occurrences", record.get("camera_occurrences", 0))),
        )

    @staticmethod
    def _should_replace_live_record(
        existing: dict[str, Any], candidate: dict[str, Any]
    ) -> bool:
        """Replace an ROI track result only when the new reading is stronger.

        OCR errors often produce a different registration key for the same
        moving plate.  Track identity is therefore considered before text
        identity, while this gate prevents a transient low-confidence frame
        from replacing an already good result.
        """

        existing_average = float(existing.get("temporal_average_quality", 0.0))
        candidate_average = float(candidate.get("temporal_average_quality", 0.0))
        existing_visual = float(existing.get("crop_visual_quality", 0.0))
        candidate_visual = float(candidate.get("crop_visual_quality", 0.0))
        existing_selection = float(existing.get("best_selection_score", 0.0))
        candidate_selection = float(candidate.get("best_selection_score", 0.0))
        existing_count = int(
            existing.get("video_occurrences", existing.get("camera_occurrences", 0))
        )
        candidate_count = int(
            candidate.get("video_occurrences", candidate.get("camera_occurrences", 0))
        )
        same_registration = ScanService._same_registration(existing, candidate, allow_lao_prefix_typo=True)
        if same_registration:
            return ScanService._record_quality(candidate) > ScanService._record_quality(existing)
        # A different text from the same physical track is accepted when it
        # has accumulated an additional confirmation, or when its combined
        # score is even modestly stronger.  These values are calibrated model
        # scores, not a ground-truth probability: waiting for a 2.5--4 point
        # jump left a better reading (for example 89.03% vs 87.77%) stuck
        # behind the first OCR result in the live table.
        return (
            candidate_average >= existing_average + 0.005
            or candidate_count > existing_count
            or (
                candidate_visual >= existing_visual + 0.12
                and candidate_average >= existing_average - 0.04
            )
            or candidate_selection >= existing_selection + 0.08
            or float(candidate.get("recognition_confidence", 0.0))
            >= float(existing.get("recognition_confidence", 0.0)) + 0.01
        )

    @staticmethod
    def _frame_bounds(record: dict[str, Any]) -> tuple[int | None, int | None]:
        first = record.get("video_first_frame", record.get("camera_first_frame"))
        last = record.get(
            "video_last_frame",
            record.get("video_confirmed_frame", record.get("camera_confirmed_frame")),
        )
        try:
            first_frame = int(first) if first is not None else None
            last_frame = int(last) if last is not None else first_frame
        except (TypeError, ValueError):
            return None, None
        return first_frame, last_frame

    @staticmethod
    def _same_registration(
        first: dict[str, Any],
        second: dict[str, Any],
        *,
        allow_lao_prefix_typo: bool = False,
    ) -> bool:
        """Compare registrations, allowing one unstable Lao prefix glyph."""

        first_country = str(first.get("country") or "")
        second_country = str(second.get("country") or "")
        first_prefix = str(first.get("plate_prefix") or "")
        second_prefix = str(second.get("plate_prefix") or "")
        first_number = str(first.get("plate_number") or "")
        second_number = str(second.get("plate_number") or "")
        if (
            not first_country
            or first_country != second_country
            or not first_number
            or first_number != second_number
        ):
            return False
        if first_prefix == second_prefix and bool(first_prefix):
            return True
        if not allow_lao_prefix_typo or first_country != "lao":
            return False
        if not first_prefix or len(first_prefix) != len(second_prefix):
            return False
        return sum(a != b for a, b in zip(first_prefix, second_prefix)) <= 1

    @classmethod
    def _find_duplicate_record(
        cls,
        records: list[dict[str, Any]],
        candidate: dict[str, Any],
        max_fuzzy_frame_gap: int,
    ) -> int | None:
        """Find an exact duplicate or a nearby Lao one-glyph OCR variant."""

        candidate_first, candidate_last = cls._frame_bounds(candidate)
        for index, existing in enumerate(records):
            if cls._same_registration(existing, candidate):
                return index
            if not cls._same_registration(
                existing,
                candidate,
                allow_lao_prefix_typo=True,
            ):
                continue
            existing_first, existing_last = cls._frame_bounds(existing)
            if None in (candidate_first, candidate_last, existing_first, existing_last):
                continue
            frame_gap = max(
                0,
                int(candidate_first) - int(existing_last),
                int(existing_first) - int(candidate_last),
            )
            if frame_gap <= max_fuzzy_frame_gap:
                return index
        return None

    @classmethod
    def _deduplicate_video_records(
        cls,
        records: list[dict[str, Any]],
        max_fuzzy_frame_gap: int,
    ) -> list[dict[str, Any]]:
        """Collapse track fragments for the same physical registration."""

        output: list[dict[str, Any]] = []
        for record in records:
            duplicate_index = cls._find_duplicate_record(
                output,
                record,
                max_fuzzy_frame_gap,
            )
            if duplicate_index is None:
                output.append(dict(record))
                continue

            existing = output[duplicate_index]
            preferred = record if cls._record_quality(record) > cls._record_quality(existing) else existing
            merged = dict(preferred)
            first_frames = [
                value
                for value in (existing.get("video_first_frame"), record.get("video_first_frame"))
                if value is not None
            ]
            last_frames = [
                value
                for value in (existing.get("video_last_frame"), record.get("video_last_frame"))
                if value is not None
            ]
            if first_frames:
                merged["video_first_frame"] = min(int(value) for value in first_frames)
            if last_frames:
                merged["video_last_frame"] = max(int(value) for value in last_frames)
            merged["video_occurrences"] = int(existing.get("video_occurrences", 0)) + int(
                record.get("video_occurrences", 0)
            )
            merged["video_track_observations"] = int(
                existing.get("video_track_observations", 0)
            ) + int(record.get("video_track_observations", 0))
            merged["video_vote_score"] = round(
                float(existing.get("video_vote_score", 0.0))
                + float(record.get("video_vote_score", 0.0)),
                4,
            )
            output[duplicate_index] = merged

        for index, record in enumerate(output, start=1):
            record["id"] = index
        return output

    @staticmethod
    def _has_complete_registration(plate: dict[str, Any]) -> bool:
        if str(plate.get("crop_status") or "full") != "full":
            return False
        validation = plate.get("validation")
        if isinstance(validation, dict) and validation.get("valid") is False:
            return False
        country = str(plate.get("country", ""))
        prefix = str(plate.get("plate_prefix") or "")
        number = str(plate.get("plate_number") or "")
        letter_body = prefix[1:] if prefix[:1].isdigit() else prefix
        thai_letter_prefix = (
            bool(letter_body)
            and 1 <= len(letter_body) <= 3
            and all("\u0e01" <= character <= "\u0e2e" for character in letter_body)
            and (not prefix[:1].isdigit() or len(letter_body) == 2)
        )
        thai_digit_prefix = prefix.isdigit() and len(prefix) in {2, 3}
        thai_motorcycle = (
            country == "thai"
            and len(number) == 4
            and number.isdigit()
            and bool(str(plate.get("province") or plate.get("province_code") or ""))
        )
        return (
            country == "thai"
            and (thai_letter_prefix or thai_digit_prefix or thai_motorcycle)
            and len(number) == 4
            and number.isdigit()
        ) or (
            country == "lao" and bool(prefix) and len(number) == 4 and number.isdigit()
        )

    @staticmethod
    def _has_usable_registration(plate: dict[str, Any]) -> bool:
        """Keep partial CCTV reads in temporal voting until a complete plate appears."""

        if ScanService._has_complete_registration(plate):
            return True
        if str(plate.get("crop_status") or "full") != "full":
            return False
        validation = plate.get("validation")
        if isinstance(validation, dict) and validation.get("valid") is False:
            return False
        country = str(plate.get("country", ""))
        prefix = str(plate.get("plate_prefix") or "")
        number = "".join(ch for ch in str(plate.get("plate_number") or "") if ch.isdigit())
        if country in {"thai", "lao"}:
            if (prefix.isdigit() and len(prefix) >= 1) or len(number) >= 2 or bool(prefix):
                return True
            return bool(plate.get("box"))
        return len(prefix) + len(number) >= 2

    @staticmethod
    def _video_plate_quality(plate: dict[str, Any]) -> float:
        """Rank a single frame while temporal voting resolves the final text."""

        ocr = plate.get("ocr", {})
        country = str(plate.get("country", ""))
        reading = plate.get("country_readings", {}).get(country, {})
        digit_confidence = float(reading.get("digit_confidence", 0.0)) if isinstance(reading, dict) else 0.0
        prefix_confidence = float(reading.get("prefix_confidence", 0.0)) if isinstance(reading, dict) else 0.0
        country_margin = max(0.0, min(1.0, float(plate.get("country_confidence_margin", 0.0))))
        digit_evidence = plate.get("cross_model_digit_evidence", {})
        evidence_status = (
            str(digit_evidence.get("status", ""))
            if isinstance(digit_evidence, dict)
            else ""
        )
        cross_model_adjustment = 0.05 if evidence_status == "agree" else -0.12 if evidence_status == "conflict" else 0.0
        return round(
            digit_confidence * 0.42
            + prefix_confidence * (0.12 if country in {"lao", "thai"} else 0.0)
            + float(ocr.get("confidence", 0.0)) * 0.10
            + float(plate.get("detection_confidence", 0.0)) * 0.16
            + float(plate.get("recognition_confidence", 0.0)) * 0.14
            + country_margin * 0.10
            + (0.20 if ScanService._has_complete_registration(plate) else 0.0)
            + cross_model_adjustment,
            4,
        )
