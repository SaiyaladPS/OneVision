"""Application service wrapping the existing licence-plate scan engine."""

from __future__ import annotations

import json
import os
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2

from .config import Settings


TEMPORAL_PROVINCE_MIN_CONFIDENCE = 0.12
TEMPORAL_PROVINCE_MIN_OBSERVATIONS = 2
TEMPORAL_PROVINCE_SCORE_MARGIN = 0.10


class ScanService:
    """Keep model/OCR details out of the Qt window."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def scan_file(self, image_path: Path) -> dict[str, Any]:
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"ไม่พบไฟล์ภาพ: {image_path}")
        if not 0 <= self.settings.padding < 1:
            raise ValueError("ค่า padding ต้องอยู่ระหว่าง 0 ถึง 1")

        # The legacy engine is imported here so opening the GUI does not load
        # OpenCV/YOLO until the user actually starts a scan.
        from scan import (
            CustomPaddleOCR,
            LicensePlateScanner,
            draw_character_boxes,
            preprocess_plate_crop,
            read_image,
            write_image,
        )

        image = read_image(image_path)
        ocr = CustomPaddleOCR(
            self.settings.ocr_source,
            self.settings.ocr_config,
            self.settings.ocr_weights,
            self.settings.torch_ocr_model,
            lao_config=self.settings.ocr_lao_config,
            lao_weights=self.settings.ocr_lao_weights,
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
        self._ensure_yolo11_country_layout(scanner)
        plates, annotated = scanner.scan(
            image,
            self.settings.detector_confidence,
            self.settings.character_confidence,
            self.settings.padding,
            self.settings.imgsz,
            vehicle_type_confidence=self.settings.vehicle_type_confidence,
            plate_type_confidence=self.settings.plate_type_confidence,
        )
        rejected_plates = [
            plate
            for plate in plates
            if not self._has_complete_registration(plate)
            or float(plate.get("recognition_confidence", 0.0)) < self.settings.temporal_min_quality
        ]
        plates = [plate for plate in plates if plate not in rejected_plates]
        for index, plate in enumerate(plates, start=1):
            plate["id"] = index

        scan_output_dir = self._dated_output_dir()
        archive_camera_id = self.settings.archive_camera_id
        annotated_path = scan_output_dir / f"{image_path.stem}_annotated.jpg"
        write_image(annotated_path, annotated)
        for plate in plates:
            x1, y1, x2, y2 = plate["box"]
            ox1, oy1, ox2, oy2 = plate["ocr_crop_box"]
            processing = plate.get("processing", {})
            parameters = (
                processing.get("preprocessing_parameters", {})
                if isinstance(processing, dict)
                else {}
            )
            full_path, crop_path, ready_path, character_path = self._save_archive_images(
                plate,
                archive_camera_id,
                image,
                image[y1:y2, x1:x2],
                preprocess_plate_crop(image[oy1:oy2, ox1:ox2], parameters),
                draw_character_boxes(
                    image[y1:y2, x1:x2].copy(),
                    self._selected_character_reading(plate),
                ),
                scanner,
                write_image,
            )
            plate["full_vehicle_image"] = str(full_path)
            plate["crop_image"] = str(crop_path)
            plate["ocr_ready_image"] = str(ready_path)
            plate["character_annotated_image"] = str(character_path)
        if self.settings.debug_enabled:
            self._write_image_debug_artifacts(
                image_path.stem,
                image,
                annotated,
                plates,
                rejected_plates,
                preprocess_plate_crop,
                write_image,
                debug_dir=scan_output_dir / "debug",
            )
        result = {
            "media_type": "image",
            "input": str(image_path),
            "plate_count": len(plates),
            "rejected_plate_count": len(rejected_plates),
            "plates": plates,
            "annotated_image": str(annotated_path),
            "output_dir": str(scan_output_dir),
        }
        result_path = scan_output_dir / f"{image_path.stem}_result.json"
        result["result_json"] = str(result_path)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

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
        output_video = scan_output_dir / f"{video_path.stem}_annotated.mp4"
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
        self._ensure_yolo11_country_layout(scanner)
        tracks: list[dict[str, Any]] = []
        frame_index = 0
        sampled_frames = 0
        effective_stride = (
            1
            if self.settings.full_accuracy_mode
            else max(
                self.settings.video_frame_stride,
                int(round(source_fps / self.settings.video_target_scans_per_second)),
            )
        )
        last_preview_at = 0.0
        preview_interval = 1.0 / self.settings.target_fps
        cancelled = False
        cached_plates: list[dict[str, Any]] = []
        live_confirmed_keys: set[str] = set()
        live_confirmed_plates: list[dict[str, Any]] = []
        live_confirmed_tracks: dict[int, int] = {}
        cached_until_frame = 0

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
                    sampled_frames += 1
                    plates, annotated = scanner.scan(
                        frame,
                        self.settings.detector_confidence,
                        self.settings.character_confidence,
                        self.settings.padding,
                        self.settings.imgsz,
                        # Structured YOLO readings are enough for temporal
                        # voting. Full Paddle/Tesseract OCR is run once when a
                        # plate is confirmed, avoiding repeated expensive OCR
                        # inference on every sampled frame.
                        run_ocr=False,
                        vehicle_type_confidence=self.settings.vehicle_type_confidence,
                        plate_type_confidence=self.settings.plate_type_confidence,
                        fast_mode=not self.settings.full_accuracy_mode,
                    )
                    current_cached_plates = [
                        plate for plate in plates if self._has_complete_registration(plate)
                    ]
                    if current_cached_plates:
                        cached_plates = current_cached_plates
                        cached_until_frame = frame_index + max(2, effective_stride * 2)
                    elif frame_index > cached_until_frame:
                        cached_plates = []
                        annotated = scanner.draw_scan_roi(frame)
                    elif cached_plates:
                        annotated = self._draw_cached_plates(scanner.draw_scan_roi(frame), cached_plates)
                    for plate in plates:
                        merged = self._merge_video_plate(
                            tracks,
                            plate,
                            frame,
                            frame_index,
                            preprocess_plate_crop,
                        )
                        if merged is None:
                            continue
                        track, candidate = merged
                        key = self._video_plate_key(plate)
                        count = int(candidate.get("count", 0))
                        average_quality = float(candidate.get("vote_score", 0.0)) / max(1, count)
                        if (
                            plate_callback is None
                            or key in live_confirmed_keys
                            or count < self.settings.video_min_confirmations
                            or average_quality < self.settings.temporal_min_quality
                        ):
                            continue
                        live_record = self._materialise_candidate(
                            candidate,
                            len(live_confirmed_plates) + 1,
                            f"{video_path.stem}_live",
                            write_image,
                            scanner,
                            run_full_ocr=False,
                            persist_archive=False,
                        )
                        if live_record is None:
                            continue
                        live_record["video_first_frame"] = track["first_frame"]
                        live_record["video_confirmed_frame"] = frame_index
                        live_record["video_last_frame"] = frame_index
                        live_record["video_occurrences"] = count
                        live_record["temporal_average_quality"] = round(average_quality, 4)
                        live_record["video_track_id"] = int(track["track_id"])
                        live_confirmed_keys.add(key)
                        track_id = int(track["track_id"])
                        known_index = live_confirmed_tracks.get(track_id)
                        if known_index is not None and known_index < len(live_confirmed_plates):
                            existing = live_confirmed_plates[known_index]
                            if not self._should_replace_live_record(existing, live_record):
                                continue
                            live_record["id"] = existing.get("id", known_index + 1)
                            live_record["live_result_key"] = existing.get(
                                "live_result_key", f"video:{known_index + 1}"
                            )
                            live_confirmed_plates[known_index] = live_record
                            plate_callback(dict(live_record))
                            continue
                        existing_index = self._find_duplicate_record(
                            live_confirmed_plates,
                            live_record,
                            max_fuzzy_frame_gap=max(60, int(round(source_fps * 10.0))),
                        )
                        if existing_index is None:
                            live_record["live_result_key"] = f"video:{len(live_confirmed_plates) + 1}"
                            live_confirmed_plates.append(live_record)
                            live_confirmed_tracks[track_id] = len(live_confirmed_plates) - 1
                            plate_callback(dict(live_record))
                        else:
                            existing = live_confirmed_plates[existing_index]
                            live_record["id"] = existing.get("id", existing_index + 1)
                            live_record["live_result_key"] = existing.get(
                                "live_result_key", f"video:{existing_index + 1}"
                            )
                            if self._record_quality(live_record) > self._record_quality(existing):
                                live_confirmed_plates[existing_index] = live_record
                                live_confirmed_tracks[track_id] = existing_index
                                plate_callback(dict(live_record))
                else:
                    annotated = scanner.draw_scan_roi(frame)
                    if cached_plates:
                        annotated = self._draw_cached_plates(annotated, cached_plates)
                writer.write(annotated)
                now = time.monotonic()
                if frame_callback is not None and (frame_index == 1 or now - last_preview_at >= preview_interval):
                    frame_callback(annotated.copy(), frame_index, frame_count)
                    last_preview_at = now
        finally:
            capture.release()
            writer.release()

        plates = self._finalise_video_tracks(
            tracks,
            video_path.stem,
            write_image,
            scanner,
            min_confirmations=self.settings.video_min_confirmations,
            dedupe_frame_gap=max(60, int(round(source_fps * 10.0))),
        )
        result = {
            "media_type": "video",
            "input": str(video_path),
            "plate_count": len(plates),
            "plates": plates,
            "frame_count": frame_index,
            "fps": round(output_fps, 3),
            "source_fps": round(source_fps, 3),
            "target_fps": round(self.settings.target_fps, 3),
            "scan_stride": effective_stride,
            "sampled_frames": sampled_frames,
            "output_video": str(output_video),
            "output_dir": str(scan_output_dir),
            "cancelled": cancelled,
        }
        result_path = scan_output_dir / f"{video_path.stem}_video_result.json"
        result["result_json"] = str(result_path)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def scan_camera(
        self,
        camera_index: int = 0,
        frame_callback: Callable[[Any, int], None] | None = None,
        plate_callback: Callable[[dict[str, Any]], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
        stop_after_first: bool = False,
    ) -> dict[str, Any]:
        """Scan a live camera until stopped, emitting only vote-confirmed plates."""

        from scan import CustomPaddleOCR, LicensePlateScanner, preprocess_plate_crop, write_image

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
        # Some camera drivers honour this request; the loop below also paces
        # reads so drivers that ignore CAP_PROP_FPS do not flood the scanner.
        capture.set(cv2.CAP_PROP_FPS, self.settings.target_fps)
        ocr = CustomPaddleOCR(
            self.settings.ocr_source,
            self.settings.ocr_config,
            self.settings.ocr_weights,
            self.settings.torch_ocr_model,
            lao_config=self.settings.ocr_lao_config,
            lao_weights=self.settings.ocr_lao_weights,
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
        self._ensure_yolo11_country_layout(scanner)
        scan_output_dir = self._dated_output_dir()
        session_stem = f"camera_{camera_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tracks: list[dict[str, Any]] = []
        confirmed_keys: set[str] = set()
        confirmed_plates: list[dict[str, Any]] = []
        confirmed_tracks: dict[int, int] = {}
        frame_index = 0
        sampled_frames = 0
        failed_reads = 0
        last_preview_at = 0.0
        last_frame: Any = None
        cached_plates: list[dict[str, Any]] = []
        cached_until_frame = 0
        next_frame_at = time.monotonic()
        # In accuracy mode do not intentionally throttle capture: inference remains the
        # limiting factor, but every frame delivered by the camera is considered.
        frame_interval = 0.0 if self.settings.full_accuracy_mode else 1.0 / self.settings.target_fps
        camera_stride = (
            1 if self.settings.full_accuracy_mode else self.settings.camera_frame_stride
        )

        try:
            while stop_requested is None or not stop_requested():
                wait_for_frame = next_frame_at - time.monotonic()
                if wait_for_frame > 0:
                    time.sleep(wait_for_frame)
                next_frame_at = max(next_frame_at + frame_interval, time.monotonic())
                ok, frame = capture.read()
                if not ok:
                    failed_reads += 1
                    if failed_reads >= 30:
                        raise RuntimeError("กล้องหยุดส่งภาพ กรุณาปิดโปรแกรมอื่นที่กำลังใช้กล้องแล้วลองใหม่")
                    continue
                failed_reads = 0
                frame_index += 1
                last_frame = frame.copy()
                annotated = frame
                if (frame_index - 1) % camera_stride == 0:
                    sampled_frames += 1
                    plates, annotated = scanner.scan(
                        frame,
                        self.settings.detector_confidence,
                        self.settings.character_confidence,
                        self.settings.padding,
                        self.settings.imgsz,
                        run_ocr=False,
                        vehicle_type_confidence=self.settings.vehicle_type_confidence,
                        plate_type_confidence=self.settings.plate_type_confidence,
                        fast_mode=not self.settings.full_accuracy_mode,
                    )
                    current_cached_plates = [
                        plate for plate in plates if self._has_complete_registration(plate)
                    ]
                    if current_cached_plates:
                        cached_plates = current_cached_plates
                        cached_until_frame = frame_index + max(2, camera_stride * 2)
                    elif frame_index > cached_until_frame:
                        cached_plates = []
                        annotated = scanner.draw_scan_roi(frame)
                    elif cached_plates:
                        annotated = self._draw_cached_plates(scanner.draw_scan_roi(frame), cached_plates)
                    for plate in plates:
                        merged = self._merge_video_plate(
                            tracks,
                            plate,
                            frame,
                            frame_index,
                            preprocess_plate_crop,
                        )
                        if merged is None:
                            continue
                        track, candidate = merged
                        key = self._video_plate_key(plate)
                        count = int(candidate.get("count", 0))
                        average_quality = float(candidate.get("vote_score", 0.0)) / max(1, count)
                        if (
                            key in confirmed_keys
                            or count < self.settings.camera_min_confirmations
                            or average_quality < self.settings.temporal_min_quality
                        ):
                            continue
                        record = self._materialise_candidate(
                            candidate,
                            len(confirmed_plates) + 1,
                            session_stem,
                            write_image,
                            scanner,
                            run_full_ocr=True,
                            archive_camera_id=str(camera_index),
                        )
                        if record is None:
                            continue
                        record["camera_first_frame"] = track["first_frame"]
                        record["camera_confirmed_frame"] = frame_index
                        record["camera_occurrences"] = count
                        record["temporal_average_quality"] = round(average_quality, 4)
                        record["camera_track_id"] = int(track["track_id"])
                        confirmed_keys.add(key)
                        track_id = int(track["track_id"])
                        known_index = confirmed_tracks.get(track_id)
                        if known_index is not None and known_index < len(confirmed_plates):
                            existing = confirmed_plates[known_index]
                            if not self._should_replace_live_record(existing, record):
                                continue
                            record["id"] = existing.get("id", known_index + 1)
                            record["live_result_key"] = existing.get(
                                "live_result_key", f"camera:{known_index + 1}"
                            )
                            confirmed_plates[known_index] = record
                            if plate_callback is not None:
                                plate_callback(dict(record))
                            continue
                        existing_index = self._find_duplicate_record(
                            confirmed_plates,
                            record,
                            max_fuzzy_frame_gap=max(60, int(round(self.settings.target_fps * 10.0))),
                        )
                        if existing_index is None:
                            record["live_result_key"] = f"camera:{len(confirmed_plates) + 1}"
                            confirmed_plates.append(record)
                            confirmed_tracks[track_id] = len(confirmed_plates) - 1
                            if plate_callback is not None:
                                plate_callback(dict(record))
                        else:
                            existing = confirmed_plates[existing_index]
                            record["id"] = existing.get("id", existing_index + 1)
                            record["live_result_key"] = existing.get(
                                "live_result_key", f"camera:{existing_index + 1}"
                            )
                            if self._record_quality(record) > self._record_quality(existing):
                                confirmed_plates[existing_index] = record
                                confirmed_tracks[track_id] = existing_index
                                if plate_callback is not None:
                                    plate_callback(dict(record))
                        if stop_after_first:
                            break
                    if stop_after_first and confirmed_plates:
                        break
                else:
                    annotated = scanner.draw_scan_roi(frame)
                    if cached_plates:
                        annotated = self._draw_cached_plates(annotated, cached_plates)

                now = time.monotonic()
                if frame_callback is not None and (frame_index == 1 or now - last_preview_at >= frame_interval):
                    frame_callback(annotated.copy(), frame_index)
                    last_preview_at = now
        finally:
            capture.release()

        snapshot_path = scan_output_dir / f"{session_stem}_last.jpg"
        if last_frame is not None:
            write_image(snapshot_path, last_frame)
        result = {
            "media_type": "camera",
            "input": f"camera://{camera_index}",
            "camera_index": camera_index,
            "plate_count": len(confirmed_plates),
            "plates": confirmed_plates,
            "frame_count": frame_index,
            "sampled_frames": sampled_frames,
            "target_fps": round(self.settings.target_fps, 3),
            "annotated_image": str(snapshot_path) if last_frame is not None else "",
            "output_dir": str(scan_output_dir),
        }
        result_path = scan_output_dir / f"{session_stem}_result.json"
        result["result_json"] = str(result_path)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def _merge_video_plate(
        self,
        tracks: list[dict[str, Any]],
        plate: dict[str, Any],
        frame: Any,
        frame_index: int,
        preprocess_plate_crop: Callable[[Any], Any],
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Add one valid reading to its spatial track and retain its best crop."""

        if not self._has_complete_registration(plate):
            return None
        country = str(plate.get("country", ""))
        box = [int(value) for value in plate["box"]]
        track = self._find_video_track(tracks, country, box, frame_index)
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
        if quality <= float(candidate["best_quality"]):
            return track, candidate

        x1, y1, x2, y2 = box
        ox1, oy1, ox2, oy2 = [int(value) for value in plate["ocr_crop_box"]]
        candidate["best_quality"] = quality
        candidate["best_plate"] = dict(plate)
        candidate["best_crop"] = frame[y1:y2, x1:x2].copy()
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
        return track, candidate

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
            candidates = [
                candidate
                for candidate in track["candidates"].values()
                if int(candidate.get("count", 0)) >= min_confirmations
                and float(candidate.get("vote_score", 0.0)) / max(1, int(candidate.get("count", 0)))
                >= self.settings.temporal_min_quality
            ]
            if not candidates:
                continue
            # Repeated readings are decisive, but a very clear single-frame
            # plate remains valid for short clips. Prefer a materially clearer
            # reading over a stale low-quality key from the same ROI track.
            winner = max(
                candidates,
                key=lambda item: (
                    float(item["vote_score"]) / max(1, int(item["count"]))
                    + min(int(item["count"]), 8) * 0.04,
                    float(item["best_quality"]),
                    int(item["count"]),
                ),
            )
            record = self._materialise_candidate(
                winner,
                len(output) + 1,
                video_stem,
                write_image,
                scanner,
                archive_camera_id=self.settings.archive_camera_id,
            )
            if record is None:
                continue
            record["video_first_frame"] = track["first_frame"]
            record["video_last_frame"] = track["last_frame"]
            record["video_occurrences"] = int(winner["count"])
            record["video_track_observations"] = int(track["observations"])
            record["video_vote_score"] = round(float(winner["vote_score"]), 4)
            output.append(record)
        return self._deduplicate_video_records(output, dedupe_frame_gap)

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
    ) -> dict[str, Any] | None:
        """Run full OCR once and persist the best temporal crop."""

        from scan import draw_character_boxes

        best = self._apply_temporal_province_vote(winner)
        if not isinstance(best, dict) or winner.get("best_crop") is None or winner.get("best_ready_crop") is None:
            return None
        record = dict(best)
        record["id"] = record_id
        country = str(record.get("country", ""))
        reading = record.get("country_readings", {}).get(country, {})
        if isinstance(reading, dict):
            ocr, prefix, number = scanner.finalise_reading(
                winner["best_ready_crop"],
                country,
                reading,
                run_ocr=run_full_ocr,
            )
            record["ocr"] = ocr
            record["plate_prefix"] = prefix
            record["plate_number"] = number
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
                }
            )
        record["confirmed"] = self._has_complete_registration(record)
        if not record["confirmed"]:
            return None
        if persist_archive:
            full_vehicle = winner.get("best_vehicle_frame")
            if full_vehicle is None:
                # Compatibility for candidates produced by older callers.
                full_vehicle = winner["best_crop"]
            character_crop = draw_character_boxes(
                winner["best_crop"].copy(),
                self._selected_character_reading(record),
            )
            full_path, crop_path, ready_path, character_path = self._save_archive_images(
                record,
                archive_camera_id,
                full_vehicle,
                winner["best_crop"],
                winner["best_ready_crop"],
                character_crop,
                scanner,
                write_image,
            )
            record["full_vehicle_image"] = str(full_path)
        else:
            scan_output_dir = self._dated_output_dir()
            crop_path = scan_output_dir / f"{output_stem}_plate_{record_id}.jpg"
            ready_path = scan_output_dir / f"{output_stem}_plate_{record_id}_ocr_ready.jpg"
            character_path = scan_output_dir / f"{output_stem}_plate_{record_id}_ocr_img_annotated.jpg"
            write_image(crop_path, winner["best_crop"])
            write_image(ready_path, winner["best_ready_crop"])
            write_image(
                character_path,
                draw_character_boxes(
                    winner["best_crop"].copy(),
                    self._selected_character_reading(record),
                ),
            )
        record["crop_image"] = str(crop_path)
        record["ocr_ready_image"] = str(ready_path)
        record["character_annotated_image"] = str(character_path)
        return record

    def _dated_output_dir(self, date_text: str | None = None) -> Path:
        """Return the output folder for one calendar day (``YYYYMMDD``)."""

        date_text = date_text or datetime.now().strftime("%Y%m%d")
        output_dir = self.settings.output_dir / self._filename_part(date_text, "unknown_date")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _archive_directories(self, date_text: str | None = None) -> tuple[Path, Path]:
        """Return and create the permanent image archive directories for a day."""

        dated_output_dir = self._dated_output_dir(date_text)
        full_vehicle = dated_output_dir / "full_vehicle"
        plate_crops = dated_output_dir / "plate_crops"
        full_vehicle.mkdir(parents=True, exist_ok=True)
        plate_crops.mkdir(parents=True, exist_ok=True)
        return full_vehicle, plate_crops

    @staticmethod
    def _filename_part(value: object, fallback: str = "unknown") -> str:
        """Make OCR/province text safe to use as one Windows filename part."""

        text = str(value or "").strip()
        text = re.sub(r"\s+", "_", text)
        text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
        text = text.strip(" .-_")
        return text or fallback

    def _next_archive_sequence(self, full_vehicle_dir: Path, date_text: str) -> int:
        """Return the next three-digit sequence for this calendar day."""

        pattern = re.compile(rf"-{re.escape(date_text)}-(\d{{3,}})\.jpg$", re.IGNORECASE)
        highest = 0
        for path in full_vehicle_dir.glob("*.jpg"):
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
        full_vehicle_dir, plate_crops_dir = self._archive_directories(date_text)
        sequence = self._next_archive_sequence(full_vehicle_dir, date_text)
        stem = self._archive_stem(plate, archive_camera_id, date_text, sequence)
        full_path = full_vehicle_dir / f"{stem}.jpg"
        crop_path = plate_crops_dir / f"{stem}.jpg"
        ready_path = plate_crops_dir / f"{stem}_ocr_ready.jpg"
        character_path = plate_crops_dir / f"{stem}_ocr_img_annotated.jpg"
        province_part = self._filename_part(
            plate.get("province_code") or plate.get("province"), "unknown_province"
        )
        prefix_part = self._filename_part(plate.get("plate_prefix"), "unknown_prefix")
        ocr_dir = full_vehicle_dir.parent / "ocr" / f"{province_part}-{prefix_part}"
        ocr_dir.mkdir(parents=True, exist_ok=True)
        ocr_ready_path = ocr_dir / ready_path.name
        ocr_character_path = ocr_dir / character_path.name
        write_image(full_path, full_vehicle)
        write_image(crop_path, crop)
        write_image(ready_path, ready_crop)
        write_image(character_path, character_annotated_crop)
        write_image(ocr_ready_path, ready_crop)
        write_image(ocr_character_path, character_annotated_crop)
        plate["archive_filename"] = full_path.name
        plate["archive_sequence"] = sequence
        plate["ocr_folder"] = str(ocr_dir)
        plate["ocr_ready_archive_image"] = str(ocr_ready_path)
        plate["ocr_character_annotated_image"] = str(ocr_character_path)
        (ocr_dir / f"{stem}.json").write_text(
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
                    "validation": plate.get("validation", {}),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        country = str(plate.get("country") or "")
        reading = self._selected_character_reading(plate)
        if country not in ("thai", "lao"):
            country = str(reading.get("script") or "")
        if country in ("thai", "lao"):
            self._write_yolo11_character_sample(
                plate,
                country,
                reading,
                crop,
                stem,
                sequence,
                scanner,
                write_image,
                full_vehicle=full_vehicle,
                archive_root=full_vehicle_dir.parent,
            )
            exports = getattr(self, "_last_dataset_exports", {})
            if isinstance(exports, dict):
                plate["roboflow_export"] = exports.get("roboflow", "")
                plate["cvat_export"] = exports.get("cvat", "")
        return full_path, crop_path, ready_path, character_path

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
    ) -> dict[str, Any] | None:
        best_track: dict[str, Any] | None = None
        best_score = 0.0
        for track in tracks:
            if frame_index - int(track["last_frame"]) > 30:
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
            elif normalised_distance <= 0.55 and size_ratio >= 0.55:
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
    def _record_quality(record: dict[str, Any]) -> tuple[float, float, float, int]:
        """Rank duplicate readings so the clearest crop/text survives."""

        return (
            float(record.get("temporal_average_quality", 0.0)),
            float(record.get("recognition_confidence", 0.0)),
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
        existing_count = int(
            existing.get("video_occurrences", existing.get("camera_occurrences", 0))
        )
        candidate_count = int(
            candidate.get("video_occurrences", candidate.get("camera_occurrences", 0))
        )
        same_registration = ScanService._same_registration(existing, candidate, allow_lao_prefix_typo=True)
        if same_registration:
            return ScanService._record_quality(candidate) > ScanService._record_quality(existing)
        # A different text from the same physical track is accepted when it is
        # materially clearer, or has accumulated an additional confirmation.
        return (
            candidate_average >= existing_average + 0.025
            or candidate_count > existing_count
            or float(candidate.get("recognition_confidence", 0.0))
            >= float(existing.get("recognition_confidence", 0.0)) + 0.04
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
        validation = plate.get("validation")
        if isinstance(validation, dict) and validation.get("valid") is False:
            return False
        country = str(plate.get("country", ""))
        prefix = str(plate.get("plate_prefix") or "")
        number = str(plate.get("plate_number") or "")
        return (
            country == "thai" and len(prefix) == 2 and prefix.isdigit() and len(number) == 4 and number.isdigit()
        ) or (
            country == "lao" and bool(prefix) and len(number) == 4 and number.isdigit()
        )

    @staticmethod
    def _video_plate_quality(plate: dict[str, Any]) -> float:
        """Rank a single frame while temporal voting resolves the final text."""

        ocr = plate.get("ocr", {})
        country = str(plate.get("country", ""))
        reading = plate.get("country_readings", {}).get(country, {})
        digit_confidence = float(reading.get("digit_confidence", 0.0)) if isinstance(reading, dict) else 0.0
        prefix_confidence = float(reading.get("prefix_confidence", 0.0)) if isinstance(reading, dict) else 0.0
        country_margin = max(0.0, min(1.0, float(plate.get("country_confidence_margin", 0.0))))
        return round(
            digit_confidence * 0.42
            + prefix_confidence * (0.12 if country == "lao" else 0.0)
            + float(ocr.get("confidence", 0.0)) * 0.10
            + float(plate.get("detection_confidence", 0.0)) * 0.16
            + float(plate.get("recognition_confidence", 0.0)) * 0.14
            + country_margin * 0.10
            + (0.20 if ScanService._has_complete_registration(plate) else 0.0),
            4,
        )
