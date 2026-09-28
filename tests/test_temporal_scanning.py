from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.car_scan.config import Settings
from src.car_scan.service import ScanService
from src.car_scan.worker import _read_first_frame, same_live_registration


class TemporalScanningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="car_scan_temporal_test_")
        self.service = ScanService(
            Settings(
                root=Path.cwd(),
                output_dir=Path(self.temporary.name),
                database_url="",
            )
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def plate() -> dict:
        return {
            "country": "thai",
            "plate_prefix": "68",
            "plate_number": "3197",
            "box": [10, 10, 90, 45],
            "ocr_crop_box": [5, 5, 95, 50],
            "detection_confidence": 0.94,
            "recognition_confidence": 0.91,
            "country_confidence_margin": 0.80,
            "ocr": {"confidence": 0.88},
            "country_readings": {
                "thai": {"digit_confidence": 0.90, "prefix_confidence": 0.0}
            },
        }

    def test_complete_plate_accumulates_temporal_votes(self) -> None:
        tracks: list[dict] = []
        frame = np.zeros((60, 110, 3), dtype=np.uint8)

        for frame_index in range(1, 4):
            merged = self.service._merge_video_plate(
                tracks,
                self.plate(),
                frame,
                frame_index,
                lambda crop: crop.copy(),
            )
            self.assertIsNotNone(merged)

        candidate = next(iter(tracks[0]["candidates"].values()))
        self.assertEqual(candidate["count"], 3)
        self.assertGreater(candidate["vote_score"] / candidate["count"], 0.65)

    def test_live_registration_identity_does_not_depend_on_track_id(self) -> None:
        first = {"id": 3, "plate_prefix": " กข ", "plate_number": "12-34"}
        second = {"id": 9, "plate_prefix": "กข", "plate_number": "1234"}

        self.assertTrue(same_live_registration(first, second))

    def test_rtsp_first_frame_is_retried_until_decoder_has_a_frame(self) -> None:
        class DelayedCapture:
            def __init__(self) -> None:
                self.reads = 0

            def read(self):
                self.reads += 1
                return (False, None) if self.reads < 3 else (True, "frame")

        capture = DelayedCapture()
        with patch("src.car_scan.worker.time.sleep"):
            self.assertEqual(_read_first_frame(capture), "frame")
        self.assertEqual(capture.reads, 3)

    def test_clipped_crop_does_not_enter_temporal_track(self) -> None:
        tracks: list[dict] = []
        frame = np.zeros((60, 110, 3), dtype=np.uint8)
        clipped = {
            **self.plate(),
            "crop_status": "clipped_by_frame",
            "plate_prefix": "",
            "plate_number": "19259",
        }
        merged = self.service._merge_video_plate(tracks, clipped, frame, 1, lambda crop: crop.copy())
        self.assertIsNone(merged)
        self.assertFalse(self.service._has_usable_registration(clipped))
        self.assertFalse(self.service._should_archive_rejection(clipped))
        tracks: list[dict] = []
        frame = np.zeros((60, 110, 3), dtype=np.uint8)
        partial = {**self.plate(), "plate_prefix": "6", "plate_number": "319"}
        merged = self.service._merge_video_plate(tracks, partial, frame, 1, lambda crop: crop.copy())
        self.assertIsNotNone(merged)
        self.assertFalse(self.service._has_complete_registration(partial))
        self.assertTrue(self.service._has_usable_registration(partial))

    def test_one_digit_outlier_contributes_to_dominant_temporal_reading(self) -> None:
        tracks: list[dict] = []
        frame = np.zeros((60, 110, 3), dtype=np.uint8)
        readings = ("3197", "3197", "3191")
        merged = None

        for frame_index, number in enumerate(readings, start=1):
            plate = {**self.plate(), "plate_number": number}
            merged = self.service._merge_video_plate(
                tracks,
                plate,
                frame,
                frame_index,
                lambda crop: crop.copy(),
            )

        self.assertIsNotNone(merged)
        _, consensus = merged
        self.assertEqual(consensus["count"], 3)
        self.assertEqual(consensus["best_plate"]["plate_number"], "3197")
        self.assertEqual(sorted(consensus["variant_support"].values()), [1, 2])

    def test_unrelated_single_frame_variants_do_not_form_consensus(self) -> None:
        tracks: list[dict] = []
        frame = np.zeros((60, 110, 3), dtype=np.uint8)
        merged = None

        for frame_index, number in enumerate(("3191", "3192", "3193"), start=1):
            merged = self.service._merge_video_plate(
                tracks,
                {**self.plate(), "plate_number": number},
                frame,
                frame_index,
                lambda crop: crop.copy(),
            )

        self.assertIsNotNone(merged)
        self.assertEqual(merged[1]["count"], 1)

    def test_sharper_frame_is_retained_for_final_ocr(self) -> None:
        tracks: list[dict] = []
        blurred = np.zeros((60, 110, 3), dtype=np.uint8)
        sharp = np.indices((60, 110)).sum(axis=0) % 2
        sharp = np.repeat((sharp * 255).astype(np.uint8)[:, :, None], 3, axis=2)

        self.service._merge_video_plate(
            tracks,
            self.plate(),
            blurred,
            1,
            lambda crop: crop.copy(),
        )
        self.service._merge_video_plate(
            tracks,
            self.plate(),
            sharp,
            2,
            lambda crop: crop.copy(),
        )

        candidate = next(iter(tracks[0]["candidates"].values()))
        self.assertGreater(candidate["best_visual_quality"], 0.5)
        self.assertGreater(float(candidate["best_crop"].std()), 0.0)

    def test_camera_preview_continues_while_inference_is_busy(self) -> None:
        class FakeCapture:
            def __init__(self, *args: object) -> None:
                del args
                self.released = False

            def isOpened(self) -> bool:
                return True

            def set(self, *args: object) -> bool:
                del args
                return True

            def read(self) -> tuple[bool, np.ndarray]:
                return True, np.zeros((48, 64, 3), dtype=np.uint8)

            def release(self) -> None:
                self.released = True

        class FakeScanner:
            def __init__(self, *args: object, **kwargs: object) -> None:
                del args, kwargs
                self.last_plate_detection_count = 0
                self.last_vehicle_types: list[dict] = []

            @staticmethod
            def draw_scan_roi(frame: np.ndarray) -> np.ndarray:
                return frame.copy()

            def scan(self, frame: np.ndarray, *args: object, **kwargs: object):
                del args, kwargs
                time.sleep(0.20)
                self.last_plate_detection_count = 0
                return [], frame.copy()

        callbacks: list[int] = []
        stop = threading.Event()

        def preview(_frame: np.ndarray, frame_index: int) -> None:
            callbacks.append(frame_index)
            if len(callbacks) >= 2:
                stop.set()

        with (
            patch("src.car_scan.service.cv2.VideoCapture", FakeCapture),
            patch("scan.CustomPaddleOCR", lambda *args, **kwargs: object()),
            patch("scan.LicensePlateScanner", FakeScanner),
        ):
            result = self.service.scan_camera(
                frame_callback=preview,
                stop_requested=stop.is_set,
            )

        self.assertGreaterEqual(len(callbacks), 2)
        self.assertGreaterEqual(result["sampled_frames"], 1)
        self.assertGreater(result["frame_count"], result["sampled_frames"])

    def test_stream_scan_uses_video_detector_settings(self) -> None:
        recorded: dict[str, object] = {}

        class Scanner:
            last_plate_detection_count = 0
            last_vehicle_types: list[dict] = []

            def scan(self, frame, detector, character, padding, imgsz, **kwargs):
                recorded["detector"] = detector
                recorded["character"] = character
                recorded["padding"] = padding
                recorded["imgsz"] = imgsz
                recorded.update(kwargs)
                return [], frame

        self.service._scan_stream_frame(
            Scanner(),
            np.zeros((8, 8, 3), dtype=np.uint8),
            sampled_frames=1,
            cached_vehicle_types=None,
            run_ocr=False,
        )
        self.assertEqual(recorded["detector"], self.service.settings.detector_confidence)
        self.assertEqual(recorded["character"], self.service.settings.character_confidence)
        self.assertEqual(recorded["padding"], self.service.settings.padding)
        self.assertEqual(recorded["imgsz"], self.service.settings.imgsz)
        self.assertFalse(recorded["run_ocr"])
        self.assertEqual(recorded["fast_mode"], (not self.service.settings.full_accuracy_mode))

        self.service._scan_stream_frame(
            Scanner(),
            np.zeros((8, 8, 3), dtype=np.uint8),
            sampled_frames=1,
            cached_vehicle_types=None,
            run_ocr=True,
        )
        self.assertTrue(recorded["run_ocr"])
        self.assertEqual(recorded["fast_mode"], (not self.service.settings.full_accuracy_mode))

    def test_image_scan_uses_video_confirm_pipeline(self) -> None:
        recorded: dict[str, object] = {}
        plate = self.plate()
        image = np.zeros((60, 110, 3), dtype=np.uint8)
        path = Path(self.temporary.name) / "still.jpg"
        path.write_bytes(b"fake")

        class Scanner:
            def __init__(self, *args: object, **kwargs: object) -> None:
                del args, kwargs
                self.last_plate_detection_count = 1
                self.last_vehicle_types: list[dict] = []

            @staticmethod
            def draw_scan_roi(frame: np.ndarray) -> np.ndarray:
                return frame.copy()

            def scan(self, frame, *args: object, **kwargs: object):
                recorded.update(kwargs)
                self.last_plate_detection_count = 1
                return [plate], frame.copy()

        with (
            patch("scan.read_image", return_value=image),
            patch("scan.CustomPaddleOCR", lambda *args, **kwargs: object()),
            patch("scan.LicensePlateScanner", Scanner),
            patch("scan.write_image", lambda path, image: None),
            patch.object(
                self.service,
                "_materialise_candidate",
                return_value={
                    "id": 1,
                    "country": "thai",
                    "plate_prefix": "68",
                    "plate_number": "3197",
                    "recognition_confidence": 0.9,
                    "ocr": {"text": "68-3197", "method": "structured-yolo", "confidence": 0.9},
                },
            ) as materialise,
        ):
            result = self.service.scan_file(path)

        self.assertFalse(recorded["run_ocr"])
        self.assertEqual(recorded["fast_mode"], (not self.service.settings.full_accuracy_mode))
        self.assertTrue(materialise.called)
        self.assertEqual(result["media_type"], "image")
        self.assertEqual(result["plate_count"], 1)
        self.assertEqual(result["plates"][0]["ocr"]["text"], "68-3197")

    def test_live_cctv_frame_follows_video_confirmation_gates(self) -> None:
        plate = self.plate()
        recorded: dict[str, object] = {}

        class Scanner:
            last_plate_detection_count = 1
            last_vehicle_types: list[dict] = []

            @staticmethod
            def draw_scan_roi(frame: np.ndarray) -> np.ndarray:
                return frame.copy()

            def scan(self, frame, *args, **kwargs):
                recorded.update(kwargs)
                self.last_plate_detection_count = 1
                return [plate], frame.copy()

        session = self.service.new_live_session()
        frame = np.zeros((60, 110, 3), dtype=np.uint8)
        with patch.object(
            self.service,
            "_materialise_candidate",
            return_value={
                "id": 1,
                "country": "thai",
                "plate_prefix": "68",
                "plate_number": "3197",
                "recognition_confidence": 0.9,
            },
        ) as materialise:
            for frame_index in range(1, self.service.settings.video_min_confirmations):
                self.service.process_live_frame(
                    session,
                    Scanner(),
                    frame,
                    frame_index,
                    write_image=lambda path, image: None,
                    preprocess_plate_crop=lambda crop, parameters=None: crop,
                    stride=2,
                    output_stem="camera_live",
                    archive_camera_id="1",
                    media="camera",
                    source_fps=15.0,
                )
                self.assertEqual(session.last_published, [])
            self.assertFalse(materialise.called)
            self.service.process_live_frame(
                session,
                Scanner(),
                frame,
                self.service.settings.video_min_confirmations,
                write_image=lambda path, image: None,
                preprocess_plate_crop=lambda crop, parameters=None: crop,
                stride=2,
                output_stem="camera_live",
                archive_camera_id="1",
                media="camera",
                source_fps=15.0,
            )
            self.assertTrue(materialise.called)
        self.assertFalse(recorded["run_ocr"])
        self.assertEqual(len(session.confirmed_plates), 1)
        self.assertEqual(session.confirmed_plates[0]["plate_number"], "3197")
        self.assertEqual(session.last_published[0]["camera_occurrences"], self.service.settings.video_min_confirmations)

    def test_camera_publishes_first_complete_yolo_reading_without_full_ocr(self) -> None:
        plate = self.plate()

        class Scanner:
            last_plate_detection_count = 1
            last_vehicle_types: list[dict] = []

            @staticmethod
            def draw_scan_roi(frame: np.ndarray) -> np.ndarray:
                return frame.copy()

            def scan(self, frame, *args, **kwargs):
                self.last_plate_detection_count = 1
                return [plate], frame.copy()

        session = self.service.new_live_session()
        frame = np.zeros((60, 110, 3), dtype=np.uint8)
        with patch.object(
            self.service,
            "_materialise_candidate",
            return_value={
                "id": 1,
                "country": "thai",
                "plate_prefix": "68",
                "plate_number": "3197",
                "recognition_confidence": 0.9,
            },
        ) as materialise:
            self.service.process_live_frame(
                session,
                Scanner(),
                frame,
                1,
                write_image=lambda path, image: None,
                preprocess_plate_crop=lambda crop, parameters=None: crop,
                stride=2,
                min_confirmations=1,
                output_stem="camera_live",
                archive_camera_id="1",
                media="camera",
                source_fps=15.0,
            )
            self.assertTrue(materialise.called)
            self.assertFalse(materialise.call_args.kwargs.get("run_full_ocr", True))
        self.assertEqual(len(session.last_published), 1)
        self.assertEqual(session.last_published[0]["plate_number"], "3197")
        self.assertEqual(len(session.pending_ocr), 1)

    def test_deferred_ocr_updates_thai_letter_prefix(self) -> None:
        record = {
            "country": "thai",
            "plate_prefix": "68",
            "plate_number": "3197",
            "province": "",
            "province_code": "",
            "confirmed": True,
            "detection_confidence": 0.9,
            "country_confidence": 0.8,
            "country_readings": {
                "thai": {
                    "plate_prefix": "68",
                    "plate_number": "3197",
                    "complete": True,
                }
            },
            "ocr": {"confidence": 0.0, "method": ""},
        }

        class Scanner:
            @staticmethod
            def finalise_country_reading(crop, country, readings, run_ocr=True):
                readings["thai"] = {
                    "plate_prefix": "กข",
                    "plate_number": "3197",
                    "province": "กรุงเทพมหานคร",
                    "province_code": "BKK",
                    "complete": True,
                }
                return (
                    country,
                    {
                        "text": "กข-3197",
                        "method": "custom-paddleocr-thai-full",
                        "confidence": 0.81,
                        "province": "กรุงเทพมหานคร",
                    },
                    "กข",
                    "3197",
                )

        self.service._scanner = Scanner()
        updated = self.service.refine_deferred_ocr(
            record, np.zeros((20, 60, 3), dtype=np.uint8)
        )
        self.assertIsNotNone(updated)
        self.assertEqual(updated["plate_prefix"], "กข")
        self.assertEqual(updated["plate_number"], "3197")
        self.assertEqual(updated["province"], "กรุงเทพมหานคร")
        self.assertTrue(updated["confirmed"])

    def test_live_frame_does_not_publish_incomplete_yolo_text(self) -> None:
        partial = {**self.plate(), "plate_prefix": "6", "plate_number": "319"}

        class Scanner:
            last_plate_detection_count = 1
            last_vehicle_types: list[dict] = []

            @staticmethod
            def draw_scan_roi(frame: np.ndarray) -> np.ndarray:
                return frame.copy()

            def scan(self, frame, *args, **kwargs):
                self.last_plate_detection_count = 1
                return [partial], frame.copy()

        session = self.service.new_live_session()
        frame = np.zeros((60, 110, 3), dtype=np.uint8)
        with patch.object(self.service, "_materialise_candidate") as materialise:
            for frame_index in range(1, 6):
                self.service.process_live_frame(
                    session,
                    Scanner(),
                    frame,
                    frame_index,
                    write_image=lambda path, image: None,
                    preprocess_plate_crop=lambda crop, parameters=None: crop,
                    stride=2,
                    output_stem="camera_live",
                    archive_camera_id="1",
                    media="camera",
                    source_fps=15.0,
                )
            self.assertFalse(materialise.called)
        self.assertEqual(session.confirmed_plates, [])
        self.assertEqual(session.last_published, [])
        self.assertGreaterEqual(len(session.tracks), 1)

    def _live_scanner(self, readings: list[dict]) -> object:
        """A scanner double that returns one queued plate per sampled frame."""

        queue = list(readings)

        class Scanner:
            last_plate_detection_count = 1
            last_vehicle_types: list[dict] = []
            context_validator = None

            @staticmethod
            def draw_scan_roi(frame: np.ndarray) -> np.ndarray:
                return frame.copy()

            def scan(self, frame, *args, **kwargs):
                self.last_plate_detection_count = 1
                plate = queue.pop(0) if queue else None
                return ([plate] if plate else []), frame.copy()

            @staticmethod
            def finalise_reading(_crop, _country, reading, run_ocr=True):
                return (
                    {"text": reading["text"], "confidence": 0.9, "method": "structured-yolo"},
                    reading["plate_prefix"],
                    reading["plate_number"],
                )

        return Scanner()

    def _live_plate(self, number: str) -> dict:
        return {
            **self.plate(),
            "plate_number": number,
            "country_readings": {
                "thai": {
                    "text": f"68-{number}",
                    "plate_prefix": "68",
                    "plate_number": number,
                    "digit_confidence": 0.90,
                    "prefix_confidence": 0.0,
                }
            },
        }

    def _run_live_frames(
        self,
        session,
        scanner,
        frame_indices: list[int],
        writes: list[Path],
        *,
        source_fps: float = 25.0,
        stride: int = 6,
    ) -> None:
        frame = np.zeros((60, 110, 3), dtype=np.uint8)

        def write(path: Path, unused_image: object) -> None:
            del unused_image
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"image")
            writes.append(path)

        for frame_index in frame_indices:
            self.service.process_live_frame(
                session,
                scanner,
                frame,
                frame_index,
                write_image=write,
                preprocess_plate_crop=lambda crop, parameters=None: crop,
                stride=stride,
                min_confirmations=3,
                output_stem="camera_live",
                archive_camera_id="1",
                media="camera",
                source_fps=source_fps,
            )

    def test_sparse_cctv_samples_keep_one_track(self) -> None:
        # A shared CPU hub sees a CCTV lane only every ~2.5 s (60 frames at
        # 25 fps). The fixed 30-frame gap used to start a new track on every
        # sample, so the plate could never collect three confirmations.
        session = self.service.new_live_session()
        scanner = self._live_scanner([self._live_plate("3197") for _ in range(3)])
        writes: list[Path] = []

        self._run_live_frames(session, scanner, [1, 61, 121], writes)

        self.assertEqual(len(session.tracks), 1)
        self.assertGreaterEqual(session.track_gap_frames, 60)
        self.assertLessEqual(session.track_gap_frames, 150)
        self.assertEqual([plate["plate_number"] for plate in session.confirmed_plates], ["3197"])
        self.assertEqual(session.confirmed_plates[0]["camera_occurrences"], 3)

    def test_later_clearer_frame_refreshes_the_confirmed_image(self) -> None:
        session = self.service.new_live_session()
        scanner = self._live_scanner([self._live_plate("3197") for _ in range(2)])
        writes: list[Path] = []
        blank = np.zeros((60, 110, 3), dtype=np.uint8)
        sharp = np.zeros((60, 110, 3), dtype=np.uint8)
        sharp[10:45, 10:90:2] = 255

        def write(path: Path, unused_image: object) -> None:
            del unused_image
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"image")
            writes.append(path)

        frames = [blank, sharp]
        for frame_index, frame in enumerate(frames, start=1):
            self.service.process_live_frame(
                session,
                scanner,
                frame,
                frame_index,
                write_image=write,
                preprocess_plate_crop=lambda crop, parameters=None: crop,
                stride=1,
                min_confirmations=1,
                output_stem="camera_live",
                archive_camera_id="1",
                media="camera",
                source_fps=15.0,
            )

        self.assertEqual(len(session.confirmed_plates), 1)
        self.assertEqual(session.confirmed_plates[0]["image_revision"], 2)
        self.assertGreater(len(writes), 3)

    def test_track_gap_stays_at_video_default_for_dense_sampling(self) -> None:
        session = self.service.new_live_session()
        scanner = self._live_scanner([self._live_plate("3197") for _ in range(3)])

        self._run_live_frames(session, scanner, [1, 2, 3], [], stride=1)

        self.assertEqual(session.track_gap_frames, 30)

    def test_minority_variant_waits_until_it_leads_its_track(self) -> None:
        # Tiny CCTV plates read as unrelated variants of one registration.
        # Only the reading that currently leads the track may be published,
        # and the losing variant must not leave archive files behind.
        session = self.service.new_live_session()
        readings = [self._live_plate("3197") for _ in range(4)] + [
            self._live_plate("3177") for _ in range(3)
        ]
        scanner = self._live_scanner(readings)
        writes: list[Path] = []

        self._run_live_frames(session, scanner, list(range(1, 8)), writes, stride=1)

        self.assertEqual(len(session.tracks), 1)
        self.assertEqual([plate["plate_number"] for plate in session.confirmed_plates], ["3197"])
        self.assertNotIn("thai|683177", session.confirmed_keys)
        self.assertTrue(all("3197" in path.name for path in writes if "3177" in path.name or "3197" in path.name))
        self.assertFalse(any("3177" in path.name for path in writes))

    def test_variant_that_loses_replacement_gate_writes_no_files(self) -> None:
        session = self.service.new_live_session()
        readings = [self._live_plate("3197") for _ in range(3)] + [
            self._live_plate("3177") for _ in range(3)
        ]
        scanner = self._live_scanner(readings)
        writes: list[Path] = []

        with patch.object(self.service, "_should_replace_live_record", return_value=False):
            self._run_live_frames(session, scanner, list(range(1, 7)), writes, stride=1)

        self.assertEqual([plate["plate_number"] for plate in session.confirmed_plates], ["3197"])
        self.assertFalse(any("3177" in path.name for path in writes))
        self.assertTrue(any("3197" in path.name for path in writes))

    def test_cross_model_digit_conflict_lowers_video_vote_quality(self) -> None:
        baseline = self.plate()
        conflicting = self.plate()
        conflicting["cross_model_digit_evidence"] = {
            "status": "conflict",
            "primary_number": "1356",
            "assistant_number": "6356",
            "resolution": "primary-retained",
        }
        agreeing = self.plate()
        agreeing["cross_model_digit_evidence"] = {"status": "agree"}

        self.assertAlmostEqual(
            self.service._video_plate_quality(baseline)
            - self.service._video_plate_quality(conflicting),
            0.12,
            places=4,
        )
        self.assertAlmostEqual(
            self.service._video_plate_quality(agreeing)
            - self.service._video_plate_quality(baseline),
            0.05,
            places=4,
        )

    def test_confirmed_video_candidate_archives_immediately_without_duplicate_files(self) -> None:
        class Scanner:
            context_validator = None

            @staticmethod
            def finalise_reading(_crop, _country, reading, run_ocr=True):
                return (
                    {"text": reading["text"], "confidence": 0.9, "method": "structured-yolo"},
                    reading["plate_prefix"],
                    reading["plate_number"],
                )

        image = np.zeros((40, 100, 3), dtype=np.uint8)
        candidate = {
            "best_plate": {
                **self.plate(),
                "country_readings": {
                    "thai": {
                        "text": "68-3197",
                        "plate_prefix": "68",
                        "plate_number": "3197",
                    }
                },
            },
            "best_crop": image,
            "best_ready_crop": image,
            "best_vehicle_frame": image,
            "province_votes": {},
        }
        writes: list[Path] = []

        def write(path: Path, unused_image: object) -> None:
            del unused_image
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"image")
            writes.append(path)

        first = self.service._materialise_candidate(candidate, 1, "live", write, Scanner())
        write_count = len(writes)
        second = self.service._materialise_candidate(candidate, 1, "final", write, Scanner())

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(len(writes), write_count)
        self.assertEqual(first["full_vehicle_image"], second["full_vehicle_image"])

    def test_archive_stem_uses_camera_province_prefix_number_date_and_sequence(self) -> None:
        stem = self.service._archive_stem(
            {
                "country": "lao",
                "province": "province_name",
                "province_code": "LNT",
                "plate_prefix": "prefix_name",
                "plate_prefix_code": "N_A",
                "plate_number": "2271",
            },
            "0",
            "20260825",
            1,
        )

        self.assertEqual(stem, "0-LNT-prefix_name-2271-20260825-001")

    def test_archive_sequence_is_scoped_to_dated_folder(self) -> None:
        dated = self.service._dated_output_dir("20260827")
        thai = dated / "thai"
        thai.mkdir(parents=True, exist_ok=True)
        (thai / "0-BKK-63-7998-20260827-028-full_vehicle.jpg").write_bytes(b"")
        (thai / "0-BKK-63-7998-20260826-099-full_vehicle.jpg").write_bytes(b"")

        self.assertEqual(self.service._next_archive_sequence(dated, "20260827"), 29)

    def test_archive_images_are_saved_under_the_daily_folder(self) -> None:
        plate = {
            "country": "thai",
            "province": "กรุงเทพมหานคร",
            "province_code": "BKK",
            "plate_prefix": "63",
            "plate_number": "7998",
            "country_readings": {},
        }

        def fake_write(path: Path, unused_image: object) -> None:
            del unused_image
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"image")

        full_path, crop_path, ready_path, character_path = self.service._save_archive_images(
            plate,
            "0",
            np.zeros((20, 30, 3), dtype=np.uint8),
            np.zeros((10, 20, 3), dtype=np.uint8),
            np.zeros((10, 20, 3), dtype=np.uint8),
            np.zeros((10, 20, 3), dtype=np.uint8),
            object(),
            fake_write,
        )

        date_text = datetime.now().strftime("%Y%m%d")
        self.assertEqual(full_path.parent.parent.name, date_text)
        self.assertEqual(full_path.parent.name, "thai")
        self.assertEqual(full_path.name, f"0-BKK-63-7998-{date_text}-001-full_vehicle.jpg")
        self.assertEqual(crop_path.name, f"0-BKK-63-7998-{date_text}-001-plate_crops.jpg")
        self.assertEqual(ready_path, crop_path)
        self.assertTrue(ready_path.is_file())
        self.assertIsNone(character_path)
        self.assertEqual(
            sorted(path.name for path in full_path.parent.glob("*.jpg")),
            sorted((full_path.name, crop_path.name)),
        )

    def test_result_manifest_has_the_same_portable_contract_for_camera(self) -> None:
        manifest_path = Path(self.temporary.name) / "camera_result.json"
        result = self.service._write_result_manifest(
            {
                "media_type": "camera",
                "input": "camera://2",
                "camera_index": 2,
                "annotated_image": "D:/scan/camera_2_annotated.jpg",
                "output_dir": "D:/scan",
                "plate_count": 0,
                "plates": [],
            },
            manifest_path,
        )

        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(result["schema_version"], "1.0")
        self.assertEqual(
            saved["source"],
            {"type": "camera", "uri": "camera://2", "camera_index": 2, "filename": "camera_2"},
        )
        self.assertEqual(saved["artifacts"]["annotated_image"], "D:/scan/camera_2_annotated.jpg")
        self.assertEqual(saved["artifacts"]["annotated_video"], "")

    def test_repeated_weak_province_candidate_is_promoted_temporally(self) -> None:
        tracks: list[dict] = []
        frame = np.zeros((60, 110, 3), dtype=np.uint8)
        plate = {
            **self.plate(),
            "country": "lao",
            "plate_prefix": "prefix",
            "plate_number": "1563",
            "country_readings": {
                "lao": {
                    "digit_confidence": 0.90,
                    "prefix_confidence": 0.80,
                    "province_candidates": [
                        {
                            "code": "VTE2",
                            "province": "province_name",
                            "confidence": 0.2467,
                            "source": "plate",
                        }
                    ],
                }
            },
        }

        for frame_index in (1, 2):
            self.service._merge_video_plate(
                tracks,
                plate,
                frame,
                frame_index,
                lambda crop: crop.copy(),
            )

        candidate = next(iter(tracks[0]["candidates"].values()))
        record = ScanService._apply_temporal_province_vote(candidate)

        self.assertIsNotNone(record)
        self.assertEqual(record["province_code"], "VTE2")
        self.assertEqual(record["province"], "province_name")
        self.assertEqual(record["country_readings"]["lao"]["province_status"], "temporal_weak_consensus")

    def test_temporal_province_vote_prefers_repeated_result(self) -> None:
        winner = {
            "best_plate": {
                "country": "lao",
                "province": "ວຽງຈັນ",
                "country_readings": {
                    "lao": {
                        "province": "ວຽງຈັນ",
                        "province_code": "VTE",
                        "province_confidence": 0.91,
                    }
                },
            },
            "province_votes": {
                "VTE": {
                    "count": 1,
                    "score": 0.91,
                    "best_confidence": 0.91,
                    "snapshot": {
                        "province": "ວຽງຈັນ",
                        "province_code": "VTE",
                        "province_confidence": 0.91,
                    },
                },
                "LPB": {
                    "count": 2,
                    "score": 1.20,
                    "best_confidence": 0.62,
                    "snapshot": {
                        "province": "ຫຼວງພະບາງ",
                        "province_code": "LPB",
                        "province_confidence": 0.62,
                    },
                },
            },
        }

        record = ScanService._apply_temporal_province_vote(winner)

        self.assertIsNotNone(record)
        self.assertEqual(record["province"], "ຫຼວງພະບາງ")
        self.assertEqual(record["country_readings"]["lao"]["province_code"], "LPB")
        self.assertEqual(record["country_readings"]["lao"]["province_status"], "temporal_consensus")

    def test_incomplete_plate_is_not_published(self) -> None:
        plate = self.plate()
        plate["plate_number"] = "319"

        merged = self.service._merge_video_plate(
            [],
            plate,
            np.zeros((60, 110, 3), dtype=np.uint8),
            1,
            lambda crop: crop.copy(),
        )

        self.assertIsNotNone(merged)
        self.assertFalse(self.service._has_complete_registration(plate))
        self.assertTrue(self.service._has_usable_registration(plate))

    def test_moving_plate_remains_in_the_same_track(self) -> None:
        tracks = [
            {
                "country": "thai",
                "last_frame": 8,
                "last_box": [10, 10, 90, 45],
            }
        ]

        matched = self.service._find_video_track(
            tracks,
            "thai",
            [42, 12, 122, 47],
            9,
        )

        self.assertIs(matched, tracks[0])

    def test_country_correction_keeps_same_overlapping_roi_track(self) -> None:
        tracks = [
            {
                "country": "thai",
                "last_frame": 8,
                "last_box": [10, 10, 90, 45],
            }
        ]

        matched = self.service._find_video_track(
            tracks,
            "lao",
            [12, 11, 92, 46],
            9,
        )

        self.assertIs(matched, tracks[0])

    def test_exact_registration_is_deduplicated_across_track_fragments(self) -> None:
        first = {
            **self.plate(),
            "video_first_frame": 10,
            "video_last_frame": 20,
            "video_occurrences": 3,
            "video_track_observations": 3,
            "video_vote_score": 2.4,
        }
        second = {
            **self.plate(),
            "video_first_frame": 80,
            "video_last_frame": 95,
            "video_occurrences": 4,
            "video_track_observations": 4,
            "video_vote_score": 3.3,
            "recognition_confidence": 0.95,
        }

        records = self.service._deduplicate_video_records([first, second], 30)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["video_first_frame"], 10)
        self.assertEqual(records[0]["video_last_frame"], 95)
        self.assertEqual(records[0]["video_occurrences"], 7)

    def test_nearby_lao_prefix_variant_updates_one_live_result(self) -> None:
        first = {
            "country": "lao",
            "plate_prefix": "ບຄ",
            "plate_number": "1356",
            "video_first_frame": 100,
            "video_last_frame": 120,
        }
        variant = {
            "country": "lao",
            "plate_prefix": "ບກ",
            "plate_number": "1356",
            "video_first_frame": 130,
            "video_last_frame": 140,
        }

        duplicate = self.service._find_duplicate_record([first], variant, 30)

        self.assertEqual(duplicate, 0)

    def test_stronger_reading_replaces_stale_live_track_result(self) -> None:
        old = {
            "country": "thai",
            "plate_prefix": "68",
            "plate_number": "3191",
            "temporal_average_quality": 0.70,
            "video_occurrences": 3,
            "recognition_confidence": 0.70,
        }
        corrected = {
            **old,
            "plate_number": "3197",
            "temporal_average_quality": 0.82,
            "video_occurrences": 3,
            "recognition_confidence": 0.86,
        }

        self.assertTrue(self.service._should_replace_live_record(old, corrected))

    def test_slightly_higher_confidence_corrects_live_one_digit_variant(self) -> None:
        wrong = {
            "country": "lao",
            "plate_prefix": "ບລ",
            "plate_number": "1356",
            "temporal_average_quality": 0.8777,
            "camera_occurrences": 3,
            "recognition_confidence": 0.8777,
        }
        corrected = {
            **wrong,
            "plate_number": "6356",
            "temporal_average_quality": 0.8903,
            "recognition_confidence": 0.8903,
        }

        self.assertTrue(self.service._should_replace_live_record(wrong, corrected))

    def test_final_track_prefers_repeated_text_over_higher_single_frame_score(self) -> None:
        first = {
            "count": 2,
            "vote_score": 1.70,
            "best_quality": 0.90,
            "best_plate": self.plate(),
            "best_crop": np.zeros((4, 4, 3), dtype=np.uint8),
            "best_ready_crop": np.zeros((4, 4, 3), dtype=np.uint8),
        }
        repeated = {
            "count": 3,
            "vote_score": 2.40,
            "best_quality": 0.82,
            "best_plate": self.plate(),
            "best_crop": np.zeros((4, 4, 3), dtype=np.uint8),
            "best_ready_crop": np.zeros((4, 4, 3), dtype=np.uint8),
        }
        tracks = [
            {
                "first_frame": 1,
                "last_frame": 5,
                "observations": 5,
                "candidates": {"first": first, "repeated": repeated},
            }
        ]

        captured: list[dict] = []
        original = self.service._materialise_candidate
        self.service._materialise_candidate = lambda winner, *_args, **_kwargs: dict(  # type: ignore[method-assign]
            winner["best_plate"]
        )
        try:
            result = self.service._finalise_video_tracks(
                tracks,
                "sample",
                lambda *_args: None,
                object(),
                min_confirmations=1,
            )
            captured.extend(result)
        finally:
            self.service._materialise_candidate = original  # type: ignore[method-assign]

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["video_occurrences"], 3)

    def test_different_thai_prefixes_are_not_fuzzy_duplicates(self) -> None:
        first = {
            "country": "thai",
            "plate_prefix": "12",
            "plate_number": "3456",
            "video_first_frame": 100,
            "video_last_frame": 120,
        }
        second = {
            "country": "thai",
            "plate_prefix": "13",
            "plate_number": "3456",
            "video_first_frame": 121,
            "video_last_frame": 140,
        }

        duplicate = self.service._find_duplicate_record([first], second, 30)

        self.assertIsNone(duplicate)


if __name__ == "__main__":
    unittest.main()
