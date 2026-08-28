from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from src.car_scan.config import Settings
from src.car_scan.service import ScanService


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
        full_vehicle = dated / "full_vehicle"
        full_vehicle.mkdir(parents=True, exist_ok=True)
        (full_vehicle / "0-BKK-63-7998-20260827-028.jpg").write_bytes(b"")
        (full_vehicle / "0-BKK-63-7998-20260826-099.jpg").write_bytes(b"")

        self.assertEqual(self.service._next_archive_sequence(full_vehicle, "20260827"), 29)

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
        self.assertEqual(full_path.name, f"0-BKK-63-7998-{date_text}-001.jpg")
        self.assertEqual(crop_path.name, full_path.name)
        self.assertTrue(ready_path.is_file())
        self.assertTrue(character_path.is_file())

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

        self.assertIsNone(merged)

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
