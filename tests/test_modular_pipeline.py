from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scan import LicensePlateScanner
from src.car_scan.config import Settings
from src.car_scan.pipeline import (
    ClassMapper,
    ContextRouter,
    ContextValidator,
    CountryClassifier,
    ProcessingContext,
    VehiclePlateAssociator,
)
from src.car_scan.service import ScanService


class ModularPipelineTests(unittest.TestCase):
    def test_class_mapping_preserves_model_raw_class(self) -> None:
        mapper = ClassMapper(
            {"bike": {"display_name": "Motorcycle", "normalized_class": "motorcycle"}}
        )

        result = mapper.map("bike", 0.87654, [1, 2, 30, 40], 2)

        self.assertEqual(result["raw_class"], "bike")
        self.assertEqual(result["display_name"], "Motorcycle")
        self.assertEqual(result["normalized_class"], "motorcycle")
        self.assertEqual(result["class_id"], 2)

    def test_country_classification_is_independent_and_can_be_unknown(self) -> None:
        classifier = CountryClassifier(unknown_score=0.60)

        uncertain = classifier.classify(
            {"thai": {"score": 0.52}, "lao": {"score": 0.49}}
        )
        clear = classifier.classify(
            {"thai": {"score": 0.41}, "lao": {"score": 0.88}}
        )

        self.assertEqual(uncertain["country"], "unknown")
        self.assertEqual(uncertain["raw_country"], "thai")
        self.assertEqual(clear["country"], "lao")

    def test_each_plate_is_associated_with_its_own_vehicle(self) -> None:
        associator = VehiclePlateAssociator()
        vehicles = [
            {
                "vehicle_id": 1,
                "raw_class": "car",
                "normalized_class": "car",
                "box": [0, 0, 200, 140],
            },
            {
                "vehicle_id": 2,
                "raw_class": "bike",
                "normalized_class": "motorcycle",
                "box": [300, 0, 430, 180],
            },
        ]

        first = associator.associate([70, 80, 130, 110], vehicles)
        second = associator.associate([330, 90, 400, 130], vehicles)

        self.assertEqual(first["vehicle_id"], 1)
        self.assertEqual(second["vehicle_id"], 2)
        self.assertEqual(second["normalized_class"], "motorcycle")

    def test_context_router_uses_specific_route_then_common_fallback(self) -> None:
        router = ContextRouter(
            {
                "preprocessing_profiles": {
                    "thai|motorcycle|*": "thai_motorcycle",
                    "*|*|*": "common",
                },
                "ocr_routes": {"*|*|*": "common_ocr"},
            }
        )

        motorcycle = ProcessingContext("thai", "motorcycle", "unknown")
        unknown = ProcessingContext("unknown", "unknown", "unknown")

        self.assertEqual(router.preprocessing_profile(motorcycle), "thai_motorcycle")
        self.assertEqual(router.preprocessing_profile(unknown), "common")
        self.assertEqual(router.ocr_model(motorcycle), "common_ocr")

    def test_validation_uses_country_vehicle_and_plate_type_rule(self) -> None:
        validator = ContextValidator(
            {
                "validation": {
                    "thai|motorcycle|private": {"allowed_pattern": r"12\d{4}"}
                }
            }
        )

        valid = validator.validate("thai", "motorcycle", "private", "12", "3456")
        invalid = validator.validate("thai", "motorcycle", "private", "34", "3456")

        self.assertTrue(valid["valid"])
        self.assertFalse(invalid["valid"])
        self.assertIn("configured_pattern_mismatch", invalid["reasons"])

    def test_service_respects_context_validation_result(self) -> None:
        plate = {
            "country": "thai",
            "plate_prefix": "12",
            "plate_number": "3456",
            "validation": {"valid": False, "reasons": ["configured_pattern_mismatch"]},
        }

        self.assertFalse(ScanService._has_complete_registration(plate))

    def test_debug_mode_creates_the_documented_stage_folders(self) -> None:
        with tempfile.TemporaryDirectory(prefix="car_scan_debug_test_") as temporary:
            service = ScanService(
                Settings(
                    root=Path.cwd(),
                    output_dir=Path(temporary),
                    database_url="",
                    debug_enabled=True,
                )
            )
            image = np.zeros((40, 80, 3), dtype=np.uint8)
            plate = {
                "box": [10, 10, 60, 30],
                "ocr_crop_box": [8, 8, 62, 32],
                "country": "thai",
                "vehicle": {"normalized_class": "car"},
                "plate_type": "unknown",
                "ocr": {"text": "12-3456"},
                "validation": {"valid": True},
                "processing": {"preprocessing_parameters": {}},
            }

            def fake_write(path: Path, unused_image: object) -> None:
                del unused_image
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"image")

            service._write_image_debug_artifacts(
                "sample",
                image,
                image,
                [plate],
                [],
                lambda crop, parameters=None: crop,
                fake_write,
            )

            expected = {
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
            }
            self.assertEqual(
                {path.name for path in service.settings.debug_dir.iterdir() if path.is_dir()},
                expected,
            )

    def test_fast_path_skips_vehicle_model_when_no_plate_is_detected(self) -> None:
        class EmptyDetector:
            def predict(self, *args: object, **kwargs: object) -> list[object]:
                del args, kwargs
                return [type("Prediction", (), {"boxes": []})()]

        scanner = LicensePlateScanner.__new__(LicensePlateScanner)
        scanner.pipeline_mode = "fast"
        scanner.detector = EmptyDetector()
        calls: list[str] = []
        scanner._vehicle_types = lambda *args: calls.append("vehicle") or []

        plates, annotated = scanner.scan(
            np.zeros((20, 30, 3), dtype=np.uint8),
            0.35,
            0.20,
            0.03,
            640,
        )

        self.assertEqual(plates, [])
        self.assertEqual(calls, [])
        self.assertEqual(annotated.shape, (20, 30, 3))

    def test_scan_roi_filters_boxes_and_draws_rectangle(self) -> None:
        scanner = LicensePlateScanner.__new__(LicensePlateScanner)
        scanner.scan_roi = scanner._normalise_scan_roi(
            {
                "enabled": True,
                "shape": "rectangle",
                "x": 0.25,
                "y": 0.25,
                "width": 0.50,
                "height": 0.50,
                "color": "#ff0000",
            }
        )
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        self.assertTrue(scanner._box_inside_scan_roi([80, 40, 120, 60], image))
        self.assertFalse(scanner._box_inside_scan_roi([0, 0, 30, 20], image))
        annotated = scanner.draw_scan_roi(image)
        self.assertGreater(int(annotated[:, :, 2].sum()), 0)

    def test_scan_roi_supports_circle(self) -> None:
        scanner = LicensePlateScanner.__new__(LicensePlateScanner)
        scanner.scan_roi = scanner._normalise_scan_roi(
            {
                "enabled": True,
                "shape": "circle",
                "x": 0.25,
                "y": 0.0,
                "width": 0.50,
                "height": 1.0,
            }
        )
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        self.assertTrue(scanner._box_inside_scan_roi([90, 45, 110, 55], image))
        self.assertFalse(scanner._box_inside_scan_roi([50, 0, 70, 20], image))


if __name__ == "__main__":
    unittest.main()
