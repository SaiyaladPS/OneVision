from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scan import (
    LicensePlateScanner,
    clamp_crop,
    complete_detector_crop,
    empty_country_reading,
)
from src.car_scan.config import Settings
from src.car_scan.pipeline import (
    UNKNOWN_CLASS,
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

    def test_thai_letter_prefix_from_ocr_is_valid(self) -> None:
        validator = ContextValidator({})
        result = validator.validate("thai", "car", "private", "กข", "1234", "กรุงเทพมหานคร")
        self.assertTrue(result["valid"], result["reasons"])
        self.assertTrue(
            ScanService._has_complete_registration(
                {"country": "thai", "plate_prefix": "กข", "plate_number": "1234"}
            )
        )

    def test_thai_motorcycle_four_digits_with_province_is_valid(self) -> None:
        validator = ContextValidator({})
        result = validator.validate("thai", "motorcycle", "unknown", "", "0059", "นครปฐม")
        self.assertTrue(result["valid"], result["reasons"])
        self.assertTrue(
            ScanService._has_complete_registration(
                {
                    "country": "thai",
                    "plate_prefix": "",
                    "plate_number": "0059",
                    "province": "นครปฐม",
                    "province_code": "NPT",
                }
            )
        )

    def test_service_respects_context_validation_result(self) -> None:
        plate = {
            "country": "thai",
            "plate_prefix": "12",
            "plate_number": "3456",
            "validation": {"valid": False, "reasons": ["configured_pattern_mismatch"]},
        }

        self.assertFalse(ScanService._has_complete_registration(plate))

    def test_service_copies_ocr_province_onto_live_record(self) -> None:
        service = ScanService(
            Settings(root=Path.cwd(), output_dir=Path.cwd() / "scan" / "data", database_url="")
        )

        class Scanner:
            context_validator = None

            @staticmethod
            def finalise_reading(_crop, _country, reading, run_ocr=True):
                reading["province"] = "ระยอง"
                reading["province_code"] = "RYG"
                return (
                    {
                        "text": "ผค-5939",
                        "raw_text": "ระยอง ผค-5939",
                        "confidence": 0.4,
                        "method": "custom-paddleocr-thai-full",
                        "province": "ระยอง",
                        "province_code": "RYG",
                    },
                    "ผค",
                    "5939",
                )

        record = {
            "country": "thai",
            "province": "",
            "province_code": "",
            "plate_prefix": "68",
            "plate_number": "5939",
            "country_readings": {
                "thai": {"text": "68-5939", "plate_prefix": "68", "plate_number": "5939"}
            },
            "detection_confidence": 0.9,
            "country_confidence": 0.8,
            "plate": {},
        }
        updated = service._apply_reading_and_quality(
            record,
            Scanner(),
            np.zeros((8, 24, 3), dtype=np.uint8),
            run_full_ocr=True,
        )
        self.assertEqual(updated["province"], "ระยอง")
        self.assertEqual(updated["province_code"], "RYG")
        self.assertEqual(updated["plate"]["province"], "ระยอง")
        self.assertTrue(updated["confirmed"])

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

    def test_country_models_are_not_called_when_detect_license_finds_nothing(self) -> None:
        class EmptyDetector:
            names = {0: "plate"}

            def predict(self, *args: object, **kwargs: object) -> list[object]:
                del args, kwargs
                return [type("Prediction", (), {"boxes": []})()]

        scanner = LicensePlateScanner.__new__(LicensePlateScanner)
        scanner.pipeline_mode = "fast"
        scanner.detector = EmptyDetector()
        scanner._vehicle_types = lambda *args: []
        country_calls: list[str] = []
        scanner._read_country_model = lambda *args, **kwargs: country_calls.append("country") or {}
        scanner._read_country_models_from_detector_crop = (
            lambda *args, **kwargs: country_calls.append("country-batch") or {}
        )

        plates, _annotated = scanner.scan(
            np.zeros((40, 80, 3), dtype=np.uint8),
            0.35,
            0.20,
            0.03,
            640,
        )

        self.assertEqual(plates, [])
        self.assertEqual(country_calls, [])

    def test_country_models_receive_detector_crop_not_full_image(self) -> None:
        class Tensor:
            def __init__(self, value: object) -> None:
                self.value = value

            def cpu(self):
                return self

            def tolist(self):
                return self.value

        class Detector:
            names = {0: "plate"}

            def predict(self, *args: object, **kwargs: object) -> list[object]:
                del args, kwargs
                boxes = type(
                    "Boxes",
                    (),
                    {
                        "xyxy": Tensor([[12.0, 10.0, 68.0, 42.0]]),
                        "conf": Tensor([0.91]),
                        "cls": Tensor([0.0]),
                        "__len__": lambda self: 1,
                    },
                )()
                return [type("Prediction", (), {"boxes": boxes})()]

        image = np.zeros((80, 120, 3), dtype=np.uint8)
        received: list[tuple[str, tuple[int, int]]] = []
        scanner = LicensePlateScanner.__new__(LicensePlateScanner)
        scanner.pipeline_mode = "fast"
        scanner.scan_roi = None
        scanner.detector = Detector()
        scanner._vehicle_types = lambda *args: []
        scanner.country_classifier = CountryClassifier()
        scanner.plate_type_classifier = type(
            "PlateType",
            (),
            {"classify": lambda *args, **kwargs: dict(UNKNOWN_CLASS)},
        )()
        scanner.context_router = ContextRouter({})
        scanner.context_validator = ContextValidator({})
        scanner.pipeline_config = {}
        scanner.finalise_reading = lambda *args, **kwargs: (
            {"text": "", "confidence": 0.0, "method": "test"},
            "",
            "",
        )

        def fake_read(crop, country, *args, **kwargs):
            received.append((country, tuple(crop.shape[:2])))
            return empty_country_reading(country)

        scanner._read_country_model = fake_read

        plates, _annotated = scanner.scan(image, 0.35, 0.20, 0.03, 640)

        expected_crop, _box, status = complete_detector_crop(
            image, [12.0, 10.0, 68.0, 42.0], 0.03
        )
        self.assertEqual(status, "full")
        self.assertEqual(len(plates), 1)
        self.assertEqual(
            [country for country, _shape in received],
            ["thai", "lao", "thai", "lao"],
        )
        for _country, shape in received:
            self.assertEqual(shape, expected_crop.shape[:2])
            self.assertNotEqual(shape, image.shape[:2])

    def test_stream_can_reuse_vehicle_classification_without_new_inference(self) -> None:
        class EmptyDetector:
            def predict(self, *args: object, **kwargs: object) -> list[object]:
                del args, kwargs
                return [type("Prediction", (), {"boxes": []})()]

        scanner = LicensePlateScanner.__new__(LicensePlateScanner)
        scanner.pipeline_mode = "fast"
        scanner.detector = EmptyDetector()
        calls: list[str] = []
        scanner._vehicle_types = lambda *args: calls.append("vehicle") or []
        cached = [
            {
                "normalized_class": "car",
                "display_name": "Car",
                "confidence": 0.9,
                "box": [0, 0, 20, 20],
            }
        ]

        scanner.scan(
            np.zeros((20, 30, 3), dtype=np.uint8),
            0.35,
            0.20,
            0.03,
            640,
            vehicle_types_override=cached,
        )

        self.assertEqual(calls, [])
        self.assertEqual(scanner.last_vehicle_types, cached)

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

    def test_plate_detection_runs_on_full_frame_then_filters_roi(self) -> None:
        class Tensor:
            def __init__(self, value: object) -> None:
                self.value = value

            def cpu(self):
                return self

            def tolist(self):
                return self.value

        received_shapes: list[tuple[int, int]] = []

        class Detector:
            names = {0: "plate"}

            def predict(self, image: np.ndarray, *args: object, **kwargs: object) -> list[object]:
                del args, kwargs
                received_shapes.append(tuple(image.shape[:2]))
                boxes = type(
                    "Boxes",
                    (),
                    {
                        "xyxy": Tensor([[80.0, 40.0, 120.0, 60.0], [0.0, 0.0, 30.0, 20.0]]),
                        "conf": Tensor([0.91, 0.88]),
                        "cls": Tensor([0.0, 0.0]),
                        "__len__": lambda self: 2,
                    },
                )()
                return [type("Prediction", (), {"boxes": boxes})()]

        scanner = LicensePlateScanner.__new__(LicensePlateScanner)
        scanner.pipeline_mode = "fast"
        scanner.scan_roi = scanner._normalise_scan_roi(
            {
                "enabled": True,
                "shape": "rectangle",
                "x": 0.25,
                "y": 0.25,
                "width": 0.50,
                "height": 0.50,
            }
        )
        scanner.detector = Detector()
        scanner._vehicle_types = lambda *args: []
        scanner.country_classifier = CountryClassifier()
        scanner.plate_type_classifier = type(
            "PlateType",
            (),
            {"classify": lambda *args, **kwargs: dict(UNKNOWN_CLASS)},
        )()
        scanner.context_router = ContextRouter({})
        scanner.context_validator = ContextValidator({})
        scanner.pipeline_config = {}
        scanner.finalise_reading = lambda *args, **kwargs: (
            {"text": "", "confidence": 0.0, "method": "test"},
            "",
            "",
        )
        scanner._read_country_model = lambda *args, **kwargs: empty_country_reading("thai")
        scanner._read_country_models_from_detector_crop = lambda *args, **kwargs: {
            "thai": empty_country_reading("thai"),
            "lao": empty_country_reading("lao"),
        }

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        plates, _annotated = scanner.scan(image, 0.35, 0.20, 0.03, 640)

        self.assertEqual(received_shapes, [(100, 200)])
        self.assertEqual(len(plates), 1)
        _crop, expected_box, status = complete_detector_crop(
            image, [80.0, 40.0, 120.0, 60.0], 0.03
        )
        self.assertEqual(status, "full")
        self.assertEqual(plates[0]["box"], expected_box)

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
