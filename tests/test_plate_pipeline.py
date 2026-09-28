from __future__ import annotations

import unittest

import cv2
import numpy as np

from scan import (
    published_plate_confidence,
    contested_script_confidence,
    CharacterDetection,
    rectify_plate_for_recognition,
    LAO_CHARACTER_NAMES,
    LAO_PROVINCE_NAMES,
    LicensePlateScanner,
    THAI_CHARACTER_NAMES,
    analyse_country_characters,
    character_glyph,
    build_plate_quality,
    complete_detector_crop,
    country_hint_from_text,
    cross_model_digit_evidence,
    country_reading_score,
    expand_detector_box_to_full_plate,
    extract_ocr_province,
    fuse_cross_model_digits,
    full_plate_crop_status,
    is_usable_detector_crop,
    resolve_plate_fields,
    split_plate_text,
    apply_country_layout_priority,
)
from src.car_scan.pipeline import CountryClassifier


def box(text: str, confidence: float, x: float, y: float, width: float = 10, height: float = 24) -> CharacterDetection:
    return CharacterDetection(text, confidence, x, y, x + width, y + height)


class StructuredPlatePipelineTests(unittest.TestCase):
    def test_country_models_do_not_run_without_a_detect_license_crop(self) -> None:
        class Model:
            def predict(self, *args: object, **kwargs: object) -> list[object]:
                raise AssertionError("Thai/Lao models must not run without a detect_license crop")

        scanner = object.__new__(LicensePlateScanner)
        scanner.country_models = {"thai": Model(), "lao": Model()}
        invalid_crops = (
            None,
            np.zeros((0, 0, 3), dtype=np.uint8),
            np.zeros((4, 10, 3), dtype=np.uint8),
        )

        for crop in invalid_crops:
            self.assertFalse(is_usable_detector_crop(crop))
            reading = scanner._read_country_model(crop, "thai", 0.20, 640)
            batch = scanner._read_country_models_from_detector_crop(crop, 0.20, 640)
            self.assertEqual(reading["inference_variant"], "skipped-no-detector-crop")
            self.assertFalse(reading["complete"])
            self.assertEqual(batch["thai"]["inference_variant"], "skipped-no-detector-crop")
            self.assertEqual(batch["lao"]["inference_variant"], "skipped-no-detector-crop")

    def test_country_models_run_only_on_a_usable_detector_crop(self) -> None:
        class Model:
            names = {0: "1"}

            def __init__(self) -> None:
                self.calls: list[object] = []

            def predict(self, image, **kwargs):
                self.calls.append(image)
                return [type("Prediction", (), {"boxes": None})()]

        scanner = object.__new__(LicensePlateScanner)
        thai = Model()
        lao = Model()
        scanner.country_models = {"thai": thai, "lao": lao}
        crop = np.zeros((24, 80, 3), dtype=np.uint8)

        readings = scanner._read_country_models_from_detector_crop(
            crop, 0.20, 640, fast_mode=True
        )

        self.assertTrue(is_usable_detector_crop(crop))
        self.assertGreater(len(thai.calls), 0)
        self.assertGreater(len(lao.calls), 0)
        self.assertEqual(readings["thai"]["script"], "thai")
        self.assertEqual(readings["lao"]["script"], "lao")
        self.assertNotEqual(readings["thai"]["inference_variant"], "skipped-no-detector-crop")

    def test_thai_plate_is_identified_before_full_country_pipeline(self) -> None:
        scanner = object.__new__(LicensePlateScanner)
        scanner.country_classifier = CountryClassifier(unknown_score=0.30)
        calls: list[tuple[str, bool]] = []

        def fake_read(
            crop: object,
            country: str,
            confidence: float,
            imgsz: int,
            fast_mode: bool = False,
            rectified: object = None,
            probe_mode: bool = False,
        ) -> dict:
            del crop, confidence, imgsz, fast_mode, rectified
            calls.append((country, probe_mode))
            if country == "thai":
                return {
                    "plate_prefix": "กข",
                    "plate_number": "0918",
                    "digit_detection_count": 4,
                    "digit_confidence": 0.91,
                    "score": 0.96,
                    "complete": True,
                    "province_code": "BKK",
                    "province": "กรุงเทพมหานคร",
                    "tokens": [],
                }
            return {
                "plate_prefix": "",
                "plate_number": "0918",
                "digit_detection_count": 4,
                "digit_confidence": 0.41,
                "score": 0.22,
                "complete": False,
                "tokens": [],
            }

        scanner._read_country_model = fake_read
        readings, decision = scanner._classify_and_read_plate(
            np.zeros((24, 80, 3), dtype=np.uint8), 0.20, 640, fast_mode=True
        )

        self.assertEqual(decision["country"], "thai")
        self.assertEqual(decision["pipeline"], "thai")
        self.assertTrue(decision["committed"])
        self.assertEqual(readings["thai"]["plate_prefix"], "กข")
        self.assertEqual(sorted(country for country, probe in calls if probe), ["lao", "thai"])
        self.assertEqual([country for country, probe in calls if not probe], ["thai"])

    def test_lao_plate_is_identified_before_full_country_pipeline(self) -> None:
        scanner = object.__new__(LicensePlateScanner)
        scanner.country_classifier = CountryClassifier(unknown_score=0.30)
        calls: list[tuple[str, bool]] = []

        def fake_read(
            crop: object,
            country: str,
            confidence: float,
            imgsz: int,
            fast_mode: bool = False,
            rectified: object = None,
            probe_mode: bool = False,
        ) -> dict:
            del crop, confidence, imgsz, fast_mode, rectified
            calls.append((country, probe_mode))
            if country == "lao":
                return {
                    "plate_prefix": "ວມ",
                    "plate_number": "7424",
                    "digit_detection_count": 4,
                    "digit_confidence": 0.92,
                    "score": 0.94,
                    "complete": True,
                    "province_code": "SVK",
                    "province": "ສະຫວັນນະເຂດ",
                    "tokens": [],
                }
            return {
                "plate_prefix": "74",
                "plate_number": "7424",
                "digit_detection_count": 4,
                "digit_confidence": 0.40,
                "score": 0.18,
                "complete": False,
                "tokens": [],
            }

        scanner._read_country_model = fake_read
        _readings, decision = scanner._classify_and_read_plate(
            np.zeros((24, 80, 3), dtype=np.uint8), 0.20, 640, fast_mode=True
        )

        self.assertEqual(decision["country"], "lao")
        self.assertEqual(decision["pipeline"], "lao")
        self.assertTrue(decision["committed"])
        self.assertEqual([country for country, probe in calls if not probe], ["lao"])

    def test_uncertain_country_probe_runs_both_full_pipelines(self) -> None:
        scanner = object.__new__(LicensePlateScanner)
        scanner.country_classifier = CountryClassifier(unknown_score=0.30)
        calls: list[tuple[str, bool]] = []

        def fake_read(
            crop: object,
            country: str,
            confidence: float,
            imgsz: int,
            fast_mode: bool = False,
            rectified: object = None,
            probe_mode: bool = False,
        ) -> dict:
            del crop, confidence, imgsz, fast_mode, rectified
            calls.append((country, probe_mode))
            return {
                "plate_prefix": "",
                "plate_number": "12",
                "digit_detection_count": 2,
                "digit_confidence": 0.40,
                "score": 0.35 if country == "thai" else 0.33,
                "complete": False,
                "tokens": [],
            }

        scanner._read_country_model = fake_read
        _readings, decision = scanner._classify_and_read_plate(
            np.zeros((24, 80, 3), dtype=np.uint8), 0.20, 640, fast_mode=True
        )

        self.assertEqual(decision["selection_stage"], "full-both")
        self.assertEqual(
            sorted(country for country, probe in calls if not probe),
            ["lao", "thai"],
        )

    def test_frame_edge_fragment_is_not_a_full_plate_crop(self) -> None:
        image = np.zeros((200, 320, 3), dtype=np.uint8)
        fragment = [0.0, 131.0, 26.0, 163.0]

        self.assertEqual(full_plate_crop_status(fragment, image.shape), "clipped_by_frame")
        crop, box, status = complete_detector_crop(image, fragment, 0.03)
        self.assertIsNone(crop)
        self.assertIsNone(box)
        self.assertEqual(status, "clipped_by_frame")

    def test_narrow_interior_box_is_expanded_to_full_plate(self) -> None:
        image = np.zeros((200, 320, 3), dtype=np.uint8)
        tight = [140.0, 80.0, 166.0, 112.0]
        expanded = expand_detector_box_to_full_plate(tight, image.shape)

        self.assertGreaterEqual(expanded[2] - expanded[0], 40)
        self.assertGreaterEqual((expanded[2] - expanded[0]) / max(expanded[3] - expanded[1], 1.0), 1.2)
        self.assertGreater(expanded[0], 2)
        self.assertLess(expanded[2], 318)
        crop, box, status = complete_detector_crop(image, tight, 0.03)
        self.assertEqual(status, "full")
        self.assertIsNotNone(crop)
        self.assertGreaterEqual(crop.shape[1], 40)

    def test_rectified_plate_copy_is_straightened_and_enlarged(self) -> None:
        # A padded CCTV crop with a plate tilted by ~4 degrees, like camera 1.
        plate = np.full((60, 160, 3), 235, dtype=np.uint8)
        for x in (30, 60, 90, 120):
            plate[14:46, x : x + 8] = 20
        canvas = np.full((110, 230, 3), 90, dtype=np.uint8)
        canvas[25:85, 35:195] = plate
        matrix = cv2.getRotationMatrix2D((115, 55), 4.0, 1.0)
        tilted = cv2.warpAffine(canvas, matrix, (230, 110), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        rectified, report = rectify_plate_for_recognition(tilted, tight_width=160)

        self.assertIsNotNone(rectified)
        self.assertTrue(report["applied"])
        self.assertEqual(report["target_width"], 320)
        self.assertGreaterEqual(rectified.shape[1], 300)
        # A visible border is fixed by the perspective warp; otherwise the
        # Hough skew estimate must have rotated the plate back.
        self.assertTrue(report["perspective_applied"] or abs(float(report["skew_angle"])) >= 1.0)
        # Either way the strokes end up vertical: the dark bars occupy the
        # same columns from top to bottom of the text band.
        gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
        band = gray[int(gray.shape[0] * 0.3) : int(gray.shape[0] * 0.7)]
        dark = band < 100
        top_left = int(np.flatnonzero(dark[0])[0])
        bottom_left = int(np.flatnonzero(dark[-1])[0])
        # In the tilted input the same bar edge drifts ~4 px per 60 rows;
        # after rectification the leftmost stroke starts in the same column.
        self.assertLessEqual(abs(top_left - bottom_left), 4)

    def test_rectified_plate_copy_is_skipped_for_unusable_crops(self) -> None:
        self.assertEqual(rectify_plate_for_recognition(None, 10), (None, {"applied": False}))
        self.assertEqual(
            rectify_plate_for_recognition(np.zeros((4, 10, 3), dtype=np.uint8), 10),
            (None, {"applied": False}),
        )

    def test_rectified_copy_joins_the_character_model_ensemble(self) -> None:
        class Model:
            names = {0: "1"}

            def __init__(self) -> None:
                self.batches: list[list[object]] = []

            def predict(self, image, **kwargs):
                images = image if isinstance(image, list) else [image]
                self.batches.append(images)
                return [type("Prediction", (), {"boxes": None})() for _ in images]

        scanner = object.__new__(LicensePlateScanner)
        model = Model()
        scanner.country_models = {"thai": model, "lao": Model()}
        crop = np.zeros((24, 80, 3), dtype=np.uint8)
        rectified = np.zeros((80, 256, 3), dtype=np.uint8)

        scanner._read_country_model(crop, "thai", 0.20, 640, fast_mode=True)
        baseline_calls = len(model.batches)
        baseline_batch = list(model.batches[-1])
        scanner._read_country_model(crop, "thai", 0.20, 640, fast_mode=True, rectified=rectified)

        # The raw-crop batch is unchanged (same shapes keep the rectangular
        # letterbox); the rectified copy is inferred in its own call.
        self.assertEqual(len(model.batches), baseline_calls + 2)
        self.assertEqual(len(model.batches[-2]), len(baseline_batch))
        self.assertEqual(len(model.batches[-1]), 1)
        self.assertIs(model.batches[-1][0], rectified)

    def test_character_variants_are_batched_with_their_own_thresholds(self) -> None:
        class Model:
            def __init__(self) -> None:
                self.calls: list[tuple[object, float]] = []

            def predict(self, image, *, conf, imgsz, verbose):
                del imgsz, verbose
                self.calls.append((image, conf))
                return [object() for _ in image]

        scanner = object.__new__(LicensePlateScanner)
        model = Model()
        variants = [
            (np.zeros((8, 12, 3), dtype=np.uint8), "raw", 0.30),
            (np.ones((8, 12, 3), dtype=np.uint8), "normalised", 0.12),
        ]

        predictions = scanner._predict_character_variants(model, variants, 960)

        self.assertEqual(len(model.calls), 1)
        self.assertIsInstance(model.calls[0][0], list)
        self.assertEqual(model.calls[0][1], 0.12)
        self.assertEqual(
            [(label, confidence) for _prediction, label, confidence in predictions],
            [("raw", 0.30), ("normalised", 0.12)],
        )

    def test_quality_contract_keeps_lao_primary_complete_when_thai_is_incomplete(self) -> None:
        readings = {
            "lao": {
                "plate_prefix": "ບບ",
                "plate_number": "9955",
                "digit_detection_count": 4,
                "digit_confidence": 1.0,
                "character_confidence": 1.0,
                "province": "ວຽງຈັນ",
                "province_code": "VTE",
                "province_confidence": 1.0,
            },
            "thai": {"digit_detection_count": 4, "plate_number": "9955"},
        }

        quality = build_plate_quality(
            country="lao",
            primary_country="lao",
            readings=readings,
            plate_prefix="ບບ",
            plate_number="9955",
            ocr={"confidence": 1.0},
            validation={"valid": True},
            detection_confidence=1.0,
            country_confidence=1.0,
            digit_evidence={"status": "agree"},
        )

        self.assertEqual(quality["qc"]["status"], "PASS")
        self.assertFalse(quality["qc"]["need_review"])
        self.assertEqual(quality["character_validation"]["primary_detected_digit_count"], 4)
        self.assertEqual(quality["character_validation"]["secondary_detected_digit_count"], 4)
        self.assertEqual(quality["dataset"]["export_status"], "PASS")

    def test_cross_model_conflict_requires_review_even_when_lao_layout_is_complete(self) -> None:
        readings = {
            "lao": {
                "plate_prefix": "ບບ", "plate_number": "1356", "digit_detection_count": 4,
                "digit_confidence": 1.0, "character_confidence": 1.0,
                "province_code": "VTE", "province_confidence": 1.0,
            },
            "thai": {"plate_number": "6356", "digit_detection_count": 6},
        }

        quality = build_plate_quality(
            country="lao", primary_country="lao", readings=readings,
            plate_prefix="ບບ", plate_number="1356", ocr={"confidence": 1.0},
            validation={"valid": True}, detection_confidence=1.0, country_confidence=1.0,
            digit_evidence={"status": "conflict"},
        )

        self.assertEqual(quality["qc"]["status"], "REVIEW")
        self.assertIn("cross_model_digit_conflict", quality["qc"]["reason"])
        self.assertEqual(quality["dataset"]["export_status"], "REJECT")

    def test_higher_score_complete_thai_truck_stays_primary_on_conflict(self) -> None:
        readings = {
            "lao": {
                "plate_prefix": "ບຄ",
                "plate_number": "1356",
                "digit_confidence": 0.86,
                "score": 0.89,
                "complete": True,
            },
            "thai": {
                "plate_prefix": "68",
                "plate_number": "6356",
                "digit_confidence": 0.95,
                "score": 0.92,
                "complete": True,
            },
        }

        result = apply_country_layout_priority(
            {"country": "thai", "raw_country": "thai", "confidence": 0.60, "margin": 0.03},
            readings,
        )
        evidence = cross_model_digit_evidence(readings, result["raw_country"])

        self.assertEqual(result["country"], "thai")
        self.assertEqual(result["selection_reason"], "thai_layout_priority")
        self.assertEqual(evidence["primary_number"], "6356")
        self.assertEqual(evidence["assistant_number"], "1356")
        self.assertEqual(evidence["status"], "conflict")
        self.assertEqual(evidence["different_positions"], [0])
        self.assertEqual(evidence["resolution"], "primary-retained")

    def test_cross_model_matching_digits_are_recorded_as_agreement(self) -> None:
        readings = {
            "lao": {"plate_number": "1356", "digit_confidence": 0.86},
            "thai": {"plate_number": "1356", "digit_confidence": 0.95},
        }

        evidence = cross_model_digit_evidence(readings, "lao")

        self.assertEqual(evidence["status"], "agree")
        self.assertEqual(evidence["different_positions"], [])

    def test_province_roi_is_upscaled_and_consistent_candidates_are_supported(self) -> None:
        class Tensor:
            def __init__(self, value):
                self.value = value

            def cpu(self):
                return self

            def tolist(self):
                return self.value

        class Boxes:
            xyxy = Tensor([[600.0, 30.0, 900.0, 150.0]])
            conf = Tensor([0.80])
            cls = Tensor([0])

        class Prediction:
            boxes = Boxes()

        class ProvinceModel:
            names = {0: "VTE"}

            def __init__(self) -> None:
                self.shapes = []

            def predict(self, image, **kwargs):
                self.shapes.append(image.shape[:2])
                return [Prediction()]

        scanner = object.__new__(LicensePlateScanner)
        model = ProvinceModel()
        scanner.country_models = {"lao": model}
        crop = np.zeros((100, 200, 3), dtype=np.uint8)
        reading = scanner._attach_province_candidates(
            {"tokens": []}, crop, "lao", 0.90, 640
        )

        self.assertEqual(model.shapes, [(330, 1200)] * 3)
        self.assertEqual(reading["province_code"], "VTE")
        self.assertEqual(reading["province_status"], "province_roi_accepted")
        self.assertEqual(reading["province_candidates"][0]["support_count"], 3)
        self.assertEqual(reading["province_candidates"][0]["box"], [100.0, 5.0, 150.0, 25.0])

    def test_ocr_document_character_mappings_are_exposed_as_unicode(self) -> None:
        self.assertEqual(THAI_CHARACTER_NAMES["A23"], "ท")
        self.assertEqual(character_glyph("A44", "thai"), "ฮ")
        self.assertEqual(LAO_CHARACTER_NAMES["J"], "ຕ")
        self.assertEqual(LAO_CHARACTER_NAMES["Y"], "ຫ")
        self.assertEqual(character_glyph("AA", "lao"), "ຮ")

    def test_ocr_document_province_names_are_used(self) -> None:
        self.assertEqual(LAO_PROVINCE_NAMES["VTE"], "ກຳແພງນະຄອນ")
        self.assertEqual(LAO_PROVINCE_NAMES["XEK"], "ເຊກອງ")

    def test_lao_ocr_transcription_extracts_province_code(self) -> None:
        self.assertEqual(
            extract_ocr_province("VTE2 ຣຄ7775", "lao"),
            ("VTE2", "ນະຄອນຫຼວງວຽງຈັນ"),
        )

    def test_thai_full_ocr_splits_province_letters_and_digits(self) -> None:
        self.assertEqual(split_plate_text("ระยอง ผค-5939", "thai"), ("ผค", "5939"))
        self.assertEqual(split_plate_text("กรุงเทพมหานคร กข-1234", "thai"), ("กข", "1234"))
        self.assertEqual(split_plate_text("68-3197", "thai"), ("68", "3197"))
        self.assertEqual(
            extract_ocr_province("ระยอง ผค-5939", "thai"),
            ("RYG", "ระยอง"),
        )
        self.assertEqual(
            extract_ocr_province("กทม กข-1234", "thai"),
            ("BKK", "กรุงเทพมหานคร"),
        )
        self.assertEqual(
            extract_ocr_province("ກຳແພງນະຄອນ ບກ0507", "lao"),
            ("VTE", "ກຳແພງນະຄອນ"),
        )
        self.assertEqual(
            extract_ocr_province("ນະຄອນຫຼວງ ກຈ6208", "lao"),
            ("VTE2", "ນະຄອນຫຼວງວຽງຈັນ"),
        )
        self.assertEqual(split_plate_text("ກຳແພງນະຄອນ ບກ0507", "lao"), ("ບກ", "0507"))
        self.assertEqual(split_plate_text("กรุงเทพมหานคร 701-1424", "thai"), ("701", "1424"))
        self.assertEqual(
            extract_ocr_province("กรุงเทพมหานคร 701-1424", "thai"),
            ("BKK", "กรุงเทพมหานคร"),
        )
        self.assertEqual(split_plate_text("ສະຫວັນນະເຂດ ວມ 7424", "lao"), ("ວມ", "7424"))
        self.assertEqual(
            extract_ocr_province("ສະຫວັນນະເຂດ ວມ 7424", "lao"),
            ("SVK", "ສະຫວັນນະເຂດ"),
        )
        self.assertEqual(country_hint_from_text("THAILAND 01 701-1424 กรุงเทพมหานคร"), "thai")
        self.assertEqual(country_hint_from_text("ສະຫວັນນະເຂດ ວມ 7424"), "lao")

    def test_thai_truck_plate_keeps_seven_digits_and_bangkok(self) -> None:
        detections = [
            box("7", 0.93, 8, 22),
            box("0", 0.92, 22, 22),
            box("1", 0.91, 36, 22),
            box("1", 0.90, 54, 22),
            box("4", 0.90, 68, 22),
            box("2", 0.89, 82, 22),
            box("4", 0.88, 96, 22),
            box("BKK", 0.86, 40, 58, 50, 12),
        ]
        reading = analyse_country_characters(detections, "thai")
        self.assertEqual(reading["plate_prefix"], "701")
        self.assertEqual(reading["plate_number"], "1424")
        self.assertEqual(reading["text"], "701-1424")
        self.assertEqual(reading["province_code"], "BKK")
        self.assertEqual(reading["province"], "กรุงเทพมหานคร")
        self.assertTrue(reading["complete"])

        chosen = apply_country_layout_priority(
            {"country": "lao", "raw_country": "lao", "confidence": 0.62, "margin": 0.04},
            {
                "thai": reading,
                "lao": {
                    "plate_prefix": "ວມ",
                    "plate_number": "7424",
                    "digit_detection_count": 4,
                    "score": 0.80,
                    "complete": True,
                },
            },
        )
        self.assertEqual(chosen["country"], "thai")
        self.assertEqual(chosen["selection_reason"], "thai_layout_priority")

    def test_thai_passenger_plate_uses_letter_prefix(self) -> None:
        detections = [
            box("A01", 0.91, 8, 22),
            box("A02", 0.90, 22, 22),
            box("0", 0.92, 48, 22),
            box("9", 0.91, 62, 22),
            box("1", 0.90, 76, 22),
            box("8", 0.89, 90, 22),
            box("BKK", 0.86, 40, 58, 50, 12),
        ]
        reading = analyse_country_characters(detections, "thai")
        self.assertEqual(reading["plate_prefix"], "กข")
        self.assertEqual(reading["plate_number"], "0918")
        self.assertEqual(reading["text"], "กข-0918")
        self.assertEqual(reading["province_code"], "BKK")
        self.assertEqual(reading["province"], "กรุงเทพมหานคร")
        self.assertTrue(reading["complete"])
        self.assertGreater(
            country_reading_score(reading, "thai"),
            country_reading_score(
                analyse_country_characters(
                    [box(value, 0.9, index * 14, 22) for index, value in enumerate("680918")],
                    "thai",
                ),
                "thai",
            ),
        )

        chosen = apply_country_layout_priority(
            {"country": "lao", "raw_country": "lao", "confidence": 0.62, "margin": 0.04},
            {
                "thai": reading,
                "lao": {
                    "plate_prefix": "",
                    "plate_number": "0918",
                    "digit_detection_count": 4,
                    "score": 0.70,
                    "complete": False,
                },
            },
        )
        self.assertEqual(chosen["country"], "thai")
        self.assertEqual(chosen["selection_reason"], "thai_layout_priority")

    def test_numbered_thai_series_stays_thai_against_lao_lookalike(self) -> None:
        detections = [
            box("4", 0.90, 4, 22),
            box("A01", 0.93, 18, 22),
            box("A24", 0.92, 32, 22),
            box("8", 0.94, 52, 22),
            box("3", 0.93, 66, 22),
            box("8", 0.92, 80, 22),
            box("6", 0.91, 94, 22),
            box("NPT", 0.84, 40, 58, 50, 12),
        ]
        reading = analyse_country_characters(detections, "thai")
        self.assertEqual(reading["plate_prefix"], "4กธ")
        self.assertEqual(reading["plate_number"], "8386")
        self.assertEqual(reading["province_code"], "NPT")
        self.assertTrue(reading["complete"])

        chosen = apply_country_layout_priority(
            {"country": "lao", "raw_country": "lao", "confidence": 0.70, "margin": 0.10},
            {
                "thai": reading,
                "lao": {
                    "plate_prefix": "ກທ",
                    "plate_number": "8386",
                    "digit_detection_count": 4,
                    "digit_confidence": 0.90,
                    "prefix_confidence": 0.95,
                    "province_code": "VTE2",
                    "province_confidence": 0.80,
                    "complete": True,
                    "score": 1.20,
                },
            },
        )
        self.assertEqual(chosen["country"], "thai")
        self.assertEqual(chosen["selection_reason"], "thai_layout_priority")

    def test_thai_letter_plate_without_province_is_not_lao(self) -> None:
        thai = {
            "plate_prefix": "กท",
            "plate_number": "8830",
            "digit_detection_count": 4,
            "expected_digit_count": 4,
            "digit_confidence": 0.91,
            "prefix_confidence": 0.90,
            "province_code": "",
            "province": "",
            "complete": True,
            "score": 1.10,
            "tokens": [],
        }
        lao = {
            "plate_prefix": "ກທ",
            "plate_number": "8830",
            "digit_detection_count": 4,
            "expected_digit_count": 4,
            "digit_confidence": 0.88,
            "prefix_confidence": 0.86,
            "province_code": "",
            "province": "",
            "complete": True,
            "score": 1.00,
            "tokens": [],
        }

        chosen = apply_country_layout_priority(
            {"country": "lao", "raw_country": "lao", "confidence": 0.60, "margin": 0.05},
            {"thai": thai, "lao": lao},
        )

        self.assertEqual(chosen["country"], "thai")
        self.assertEqual(chosen["selection_reason"], "thai_layout_priority")

    def test_thai_motorcycle_uses_province_line_not_lao_digits(self) -> None:
        detections = [
            box("A01", 0.90, 8, 4, 12, 12),
            box("A23", 0.89, 22, 4, 12, 12),
            box("A33", 0.88, 36, 4, 12, 12),
            box("2", 0.93, 10, 26),
            box("5", 0.92, 24, 26),
            box("8", 0.91, 38, 26),
            box("8", 0.90, 52, 26),
        ]
        reading = analyse_country_characters(detections, "thai")
        self.assertEqual(reading["plate_prefix"], "")
        self.assertEqual(reading["plate_number"], "2588")
        self.assertEqual(reading["province_code"], "BKK")
        self.assertEqual(reading["province"], "กรุงเทพมหานคร")
        self.assertTrue(reading["complete"])

        lao = analyse_country_characters(
            [box(value, 0.9, 10 + index * 14, 26) for index, value in enumerate("2588")],
            "lao",
        )
        chosen = apply_country_layout_priority(
            {"country": "lao", "raw_country": "lao", "confidence": 0.70, "margin": 0.20},
            {
                "thai": {**reading, "score": country_reading_score(reading, "thai")},
                "lao": {**lao, "score": country_reading_score(lao, "lao")},
            },
        )
        self.assertEqual(chosen["country"], "thai")

    def test_thai_motorcycle_province_code_keeps_four_digits(self) -> None:
        detections = [
            box("NPT", 0.88, 20, 4, 50, 12),
            box("0", 0.92, 10, 26),
            box("0", 0.91, 24, 26),
            box("5", 0.90, 38, 26),
            box("9", 0.89, 52, 26),
        ]
        reading = analyse_country_characters(detections, "thai")
        self.assertEqual(reading["plate_prefix"], "")
        self.assertEqual(reading["plate_number"], "0059")
        self.assertEqual(reading["province_code"], "NPT")
        self.assertTrue(reading["complete"])

    def test_lao_savannakhet_plate_keeps_prefix_and_province(self) -> None:
        detections = [
            box("SVK", 0.88, 40, 4, 50, 12),
            box("X", 0.91, 10, 26),
            box("T", 0.90, 24, 26),
            box("7", 0.92, 48, 26),
            box("4", 0.91, 62, 26),
            box("2", 0.90, 76, 26),
            box("4", 0.89, 90, 26),
        ]
        reading = analyse_country_characters(detections, "lao")
        self.assertEqual(reading["plate_prefix"], "ວມ")
        self.assertEqual(reading["plate_number"], "7424")
        self.assertEqual(reading["province_code"], "SVK")
        self.assertEqual(reading["province"], "ສະຫວັນນະເຂດ")
        self.assertTrue(reading["complete"])

    def test_complete_lao_beats_incomplete_thai_with_false_province(self) -> None:
        thai = {
            "plate_prefix": "บ",
            "plate_number": "9899",
            "digit_detection_count": 4,
            "expected_digit_count": 4,
            "digit_confidence": 0.90,
            "prefix_confidence": 0.90,
            "province_code": "KPT",
            "province": "กำแพงเพชร",
            "province_confidence": 0.66,
            "complete": False,
            "score": 0.46,
            "tokens": [],
        }
        lao = {
            "plate_prefix": "ບຈ",
            "plate_number": "9899",
            "digit_detection_count": 4,
            "expected_digit_count": 4,
            "digit_confidence": 0.91,
            "prefix_confidence": 0.92,
            "province_code": "VTE2",
            "province": "ນະຄອນຫຼວງວຽງຈັນ",
            "province_confidence": 0.85,
            "complete": True,
            "score": 1.30,
            "tokens": [],
        }

        chosen = apply_country_layout_priority(
            {
                "country": "thai",
                "raw_country": "thai",
                "confidence": 0.55,
                "margin": -0.10,
            },
            {"thai": thai, "lao": lao},
        )

        self.assertEqual(chosen["country"], "lao")
        self.assertEqual(chosen["selection_reason"], "lao_layout_priority")

    def test_complete_lao_beats_thai_letter_lookalike_without_province(self) -> None:
        thai = {
            "plate_prefix": "บจ",
            "plate_number": "9899",
            "digit_detection_count": 4,
            "expected_digit_count": 4,
            "digit_confidence": 0.90,
            "prefix_confidence": 0.90,
            "province_code": "",
            "province": "",
            "complete": True,
            "score": 1.20,
            "tokens": [],
        }
        lao = {
            "plate_prefix": "ບຈ",
            "plate_number": "9899",
            "digit_detection_count": 4,
            "expected_digit_count": 4,
            "digit_confidence": 0.91,
            "prefix_confidence": 0.92,
            "province_code": "VTE2",
            "province": "ນະຄອນຫຼວງວຽງຈັນ",
            "province_confidence": 0.85,
            "complete": True,
            "score": 1.30,
            "tokens": [],
        }

        chosen = apply_country_layout_priority(
            {
                "country": "thai",
                "raw_country": "thai",
                "confidence": 0.55,
                "margin": -0.10,
            },
            {"thai": thai, "lao": lao},
        )

        self.assertEqual(chosen["country"], "lao")
        self.assertEqual(chosen["selection_reason"], "lao_script_priority")

    def test_lao_plate_beats_thai_letter_lookalike_with_false_bangkok(self) -> None:
        thai = {
            "plate_prefix": "บข",
            "plate_number": "8634",
            "digit_detection_count": 4,
            "expected_digit_count": 4,
            "digit_confidence": 0.91,
            "prefix_confidence": 0.88,
            "province_code": "BKK",
            "province": "กรุงเทพมหานคร",
            "province_confidence": 0.74,
            "registration_confidence": 0.915,
            "complete": True,
            "score": 1.45,
            "tokens": [],
        }
        lao = {
            "plate_prefix": "ບຂ",
            "plate_number": "8634",
            "digit_detection_count": 4,
            "expected_digit_count": 4,
            "digit_confidence": 0.90,
            "prefix_confidence": 0.91,
            "province_code": "VTE2",
            "province": "ນະຄອນຫຼວງວຽງຈັນ",
            "province_confidence": 0.86,
            "registration_confidence": 0.90,
            "complete": True,
            "score": 1.40,
            "tokens": [],
        }
        readings = {"thai": thai, "lao": lao}

        chosen = apply_country_layout_priority(
            {
                "country": "thai",
                "raw_country": "thai",
                "confidence": 0.70,
                "margin": 0.05,
            },
            readings,
        )

        self.assertEqual(chosen["country"], "lao")
        self.assertEqual(chosen["selection_reason"], "lao_script_priority")
        self.assertEqual(
            contested_script_confidence(0.915, readings, "thai"),
            0.62,
        )
        self.assertEqual(
            contested_script_confidence(0.90, readings, "lao"),
            0.90,
        )

    def test_lao_province_does_not_override_complete_thai_truck(self) -> None:
        thai = {
            "plate_prefix": "70",
            "plate_number": "7779",
            "digit_detection_count": 6,
            "expected_digit_count": 6,
            "digit_confidence": 0.91,
            "prefix_confidence": 0.0,
            "province_code": "",
            "province": "",
            "complete": True,
            "score": 0.93,
            "tokens": [],
        }
        lao = {
            "plate_prefix": "",
            "plate_number": "079",
            "digit_detection_count": 3,
            "expected_digit_count": 4,
            "digit_confidence": 0.67,
            "prefix_confidence": 0.0,
            "province_code": "CPS",
            "province": "ຈຳປາສັກ",
            "province_confidence": 0.55,
            "complete": False,
            "score": 0.44,
            "tokens": [],
        }

        chosen = apply_country_layout_priority(
            {
                "country": "lao",
                "raw_country": "lao",
                "confidence": 0.50,
                "margin": -0.49,
            },
            {"thai": thai, "lao": lao},
        )

        self.assertEqual(chosen["country"], "thai")
        self.assertEqual(chosen["selection_reason"], "thai_layout_priority")

    def test_thai_letter_plate_quality_has_complete_prefix(self) -> None:
        thai = analyse_country_characters(
            [
                box("A26", 0.90, 10, 25),
                box("A23", 0.88, 24, 25),
                box("1", 0.91, 48, 25),
                box("5", 0.90, 62, 25),
                box("8", 0.89, 76, 25),
                box("7", 0.88, 90, 25),
                box("BKK", 0.80, 35, 4, 45, 12),
            ],
            "thai",
        )
        quality = build_plate_quality(
            country="thai",
            primary_country="thai",
            readings={"thai": thai, "lao": {}},
            plate_prefix="บท",
            plate_number="1587",
            ocr={"confidence": 0.88},
            validation={"valid": True},
            detection_confidence=0.80,
            country_confidence=0.80,
            digit_evidence={"status": "assistant_unavailable"},
        )

        self.assertTrue(quality["character_validation"]["prefix_complete"])
        self.assertEqual(
            quality["character_validation"]["primary_expected_digit_count"], 4
        )
        self.assertNotIn("prefix_missing", quality["qc"]["reason"])

    def test_thai_three_digit_truck_prefix_is_complete(self) -> None:
        thai = {
            "plate_prefix": "170",
            "plate_number": "3231",
            "digit_detection_count": 7,
            "expected_digit_count": 7,
            "digit_confidence": 0.90,
            "character_confidence": 0.90,
            "province": "สกลนคร",
            "province_code": "SKN",
            "province_confidence": 0.80,
            "complete": True,
            "tokens": [],
        }
        quality = build_plate_quality(
            country="thai",
            primary_country="thai",
            readings={"thai": thai, "lao": {}},
            plate_prefix="170",
            plate_number="3231",
            ocr={"confidence": 0.90},
            validation={"valid": True},
            detection_confidence=0.85,
            country_confidence=0.85,
            digit_evidence={"status": "assistant_unavailable"},
        )

        self.assertTrue(quality["character_validation"]["prefix_complete"])
        self.assertEqual(
            quality["character_validation"]["primary_expected_digit_count"], 7
        )
        self.assertNotIn("prefix_missing", quality["qc"]["reason"])

    def test_digit_prefix_confidence_uses_its_own_digits(self) -> None:
        reading = analyse_country_characters(
            [
                box("7", 0.91, 10, 25),
                box("0", 0.90, 24, 25),
                box("4", 0.89, 48, 25),
                box("1", 0.88, 62, 25),
                box("2", 0.87, 76, 25),
                box("2", 0.86, 90, 25),
            ],
            "thai",
        )

        self.assertEqual(reading["plate_prefix"], "70")
        self.assertEqual(reading["plate_number"], "4122")
        self.assertAlmostEqual(reading["prefix_confidence"], 0.905, places=3)
        self.assertGreater(reading["registration_confidence"], 0.85)
        self.assertEqual(
            published_plate_confidence(reading),
            reading["registration_confidence"],
        )

    def test_weak_glyph_cannot_publish_a_high_confidence(self) -> None:
        reading = analyse_country_characters(
            [
                box("7", 0.41, 10, 25),
                box("3", 0.90, 24, 25),
                box("1", 0.89, 48, 25),
                box("3", 0.88, 62, 25),
                box("7", 0.87, 76, 25),
                box("0", 0.86, 90, 25),
                box("BKK", 0.80, 35, 4, 45, 12),
            ],
            "thai",
        )
        confidence = published_plate_confidence(reading)

        self.assertLess(reading["weakest_glyph_confidence"], 0.50)
        self.assertLess(confidence, 0.80)

    def test_ocr_province_fills_blank_yolo_province(self) -> None:
        class Engine:
            def recognise(self, *args: object, **kwargs: object) -> dict:
                return {
                    "status": "ok",
                    "text": "ระยอง ผค-5939",
                    "confidence": 0.22,
                    "method": "custom-paddleocr-thai-full",
                }

        scanner = object.__new__(LicensePlateScanner)
        scanner.ocr = Engine()
        reading = {
            "text": "68-5939",
            "confidence": 0.80,
            "plate_prefix": "68",
            "plate_number": "5939",
        }

        ocr, prefix, number = scanner.finalise_reading(
            np.zeros((12, 40, 3), dtype=np.uint8),
            "thai",
            reading,
        )

        self.assertEqual((prefix, number), ("ผค", "5939"))
        self.assertEqual(reading["province"], "ระยอง")
        self.assertEqual(reading["province_code"], "RYG")
        self.assertEqual(ocr["province"], "ระยอง")
        self.assertEqual(ocr["raw_text"], "ระยอง ผค-5939")

    def test_trained_thai_ocr_overrides_digit_only_yolo_prefix(self) -> None:
        reading = {
            "plate_prefix": "68",
            "plate_number": "3197",
            "prefix_confidence": 0.91,
            "digit_confidence": 0.90,
        }
        ocr = {"text": "กรุงเทพมหานคร กข-3197", "confidence": 0.96, "method": "custom-paddleocr-thai-full"}

        prefix, number, source = resolve_plate_fields("thai", reading, ocr)

        self.assertEqual((prefix, number), ("กข", "3197"))
        self.assertEqual(source, "ocr+yolo-agree")

    def test_lower_confidence_ocr_does_not_replace_yolo_prefix(self) -> None:
        reading = {
            "plate_prefix": "68",
            "plate_number": "3197",
            "prefix_confidence": 0.91,
            "digit_confidence": 0.90,
        }
        ocr = {
            "text": "กรุงเทพมหานคร กข-3197",
            "confidence": 0.81,
            "method": "custom-paddleocr-thai-full",
        }

        prefix, number, source = resolve_plate_fields("thai", reading, ocr)

        self.assertEqual((prefix, number), ("68", "3197"))
        self.assertIn(source, {"structured-yolo", "yolo+ocr-agree"})

    def test_thai_model_codes_do_not_contaminate_registration_digits(self) -> None:
        detections = [
            box("A23", 0.82, 35, 4, 12, 10),
            box("6", 0.91, 10, 25),
            box("8", 0.90, 24, 25),
            box("3", 0.88, 45, 25),
            box("1", 0.87, 59, 25),
            box("9", 0.86, 73, 25),
            box("7", 0.85, 87, 25),
            box("BKK", 0.81, 30, 58, 50, 12),
        ]

        reading = analyse_country_characters(detections, "thai")

        self.assertEqual(reading["plate_prefix"], "68")
        self.assertEqual(reading["plate_number"], "3197")
        self.assertEqual(reading["text"], "68-3197")
        self.assertEqual(reading["province_code"], "BKK")
        self.assertTrue(reading["complete"])

    def test_lao_aliases_are_converted_and_province_is_separate(self) -> None:
        detections = [
            box("LNT", 0.91, 40, 3, 45, 14),
            box("N", 0.92, 8, 28),
            box("A", 0.90, 22, 28),
            box("2", 0.91, 45, 28),
            box("2", 0.92, 59, 28),
            box("7", 0.90, 73, 28),
            box("1", 0.89, 87, 28),
        ]

        reading = analyse_country_characters(detections, "lao")

        self.assertEqual(reading["plate_prefix"], "ບກ")
        self.assertEqual(reading["plate_number"], "2271")
        self.assertEqual(reading["text"], "ບກ 2271")
        self.assertEqual(reading["province_code"], "LNT")
        self.assertEqual(reading["plate_prefix_code"], "N_A")
        self.assertEqual(reading["detected_characters"][0]["text"], "ບ")
        self.assertTrue(reading["complete"])

    def test_lao_province_on_number_line_is_rejected(self) -> None:
        detections = [
            box("LNT", 0.95, 40, 28, 45, 14),
            box("N", 0.92, 8, 28),
            box("A", 0.90, 22, 28),
            box("2", 0.91, 45, 28),
            box("2", 0.92, 59, 28),
            box("7", 0.90, 73, 28),
            box("1", 0.89, 87, 28),
        ]

        reading = analyse_country_characters(detections, "lao")

        self.assertEqual(reading["province"], "")
        self.assertEqual(reading["province_code"], "")
        self.assertEqual(reading["province_status"], "not_top_line")

    def test_low_confidence_lao_province_is_not_displayed(self) -> None:
        detections = [
            box("LNT", 0.40, 40, 3, 45, 14),
            box("N", 0.92, 8, 28),
            box("A", 0.90, 22, 28),
            box("2", 0.91, 45, 28),
            box("2", 0.92, 59, 28),
            box("7", 0.90, 73, 28),
            box("1", 0.89, 87, 28),
        ]

        reading = analyse_country_characters(detections, "lao")

        self.assertEqual(reading["province"], "")
        self.assertEqual(reading["province_status"], "below_confidence")

    def test_country_score_penalises_wrong_digit_shape(self) -> None:
        thai = analyse_country_characters(
            [box(value, 0.9, index * 14, 20) for index, value in enumerate("683197")],
            "thai",
        )
        lao = analyse_country_characters(
            [box(value, 0.9, index * 14, 20) for index, value in enumerate("683197")],
            "lao",
        )

        self.assertGreater(country_reading_score(thai, "thai"), country_reading_score(lao, "lao"))

    def test_strong_structured_lao_prefix_beats_disagreeing_tesseract(self) -> None:
        reading = {
            "plate_prefix": "ບກ",
            "plate_number": "2271",
            "prefix_confidence": 0.91,
            "digit_confidence": 0.90,
        }
        ocr = {"text": "ສປ2271", "confidence": 0.83}

        prefix, number, source = resolve_plate_fields("lao", reading, ocr)

        self.assertEqual((prefix, number), ("ບກ", "2271"))
        self.assertEqual(source, "yolo+ocr-agree")

    def test_yolo_digits_stay_when_ocr_confidence_is_lower(self) -> None:
        reading = {
            "plate_prefix": "69",
            "plate_number": "2679",
            "prefix_confidence": 0.90,
            "digit_confidence": 0.88,
            "tokens": [
                {"label": "6", "kind": "digit", "confidence": 0.91, "box": [10, 0, 20, 20]},
                {"label": "9", "kind": "digit", "confidence": 0.90, "box": [22, 0, 32, 20]},
                {"label": "2", "kind": "digit", "confidence": 0.88, "box": [40, 0, 50, 20]},
                {"label": "6", "kind": "digit", "confidence": 0.89, "box": [52, 0, 62, 20]},
                {"label": "7", "kind": "digit", "confidence": 0.87, "box": [64, 0, 74, 20]},
                {"label": "9", "kind": "digit", "confidence": 0.86, "box": [76, 0, 86, 20]},
            ],
        }
        ocr = {"text": "8787", "confidence": 0.495, "method": "custom-paddleocr"}

        prefix, number, source = resolve_plate_fields("thai", reading, ocr)

        self.assertEqual((prefix, number), ("69", "2679"))
        self.assertEqual(source, "structured-yolo")
        self.assertGreater(ocr["confidence"], 0.85)

    def test_higher_ocr_confidence_replaces_one_weak_yolo_digit(self) -> None:
        reading = {
            "plate_prefix": "70",
            "plate_number": "4293",
            "digit_confidence": 0.70,
            "tokens": [
                {"label": "7", "kind": "digit", "confidence": 0.91, "box": [10, 0, 20, 20]},
                {"label": "0", "kind": "digit", "confidence": 0.90, "box": [22, 0, 32, 20]},
                {"label": "4", "kind": "digit", "confidence": 0.88, "box": [40, 0, 50, 20]},
                {"label": "2", "kind": "digit", "confidence": 0.22, "box": [52, 0, 62, 20]},
                {"label": "9", "kind": "digit", "confidence": 0.87, "box": [64, 0, 74, 20]},
                {"label": "3", "kind": "digit", "confidence": 0.86, "box": [76, 0, 86, 20]},
            ],
        }
        ocr = {
            "text": "70-4393",
            "confidence": 0.40,
            "character_confidences": [0.10, 0.10, 0.10, 0.20, 0.95, 0.20, 0.20],
            "confidence_mode": "per-character-ctc",
        }

        prefix, number, source = resolve_plate_fields("thai", reading, ocr)

        self.assertEqual((prefix, number), ("70", "4393"))
        self.assertEqual(source, "ocr+yolo-agree")
        replaced = next(
            item
            for item in ocr["character_fusion"]
            if item["field"] == "number" and item["source"] == "ocr"
        )
        self.assertEqual(replaced["char"], "3")
        self.assertEqual(replaced["confidence"], 0.95)

    def test_other_country_model_fills_missing_lao_digits_only(self) -> None:
        readings = {
            "lao": {
                "plate_prefix": "",
                "plate_number": "079",
                "province": "ຈຳປາສັກ",
                "province_code": "CPS",
                "digit_confidence": 0.67,
                "tokens": [
                    {
                        "label": "0",
                        "kind": "digit",
                        "confidence": 0.48,
                        "box": [44, 40, 64, 80],
                    },
                    {
                        "label": "7",
                        "kind": "digit",
                        "confidence": 0.66,
                        "box": [131, 43, 153, 82],
                    },
                    {
                        "label": "9",
                        "kind": "digit",
                        "confidence": 0.86,
                        "box": [154, 44, 175, 83],
                    },
                ],
            },
            "thai": {
                "plate_prefix": "70",
                "plate_number": "7779",
                "digit_confidence": 0.91,
                "tokens": [
                    {
                        "label": char,
                        "kind": "digit",
                        "confidence": confidence,
                        "box": [index * 20, 40, index * 20 + 15, 80],
                    }
                    for index, (char, confidence) in enumerate(
                        zip("707779", (0.90, 0.93, 0.91, 0.91, 0.90, 0.89))
                    )
                ],
            },
        }

        fusion = fuse_cross_model_digits(readings, "lao")

        self.assertEqual(readings["lao"]["plate_number"], "7779")
        self.assertEqual(
            fusion["assistant_model"], "thai_license_plate"
        )
        self.assertTrue(
            all(item["source"] == "thai" for item in fusion["characters"])
        )

    def test_shifted_lao_digit_does_not_replace_thai_one(self) -> None:
        readings = {
            "thai": {
                "plate_prefix": "72",
                "plate_number": "6614",
                "digit_confidence": 0.8481,
                "prefix_confidence": 0.8858,
                "province_code": "SRI",
                "tokens": [
                    {"label": "7", "kind": "digit", "confidence": 0.8838, "box": [15.69, 45.53, 27.99, 69.69]},
                    {"label": "2", "kind": "digit", "confidence": 0.8947, "box": [29.85, 46.33, 42.06, 70.21]},
                    {"label": "6", "kind": "digit", "confidence": 0.8745, "box": [53.11, 46.52, 65.22, 69.76]},
                    {"label": "6", "kind": "digit", "confidence": 0.8634, "box": [66.97, 46.83, 79.16, 69.87]},
                    {"label": "1", "kind": "digit", "confidence": 0.7767, "box": [84.58, 47.64, 88.35, 70.33]},
                    {"label": "4", "kind": "digit", "confidence": 0.8508, "box": [94.63, 47.67, 106.87, 71.32]},
                ],
            },
            "lao": {
                "plate_prefix": "",
                "plate_number": "2664",
                "digit_confidence": 0.8616,
                "tokens": [
                    {"label": "7", "kind": "digit", "confidence": 0.3426, "box": [11.89, 15.05, 23.72, 29.48]},
                    {"label": "6", "kind": "digit", "confidence": 0.3487, "box": [48.63, 15.73, 57.39, 30.58]},
                    {"label": "3", "kind": "digit", "confidence": 0.8389, "box": [81.02, 16.65, 91.01, 31.45]},
                    {"label": "7", "kind": "digit", "confidence": 0.5445, "box": [14.58, 45.9, 28.34, 83.36]},
                    {"label": "2", "kind": "digit", "confidence": 0.7953, "box": [29.66, 46.19, 43.54, 74.71]},
                    {"label": "6", "kind": "digit", "confidence": 0.8717, "box": [53.29, 46.94, 66.29, 71.15]},
                    {"label": "6", "kind": "digit", "confidence": 0.8896, "box": [67.23, 47.38, 80.45, 71.49]},
                    {"label": "4", "kind": "digit", "confidence": 0.8898, "box": [94.68, 48.85, 108.33, 72.54]},
                ],
            },
        }

        fusion = fuse_cross_model_digits(readings, "thai")

        self.assertEqual(readings["thai"]["plate_number"], "6614")
        replaced = fusion["characters"][2]
        self.assertEqual(replaced["char"], "1")
        self.assertEqual(replaced["source"], "thai")
        self.assertEqual(replaced["confidence"], 0.7767)
        self.assertLess(readings["thai"]["digit_confidence"], 0.87)


if __name__ == "__main__":
    unittest.main()
