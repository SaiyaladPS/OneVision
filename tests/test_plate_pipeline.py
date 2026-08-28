from __future__ import annotations

import unittest

from scan import (
    CharacterDetection,
    LAO_CHARACTER_NAMES,
    LAO_PROVINCE_NAMES,
    THAI_CHARACTER_NAMES,
    analyse_country_characters,
    character_glyph,
    country_reading_score,
    extract_ocr_province,
    resolve_plate_fields,
)


def box(text: str, confidence: float, x: float, y: float, width: float = 10, height: float = 24) -> CharacterDetection:
    return CharacterDetection(text, confidence, x, y, x + width, y + height)


class StructuredPlatePipelineTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
