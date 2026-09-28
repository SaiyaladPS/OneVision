from __future__ import annotations

import unittest
from pathlib import Path

from scan import CustomPaddleOCR
from src.car_scan.config import Settings
from src.car_scan.service import ScanService
from src.car_scan.worker import GpuHub


class ThaiOcrRuntimeTests(unittest.TestCase):
    def test_settings_select_latest_thai_full_checkpoint(self) -> None:
        settings = Settings.from_env()
        weights = settings.ocr_thai_weights
        if weights is None or not weights.with_suffix(".pdparams").is_file():
            self.skipTest("no trained thai-full OCR checkpoint under runs/ocr")
        self.assertIn("thai-full", str(weights).replace("\\", "/"))
        self.assertTrue(settings.ocr_thai_config.is_file())

    def test_web_scan_ocr_engine_uses_trained_thai_full_weights(self) -> None:
        settings = Settings.from_env()
        weights = settings.ocr_thai_weights
        if weights is None or not weights.with_suffix(".pdparams").is_file():
            self.skipTest("no trained thai-full OCR checkpoint under runs/ocr")

        ocr = CustomPaddleOCR(
            settings.ocr_source,
            settings.ocr_config,
            settings.ocr_weights,
            settings.torch_ocr_model,
            lao_config=settings.ocr_lao_config,
            lao_weights=settings.ocr_lao_weights,
            thai_config=settings.ocr_thai_config,
            thai_weights=settings.ocr_thai_weights,
        )

        self.assertTrue(ocr.thai_available)
        self.assertEqual(Path(ocr.thai_weights), Path(weights))
        self.assertTrue(ocr.thai_config.is_file())

        status = ScanService(settings).ocr_engine_status()
        self.assertTrue(status["thai_full"])
        self.assertIn("tesseract_thai", status)
        self.assertIn("thai-full", status["thai_weights"].replace("\\", "/"))

        hub = GpuHub(settings)
        gpu = hub.status()
        self.assertTrue(gpu["ocr"]["thai_full"])
        self.assertEqual(Path(gpu["ocr"]["thai_weights"]), Path(weights))

    def test_clean_ocr_line_keeps_thai_letters(self) -> None:
        from scan import clean_ocr_line

        self.assertEqual(clean_ocr_line("กก-1234!"), "กก1234")
        self.assertEqual(clean_ocr_line("กรุงเทพมหานคร 60-0514"), "กรุงเทพมหานคร600514")

    def test_thai_tesseract_language_pack_is_configured(self) -> None:
        from scan import TESSERACT_LANGUAGES, _tesseract_model_dir

        spec = TESSERACT_LANGUAGES["thai"]
        self.assertEqual(spec["code"], "tha")
        self.assertEqual(spec["method"], "tesseract-thai")
        self.assertEqual(spec["model"], "tha.traineddata")
        model_dir = _tesseract_model_dir("tha.traineddata")
        if model_dir is None:
            self.skipTest("tha.traineddata is not installed")
        self.assertTrue((model_dir / "tha.traineddata").is_file())


if __name__ == "__main__":
    unittest.main()
