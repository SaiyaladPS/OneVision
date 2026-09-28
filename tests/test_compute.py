from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.car_scan.compute import (
    load_saved_compute_mode,
    normalise_compute_mode,
    resolve_compute,
    save_compute_mode,
    yolo_predict,
)


class ComputePlanTests(unittest.TestCase):
    def test_aliases_map_to_hybrid(self) -> None:
        self.assertEqual(normalise_compute_mode("GPU+CPU"), "hybrid")
        self.assertEqual(normalise_compute_mode("both"), "hybrid")
        self.assertEqual(normalise_compute_mode("cuda"), "gpu")
        self.assertEqual(normalise_compute_mode("weird"), "auto")

    def test_cpu_mode_pins_both_devices(self) -> None:
        plan = resolve_compute("cpu", has_cuda=True)
        self.assertEqual(plan.mode, "cpu")
        self.assertEqual(plan.yolo_device, "cpu")
        self.assertEqual(plan.ocr_device, "cpu")
        self.assertFalse(plan.hybrid_cpu_yolo)

    def test_gpu_mode_uses_cuda_when_available(self) -> None:
        plan = resolve_compute("gpu", has_cuda=True)
        self.assertEqual(plan.yolo_device, "cuda:0")
        self.assertEqual(plan.ocr_device, "gpu:0")
        self.assertFalse(plan.fallback)

    def test_gpu_mode_falls_back_without_cuda(self) -> None:
        plan = resolve_compute("gpu", has_cuda=False)
        self.assertEqual(plan.mode, "cpu")
        self.assertTrue(plan.fallback)
        self.assertEqual(plan.yolo_device, "cpu")

    def test_auto_keeps_ocr_on_cpu_when_cuda_exists(self) -> None:
        plan = resolve_compute("auto", has_cuda=True)
        self.assertEqual(plan.yolo_device, "cuda:0")
        self.assertEqual(plan.ocr_device, "cpu")
        self.assertFalse(plan.hybrid_cpu_yolo)

    def test_hybrid_splits_yolo_and_ocr_and_enables_cpu_overflow(self) -> None:
        plan = resolve_compute("hybrid", has_cuda=True)
        self.assertEqual(plan.mode, "hybrid")
        self.assertEqual(plan.yolo_device, "cuda:0")
        self.assertEqual(plan.ocr_device, "cpu")
        self.assertTrue(plan.hybrid_cpu_yolo)

    def test_saved_compute_mode_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            save_compute_mode(output, "hybrid")
            self.assertEqual(load_saved_compute_mode(output), "hybrid")

    def test_yolo_predict_ignores_device_when_model_rejects_it(self) -> None:
        class Model:
            def predict(self, source, *, conf=0.0):
                return [source, conf]

        result = yolo_predict(Model(), "frame", device="cuda:0", conf=0.2)
        self.assertEqual(result, ["frame", 0.2])


if __name__ == "__main__":
    unittest.main()
