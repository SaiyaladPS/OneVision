from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.build_training_dataset import export_cvat, export_roboflow, load_samples, selected_dates


class TrainingExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="car_scan_training_export_")
        self.root = Path(self.temporary.name)
        self.day = self.root / "20260802"
        for name in ("laos", "json", "log", "video", "thai"):
            (self.day / name).mkdir(parents=True, exist_ok=True)
        self.full = self.day / "laos" / "0-LPB-prefix-1356-20260802-001-full_vehicle.jpg"
        self.crop = self.day / "laos" / "0-LPB-prefix-1356-20260802-001-plate_crops.jpg"
        self.ready = self.day / "laos" / "0-LPB-prefix-1356-20260802-001-ocr_ready.jpg"
        image = np.zeros((40, 120, 3), dtype=np.uint8)
        cv2.imwrite(str(self.full), image)
        cv2.imwrite(str(self.crop), image)
        cv2.imwrite(str(self.ready), image)
        (self.day / "json" / "sample_result.json").write_text(
            json.dumps(
                {
                    "plates": [
                        {
                            "country": "lao",
                            "confirmed": True,
                            "plate_prefix": "ບຄ",
                            "plate_number": "1356",
                            "box": [10, 10, 80, 30],
                            "full_vehicle_image": str(self.full),
                            "crop_image": str(self.crop),
                            "ocr_ready_image": str(self.ready),
                            "country_readings": {
                                "lao": {
                                    "tokens": [
                                        {"label": "1", "box": [10, 8, 20, 30]},
                                        {"label": "3", "box": [25, 8, 35, 30]},
                                    ]
                                }
                            },
                        }
                    ],
                    "rejected_plates": [
                        {
                            "country": "lao",
                            "training_status": "REJECT",
                            "plate_prefix": "ບຄ",
                            "plate_number": "6356",
                            "box": [10, 10, 80, 30],
                            "full_vehicle_image": str(self.full),
                            "crop_image": str(self.crop),
                            "ocr_ready_image": str(self.ready),
                            "country_readings": {"lao": {"tokens": []}},
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_selected_date_exports_cvat_and_roboflow_layouts(self) -> None:
        dates = selected_dates("20260802", self.root)
        samples = load_samples(self.root, dates)
        self.assertEqual(len(samples), 2)

        cvat = self.root / "cvat" / "cvat-20260802"
        export_cvat(cvat, samples, "20260802")
        self.assertTrue((cvat / "dataset_lao_license_plate" / "PASS" / "images").is_dir())
        self.assertTrue((cvat / "dataset_lao_license_plate" / "REJECT" / "images").is_dir())
        self.assertTrue((cvat / "dataset_detect_license" / "PASS" / "labels").is_dir())

        roboflow = self.root / "roboflow" / "roboflow-20260802"
        export_roboflow(roboflow, samples, "20260802")
        self.assertTrue((roboflow / "dataset_lao_license_plate" / "PASS" / "data.yaml").is_file())
        self.assertTrue((roboflow / "dataset_ocr" / "PASS" / "rec_lao_full" / "train_list.txt").is_file())


if __name__ == "__main__":
    unittest.main()
