from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.car_scan import report_storage


class ReportStorageTests(unittest.TestCase):
    def test_enqueue_artifacts_accepts_endpoint_project_token_config(self) -> None:
        with tempfile.TemporaryDirectory(prefix="report_storage_test_") as directory:
            root = Path(directory)
            artifact = root / "20261006" / "json" / "scan.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("{}", encoding="utf-8")

            with (
                patch.object(
                    report_storage,
                    "_config",
                    return_value=("http://storage.test", "onevision", "test-token"),
                ),
                patch.object(report_storage, "_WORKER_STARTED", True),
                patch.object(report_storage, "_UPLOADS") as uploads,
            ):
                report_storage.enqueue_artifacts(root, artifact)

            uploads.put_nowait.assert_called_once_with((root, [artifact]))

    def test_relative_files_accepts_camera_country_archive_layout(self) -> None:
        with tempfile.TemporaryDirectory(prefix="report_storage_test_") as directory:
            root = Path(directory)
            image = root / "LED01" / "2026" / "10" / "07" / "thai" / "0001-LED01-BKK-69-1690-20261007_plate_crops.webp"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"image")

            self.assertEqual(
                report_storage._relative_files(root, [image]),
                [(image.resolve(), "LED01/2026/10/07/thai/0001-LED01-BKK-69-1690-20261007_plate_crops.webp")],
            )

    def test_relative_files_rejects_unexpected_camera_archive_layout(self) -> None:
        with tempfile.TemporaryDirectory(prefix="report_storage_test_") as directory:
            root = Path(directory)
            image = root / "LED01" / "2026" / "10" / "07" / "other" / "plate.webp"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"image")

            self.assertEqual(report_storage._relative_files(root, [image]), [])


if __name__ == "__main__":
    unittest.main()
