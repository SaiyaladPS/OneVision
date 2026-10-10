from __future__ import annotations

import tempfile
import unittest
import zipfile
import os
import json
from unittest.mock import patch
from io import BytesIO
from PIL import Image
from pathlib import Path
import yaml

from src.car_scan.training import TrainingRunManager


class TrainingRunManagerTests(unittest.TestCase):
    def make_training_root(self, root: Path) -> None:
        (root / "train.py").write_text("# launcher\n", encoding="utf-8")
        target = root / "thai_license_plate-train" / "dataset"
        target.mkdir(parents=True)
        (target / "train.py").write_text("# model trainer\n", encoding="utf-8")
        invalid = root / "not-a-model" / "dataset"
        invalid.mkdir(parents=True)
        (invalid / "train.py").write_text("# not an allow-listed folder\n", encoding="utf-8")

    @staticmethod
    def make_dataset_zip(entries: dict[str, str | bytes] | None = None) -> bytes:
        if entries is None:
            image = BytesIO()
            Image.new("RGB", (1, 1), color="white").save(image, format="PNG")
            image_bytes = image.getvalue()
            entries = {
                "data.yaml": "path: .\ntrain: train.txt\nnc: 1\nnames: [plate]\n",
                "train.txt": "data/images/train/car.png\n",
                "images/train/car.png": image_bytes,
                "labels/train/car.txt": "0 0.5 0.5 0.4 0.3\n",
            }
        image = BytesIO()
        Image.new("RGB", (1, 1), color="white").save(image, format="PNG")
        image_bytes = image.getvalue()
        stream = BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, value in entries.items():
                if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")) and value == b"image":
                    value = image_bytes
                archive.writestr(name, value)
        return stream.getvalue()

    def test_discovers_only_real_model_training_entry_points(self) -> None:
        with tempfile.TemporaryDirectory(prefix="car_scan_training_test_") as directory:
            root = Path(directory)
            self.make_training_root(root)
            status = TrainingRunManager(root).status()

        self.assertTrue(status["available"])
        self.assertEqual([target["id"] for target in status["models"]], ["thai_license_plate"])
        self.assertIsNone(status["run"])

    def test_start_rejects_unknown_model_before_spawning(self) -> None:
        with tempfile.TemporaryDirectory(prefix="car_scan_training_test_") as directory:
            root = Path(directory)
            self.make_training_root(root)
            with self.assertRaisesRegex(ValueError, "unknown model"):
                TrainingRunManager(root).start(model="../../something")

    def test_start_rejects_out_of_range_training_parameters(self) -> None:
        with tempfile.TemporaryDirectory(prefix="car_scan_training_test_") as directory:
            root = Path(directory)
            self.make_training_root(root)
            with self.assertRaisesRegex(ValueError, "epochs or batch"):
                TrainingRunManager(root).start(model="thai_license_plate", epochs=0)

    def test_worker_list_exposes_only_valid_configured_compute_workers(self) -> None:
        with tempfile.TemporaryDirectory(prefix="car_scan_training_test_") as directory:
            root = Path(directory)
            self.make_training_root(root)
            configuration = [{
                "id": "cpu-worker", "name": "CPU Worker", "host": "10.0.0.12", "user": "trainer",
                "dockerImage": "car-scan-web:local", "device": "cpu", "cpus": 4, "memory": "6g",
                "password": "never-return-this",
            }, {"id": "bad", "host": "invalid host", "user": "x", "dockerImage": "bad image"}]
            with patch.dict(os.environ, {"CAR_SCAN_TRAIN_WORKERS": json.dumps(configuration)}):
                workers = TrainingRunManager(root).workers()

        self.assertEqual([item["id"] for item in workers], ["local", "cpu-worker"])
        self.assertEqual(workers[1]["cpus"], 4)
        self.assertNotIn("password", workers[1])

    def test_remote_bundle_contains_only_selected_dataset_and_no_wrapper_prefix(self) -> None:
        with tempfile.TemporaryDirectory(prefix="car_scan_training_test_") as directory:
            root = Path(directory) / "model" / "train"
            root.mkdir(parents=True)
            self.make_training_root(root)
            model_folder = root.parent / "thai_license_plate"
            model_folder.mkdir()
            (model_folder / "weights.pt").write_text("weights", encoding="utf-8")
            target = root / "thai_license_plate-train"
            for dataset_id in ("chosen", "other"):
                folder = target / "dataset" / dataset_id
                folder.mkdir(parents=True)
                (folder / "data.yaml").write_text("names: [plate]\n", encoding="utf-8")
            (target / "runs" / "old").mkdir(parents=True)
            (target / "runs" / "old" / "metrics.csv").write_text("old", encoding="utf-8")
            manager = TrainingRunManager(root)

            with tempfile.TemporaryDirectory(prefix="training_bundle_test_") as bundle_dir:
                bundle = manager._create_worker_bundle(
                    model="thai_license_plate", directory="thai_license_plate-train", dataset="chosen",
                    resume=False, destination=Path(bundle_dir),
                )
                with zipfile.ZipFile(bundle) as archive:
                    names = set(archive.namelist())

        self.assertIn("model/train/train.py", names)
        self.assertIn("model/train/thai_license_plate-train/dataset/chosen/data.yaml", names)
        self.assertIn("model/thai_license_plate/weights.pt", names)
        self.assertFalse(any("other" in name for name in names))
        self.assertFalse(any("runs/old" in name for name in names))

    def test_dataset_inventory_counts_splits_and_validates_classes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="car_scan_training_test_") as directory:
            root = Path(directory)
            self.make_training_root(root)
            dataset = root / "thai_license_plate-train" / "dataset" / "set-one"
            for split in ("train", "valid", "test"):
                image_dir = dataset / "images" / split
                label_dir = dataset / "labels" / split
                image_dir.mkdir(parents=True)
                label_dir.mkdir(parents=True)
                (image_dir / "sample.jpg").touch()
                (label_dir / "sample.txt").touch()
            (dataset / "data.yaml").write_text(
                "train: images/train\nval: images/valid\ntest: images/test\nnc: 1\nnames: [plate]\n",
                encoding="utf-8",
            )

            catalog = TrainingRunManager(root).datasets(force_refresh=True)

        item = catalog["models"][0]["datasets"][0]
        self.assertEqual(item["id"], "set-one")
        self.assertTrue(item["ready"])
        self.assertEqual(item["splits"], {"train": 1, "val": 1, "test": 1})
        self.assertEqual(item["classes"], ["plate"])

    def test_start_rejects_dataset_not_in_selected_model(self) -> None:
        with tempfile.TemporaryDirectory(prefix="car_scan_training_test_") as directory:
            root = Path(directory)
            self.make_training_root(root)
            with self.assertRaisesRegex(ValueError, "unknown dataset"):
                TrainingRunManager(root).start(model="thai_license_plate", dataset="../../outside")

    def test_resume_and_result_history_are_scoped_to_model_runs_folder(self) -> None:
        with tempfile.TemporaryDirectory(prefix="car_scan_training_test_") as directory:
            root = Path(directory)
            self.make_training_root(root)
            target = root / "thai_license_plate-train"
            dataset_checkpoint = target / "dataset" / "weights" / "last.pt"
            dataset_checkpoint.parent.mkdir(parents=True)
            dataset_checkpoint.touch()
            manager = TrainingRunManager(root)
            self.assertFalse(manager.status()["models"][0]["resumeAvailable"])

            run = target / "runs" / "thai_plate_detect"
            (run / "weights").mkdir(parents=True)
            (run / "weights" / "best.pt").touch()
            (run / "weights" / "last.pt").touch()
            (run / "results.csv").write_text(
                "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B)\n2,0.9,0.8,0.85\n",
                encoding="utf-8",
            )

            model = manager.results(force_refresh=True)["models"][0]

        self.assertEqual(len(model["runs"]), 1)
        self.assertEqual(model["runs"][0]["epochsCompleted"], 3)
        self.assertTrue(model["runs"][0]["hasBestWeights"])
        self.assertTrue(model["runs"][0]["hasLastWeights"])

    def test_upload_saves_only_a_validated_dataset_under_its_model(self) -> None:
        with tempfile.TemporaryDirectory(prefix="car_scan_training_test_") as directory:
            root = Path(directory)
            self.make_training_root(root)
            result = TrainingRunManager(root).upload_dataset(
                model="thai_license_plate",
                dataset_name="new_set_01",
                archive=self.make_dataset_zip(),
            )
            destination = root / "thai_license_plate-train" / "dataset" / "new_set_01"
            destination_exists = destination.is_dir()

        self.assertTrue(destination_exists)
        self.assertTrue(result["dataset"]["ready"])
        self.assertEqual(result["validation"]["splits"]["train"]["images"], 1)
        self.assertEqual(result["validation"]["splits"]["val"]["labelFiles"], 0)

    def test_upload_accepts_cvat_train_manifest_without_validation_or_test(self) -> None:
        with tempfile.TemporaryDirectory(prefix="car_scan_training_test_") as directory:
            root = Path(directory)
            self.make_training_root(root)
            entries = {
                "data.yaml": "path: .\ntrain: train.txt\nnc: 1\nnames: [plate]\n",
                "train.txt": "data/images/train/car.png\n",
                "images/train/car.png": b"image",
                "labels/train/car.txt": "0 0.5 0.5 0.4 0.3\n",
            }

            result = TrainingRunManager(root).upload_dataset(
                model="thai_license_plate",
                dataset_name="roboflow_set",
                archive=self.make_dataset_zip(entries),
            )

        self.assertTrue(result["dataset"]["ready"])
        self.assertEqual(result["validation"]["splits"]["train"]["images"], 1)
        self.assertEqual(result["validation"]["splits"]["val"]["images"], 0)
        self.assertEqual(result["validation"]["splits"]["test"]["images"], 0)

    def test_upload_rejects_bad_labels_and_never_creates_destination(self) -> None:
        with tempfile.TemporaryDirectory(prefix="car_scan_training_test_") as directory:
            root = Path(directory)
            self.make_training_root(root)
            entries = {
                "data.yaml": "path: .\ntrain: train.txt\nnc: 1\nnames: [plate]\n",
                "train.txt": "data/images/train/car.jpg\n",
                "images/train/car.jpg": b"image",
                "labels/train/car.txt": "1 0.5 0.5 0.4 0.3\n",
            }
            with self.assertRaisesRegex(ValueError, "class id"):
                TrainingRunManager(root).upload_dataset(
                    model="thai_license_plate",
                    dataset_name="invalid_set",
                    archive=self.make_dataset_zip(entries),
                )
            self.assertFalse((root / "thai_license_plate-train" / "dataset" / "invalid_set").exists())

    def test_upload_rejects_non_cvat_yolo_split_layout(self) -> None:
        with tempfile.TemporaryDirectory(prefix="car_scan_training_test_") as directory:
            root = Path(directory)
            self.make_training_root(root)
            entries = {
                "data.yaml": "train: images/train\nval: images/val\nnc: 1\nnames: [plate]\n",
                "images/train/car.jpg": b"image",
                "images/val/car.jpg": b"image",
                "labels/train/car.txt": "0 0.5 0.5 0.4 0.3\n",
                "labels/val/car.txt": "0 0.5 0.5 0.4 0.3\n",
            }
            with self.assertRaisesRegex(ValueError, "train.txt"):
                TrainingRunManager(root).upload_dataset(
                    model="thai_license_plate",
                    dataset_name="other_yolo_layout",
                    archive=self.make_dataset_zip(entries),
                )

    def test_delete_dataset_is_scoped_to_selected_model_and_invalidates_catalog(self) -> None:
        with tempfile.TemporaryDirectory(prefix="car_scan_training_test_") as directory:
            root = Path(directory)
            self.make_training_root(root)
            manager = TrainingRunManager(root)
            manager.upload_dataset(
                model="thai_license_plate",
                dataset_name="delete_me",
                archive=self.make_dataset_zip(),
            )
            destination = root / "thai_license_plate-train" / "dataset" / "delete_me"
            self.assertTrue(destination.is_dir())

            result = manager.delete_dataset(model="thai_license_plate", dataset_id="delete_me")

            self.assertEqual(result["deleted"], "delete_me")
            self.assertFalse(destination.exists())
            self.assertEqual(manager.datasets(force_refresh=True)["datasetCount"], 0)

    def test_delete_rejects_dataset_traversal(self) -> None:
        with tempfile.TemporaryDirectory(prefix="car_scan_training_test_") as directory:
            root = Path(directory)
            self.make_training_root(root)
            with self.assertRaisesRegex(ValueError, "invalid dataset id"):
                TrainingRunManager(root).delete_dataset(model="thai_license_plate", dataset_id="../outside")

    def test_upload_rejects_zip_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory(prefix="car_scan_training_test_") as directory:
            root = Path(directory)
            self.make_training_root(root)
            with self.assertRaisesRegex(ValueError, "unsafe path"):
                TrainingRunManager(root).upload_dataset(
                    model="thai_license_plate",
                    dataset_name="unsafe_set",
                    archive=self.make_dataset_zip({"../escaped.txt": "no"}),
                )

    def test_upload_splits_dataset_by_percentages_and_updates_ultralytics_yaml(self) -> None:
        with tempfile.TemporaryDirectory(prefix="car_scan_training_test_") as directory:
            root = Path(directory)
            self.make_training_root(root)
            entries: dict[str, str | bytes] = {
                "data.yaml": "path: .\ntrain: train.txt\nnc: 1\nnames: [plate]\n",
            }
            manifest = []
            for index in range(10):
                name = f"car-{index:02}.jpg"
                manifest.append(f"data/images/train/{name}")
                entries[f"images/train/{name}"] = b"image"
                entries[f"labels/train/{Path(name).with_suffix('.txt').name}"] = "0 0.5 0.5 0.4 0.3\n"
            entries["train.txt"] = "\n".join(manifest) + "\n"

            manager = TrainingRunManager(root)
            uploaded = manager.upload_dataset(
                model="thai_license_plate",
                dataset_name="split_set",
                archive=self.make_dataset_zip(entries),
            )
            self.assertEqual(uploaded["validation"]["splits"]["train"]["images"], 10)
            self.assertEqual(uploaded["validation"]["splits"]["val"]["images"], 0)
            result = manager.split_dataset(
                model="thai_license_plate",
                dataset_id="split_set",
                percentages={"train": 80, "val": 10, "test": 10},
            )
            dataset = root / "thai_license_plate-train" / "dataset" / "split_set"
            config = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
            repeated = manager.split_dataset(
                model="thai_license_plate",
                dataset_id="split_set",
                percentages={"train": 50, "val": 50, "test": 0},
            )

        self.assertEqual(result["validation"]["splits"], {
            "train": {"images": 8, "labelFiles": 8},
            "val": {"images": 1, "labelFiles": 1},
            "test": {"images": 1, "labelFiles": 1},
        })
        self.assertEqual(config["train"], "images/train")
        self.assertEqual(config["val"], "images/val")
        self.assertEqual(config["test"], "images/test")
        self.assertEqual(repeated["validation"]["splits"]["train"]["images"], 5)
        self.assertEqual(repeated["validation"]["splits"]["val"]["images"], 5)
        self.assertEqual(repeated["validation"]["splits"]["test"]["images"], 0)

    def test_upload_rejects_invalid_split_percentages(self) -> None:
        with tempfile.TemporaryDirectory(prefix="car_scan_training_test_") as directory:
            root = Path(directory)
            self.make_training_root(root)
            manager = TrainingRunManager(root)
            manager.upload_dataset(
                    model="thai_license_plate",
                    dataset_name="bad_percentages",
                    archive=self.make_dataset_zip(),
                )
            with self.assertRaisesRegex(ValueError, "must total 100"):
                manager.split_dataset(
                    model="thai_license_plate",
                    dataset_id="bad_percentages",
                    percentages={"train": 70, "val": 20, "test": 0},
                )


if __name__ == "__main__":
    unittest.main()
