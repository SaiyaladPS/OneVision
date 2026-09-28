from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


def _load_train_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "model"
        / "train"
        / "detect_license-train"
        / "dataset"
        / "train.py"
    )
    spec = importlib.util.spec_from_file_location("detect_license_train", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train = _load_train_module()


class DetectLicenseTrainTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_roboflow(self, name: str) -> Path:
        dataset = self.root / name
        for split in ("train", "valid", "test"):
            (dataset / split / "images").mkdir(parents=True)
            (dataset / split / "labels").mkdir(parents=True)
        (dataset / "data.yaml").write_text(
            "train: ../train/images\nval: ../valid/images\ntest: ../test/images\nnc: 1\nnames: ['License_Plate']\n",
            encoding="utf-8",
        )
        return dataset

    def _write_detect_license(self) -> Path:
        dataset = self.root / "detect_license"
        for split in ("Train", "Validation", "Test"):
            (dataset / "images" / split).mkdir(parents=True)
            (dataset / "labels" / split).mkdir(parents=True)
        (dataset / "images" / "Train" / "car.jpg").write_bytes(b"x")
        (dataset / "data.yaml").write_text(
            "Train: Train.txt\nValidation: Validation.txt\nTest: Test.txt\n"
            "names:\n  0: license-plate\npath: .\ntrain: train.txt\n",
            encoding="utf-8",
        )
        return dataset

    def test_dataset_all_includes_detect_license_layout(self) -> None:
        roboflow = self._write_roboflow("Lpr")
        native = self._write_detect_license()
        selected = train.select_datasets("all", root=self.root)
        self.assertEqual({path.name for path in selected}, {"Lpr", "detect_license"})
        manifest = train.build_combined_data(selected, output=self.root / "combined.yaml")
        data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        self.assertEqual(data["nc"], 1)
        self.assertEqual(data["names"], ["License_Plate"])
        self.assertIn(str(roboflow / "train" / "images").replace("\\", "/"), data["train"])
        self.assertIn(str(native / "images" / "Train").replace("\\", "/"), data["train"])
        self.assertIn(str(native / "images" / "Validation").replace("\\", "/"), data["val"])

    def test_lpr2_alias_selects_lpr_folder(self) -> None:
        self._write_roboflow("Lpr")
        selected = train.select_datasets("Lpr-2", root=self.root)
        self.assertEqual([path.name for path in selected], ["Lpr"])

    def test_skips_incompatible_datasets_when_combining(self) -> None:
        self._write_roboflow("Lpr")
        other = self.root / "ocr-chars"
        (other / "train" / "images").mkdir(parents=True)
        (other / "data.yaml").write_text("nc: 2\nnames: ['A', 'B']\ntrain: train/images\n", encoding="utf-8")
        selected = train.select_datasets("all", root=self.root)
        self.assertEqual([path.name for path in selected], ["Lpr"])

    def _write_cvat_pass(self, wrapper: str = "dataset_detect_license") -> Path:
        leaf = self.root / wrapper / "PASS"
        (leaf / "images").mkdir(parents=True)
        (leaf / "labels").mkdir(parents=True)
        (leaf / "images" / "gate.jpg").write_bytes(b"x")
        (leaf / "labels" / "gate.txt").write_text("0 0.5 0.5 0.2 0.1\n", encoding="utf-8")
        (leaf / "obj.names").write_text("license_plate\n", encoding="utf-8")
        (leaf / "train.txt").write_text("images/gate.jpg\n", encoding="utf-8")
        reject = self.root / wrapper / "REJECT"
        (reject / "images").mkdir(parents=True)
        (reject / "obj.names").write_text("license_plate\n", encoding="utf-8")
        return leaf

    def _write_cvat_obj_train_data(self) -> Path:
        dataset = self.root / "cvat_yolo11"
        mixed = dataset / "obj_train_data"
        mixed.mkdir(parents=True)
        (mixed / "car.jpg").write_bytes(b"x")
        (mixed / "car.txt").write_text("0 0.5 0.5 0.4 0.3\n", encoding="utf-8")
        (dataset / "obj.names").write_text("License_Plate\n", encoding="utf-8")
        (dataset / "obj.data").write_text("classes = 1\nnames = obj.names\n", encoding="utf-8")
        return dataset

    def test_dataset_all_includes_cvat_and_keeps_canonical_class(self) -> None:
        roboflow = self._write_roboflow("Lpr")
        cvat = self._write_cvat_pass()
        selected = train.select_datasets("all", root=self.root)
        self.assertIn(cvat.resolve(), selected)
        self.assertTrue(any(path.name == "Lpr" for path in selected))
        self.assertFalse(any(path.name.lower() == "reject" for path in selected))
        data = yaml.safe_load(train.build_combined_data(selected, output=self.root / "cvat.yaml").read_text(encoding="utf-8"))
        self.assertEqual(data["names"], ["License_Plate"])
        self.assertEqual(data["nc"], 1)
        self.assertIn(str(roboflow / "train" / "images").replace("\\", "/"), data["train"])
        self.assertIn(str(cvat / "images").replace("\\", "/"), data["train"])

    def test_cvat_obj_train_data_is_exposed_as_ultralytics_images(self) -> None:
        dataset = self._write_cvat_obj_train_data()
        selected = train.select_datasets("cvat_yolo11", root=self.root)
        self.assertEqual(selected, [dataset.resolve()])
        data = yaml.safe_load(train.build_combined_data(selected, output=self.root / "mixed.yaml").read_text(encoding="utf-8"))
        self.assertEqual(data["names"], ["License_Plate"])
        train_dir = Path(data["train"][0])
        self.assertTrue((train_dir / "car.jpg").is_file())
        self.assertTrue((train_dir.parent / "labels" / "car.txt").is_file())

    def test_cvat_wrapper_name_selects_pass_only(self) -> None:
        leaf = self._write_cvat_pass()
        selected = train.select_datasets("dataset_detect_license", root=self.root)
        self.assertEqual(selected, [leaf.resolve()])

    def test_real_training_folder_all_includes_detect_license(self) -> None:
        dataset_root = Path(train.SCRIPT_DIR)
        if not (dataset_root / "detect_license" / "data.yaml").is_file():
            self.skipTest("detect_license dataset is not checked out")
        selected = train.select_datasets("all", root=dataset_root)
        names = {path.name for path in selected}
        self.assertIn("detect_license", names)
        self.assertTrue({"Lpr", "Plate Detect"} & names)
        manifest = yaml.safe_load(train.build_combined_data(selected, output=self.root / "real.yaml").read_text(encoding="utf-8"))
        joined = " ".join(manifest["train"])
        self.assertIn("detect_license/images/Train", joined.replace("\\", "/"))


if __name__ == "__main__":
    unittest.main()
