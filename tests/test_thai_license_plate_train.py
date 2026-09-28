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
        / "thai_license_plate-train"
        / "dataset"
        / "train.py"
    )
    spec = importlib.util.spec_from_file_location("thai_license_plate_train", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train = _load_train_module()


class ThaiLicensePlateTrainTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_job(self, name: str, class_names: list[str] | None = None) -> Path:
        names = class_names or ["0", "1", "A01", "BKK"]
        dataset = self.root / name
        image_dir = dataset / "images" / "train" / "obj_train_data"
        label_dir = dataset / "labels" / "train" / "obj_train_data"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        (image_dir / f"{name}.jpg").write_bytes(b"x")
        (label_dir / f"{name}.txt").write_text("0 0.5 0.5 0.2 0.1\n", encoding="utf-8")
        (dataset / "train.txt").write_text(
            f"data/images/train/obj_train_data/{name}.jpg\n",
            encoding="utf-8",
        )
        names_yaml = "\n".join(f"  {index}: {item}" for index, item in enumerate(names))
        (dataset / "data.yaml").write_text(
            f"names:\n{names_yaml}\npath: .\ntrain: train.txt\n",
            encoding="utf-8",
        )
        return dataset

    def test_job_train_txt_is_ignored_in_favor_of_image_folder(self) -> None:
        job = self._write_job("job-163")
        selected = train.select_datasets("job-163", root=self.root)
        self.assertEqual(selected, [job.resolve()])
        data = yaml.safe_load(train.build_data_manifest(selected, output=self.root / "out.yaml").read_text(encoding="utf-8"))
        train_dir = Path(data["train"][0])
        self.assertEqual(train_dir, job / "images" / "train" / "obj_train_data")
        self.assertTrue((train_dir / "job-163.jpg").is_file())
        self.assertTrue((job / "labels" / "train" / "obj_train_data" / "job-163.txt").is_file())
        self.assertEqual(data["nc"], 4)
        self.assertEqual(data["names"], ["0", "1", "A01", "BKK"])

    def test_dataset_all_combines_jobs_and_holds_out_last_as_val(self) -> None:
        first = self._write_job("job-161")
        second = self._write_job("job-165")
        selected = train.select_datasets("all", root=self.root)
        self.assertEqual([path.name for path in selected], ["job-161", "job-165"])
        data = yaml.safe_load(train.build_data_manifest(selected, output=self.root / "all.yaml").read_text(encoding="utf-8"))
        self.assertEqual(data["train"], [str(first / "images" / "train" / "obj_train_data").replace("\\", "/")])
        self.assertEqual(data["val"], [str(second / "images" / "train" / "obj_train_data").replace("\\", "/")])

    def test_real_jobs_are_discovered_and_point_at_obj_train_data(self) -> None:
        dataset_root = Path(train.SCRIPT_DIR)
        if not (dataset_root / "job-163" / "data.yaml").is_file():
            self.skipTest("Thai CVAT jobs are not checked out")
        selected = train.select_datasets("all", root=dataset_root)
        self.assertIn("job-163", {path.name for path in selected})
        self.assertGreaterEqual(len(selected), 2)
        data = yaml.safe_load(
            train.build_data_manifest(selected, output=self.root / "real.yaml").read_text(encoding="utf-8")
        )
        joined = " ".join(data["train"] + data["val"])
        self.assertIn("images/train/obj_train_data", joined.replace("\\", "/"))
        self.assertNotIn("train.txt", joined)
        self.assertEqual(data["nc"], len(data["names"]))
        self.assertGreater(data["nc"], 10)


if __name__ == "__main__":
    unittest.main()
