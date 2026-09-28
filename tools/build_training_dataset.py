"""Build reviewable CVAT or Roboflow training sets from saved scan dates.

Examples:
    python tools/build_training_dataset.py
    python tools/build_training_dataset.py --format cvat --dates 20260802-20260803
    python tools/build_training_dataset.py --format roboflow --dates 20260802,20260805

The scan archive is evidence, not ground truth.  ``PASS`` therefore contains
confirmed scanner results as *pre-labels* for review; ``REJECT`` is created for
items explicitly stored as rejected.  Correct labels in CVAT/Roboflow before
using either group to train a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "scan" / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CVAT or Roboflow datasets from scan dates.")
    parser.add_argument("--format", choices=("cvat", "roboflow"), help="Target annotation/training format.")
    parser.add_argument(
        "--dates",
        help="Dates as YYYYMMDD-YYYYMMDD, or a comma-separated list such as 20260802,20260805.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, help="Defaults to <project>/cvat or <project>/roboflow.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing export with the same name.")
    return parser.parse_args()


def prompt(value: str | None, text: str, allowed: set[str] | None = None) -> str:
    while not value:
        value = input(text).strip().lower()
        if not allowed or value in allowed:
            break
        print(f"Choose one of: {', '.join(sorted(allowed))}")
        value = ""
    return value


def selected_dates(specification: str, data_root: Path) -> list[str]:
    values: set[str] = set()
    for piece in (part.strip() for part in specification.split(",") if part.strip()):
        if "-" not in piece:
            values.add(validate_date(piece))
            continue
        start_text, end_text = (validate_date(part) for part in piece.split("-", 1))
        start = datetime.strptime(start_text, "%Y%m%d").date()
        end = datetime.strptime(end_text, "%Y%m%d").date()
        if end < start:
            raise ValueError(f"End date is before start date: {piece}")
        while start <= end:
            values.add(start.strftime("%Y%m%d"))
            start += timedelta(days=1)
    missing = sorted(date for date in values if not (data_root / date).is_dir())
    if missing:
        raise FileNotFoundError(f"No saved scan data for: {', '.join(missing)}")
    return sorted(values)


def validate_date(value: str) -> str:
    datetime.strptime(value, "%Y%m%d")
    return value


def archive_name(dates: list[str]) -> str:
    return f"{dates[0]}-{dates[-1]}" if len(dates) > 1 and dates == list_date_range(dates[0], dates[-1]) else "-".join(dates)


def list_date_range(first: str, last: str) -> list[str]:
    value = datetime.strptime(first, "%Y%m%d").date()
    end = datetime.strptime(last, "%Y%m%d").date()
    output: list[str] = []
    while value <= end:
        output.append(value.strftime("%Y%m%d"))
        value += timedelta(days=1)
    return output


def load_samples(data_root: Path, dates: Iterable[str]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for date in dates:
        for manifest_path in sorted((data_root / date / "json").glob("*_result.json")):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                print(f"Skipping unreadable manifest {manifest_path}: {error}")
                continue
            entries = [
                ("PASS", plate)
                for plate in manifest.get("plates", [])
            ] + [
                ("REJECT", plate)
                for plate in manifest.get("rejected_plates", [])
            ]
            for index, (status, plate) in enumerate(entries, start=1):
                if not isinstance(plate, dict):
                    continue
                country = "thai" if plate.get("country") == "thai" else "laos"
                crop = Path(str(plate.get("crop_image") or ""))
                full = Path(str(plate.get("full_vehicle_image") or ""))
                ready = Path(str(plate.get("ocr_ready_image") or crop))
                if not crop.is_file() or not full.is_file():
                    continue
                sample_id = f"{date}-{manifest_path.stem}-{index}-{plate.get('archive_filename') or crop.stem}"
                dataset = plate.get("dataset", {})
                dataset_status = (
                    str(dataset.get("export_status") or "")
                    if isinstance(dataset, dict)
                    else ""
                )
                samples.append(
                    {
                        "id": sample_id,
                        "date": date,
                        "country": country,
                        # QC is calculated in the scan pipeline.  Keep its
                        # original PASS/REVIEW/REJECT value on the plate and
                        # use its explicit export mapping here, rather than
                        # independently recalculating confidence in exporter.
                        "status": dataset_status or str(plate.get("training_status") or status),
                        "plate": plate,
                        "crop": crop,
                        "full": full,
                        "ready": ready if ready.is_file() else crop,
                        "manifest": manifest_path,
                    }
                )
    return samples


def reading_for(sample: dict[str, Any]) -> dict[str, Any]:
    plate = sample["plate"]
    country = "lao" if sample["country"] == "laos" else "thai"
    readings = plate.get("country_readings", {})
    return readings.get(country, {}) if isinstance(readings, dict) else {}


def character_classes(samples: Iterable[dict[str, Any]], country: str) -> list[str]:
    labels = {
        str(token.get("label"))
        for sample in samples
        if sample["country"] == country
        for token in reading_for(sample).get("tokens", [])
        if isinstance(token, dict) and token.get("label")
    }
    return sorted(labels)


def yolo_character_labels(sample: dict[str, Any], classes: list[str]) -> list[str]:
    image = cv2.imread(str(sample["crop"]))
    if image is None:
        return []
    height, width = image.shape[:2]
    if not width or not height:
        return []
    class_ids = {name: index for index, name in enumerate(classes)}
    labels: list[str] = []
    for token in reading_for(sample).get("tokens", []):
        if not isinstance(token, dict) or str(token.get("label")) not in class_ids:
            continue
        box = token.get("box")
        if not isinstance(box, list) or len(box) != 4:
            continue
        left, top, right, bottom = (float(value) for value in box)
        left, right = max(0.0, left), min(float(width), right)
        top, bottom = max(0.0, top), min(float(height), bottom)
        if right <= left or bottom <= top:
            continue
        labels.append(
            f"{class_ids[str(token['label'])]} {(left + right) / 2 / width:.6f} "
            f"{(top + bottom) / 2 / height:.6f} {(right - left) / width:.6f} {(bottom - top) / height:.6f}"
        )
    return labels


def yolo_detector_labels(sample: dict[str, Any]) -> list[str]:
    image = cv2.imread(str(sample["full"]))
    box = sample["plate"].get("box")
    if image is None or not isinstance(box, list) or len(box) != 4:
        return []
    height, width = image.shape[:2]
    left, top, right, bottom = (float(value) for value in box)
    left, right = max(0.0, left), min(float(width), right)
    top, bottom = max(0.0, top), min(float(height), bottom)
    if right <= left or bottom <= top:
        return []
    return [f"0 {(left + right) / 2 / width:.6f} {(top + bottom) / 2 / height:.6f} {(right - left) / width:.6f} {(bottom - top) / height:.6f}"]


def split_for(sample_id: str) -> str:
    bucket = int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "valid" if bucket < 90 else "test"


def copy_labelled(image: Path, labels: list[str], image_dir: Path, label_dir: Path, name: str) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    target = image_dir / f"{name}{image.suffix.lower()}"
    shutil.copy2(image, target)
    (label_dir / f"{name}.txt").write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")


def write_data_yaml(path: Path, names: list[str]) -> None:
    name_rows = "\n".join(f"  {index}: {json.dumps(name, ensure_ascii=False)}" for index, name in enumerate(names))
    path.write_text(
        "train: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: {len(names)}\nnames:\n{name_rows}\n",
        encoding="utf-8",
    )


def export_roboflow(output: Path, samples: list[dict[str, Any]], selection: str) -> None:
    classes = {country: character_classes(samples, country) for country in ("thai", "laos")}
    jobs = (("dataset_thai_license_plate", "thai"), ("dataset_lao_license_plate", "laos"))
    for dataset_name, country in jobs:
        for status in ("PASS", "REJECT"):
            root = output / dataset_name / status
            for split in ("train", "valid", "test"):
                (root / split / "images").mkdir(parents=True, exist_ok=True)
                (root / split / "labels").mkdir(parents=True, exist_ok=True)
            for sample in samples:
                if sample["country"] != country or sample["status"] != status:
                    continue
                split = split_for(sample["id"])
                copy_labelled(sample["crop"], yolo_character_labels(sample, classes[country]), root / split / "images", root / split / "labels", sample["id"])
            write_data_yaml(root / "data.yaml", classes[country])

    for status in ("PASS", "REJECT"):
        root = output / "dataset_detect_license" / status
        for split in ("train", "valid", "test"):
            (root / split / "images").mkdir(parents=True, exist_ok=True)
            (root / split / "labels").mkdir(parents=True, exist_ok=True)
        for sample in samples:
            if sample["status"] != status:
                continue
            split = split_for(sample["id"])
            copy_labelled(sample["full"], yolo_detector_labels(sample), root / split / "images", root / split / "labels", sample["id"])
        write_data_yaml(root / "data.yaml", ["license_plate"])
    export_ocr(output / "dataset_ocr", samples)
    write_manifest(output, samples, selection, "roboflow")


def export_cvat(output: Path, samples: list[dict[str, Any]], selection: str) -> None:
    classes = {country: character_classes(samples, country) for country in ("thai", "laos")}
    jobs = (("dataset_thai_license_plate", "thai"), ("dataset_lao_license_plate", "laos"))
    for dataset_name, country in jobs:
        for status in ("PASS", "REJECT"):
            root = output / dataset_name / status
            root.mkdir(parents=True, exist_ok=True)
            image_dir, label_dir = root / "images", root / "labels"
            lines: list[str] = []
            for sample in samples:
                if sample["country"] != country or sample["status"] != status:
                    continue
                copy_labelled(sample["crop"], yolo_character_labels(sample, classes[country]), image_dir, label_dir, sample["id"])
                lines.append(f"images/{sample['id']}{sample['crop'].suffix.lower()}")
            (root / "obj.names").write_text("\n".join(classes[country]) + "\n", encoding="utf-8")
            (root / "train.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    for status in ("PASS", "REJECT"):
        root = output / "dataset_detect_license" / status
        root.mkdir(parents=True, exist_ok=True)
        image_dir, label_dir = root / "images", root / "labels"
        lines: list[str] = []
        for sample in samples:
            if sample["status"] != status:
                continue
            copy_labelled(sample["full"], yolo_detector_labels(sample), image_dir, label_dir, sample["id"])
            lines.append(f"images/{sample['id']}{sample['full'].suffix.lower()}")
        (root / "obj.names").write_text("license_plate\n", encoding="utf-8")
        (root / "train.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    export_ocr(output / "dataset_ocr", samples)
    write_manifest(output, samples, selection, "cvat")


def export_ocr(root: Path, samples: list[dict[str, Any]]) -> None:
    for status in ("PASS", "REJECT"):
        for country in ("thai", "laos"):
            folder = root / status / f"rec_{'lao' if country == 'laos' else 'thai'}_full"
            folder.mkdir(parents=True, exist_ok=True)
            image_dir = folder / "images"
            lines: dict[str, list[str]] = defaultdict(list)
            for sample in samples:
                if sample["country"] != country or sample["status"] != status:
                    continue
                plate = sample["plate"]
                text = f"{plate.get('plate_prefix') or ''}{plate.get('plate_number') or ''}".strip()
                if not text:
                    continue
                target = image_dir / f"{sample['id']}{sample['ready'].suffix.lower()}"
                image_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sample["ready"], target)
                lines[split_for(sample["id"])].append(f"images/{target.name}\t{text}")
            for split, filename in (("train", "train_list.txt"), ("valid", "val_list.txt")):
                (folder / filename).write_text("\n".join(lines[split]) + ("\n" if lines[split] else ""), encoding="utf-8")
            (folder / "rejected.txt").write_text("", encoding="utf-8")


def write_manifest(output: Path, samples: list[dict[str, Any]], selection: str, target: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": target,
        "date_selection": selection,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sample_count": len(samples),
        "pass_count": sum(sample["status"] == "PASS" for sample in samples),
        "reject_count": sum(sample["status"] == "REJECT" for sample in samples),
        "review_required": True,
        "samples": [{key: str(value) if isinstance(value, Path) else value for key, value in sample.items() if key != "plate"} for sample in samples],
    }
    (output / "export_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.txt").write_text(
        "PASS labels are scanner pre-labels and must be reviewed before training.\n"
        "REJECT is for confirmed bad/uncertain scans and must not be used as OCR truth without manual labels.\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    target = prompt(args.format, "Export format (cvat/roboflow): ", {"cvat", "roboflow"})
    specification = prompt(args.dates, "Dates (YYYYMMDD-YYYYMMDD or YYYYMMDD,YYYYMMDD): ")
    data_root = args.data_root.resolve()
    dates = selected_dates(specification, data_root)
    selection = archive_name(dates)
    base = (args.output_root or ROOT / target).resolve()
    output = base / f"{target}-{selection}"
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Export already exists: {output}. Use --overwrite to replace it.")
        shutil.rmtree(output)
    samples = load_samples(data_root, dates)
    if target == "cvat":
        export_cvat(output, samples, selection)
    else:
        export_roboflow(output, samples, selection)
    print(f"Created {target} export: {output}")
    print(f"Selected dates: {', '.join(dates)} | samples: {len(samples)}")


if __name__ == "__main__":
    main()
