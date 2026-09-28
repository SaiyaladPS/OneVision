"""Prepare full-plate Thai OCR datasets for PaddleOCR.

Reads local Thai character-detector jobs (``job-*`` CVAT exports) and writes
canonical labels such as ``กรุงเทพมหานคร กข-1234``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image

from scan import THAI_CHARACTER_NAMES, THAI_PROVINCE_NAMES


ROOT = Path(__file__).resolve().parent
JOB_ROOT = ROOT / "model" / "train" / "thai_license_plate-train" / "dataset"
OCR_ROOT = ROOT / "model" / "paddleocr_train2"
OUTPUT_ROOT = OCR_ROOT / "train_data" / "rec_thai_full"
IMAGE_OUT_DIR = OUTPUT_ROOT / "images"
TRAIN_LIST = OUTPUT_ROOT / "train_list.txt"
VAL_LIST = OUTPUT_ROOT / "val_list.txt"
REPORT_PATH = OUTPUT_ROOT / "rejected.txt"
DICT_PATH = OCR_ROOT / "plate_thai_full_dict.txt"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True)
class Box:
    label: str
    x: float
    y: float
    width: float
    height: float


def class_names(dataset_dir: Path) -> list[str]:
    data = yaml.safe_load((dataset_dir / "data.yaml").read_text(encoding="utf-8")) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    return [str(name) for name in names]


def parse_boxes(label_path: Path, names: list[str]) -> list[Box]:
    boxes: list[Box] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        class_id = int(fields[0])
        if class_id < 0 or class_id >= len(names):
            continue
        boxes.append(
            Box(names[class_id], float(fields[1]), float(fields[2]), float(fields[3]), float(fields[4]))
        )
    return boxes


def transcription(boxes: list[Box]) -> tuple[str | None, str]:
    provinces = [box for box in boxes if box.label in THAI_PROVINCE_NAMES]
    letters = [box for box in boxes if box.label in THAI_CHARACTER_NAMES]
    digits = [box for box in boxes if box.label.isdigit() and len(box.label) == 1]
    if not provinces:
        return None, "missing province token"
    if len(digits) < 4:
        return None, "registration line does not have at least four digits"
    provinces.sort(key=lambda box: box.x)
    letters.sort(key=lambda box: box.x)
    digits.sort(key=lambda box: box.x)
    province = THAI_PROVINCE_NAMES[provinces[-1].label]
    glyphs = "".join(THAI_CHARACTER_NAMES[box.label] for box in letters)
    number = "".join(box.label for box in digits)
    if glyphs:
        return f"{province} {glyphs}-{number}", ""
    return f"{province} {number}", ""


def image_label_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
    layouts = (
        (dataset_dir / "images" / "train" / "obj_train_data", dataset_dir / "labels" / "train" / "obj_train_data"),
        (dataset_dir / "train" / "images", dataset_dir / "train" / "labels"),
        (dataset_dir / "images", dataset_dir / "labels"),
    )
    for image_dir, label_dir in layouts:
        if not image_dir.is_dir() or not label_dir.is_dir():
            continue
        pairs = []
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.is_file():
                pairs.append((image_path, label_path))
        if pairs:
            return pairs
    return []


def discover_jobs() -> list[Path]:
    if not JOB_ROOT.is_dir():
        return []
    return sorted(
        path
        for path in JOB_ROOT.iterdir()
        if path.is_dir() and (path / "data.yaml").is_file()
    )


def write_list(path: Path, items: list[tuple[str, str]]) -> None:
    path.write_text(
        "".join(f"rec_thai_full/images/{name}\t{text}\n" for name, text in items),
        encoding="utf-8",
    )


def write_dictionary(items: list[tuple[str, str]]) -> None:
    values = {character for _, text in items for character in text if not character.isspace()}
    ordered = ["-"] + list("0123456789")
    ordered += sorted(character for character in values if "\u0e00" <= character <= "\u0e7f")
    if DICT_PATH.is_file():
        existing = [line for line in DICT_PATH.read_text(encoding="utf-8").splitlines() if line]
        ordered = list(dict.fromkeys(existing + ordered))
    DICT_PATH.write_text("\n".join(ordered) + "\n", encoding="utf-8")


def main() -> None:
    jobs = discover_jobs()
    if not jobs:
        raise FileNotFoundError(f"No Thai job datasets with data.yaml were found in {JOB_ROOT}")

    IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    rejected: list[str] = []
    rows: list[tuple[str, str]] = []
    for job in jobs:
        names = class_names(job)
        for image_path, label_path in image_label_pairs(job):
            text, reason = transcription(parse_boxes(label_path, names))
            if text is None:
                rejected.append(f"{job.name}/{image_path.name}\t{reason}")
                continue
            name = f"{job.name}_{image_path.name}"
            Image.open(image_path).convert("RGB").save(IMAGE_OUT_DIR / name, quality=95)
            rows.append((name, text))

    if len(rows) < 2:
        raise FileNotFoundError(f"Not enough complete Thai plates were found under {JOB_ROOT}")

    val_count = max(1, len(rows) // 10)
    val_items = rows[-val_count:]
    train_items = rows[:-val_count]
    write_list(TRAIN_LIST, train_items)
    write_list(VAL_LIST, val_items)
    write_dictionary(rows)
    REPORT_PATH.write_text("\n".join(rejected) + ("\n" if rejected else ""), encoding="utf-8")
    print(f"Thai OCR train: {len(train_items)}")
    print(f"Thai OCR val: {len(val_items)}")
    print(f"Rejected: {len(rejected)}")
    print(f"Wrote {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
