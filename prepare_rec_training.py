"""Prepare the PaddleOCR number-recognition dataset from ``Images``.

The source files use names such as ``00003_ບກ1356_..._ocr_crop.jpg`` and
``00003_619335_..._ocr_crop.jpg``.  The province/prefix is deliberately not
trained here: the target is the plate number line only.
"""

from __future__ import annotations

import re
import random
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "Images"
OCR_ROOT = ROOT / "model" / "paddleocr_train2"
DATA_DIR = OCR_ROOT / "train_data" / "rec_plate"
IMAGE_OUT_DIR = DATA_DIR / "images"
TRAIN_LIST = DATA_DIR / "train_list.txt"
VAL_LIST = DATA_DIR / "val_list.txt"
DICT_PATH = OCR_ROOT / "plate_dict.txt"


def plate_text_from_token(token: str) -> str | None:
    """Convert the filename token into the visible number-line label."""

    # The user-defined convention is: everything after ``00003_`` is the
    # plate number.  Do not infer a province code or insert punctuation.
    if re.fullmatch(r"[0-9]{6}", token):
        return token
    match = re.search(r"([0-9]{4})$", token)
    return match.group(1) if match else None


def read_rows() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    pattern = re.compile(r"^00003_(?P<token>[^_]+)_[^_]+_ocr_crop\.(?:jpg|jpeg|png|bmp|webp)$", re.IGNORECASE)
    for source in sorted(IMAGE_DIR.glob("*_ocr_crop.*")):
        match = pattern.match(source.name)
        if not match:
            continue
        plate_text = plate_text_from_token(match.group("token"))
        if plate_text:
            rows.append((source.name, plate_text))
    return rows


def crop_plate_line(source: Path, target: Path) -> None:
    with Image.open(source) as image:
        image = image.convert("RGB")
        width, height = image.size
        # Keep the large plate-number line and remove the small top/bottom text.
        top = max(0, int(height * 0.38))
        bottom = min(height, max(top + 1, int(height * 0.90)))
        cropped = image.crop((0, top, width, bottom))
        cropped.save(target, quality=95)


def write_list(path: Path, items: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for image_name, text in items:
            file.write(f"rec_plate/images/{image_name}\t{text}\n")


def main() -> None:
    rows = read_rows()
    if not rows:
        raise RuntimeError("No usable rows found in image/labels.csv")

    if IMAGE_OUT_DIR.exists():
        shutil.rmtree(IMAGE_OUT_DIR)
    for label_file in (TRAIN_LIST, VAL_LIST):
        if label_file.exists():
            label_file.unlink()
    IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    prepared: list[tuple[str, str]] = []
    for index, (filename, plate_text) in enumerate(rows):
        output_name = f"{index:04d}.jpg"
        crop_plate_line(IMAGE_DIR / filename, IMAGE_OUT_DIR / output_name)
        prepared.append((output_name, plate_text))

    # Keep the same plate number in one split to avoid validation leakage.
    groups: dict[str, list[tuple[str, str]]] = {}
    for item in prepared:
        groups.setdefault(item[1], []).append(item)
    group_names = sorted(groups)
    random.Random(20260817).shuffle(group_names)
    val_groups = set(group_names[: max(1, round(len(group_names) * 0.2))])
    train_items = [item for item in prepared if item[1] not in val_groups]
    val_items = [item for item in prepared if item[1] in val_groups]

    write_list(TRAIN_LIST, train_items)
    write_list(VAL_LIST, val_items)

    required_characters = sorted({char for _, text in prepared for char in text if char != " "})
    existing_characters = []
    if DICT_PATH.is_file():
        existing_characters = [line.strip() for line in DICT_PATH.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    characters = existing_characters + [char for char in required_characters if char not in existing_characters]
    DICT_PATH.write_text("\n".join(characters) + "\n", encoding="utf-8")

    print(f"Prepared samples: {len(prepared)}")
    print(f"Train samples: {len(train_items)}")
    print(f"Validation samples: {len(val_items)}")
    print(f"Characters in dictionary: {len(characters)}")
    print(f"Training data: {DATA_DIR}")
    print(f"Dictionary: {DICT_PATH}")


if __name__ == "__main__":
    main()
