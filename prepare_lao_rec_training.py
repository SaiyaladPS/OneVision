"""Prepare full-plate and focused Lao OCR datasets for PaddleOCR.

The detection dataset labels one province token and the characters on the
registration line.  The regular output contains a transcription such as
``VTE ກຈ6208``.  The ``detail`` output additionally crops each annotated digit
or Lao glyph, plus a focused province-name strip, so the recogniser can learn
the shapes without neighbouring plate content.  The province code remains
intentional in the regular output because the application maps it through
``LAO_PROVINCE_NAMES`` to the full Lao province name.
"""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image

from scan import LAO_CHARACTER_NAMES, LAO_PROVINCE_NAMES


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "model" / "train" / "lao_license_plate-train" / "dataset" / "lao-plate-detect-1"
OCR_ROOT = ROOT / "model" / "paddleocr_train2"
OUTPUT_ROOT = OCR_ROOT / "train_data" / "rec_lao_full"
DETAIL_OUTPUT_ROOT = OUTPUT_ROOT / "detail"
DETAIL_IMAGE_OUT_DIR = DETAIL_OUTPUT_ROOT / "images"
DETAIL_TRAIN_LIST = DETAIL_OUTPUT_ROOT / "train_list.txt"
DETAIL_VAL_LIST = DETAIL_OUTPUT_ROOT / "val_list.txt"
DETAIL_DICT_PATH = OCR_ROOT / "plate_lao_detail_dict.txt"
IMAGE_OUT_DIR = OUTPUT_ROOT / "images"
TRAIN_LIST = OUTPUT_ROOT / "train_list.txt"
VAL_LIST = OUTPUT_ROOT / "val_list.txt"
REPORT_PATH = OUTPUT_ROOT / "rejected.txt"
DICT_PATH = OCR_ROOT / "plate_lao_full_dict.txt"


@dataclass(frozen=True)
class Box:
    label: str
    x: float
    y: float
    width: float
    height: float


def class_names() -> list[str]:
    data = yaml.safe_load((DATASET_ROOT / "data.yaml").read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        return [names[index] for index in range(len(names))]
    return list(names)


def parse_boxes(label_path: Path, names: list[str]) -> list[Box]:
    boxes: list[Box] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        class_id = int(fields[0])
        boxes.append(
            Box(
                names[class_id],
                float(fields[1]),
                float(fields[2]),
                float(fields[3]),
                float(fields[4]),
            )
        )
    return boxes


def transcription(boxes: list[Box]) -> tuple[str | None, str]:
    province_boxes = [box for box in boxes if box.label in LAO_PROVINCE_NAMES]
    if not province_boxes:
        return None, "missing province token"
    province_code = max(province_boxes, key=lambda box: box.y).label
    # Recognition ground truth uses the human-readable Lao province name,
    # not the detector's ASCII class code.  The runtime keeps the code
    # separately in ``province_code`` after OCR.
    province = LAO_PROVINCE_NAMES[province_code]

    line_boxes = [box for box in boxes if box not in province_boxes]
    line_boxes.sort(key=lambda box: box.x)
    if not line_boxes:
        return None, "missing registration line"

    characters: list[str] = []
    for box in line_boxes:
        if box.label.isdigit() and len(box.label) == 1:
            characters.append(box.label)
        elif box.label in LAO_CHARACTER_NAMES:
            characters.append(LAO_CHARACTER_NAMES[box.label])
        else:
            return None, f"unmapped character {box.label}"

    digits = [character for character in characters if character.isdigit()]
    lao = [character for character in characters if character in LAO_CHARACTER_NAMES.values()]
    if len(digits) != 4 or len(lao) < 2:
        return None, "registration line is not two Lao characters plus four digits"
    return f"{province} {''.join(characters)}", ""


def write_list(path: Path, items: list[tuple[str, str]]) -> None:
    path.write_text(
        "".join(f"rec_lao_full/images/{name}\t{text}\n" for name, text in items),
        encoding="utf-8",
    )


def write_detail_list(path: Path, items: list[tuple[str, str]]) -> None:
    path.write_text(
        "".join(f"rec_lao_full/detail/images/{name}\t{text}\n" for name, text in items),
        encoding="utf-8",
    )


def write_dictionary(items: list[tuple[str, str]]) -> None:
    values = {character for _, text in items for character in text if character != " "}
    # Keep only characters that can occur in the canonical full-name format:
    # ``Lao province name + space + Lao prefix + four digits``.
    ordered = ["-"] + list("0123456789")
    ordered += sorted(character for character in values if "\u0e80" <= character <= "\u0eff")
    DICT_PATH.write_text("\n".join(dict.fromkeys(ordered)) + "\n", encoding="utf-8")


def collect_split(split: str, names: list[str], rejected: list[str]) -> list[tuple[Path, str]]:
    image_dir = DATASET_ROOT / split / "images"
    label_dir = DATASET_ROOT / split / "labels"
    rows: list[tuple[Path, str]] = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            rejected.append(f"{split}/{image_path.name}\tmissing label")
            continue
        text, reason = transcription(parse_boxes(label_path, names))
        if text is None:
            rejected.append(f"{split}/{image_path.name}\t{reason}")
            continue
        rows.append((image_path, text))
    return rows


def detail_label(box: Box) -> str | None:
    """Return the Unicode transcription for one annotated component."""

    if box.label in LAO_PROVINCE_NAMES:
        return LAO_PROVINCE_NAMES[box.label]
    if box.label in LAO_CHARACTER_NAMES:
        return LAO_CHARACTER_NAMES[box.label]
    if box.label.isdigit() and len(box.label) == 1:
        return box.label
    return None


def collect_detail_split(
    split: str,
    names: list[str],
    output_index: int,
    rejected: list[str],
) -> tuple[list[tuple[str, str]], int]:
    """Crop province strips and individual registration glyphs for OCR."""

    image_dir = DATASET_ROOT / split / "images"
    label_dir = DATASET_ROOT / split / "labels"
    rows: list[tuple[str, str]] = []
    DETAIL_IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        boxes = parse_boxes(label_path, names)
        try:
            image = Image.open(image_path).convert("RGB")
        except (OSError, ValueError) as error:
            rejected.append(f"{split}/{image_path.name}\tcannot read image: {error}")
            continue
        image_width, image_height = image.size
        for box in boxes:
            label = detail_label(box)
            if not label:
                rejected.append(f"{split}/{image_path.name}\tunknown detail label {box.label}")
                continue
            centre_x, centre_y = box.x * image_width, box.y * image_height
            box_width, box_height = box.width * image_width, box.height * image_height
            # A small border protects strokes at the edge while keeping the
            # crop focused on one character or the province strip.
            pad_x = max(2.0, box_width * 0.12)
            pad_y = max(2.0, box_height * 0.18)
            left = max(0, int(round(centre_x - box_width / 2 - pad_x)))
            top = max(0, int(round(centre_y - box_height / 2 - pad_y)))
            right = min(image_width, int(round(centre_x + box_width / 2 + pad_x)))
            bottom = min(image_height, int(round(centre_y + box_height / 2 + pad_y)))
            if right - left < 3 or bottom - top < 3:
                continue
            filename = f"{split}_{output_index:06d}.jpg"
            image.crop((left, top, right, bottom)).save(DETAIL_IMAGE_OUT_DIR / filename, quality=95)
            rows.append((filename, label))
            output_index += 1
    return rows, output_index


def write_detail_dictionary(items: list[tuple[str, str]]) -> None:
    values = {character for _, text in items for character in text if character != " "}
    ordered = list("0123456789")
    ordered += sorted(character for character in values if "\u0e80" <= character <= "\u0eff")
    DETAIL_DICT_PATH.write_text("\n".join(dict.fromkeys(ordered)) + "\n", encoding="utf-8")


def main() -> None:
    names = class_names()
    rejected: list[str] = []
    train_rows = collect_split("train", names, rejected)
    val_rows = collect_split("valid", names, rejected)
    if not train_rows or not val_rows:
        raise RuntimeError("The Lao dataset did not produce usable train/validation samples")

    if IMAGE_OUT_DIR.exists():
        shutil.rmtree(IMAGE_OUT_DIR)
    IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, _ in train_rows + val_rows:
        output_name = f"{len(list(IMAGE_OUT_DIR.iterdir())):04d}.jpg"
        with Image.open(path) as image:
            image.convert("RGB").save(IMAGE_OUT_DIR / output_name, quality=95)
        if path in [row[0] for row in train_rows]:
            index = next(index for index, row in enumerate(train_rows) if row[0] == path)
            train_rows[index] = (IMAGE_OUT_DIR / output_name, train_rows[index][1])
        else:
            index = next(index for index, row in enumerate(val_rows) if row[0] == path)
            val_rows[index] = (IMAGE_OUT_DIR / output_name, val_rows[index][1])

    train_items = [(path.name, text) for path, text in train_rows]
    val_items = [(path.name, text) for path, text in val_rows]
    write_list(TRAIN_LIST, train_items)
    write_list(VAL_LIST, val_items)
    write_dictionary(train_items + val_items)
    REPORT_PATH.write_text("\n".join(rejected) + ("\n" if rejected else ""), encoding="utf-8")
    # Also build a focused recognition set. Each image contains exactly one
    # digit/glyph or one province-name strip, so the model can learn fine
    # character shapes without the neighbouring plate content confusing it.
    if DETAIL_IMAGE_OUT_DIR.exists():
        shutil.rmtree(DETAIL_IMAGE_OUT_DIR)
    detail_rejected: list[str] = []
    detail_train, next_index = collect_detail_split("train", names, 0, detail_rejected)
    detail_val, _ = collect_detail_split("valid", names, next_index, detail_rejected)
    if not detail_train or not detail_val:
        raise RuntimeError("The Lao dataset did not produce usable detail OCR crops")
    write_detail_list(DETAIL_TRAIN_LIST, detail_train)
    write_detail_list(DETAIL_VAL_LIST, detail_val)
    write_detail_dictionary(detail_train + detail_val)
    (DETAIL_OUTPUT_ROOT / "rejected.txt").write_text(
        "\n".join(detail_rejected) + ("\n" if detail_rejected else ""), encoding="utf-8"
    )
    print(f"Train samples: {len(train_items)}")
    print(f"Validation samples: {len(val_items)}")
    print(f"Rejected samples: {len(rejected)}")
    print(f"Training data: {OUTPUT_ROOT}")
    print(f"Dictionary: {DICT_PATH}")
    print(f"Detail train samples: {len(detail_train)}")
    print(f"Detail validation samples: {len(detail_val)}")
    print(f"Detail data: {DETAIL_OUTPUT_ROOT}")
    print(f"Detail dictionary: {DETAIL_DICT_PATH}")


if __name__ == "__main__":
    main()
