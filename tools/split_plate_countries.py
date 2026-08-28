"""Separate cropped plate images into Thai and Lao groups.

The default mode follows the project's filename convention: a Lao Unicode
character in the filename means Lao; all other filenames are Thai. A model
mode remains available for diagnostics but is not used for the dataset split.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scan import (
    LAO_PROVINCE_NAMES,
    THAI_PROVINCE_NAMES,
    analyse_country_characters,
    character_kind,
    country_reading_score,
)


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split plate crops into Thai/Lao/uncertain folders.")
    parser.add_argument("--source", type=Path, default=ROOT / "auto" / "dataset" / "source" / "plate_crops")
    parser.add_argument("--full-source", type=Path, default=ROOT / "auto" / "dataset" / "source" / "full_vehicle")
    parser.add_argument("--output", type=Path, default=ROOT / "auto" / "dataset" / "countries")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--min-margin", type=float, default=0.12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("filename", "model"), default="filename")
    parser.add_argument("--clear", action="store_true")
    return parser.parse_args()


def model_items(result: Any, model: YOLO) -> list[Any]:
    from scan import CharacterDetection

    boxes = result.boxes
    if boxes is None:
        return []
    return [
        CharacterDetection(
            str(model.names[int(class_id)]),
            float(confidence),
            *map(float, xyxy),
        )
        for xyxy, confidence, class_id in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.conf.cpu().tolist(),
            boxes.cls.cpu().tolist(),
        )
    ]


def classify(
    image: Any,
    models: dict[str, YOLO],
    confidence: float,
    imgsz: int,
    min_margin: float,
    device: str,
) -> dict[str, Any]:
    readings: dict[str, dict[str, Any]] = {}
    evidence: dict[str, dict[str, int]] = {}
    for country, model in models.items():
        result = model.predict(image, conf=confidence, imgsz=imgsz, device=device, verbose=False)[0]
        items = model_items(result, model)
        reading = analyse_country_characters(items, country)
        reading["score"] = country_reading_score(reading, country)
        readings[country] = reading
        province_names = THAI_PROVINCE_NAMES if country == "thai" else LAO_PROVINCE_NAMES
        evidence[country] = {
            # A weak province box is not country evidence. In particular,
            # the Lao model sometimes sees VTE on a Thai plate background.
            "province": sum(
                item.text in province_names and item.confidence >= 0.45
                for item in items
            ),
            "character": sum(
                character_kind(item, country) == "character" and item.confidence >= 0.35
                for item in items
            ),
            "digits": sum(item.text.isdigit() and len(item.text) == 1 for item in items),
        }

    adjusted: dict[str, float] = {}
    for country in ("thai", "lao"):
        score = float(readings[country]["score"])
        score += min(0.30, evidence[country]["province"] * 0.15)
        score += min(0.18, evidence[country]["character"] * 0.06)
        adjusted[country] = round(score, 4)

    ranked = sorted(adjusted, key=adjusted.get, reverse=True)
    winner, runner = ranked[0], ranked[1]
    margin = adjusted[winner] - adjusted[runner]
    unique_province = (
        evidence[winner]["province"] > 0
        and evidence[runner]["province"] == 0
    )
    has_country_structure = evidence[winner]["digits"] >= 4 and (
        evidence[winner]["province"] > 0
        or evidence[winner]["character"] >= (1 if winner == "thai" else 2)
    )
    accepted = (
        adjusted[winner] >= 0.30
        and has_country_structure
        and (margin >= min_margin or unique_province)
    )
    country = winner if accepted else "uncertain"
    return {
        "country": country,
        "raw_winner": winner,
        "score_thai": adjusted["thai"],
        "score_lao": adjusted["lao"],
        "margin": round(margin, 4),
        "evidence": evidence,
        "readings": readings,
    }


def classify_by_filename(image_path: Path) -> dict[str, Any]:
    """Apply the dataset naming convention instead of guessing from pixels."""

    # Lao occupies U+0E80-U+0EFF. Numeric-only filenames in this dataset are
    # Thai plates, so every non-Lao filename belongs to the Thai group.
    country = "lao" if re.search(r"[\u0e80-\u0eff]", image_path.name) else "thai"
    return {
        "country": country,
        "raw_winner": "filename",
        "score_thai": 1.0 if country == "thai" else 0.0,
        "score_lao": 1.0 if country == "lao" else 0.0,
        "margin": 1.0,
        "evidence": {"source": "filename"},
        "readings": {},
    }


def copy_group(image: Path, full_source: Path, output: Path, country: str) -> None:
    target = output / "plate_crops" / country
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, target / image.name)
    full_name = re.sub(r"_ocr_crop(?=\.[^.]+$)", "_ocr_img", image.name)
    full_image = full_source / full_name
    if full_image.is_file():
        full_target = output / "full_vehicle" / country
        full_target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(full_image, full_target / full_image.name)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    full_source = args.full_source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if args.clear and output.exists():
        if output == ROOT or ROOT not in output.parents:
            raise ValueError(f"Unsafe output path: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    models = {}
    if args.mode == "model":
        models = {
            "thai": YOLO(str(ROOT / "model" / "thai_license_plate" / "weights" / "best.pt")),
            "lao": YOLO(str(ROOT / "model" / "lao_license_plate" / "weights" / "best.pt")),
        }
    images = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    rows: list[dict[str, Any]] = []
    counts = {"thai": 0, "lao": 0, "uncertain": 0}
    for index, image_path in enumerate(images, start=1):
        if args.mode == "filename":
            result = classify_by_filename(image_path)
        else:
            image = cv2.imread(str(image_path))
            if image is None:
                result = {"country": "uncertain", "raw_winner": "", "score_thai": 0.0, "score_lao": 0.0, "margin": 0.0, "evidence": {}, "readings": {}}
            else:
                result = classify(image, models, args.confidence, args.imgsz, args.min_margin, args.device)
        country = result["country"]
        counts[country] += 1
        if country in ("thai", "lao"):
            copy_group(image_path, full_source, output, country)
        else:
            uncertain = output / "plate_crops" / "uncertain"
            uncertain.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, uncertain / image_path.name)
        rows.append(
            {
                "file": image_path.name,
                "country": country,
                "raw_winner": result["raw_winner"],
                "score_thai": result["score_thai"],
                "score_lao": result["score_lao"],
                "margin": result["margin"],
                "thai_evidence": result["evidence"].get("thai", {}),
                "lao_evidence": result["evidence"].get("lao", {}),
            }
        )
        if index % 50 == 0 or index == len(images):
            print(f"processed {index}/{len(images)}: {counts}")

    manifest_json = output / "country_split_manifest.json"
    manifest_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "country_split_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["file", "country", "raw_winner", "score_thai", "score_lao", "margin", "thai_evidence", "lao_evidence"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"output={output}")
    print(f"counts={counts}")


if __name__ == "__main__":
    main()
