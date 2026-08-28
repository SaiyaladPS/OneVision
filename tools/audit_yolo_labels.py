"""Audit a YOLO detection dataset and render high-confidence label conflicts.

The script never changes labels. It reports structural issues, checks split
leakage by source name, and optionally compares labels with a trained model.
Review images show green ground truth and red model predictions.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--max-review", type=int, default=80)
    parser.add_argument("--splits", nargs="+", choices=("train", "valid", "test"), default=("train", "valid", "test"))
    return parser.parse_args()


def load_names(dataset: Path) -> list[str]:
    import yaml

    config = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    names = config.get("names", []) if isinstance(config, dict) else []
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    return [str(name) for name in names]


def source_key(path: Path) -> str:
    """Remove Roboflow's augmentation suffix to identify a source image."""

    return path.stem.split(".rf.", 1)[0].casefold()


def parse_labels(path: Path, class_count: int) -> tuple[list[dict[str, Any]], list[str]]:
    boxes: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            errors.append(f"{path}:{line_number}: expected 5 fields")
            continue
        try:
            class_id = int(values[0])
            x, y, width, height = (float(value) for value in values[1:])
        except ValueError:
            errors.append(f"{path}:{line_number}: non-numeric value")
            continue
        if class_id < 0 or class_id >= class_count:
            errors.append(f"{path}:{line_number}: invalid class {class_id}")
            continue
        if not (0 < width <= 1 and 0 < height <= 1 and 0 <= x <= 1 and 0 <= y <= 1):
            errors.append(f"{path}:{line_number}: invalid normalized box")
            continue
        if x - width / 2 < 0 or x + width / 2 > 1 or y - height / 2 < 0 or y + height / 2 > 1:
            errors.append(f"{path}:{line_number}: box exceeds image bounds")
            continue
        boxes.append({"class_id": class_id, "xywh": [x, y, width, height]})
    return boxes, errors


def to_xyxy(box: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, box_width, box_height = box
    return (
        (x - box_width / 2) * width,
        (y - box_height / 2) * height,
        (x + box_width / 2) * width,
        (y + box_height / 2) * height,
    )


def iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def draw_box(image: Any, box: tuple[float, float, float, float], text: str, colour: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
    cv2.putText(image, text, (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    names = load_names(dataset)
    audit_dir = dataset / "audit"
    audit_dir.mkdir(exist_ok=True)
    report: dict[str, Any] = {"dataset": str(dataset), "classes": names, "splits": {}, "errors": [], "duplicates": {}}
    source_splits: dict[str, set[str]] = defaultdict(set)
    records: list[dict[str, Any]] = []

    for split in args.splits:
        images_dir, labels_dir = dataset / split / "images", dataset / split / "labels"
        images = {path.stem: path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
        labels = {path.stem: path for path in labels_dir.glob("*.txt")}
        missing_labels = sorted(set(images) - set(labels))
        orphan_labels = sorted(set(labels) - set(images))
        report["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "missing_labels": missing_labels,
            "orphan_labels": orphan_labels,
            "class_boxes": Counter(),
        }
        for stem in sorted(set(images) & set(labels)):
            boxes, errors = parse_labels(labels[stem], len(names))
            report["errors"].extend(errors)
            report["splits"][split]["class_boxes"].update(names[box["class_id"]] for box in boxes)
            records.append({"split": split, "image": images[stem], "label": labels[stem], "boxes": boxes})
            source_splits[source_key(images[stem])].add(split)

    report["duplicates"] = {
        source: sorted(splits) for source, splits in source_splits.items() if len(splits) > 1
    }
    for split in report["splits"].values():
        split["class_boxes"] = dict(sorted(split["class_boxes"].items()))

    review: list[dict[str, Any]] = []
    if args.weights:
        from ultralytics import YOLO

        model = YOLO(str(args.weights.resolve()))
        for index, record in enumerate(records, start=1):
            result = model.predict(str(record["image"]), imgsz=args.imgsz, conf=args.confidence, verbose=False)[0]
            image = cv2.imread(str(record["image"]))
            if image is None:
                report["errors"].append(f"cannot read image: {record['image']}")
                continue
            height, width = image.shape[:2]
            truth = [
                {"class_id": box["class_id"], "box": to_xyxy(box["xywh"], width, height)}
                for box in record["boxes"]
            ]
            predictions: list[dict[str, Any]] = []
            if result.boxes is not None:
                for box, score, class_id in zip(
                    result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()
                ):
                    predictions.append({"class_id": int(class_id), "confidence": float(score), "box": tuple(box)})
            conflicts: list[dict[str, Any]] = []
            for expected in truth:
                overlaps = [(iou(expected["box"], predicted["box"]), predicted) for predicted in predictions]
                best_iou, prediction = max(overlaps, key=lambda item: item[0], default=(0.0, None))
                if prediction and best_iou >= 0.5 and prediction["class_id"] != expected["class_id"] and prediction["confidence"] >= 0.75:
                    conflicts.append({"type": "class_conflict", "truth": expected, "prediction": prediction, "iou": round(best_iou, 4)})
            if conflicts:
                review.append({"record": record, "conflicts": conflicts, "image": image, "truth": truth, "predictions": predictions})
            if index % 100 == 0:
                print(f"Compared {index}/{len(records)} images")

    review.sort(key=lambda item: max(conflict["prediction"]["confidence"] for conflict in item["conflicts"]), reverse=True)
    review_dir = audit_dir / "review"
    if review_dir.exists():
        shutil.rmtree(review_dir)
    review_dir.mkdir(exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for number, item in enumerate(review[: args.max_review], start=1):
        image = item["image"].copy()
        for truth in item["truth"]:
            draw_box(image, truth["box"], f"GT:{names[truth['class_id']]}", (0, 200, 0))
        for prediction in item["predictions"]:
            draw_box(image, prediction["box"], f"P:{names[prediction['class_id']]} {prediction['confidence']:.2f}", (0, 0, 255))
        output = review_dir / f"{number:03d}_{item['record']['split']}_{item['record']['image'].name}"
        cv2.imwrite(str(output), image)
        manifest.append(
            {
                "review_image": str(output),
                "split": item["record"]["split"],
                "image": str(item["record"]["image"]),
                "label": str(item["record"]["label"]),
                "conflicts": item["conflicts"],
            }
        )
    report["model_review"] = {"conflicts": len(review), "rendered": manifest}
    (audit_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"errors": len(report["errors"]), "split_duplicates": len(report["duplicates"]), "model_conflicts": len(review), "rendered": len(manifest)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
