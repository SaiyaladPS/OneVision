"""Convert mixed YOLO segmentation/detection labels into detection labels.

Segment polygons are converted to their tight, clipped axis-aligned bounding
boxes. The original files are backed up before any write. Source-image split
leaks (Roboflow filename before ``.rf.``) can be moved from validation to
training so validation remains independent.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--apply", action="store_true", help="write converted labels and move leaked valid samples")
    return parser.parse_args()


def source_key(path: Path) -> str:
    return path.stem.split(".rf.", 1)[0].casefold()


def normalise_line(line: str, class_count: int) -> tuple[str | None, str | None, bool]:
    values = line.split()
    if not values:
        return "", None, False
    if len(values) < 5 or len(values) % 2 == 0:
        return None, "expected class plus xywh or polygon pairs", False
    try:
        class_id = int(values[0])
        coordinates = [float(value) for value in values[1:]]
    except ValueError:
        return None, "non-numeric label", False
    if class_id < 0 or class_id >= class_count:
        return None, f"invalid class {class_id}", False
    if len(values) == 5:
        x, y, width, height = coordinates
        changed = False
    else:
        xs, ys = coordinates[::2], coordinates[1::2]
        left, right = max(0.0, min(xs)), min(1.0, max(xs))
        top, bottom = max(0.0, min(ys)), min(1.0, max(ys))
        x, y = (left + right) / 2, (top + bottom) / 2
        width, height = right - left, bottom - top
        changed = True
    left, right = max(0.0, x - width / 2), min(1.0, x + width / 2)
    top, bottom = max(0.0, y - height / 2), min(1.0, y + height / 2)
    if right <= left or bottom <= top:
        return None, "empty box after clipping", changed
    normalised = ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)
    changed = changed or any(
        abs(original - converted) > 1e-9
        for original, converted in zip((x, y, width, height), normalised)
    )
    result = f"{class_id} {normalised[0]:.8f} {normalised[1]:.8f} {normalised[2]:.8f} {normalised[3]:.8f}"
    return result, None, changed


def class_count(dataset: Path) -> int:
    import yaml

    data = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    names = data.get("names", []) if isinstance(data, dict) else []
    return len(names)


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    count = class_count(dataset)
    audit_dir = dataset / "audit"
    backup_dir = audit_dir / "original_segment_labels"
    report: dict[str, object] = {"converted_files": [], "converted_lines": 0, "invalid": [], "moved_from_valid": []}

    for split in SPLITS:
        labels_dir = dataset / split / "labels"
        for label_path in sorted(labels_dir.glob("*.txt")):
            output: list[str] = []
            changed = False
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                normalised, error, was_polygon = normalise_line(line, count)
                if error:
                    report["invalid"].append(f"{label_path.relative_to(dataset)}:{line_number}: {error}")
                    continue
                if normalised:
                    output.append(normalised)
                changed = changed or was_polygon
                if was_polygon:
                    report["converted_lines"] = int(report["converted_lines"]) + 1
            if changed:
                report["converted_files"].append(str(label_path.relative_to(dataset)))
                if args.apply:
                    backup_path = backup_dir / label_path.relative_to(dataset)
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(label_path, backup_path)
                    label_path.write_text("\n".join(output) + ("\n" if output else ""), encoding="utf-8")

    source_splits: dict[str, dict[str, list[Path]]] = {}
    for split in SPLITS:
        for image_path in (dataset / split / "images").glob("*"):
            if not image_path.is_file():
                continue
            source_splits.setdefault(source_key(image_path), {}).setdefault(split, []).append(image_path)
    for source, locations in sorted(source_splits.items()):
        if "train" not in locations or "valid" not in locations:
            continue
        for image_path in locations["valid"]:
            label_path = dataset / "valid" / "labels" / f"{image_path.stem}.txt"
            destination_image = dataset / "train" / "images" / image_path.name
            destination_label = dataset / "train" / "labels" / label_path.name
            report["moved_from_valid"].append(
                {"source": source, "image": str(image_path.relative_to(dataset)), "label": str(label_path.relative_to(dataset))}
            )
            if args.apply:
                if destination_image.exists() or destination_label.exists():
                    raise FileExistsError(f"destination already exists: {destination_image}")
                shutil.move(str(image_path), str(destination_image))
                shutil.move(str(label_path), str(destination_label))

    audit_dir.mkdir(exist_ok=True)
    report_path = audit_dir / "normalisation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: len(value) if isinstance(value, list) else value for key, value in report.items()}, ensure_ascii=False))
    print(f"Report: {report_path}")
    return 0 if not report["invalid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
