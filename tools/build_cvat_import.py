"""Create a CVAT YOLO 1.1 import package from a YOLO detection dataset.

The source labels are never changed.  Polygon rows are converted to their
tight, normalised detection boxes only in the generated annotation archive.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
ALL_SUBSETS = ("train", "valid", "test")


@dataclass(frozen=True)
class Sample:
    subset: str
    image: Path
    label_name: str
    annotations: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--subset", default="all", help="train, valid, test, or all (default).")
    parser.add_argument("--output", type=Path, help="Defaults to <dataset>/cvat_import/<subset>.")
    parser.add_argument(
        "--format",
        choices=("cvat-compatible", "ultralytics", "yolo1.1"),
        default="cvat-compatible",
        help="CVAT import format. cvat-compatible is the safest direct-upload ZIP.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def class_names(dataset: Path) -> list[str]:
    data = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    names = data.get("names") if isinstance(data, dict) else None
    if isinstance(names, list):
        result = [str(name) for name in names]
    elif isinstance(names, dict):
        result = [str(names[index]) for index in sorted(int(key) for key in names)]
    else:
        raise ValueError("data.yaml must contain names")
    if int(data.get("nc", len(result))) != len(result):
        raise ValueError("data.yaml nc does not match names")
    return result


def normalise_row(raw: str, class_count: int, location: str) -> tuple[str, bool] | None:
    values = raw.split()
    if not values:
        return None
    if len(values) < 5 or len(values) % 2 == 0:
        raise ValueError(f"{location}: expected class + xywh or polygon coordinate pairs")
    try:
        class_id = int(values[0])
        coordinates = [float(value) for value in values[1:]]
    except ValueError as error:
        raise ValueError(f"{location}: non-numeric label") from error
    if not 0 <= class_id < class_count:
        raise ValueError(f"{location}: invalid class {class_id}")
    polygon = len(values) != 5
    if polygon:
        xs, ys = coordinates[::2], coordinates[1::2]
        left, right = max(0.0, min(xs)), min(1.0, max(xs))
        top, bottom = max(0.0, min(ys)), min(1.0, max(ys))
    else:
        x, y, width, height = coordinates
        left, right = max(0.0, x - width / 2), min(1.0, x + width / 2)
        top, bottom = max(0.0, y - height / 2), min(1.0, y + height / 2)
    if right <= left or bottom <= top:
        raise ValueError(f"{location}: empty/out-of-bounds box")
    return f"{class_id} {(left + right) / 2:.8f} {(top + bottom) / 2:.8f} {right - left:.8f} {bottom - top:.8f}", polygon


def load_samples(dataset: Path, subsets: tuple[str, ...], class_count: int) -> tuple[list[Sample], list[Path], int]:
    samples: list[Sample] = []
    skipped: list[Path] = []
    polygons = 0
    seen_names: set[str] = set()
    for subset in subsets:
        images_dir, labels_dir = dataset / subset / "images", dataset / subset / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(f"Missing YOLO split folders: {images_dir}, {labels_dir}")
        for image in sorted(path for path in images_dir.iterdir() if path.is_file()):
            label = labels_dir / f"{image.stem}.txt"
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                skipped.append(image)
                continue
            if not label.is_file():
                raise FileNotFoundError(f"Missing label for {image.name}")
            if image.name in seen_names:
                raise ValueError(f"Duplicate image name across splits: {image.name}")
            seen_names.add(image.name)
            rows: list[str] = []
            for line_number, raw in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
                result = normalise_row(raw, class_count, f"{label}:{line_number}")
                if result is not None:
                    row, was_polygon = result
                    rows.append(row)
                    polygons += int(was_polygon)
            samples.append(Sample(subset, image, label.name, tuple(rows)))
    return samples, skipped, polygons


def build(output: Path, names: list[str], samples: list[Sample], skipped: list[Path], polygons: int, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output}; use --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "obj.names").write_text("\n".join(names) + "\n", encoding="utf-8")
    (output / "obj.data").write_text(f"classes = {len(names)}\nnames = obj.names\n", encoding="utf-8")
    with zipfile.ZipFile(output / "images.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for sample in samples:
            archive.write(sample.image, sample.image.name)
    with zipfile.ZipFile(output / "yolo_1.1_annotations.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(output / "obj.names", "obj.names")
        archive.write(output / "obj.data", "obj.data")
        for sample in samples:
            archive.writestr(sample.label_name, "\n".join(sample.annotations) + ("\n" if sample.annotations else ""))
    (output / "SKIPPED_FILES.txt").write_text(
        "Unsupported source files excluded from CVAT archives; convert to JPG/PNG then re-run.\n"
        + "\n".join(str(path) for path in skipped) + "\n",
        encoding="utf-8",
    )
    (output / "IMPORT_TO_CVAT.md").write_text(
        "# CVAT import\n\n"
        "1. Create an annotation task and upload `images.zip` as task data.\n"
        "2. Add the labels listed in `obj.names`, preserving their order.\n"
        "3. Upload `yolo_1.1_annotations.zip` using **YOLO 1.1**.\n\n"
        f"Validated {len(samples)} images. Converted {polygons} polygon labels to detection boxes only in this package.\n",
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(json.dumps({
        "images": len(samples), "classes": len(names), "polygon_rows_converted": polygons,
        "skipped_unsupported_images": [str(path) for path in skipped],
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def build_ultralytics(output: Path, names: list[str], samples: list[Sample], skipped: list[Path], polygons: int, overwrite: bool) -> None:
    """Create one Roboflow-compatible Ultralytics ZIP for CVAT upload."""

    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output}; use --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    present = {sample.subset for sample in samples}
    lines = ["path: ./"]
    if "train" in present:
        lines.append("train: train.txt")
    if "valid" in present:
        lines.append("val: valid.txt")
    if "test" in present:
        lines.append("test: test.txt")
    lines.extend(["nc: " + str(len(names)), "names:"])
    lines.extend(f"  {index}: {json.dumps(name, ensure_ascii=False)}" for index, name in enumerate(names))
    archive_path = output / "lao_plate_detect_ultralytics_yolo.zip"
    rows: dict[str, list[str]] = {subset: [] for subset in present}
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data.yaml", "\n".join(lines) + "\n")
        for sample in samples:
            image_name = f"{sample.subset}/images/{sample.image.name}"
            label_name = f"{sample.subset}/labels/{sample.label_name}"
            archive.write(sample.image, image_name)
            archive.writestr(label_name, "\n".join(sample.annotations) + ("\n" if sample.annotations else ""))
            rows[sample.subset].append(image_name)
        for subset, paths in rows.items():
            filename = "valid.txt" if subset == "valid" else f"{subset}.txt"
            archive.writestr(filename, "\n".join(paths) + "\n")
    (output / "IMPORT_TO_CVAT.md").write_text(
        "# Direct CVAT import\n\n"
        "Create/import a dataset in CVAT, select **Ultralytics YOLO** as the format, then upload `lao_plate_detect_ultralytics_yolo.zip`.\n"
        "The ZIP preserves Roboflow's `train/images`, `train/labels`, `valid/...`, and `test/...` layout.\n"
        f"Validated {len(samples)} images and converted {polygons} polygon labels to detection boxes inside the ZIP.\n",
        encoding="utf-8",
    )
    (output / "SKIPPED_FILES.txt").write_text(
        "Unsupported source files excluded from the CVAT archive; convert to JPG/PNG then re-run.\n"
        + "\n".join(str(path) for path in skipped) + "\n",
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(json.dumps({
        "format": "Ultralytics YOLO", "archive": archive_path.name, "images": len(samples),
        "classes": len(names), "polygon_rows_converted": polygons,
        "skipped_unsupported_images": [str(path) for path in skipped],
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def build_cvat_compatible(output: Path, names: list[str], samples: list[Sample], skipped: list[Path], polygons: int, overwrite: bool) -> None:
    """Build the conservative structure accepted by older CVAT Datumaro importers.

    These importers treat unknown ``data.yaml`` keys as dataset subsets.  In
    particular, an ``nc: 50`` field is interpreted as a subset with an integer
    value and causes ``'int' object is not iterable``.  Keep one train subset
    because CVAT annotation tasks do not require a train/valid/test split.
    """

    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output}; use --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    archive_path = output / "lao_plate_detect_cvat_compatible.zip"
    yaml_rows = ["train: train.txt", "names:"]
    yaml_rows.extend(f"  {index}: {json.dumps(name, ensure_ascii=False)}" for index, name in enumerate(names))
    train_rows: list[str] = []
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data.yaml", "\n".join(yaml_rows) + "\n")
        for sample in samples:
            image_name = f"train/images/{sample.image.name}"
            label_name = f"train/labels/{sample.label_name}"
            archive.write(sample.image, image_name)
            archive.writestr(label_name, "\n".join(sample.annotations) + ("\n" if sample.annotations else ""))
            train_rows.append(image_name)
        archive.writestr("train.txt", "\n".join(train_rows) + "\n")
    (output / "IMPORT_TO_CVAT.md").write_text(
        "# Direct CVAT import\n\n"
        "In CVAT choose **Ultralytics YOLO Detection** and upload `lao_plate_detect_cvat_compatible.zip`.\n"
        "This archive intentionally has one `train` subset and no `nc`, `path`, `val`, or `test` YAML fields, which avoids the Dataset Manager `int object is not iterable` importer error.\n"
        f"Validated {len(samples)} images and converted {polygons} polygon rows to detection boxes.\n",
        encoding="utf-8",
    )
    (output / "SKIPPED_FILES.txt").write_text(
        "Unsupported source files excluded from the CVAT archive; convert to JPG/PNG then re-run.\n"
        + "\n".join(str(path) for path in skipped) + "\n",
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(json.dumps({
        "format": "Ultralytics YOLO Detection (CVAT-compatible)", "archive": archive_path.name,
        "images": len(samples), "classes": len(names), "polygon_rows_converted": polygons,
        "skipped_unsupported_images": [str(path) for path in skipped],
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    subsets = ALL_SUBSETS if args.subset == "all" else (args.subset,)
    names = class_names(dataset)
    samples, skipped, polygons = load_samples(dataset, subsets, len(names))
    default_name = f"{args.subset}_{args.format}"
    output = (args.output or dataset / "cvat_import" / default_name).resolve()
    if args.format == "cvat-compatible":
        build_cvat_compatible(output, names, samples, skipped, polygons, args.overwrite)
    elif args.format == "ultralytics":
        build_ultralytics(output, names, samples, skipped, polygons, args.overwrite)
    else:
        build(output, names, samples, skipped, polygons, args.overwrite)
    print(f"Created CVAT package: {output}")
    print(f"Images={len(samples)} Classes={len(names)} Polygon labels converted={polygons} Skipped={len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
