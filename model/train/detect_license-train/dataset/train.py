"""Fine-tune the licence-plate detector from the existing ``best.pt``.

By default ``--dataset all`` combines every local one-class plate dataset that
can share the deployed ``License_Plate`` head: Roboflow folders, the native
``detect_license`` layout, and CVAT YOLO 1.1 exports (``obj.names``). Source
class names such as ``license_plate`` are accepted as aliases and rewritten to
``License_Plate`` only in the generated Ultralytics manifest.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import os
import re
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils import LOGGER as ULTRALYTICS_LOGGER


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent
MODEL_ROOT = SCRIPT_DIR.parents[3]
DEPLOYED_DIR = MODEL_ROOT / "detect_license"
RUNS_DIR = TRAINING_DIR / "runs"
COMBINED_DATA = SCRIPT_DIR / "combined_data.generated.yaml"
DEFAULT_DATASET = "all"
CANONICAL_CLASS = "License_Plate"
CLASS_ALIASES = {
    "license_plate",
    "license-plate",
    "licence_plate",
    "licence-plate",
    "licenseplate",
    "licenceplate",
}
DATASET_ALIASES = {
    "lpr-2": "Lpr",
    "lpr2": "Lpr",
}
SPLIT_CANDIDATES = {
    "train": ("train", "Train"),
    "val": ("val", "valid", "Validation", "validation"),
    "test": ("test", "Test"),
}
IMAGE_DIR_LAYOUTS = {
    "train": ("train/images", "images/Train", "images/train", "images"),
    "val": ("valid/images", "val/images", "images/Validation", "images/valid", "images/val"),
    "test": ("test/images", "images/Test", "images/test"),
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SKIP_DIR_NAMES = {
    "__pycache__",
    ".ultralytics_view",
    "runs",
    "weights",
    "images",
    "labels",
    "obj_train_data",
    "train",
    "valid",
    "val",
    "test",
}
CVAT_VIEW_DIR = ".ultralytics_view"


class TeeStream:
    """Write training output to the terminal and its persistent log file."""

    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, message: str) -> int:
        for stream in self.streams:
            stream.write(message)
        return len(message)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def training_log_path(log_dir: Path, run_name: str) -> Path:
    """Create a timestamped, filesystem-safe log path for one train invocation."""

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name).strip("._") or "training"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{safe_name}_{timestamp}.log"


def find_default_weights() -> Path:
    """Prefer the deployed detect_license checkpoint, then the local training copy."""

    candidates = (
        DEPLOYED_DIR / "weights" / "best.pt",
        TRAINING_DIR / "weights" / "best.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


WEIGHTS = find_default_weights()


def _normalize_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, list):
        return [str(name) for name in names]
    return []


def _is_plate_class(name: str) -> bool:
    compact = name.strip().lower().replace(" ", "_").replace("-", "_")
    return compact == CANONICAL_CLASS.lower() or compact.replace("_", "") in {
        alias.replace("-", "").replace("_", "") for alias in CLASS_ALIASES
    } or compact in {alias.replace("-", "_") for alias in CLASS_ALIASES}


def is_compatible_plate_dataset(config: dict[str, Any]) -> bool:
    names = _normalize_names(config.get("names", []))
    raw_nc = config.get("nc", len(names))
    try:
        class_count = int(raw_nc)
    except (TypeError, ValueError):
        class_count = len(names)
    return class_count == 1 and len(names) == 1 and _is_plate_class(names[0])


def _read_name_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _is_cvat_leaf(path: Path) -> bool:
    if not path.is_dir() or not (path / "obj.names").is_file():
        return False
    return (
        (path / "images").is_dir()
        or (path / "labels").is_dir()
        or (path / "obj_train_data").is_dir()
        or (path / "train.txt").is_file()
    )


def _is_leaf_dataset(path: Path) -> bool:
    if not path.is_dir() or path.name.startswith(".") or path.name in SKIP_DIR_NAMES:
        return False
    if (path / "data.yaml").is_file():
        return True
    if (path / "images" / "Train").is_dir() and (path / "labels" / "Train").is_dir():
        return True
    return _is_cvat_leaf(path)


def _looks_like_dataset(path: Path) -> bool:
    return _is_leaf_dataset(path) or bool(list(_iter_dataset_leaves(path, include_reject=True, max_depth=3)))


def _iter_dataset_leaves(root: Path, *, include_reject: bool, max_depth: int, depth: int = 0) -> Iterable[Path]:
    if not root.is_dir():
        return
    if _is_leaf_dataset(root):
        yield root
        return
    if depth >= max_depth:
        return
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.startswith(".") or path.name in SKIP_DIR_NAMES:
            continue
        if not include_reject and path.name.lower() == "reject":
            continue
        yield from _iter_dataset_leaves(path, include_reject=include_reject, max_depth=max_depth, depth=depth + 1)


def _canonical_plate_config(original_names: list[str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    config = {
        "nc": 1,
        "names": [CANONICAL_CLASS],
        "original_names": original_names,
    }
    if extra:
        config.update(extra)
    return config


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def materialize_cvat_image_dir(dataset_dir: Path) -> Path | None:
    """Expose a mixed CVAT ``obj_train_data`` folder as Ultralytics images/labels."""

    mixed = dataset_dir / "obj_train_data"
    if not mixed.is_dir():
        return None
    images_already = dataset_dir / "images"
    if images_already.is_dir() and (dataset_dir / "labels").is_dir():
        return images_already
    view = dataset_dir / CVAT_VIEW_DIR / "train"
    image_dir = view / "images"
    label_dir = view / "labels"
    for item in mixed.iterdir():
        if not item.is_file():
            continue
        if item.suffix.lower() in IMAGE_SUFFIXES:
            _link_or_copy(item, image_dir / item.name)
            label = mixed / f"{item.stem}.txt"
            if label.is_file():
                _link_or_copy(label, label_dir / label.name)
    if not image_dir.is_dir() or not any(image_dir.iterdir()):
        return None
    return image_dir


def _read_cvat_config(dataset_dir: Path) -> dict[str, Any]:
    names = _read_name_list(dataset_dir / "obj.names")
    extra: dict[str, Any] = {"format": "cvat-yolo1.1"}
    if (dataset_dir / "images").is_dir():
        extra["train"] = "images"
        if (dataset_dir / "images" / "Validation").is_dir():
            extra["val"] = "images/Validation"
        if (dataset_dir / "images" / "Test").is_dir():
            extra["test"] = "images/Test"
    generated = materialize_cvat_image_dir(dataset_dir)
    if generated is not None and "train" not in extra:
        extra["train"] = str(generated.relative_to(dataset_dir)).replace("\\", "/")
    return _canonical_plate_config(names, extra)


def _read_dataset_config(dataset_dir: Path) -> dict[str, Any]:
    config_path = dataset_dir / "data.yaml"
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        if not isinstance(config, dict):
            raise ValueError(f"{config_path} must contain a YAML mapping")
        names = _normalize_names(config.get("names", []))
        if is_compatible_plate_dataset(config):
            config = {**config, **_canonical_plate_config(names)}
        return config
    if _is_cvat_leaf(dataset_dir):
        return _read_cvat_config(dataset_dir)
    if (dataset_dir / "images" / "Train").is_dir():
        return _canonical_plate_config(
            [CANONICAL_CLASS],
            {"train": "images/Train", "val": "images/Validation", "test": "images/Test"},
        )
    raise FileNotFoundError(f"Dataset config not found: {config_path}")


def discover_datasets(root: Path | None = None, *, include_reject: bool = False) -> list[Path]:
    """Find Roboflow, native detect_license, and CVAT YOLO 1.1 plate datasets."""

    search_root = (root or SCRIPT_DIR).resolve()
    datasets = sorted(
        {
            path.resolve()
            for path in _iter_dataset_leaves(search_root, include_reject=include_reject, max_depth=3)
        }
    )
    if not datasets:
        raise FileNotFoundError(f"No YOLO or CVAT datasets were found in {search_root}")
    return datasets


def select_datasets(dataset_name: str, root: Path | None = None) -> list[Path]:
    """Select one dataset (or a CVAT wrapper), or all compatible plate datasets."""

    search_root = (root or SCRIPT_DIR).resolve()
    requested = dataset_name.strip()
    include_reject = requested.lower() == "reject"
    datasets = discover_datasets(search_root, include_reject=include_reject)
    if requested.lower() == "all":
        selected: list[Path] = []
        skipped: list[str] = []
        for dataset in datasets:
            config = _read_dataset_config(dataset)
            if is_compatible_plate_dataset(config):
                selected.append(dataset)
            else:
                skipped.append(str(dataset.relative_to(search_root)) if dataset.is_relative_to(search_root) else dataset.name)
        if skipped:
            print(
                "Skipping datasets that are not a one-class plate schema: "
                + ", ".join(skipped)
            )
        if not selected:
            available = ", ".join(path.name for path in datasets)
            raise ValueError(
                f"No compatible one-class plate datasets found in {search_root}. "
                f"Available folders: {available}"
            )
        return selected

    alias = DATASET_ALIASES.get(requested.lower(), requested)
    selected_dir = search_root / alias
    if not selected_dir.is_dir():
        matches = [path for path in datasets if path.name.lower() == alias.lower()]
        if len(matches) == 1:
            selected_dir = matches[0]
        else:
            available = ", ".join(path.name for path in datasets)
            raise FileNotFoundError(
                f"Dataset '{dataset_name}' was not found in {search_root}. "
                f"Available datasets: {available}"
            )
    leaves = list(_iter_dataset_leaves(selected_dir, include_reject=include_reject, max_depth=3))
    if not leaves:
        raise FileNotFoundError(f"Dataset '{dataset_name}' has no train-ready YOLO or CVAT files")
    selected: list[Path] = []
    for leaf in leaves:
        config = _read_dataset_config(leaf)
        if not is_compatible_plate_dataset(config):
            raise ValueError(
                f"{leaf} must be a one-class plate dataset "
                f"(License_Plate or license_plate). Found names={config.get('names')!r}"
            )
        selected.append(leaf)
    return selected


def _existing_dir(dataset_dir: Path, value: str) -> Path | None:
    path = (dataset_dir / value).resolve()
    if path.is_dir():
        return path
    if value.startswith("../"):
        fallback = (dataset_dir / value[3:]).resolve()
        if fallback.is_dir():
            return fallback
    return None


def resolve_split(dataset_dir: Path, config: dict[str, Any], split: str) -> Path:
    """Resolve train/val/test image folders for Roboflow, detect_license, and CVAT."""

    keys = SPLIT_CANDIDATES[split]
    candidates: list[str] = []
    for key in keys:
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    candidates.extend(IMAGE_DIR_LAYOUTS[split])
    if split == "train":
        candidates.append(f"{CVAT_VIEW_DIR}/train/images")

    for value in candidates:
        if value.lower().endswith(".txt"):
            continue
        found = _existing_dir(dataset_dir, value)
        if found is not None:
            return found

    if split == "train":
        generated = materialize_cvat_image_dir(dataset_dir)
        if generated is not None:
            return generated

    raise FileNotFoundError(
        f"Dataset split '{split}' does not exist under {dataset_dir}. "
        "Expected Roboflow train/images, detect_license images/Train, or CVAT images/obj_train_data."
    )


def build_combined_data(datasets: list[Path], output: Path | None = None) -> Path:
    """Create one YOLO data file whose splits contain every selected dataset."""

    train_paths: list[str] = []
    val_paths: list[str] = []
    test_paths: list[str] = []
    for dataset in datasets:
        config = _read_dataset_config(dataset)
        train_path = str(resolve_split(dataset, config, "train")).replace("\\", "/")
        train_paths.append(train_path)
        try:
            val_paths.append(str(resolve_split(dataset, config, "val")).replace("\\", "/"))
        except FileNotFoundError:
            val_paths.append(train_path)
        try:
            test_paths.append(str(resolve_split(dataset, config, "test")).replace("\\", "/"))
        except FileNotFoundError:
            pass

    combined = {
        "train": train_paths,
        "val": val_paths,
        "nc": 1,
        "names": [CANONICAL_CLASS],
    }
    if test_paths:
        combined["test"] = test_paths
    manifest = output or COMBINED_DATA
    with manifest.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(combined, stream, sort_keys=False, allow_unicode=True)
    return manifest


def select_device(requested: str | None) -> str | int:
    if requested:
        return int(requested) if requested.isdigit() else requested
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu:0"
    if torch.cuda.is_available():
        return 0
    return "cpu"


def _completed_epochs(run_dir: Path) -> int | None:
    """Return the number of completed epochs recorded for a training run."""

    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        return None

    try:
        with results_path.open("r", encoding="utf-8", newline="") as stream:
            rows = csv.DictReader(stream)
            epochs = [int(float(row["epoch"])) for row in rows if row.get("epoch")]
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return max(epochs) + 1 if epochs else None


def find_incomplete_runs(name: str) -> list[Path]:
    """Find unfinished runs with a usable ``last.pt`` checkpoint."""

    if not RUNS_DIR.is_dir():
        return []

    incomplete: list[Path] = []
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir() or not (run_dir / "weights" / "last.pt").is_file():
            continue

        args_path = run_dir / "args.yaml"
        try:
            with args_path.open("r", encoding="utf-8") as stream:
                run_args = yaml.safe_load(stream) or {}
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(run_args, dict) or run_args.get("name") != name:
            continue

        target_epochs = run_args.get("epochs")
        completed = _completed_epochs(run_dir)
        try:
            finished = completed is not None and completed >= int(target_epochs)
        except (TypeError, ValueError):
            finished = False
        if not finished:
            incomplete.append(run_dir)

    return sorted(incomplete, key=lambda path: path.stat().st_mtime, reverse=True)


def _run_number(path: Path, base_name: str) -> int | None:
    if path.name == base_name:
        return 0
    match = re.fullmatch(re.escape(base_name) + r"_(\d+)", path.name)
    return int(match.group(1)) if match else None


def latest_run(name: str) -> Path | None:
    """Return the newest numbered run with a usable last checkpoint."""
    candidates: list[Path] = []
    if RUNS_DIR.is_dir():
        for path in RUNS_DIR.iterdir():
            if _run_number(path, name) is None:
                continue
            checkpoint = path / "weights" / "last.pt"
            if path.is_dir() and checkpoint.is_file() and checkpoint.stat().st_size > 0:
                candidates.append(path)
    return max(candidates, key=lambda path: (_run_number(path, name) or 0, path.stat().st_mtime), default=None)


def next_run_name(name: str) -> str:
    """Return ``name`` or the next available ``name_N`` without overwriting runs."""
    if not (RUNS_DIR / name).exists():
        return name
    index = 1
    while (RUNS_DIR / f"{name}_{index}").exists():
        index += 1
    return f"{name}_{index}"


def choose_resume_run(runs: list[Path]) -> Path | None:
    """Ask whether to resume the newest unfinished run or start a new one."""

    if not runs:
        return None

    print("พบการ train ที่ยังไม่เสร็จ:")
    for run in runs:
        print(f"  - {run}")
    print("เลือก train ต่อเพื่อใช้ checkpoint เดิม หรือเริ่มใหม่เพื่อสร้าง run ใหม่")
    while True:
        try:
            answer = input("Train ต่อ (c) หรือเริ่มใหม่ (n)? [c]: ").strip().lower()
        except EOFError as error:
            raise RuntimeError(
                "An unfinished training run was found, but no interactive input is available. "
                "Run the script interactively and choose resume or restart."
            ) from error

        if answer in {"", "c", "continue", "resume", "ต่อ", "y", "yes"}:
            return runs[0]
        if answer in {"n", "new", "restart", "เริ่ม", "เริ่มใหม่", "no"}:
            return None
        print("กรุณาเลือก c เพื่อ train ต่อ หรือ n เพื่อเริ่มใหม่")


def default_run_name(dataset_name: str) -> str:
    if dataset_name.strip().lower() == "all":
        return "detect_license_all_finetune"
    slug = re.sub(r"[^A-Za-z0-9]+", "_", dataset_name).strip("_").lower() or "dataset"
    return f"detect_license_{slug}_finetune"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune detect_license from best.pt on local YOLO datasets."
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv("CAR_SCAN_TRAIN_DATASET", DEFAULT_DATASET),
        help="Dataset folder, CVAT wrapper, or 'all' to combine Roboflow, detect_license, and CVAT plate sets.",
    )
    parser.add_argument("--epochs", type=int, default=int(os.getenv("CAR_SCAN_TRAIN_EPOCHS", "50")))
    parser.add_argument("--batch", type=int, default=int(os.getenv("CAR_SCAN_TRAIN_BATCH", "16")))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu, 0, 1, or xpu:0")
    parser.add_argument("--name", default=None, help="Ultralytics run name under detect_license-train/runs.")
    parser.add_argument("--new-run", action="store_true", help="Start a fresh numbered run instead of resuming the latest run.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=RUNS_DIR / "training_logs",
        help="Directory for timestamped training console logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.name = args.name or default_run_name(args.dataset)
    resume_mode = os.getenv("CAR_SCAN_TRAIN_RESUME", "auto").strip().lower()
    start_new = args.new_run or resume_mode in {"no", "false", "0", "new"}
    latest = None if start_new else latest_run(args.name)
    if latest is None:
        args.name = next_run_name(args.name)
    elif latest is not None:
        args.name = latest.name
    log_path = training_log_path(args.log_dir, args.name)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        ultralytics_handler = logging.StreamHandler(log_file)
        ultralytics_handler.setFormatter(logging.Formatter("%(message)s"))
        ULTRALYTICS_LOGGER.addHandler(ultralytics_handler)
        stdout = TeeStream(sys.stdout, log_file)
        stderr = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            print(f"Training log: {log_path}")
            print(f"Started: {datetime.now().astimezone().isoformat(timespec='seconds')}")
            print(f"Arguments: {vars(args)}")
            try:
                if args.epochs < 1 or args.batch < 1 or args.imgsz < 32:
                    raise ValueError("epochs, batch, and imgsz must be positive (imgsz >= 32)")

                incomplete_runs = find_incomplete_runs(args.name)
                if resume_mode in {"yes", "true", "1"}:
                    requested_checkpoint = Path(os.getenv("CAR_SCAN_TRAIN_CHECKPOINT", ""))
                    if requested_checkpoint.is_file() and requested_checkpoint.name == "last.pt":
                        resume_run = requested_checkpoint.parent.parent
                    elif incomplete_runs:
                        resume_run = incomplete_runs[0]
                    else:
                        raise FileNotFoundError("No unfinished run with last.pt is available to resume")
                elif start_new:
                    resume_run = None
                else:
                    resume_run = latest or (incomplete_runs[0] if incomplete_runs else None)
                resume_checkpoint = resume_run / "weights" / "last.pt" if resume_run else None
                if resume_checkpoint is None and not WEIGHTS.is_file():
                    raise FileNotFoundError(f"Starting weights not found: {WEIGHTS}")

                datasets = select_datasets(args.dataset)
                data_path = build_combined_data(datasets)
                device = select_device(args.device)
                print("Datasets:")
                for dataset in datasets:
                    print(f"  - {dataset}")
                print(f"Training data manifest: {data_path}")
                print(f"Deployed detect_license weights: {DEPLOYED_DIR / 'weights' / 'best.pt'}")
                if resume_checkpoint:
                    print(f"Resuming from checkpoint: {resume_checkpoint}")
                else:
                    print(f"Starting weights: {WEIGHTS}")
                print(f"Device: {device}")

                model = YOLO(str(resume_checkpoint or WEIGHTS))
                model.train(
                    data=str(data_path),
                    epochs=args.epochs,
                    patience=20,
                    batch=args.batch,
                    imgsz=args.imgsz,
                    workers=args.workers,
                    device=device,
                    project=str(RUNS_DIR),
                    name=resume_run.name if resume_run else args.name,
                    exist_ok=resume_run is not None,
                    save=True,
                    save_period=5,
                    close_mosaic=10,
                    lr0=0.001,
                    lrf=0.01,
                    warmup_epochs=1.0,
                    optimizer="AdamW",
                    pretrained=False,
                    resume=bool(resume_checkpoint),
                )
            except BaseException:
                traceback.print_exc()
                print(f"Training failed: {datetime.now().astimezone().isoformat(timespec='seconds')}")
                raise
            else:
                print(f"Training completed: {datetime.now().astimezone().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
