"""Train or resume the Vietnamese licence-plate character detector.

The default ``--resume auto`` continues from ``runs/<name>/weights/last.pt``.
It restores optimizer state when available; stripped checkpoints continue from
their model weights and saved epoch with a newly initialized optimizer.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
GENERATED_DATA = SCRIPT_DIR / "vietnam_plate_data.generated.yaml"
DOWNLOAD_SCRIPT = SCRIPT_DIR / "download_roboflow.py"
RUNS_DIR = TRAINING_DIR / "runs"
SKIP_DIR_NAMES = {"__pycache__", "runs", "weights", ".ultralytics_view"}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IMAGE_DIR_LAYOUTS = {
    "train": ("images/train/obj_train_data", "images/Train/obj_train_data", "images/train", "images/Train", "train/images", "obj_train_data"),
    "val": ("images/valid/obj_train_data", "images/val/obj_train_data", "images/valid", "images/val", "valid/images", "val/images"),
    "test": ("images/test/obj_train_data", "images/test", "test/images"),
}


class ResumableDetectionTrainer(DetectionTrainer):
    """Respect a caller supplied epoch target when resuming a run."""

    def check_resume(self, overrides: dict[str, Any]) -> None:
        super().check_resume(overrides)
        if self.resume:
            checkpoint = Path(self.args.model).expanduser().resolve()
            run_dir = checkpoint.parent.parent
            # Ultralytics may increment an existing run name (for example
            # ``run-2``) when the checkpoint's saved args have exist_ok=False.
            # Pin resume output back to the exact directory holding last.pt.
            self.args.save_dir = str(run_dir)
            self.args.name = run_dir.name
            self.args.exist_ok = True
            if "epochs" in overrides:
                self.args.epochs = int(overrides["epochs"])

    def resume_training(self, ckpt: dict[str, Any] | None) -> None:
        """Resume model/epoch state even when a checkpoint lacks optimizer state."""

        if ckpt is not None and self.resume and int(ckpt.get("epoch", -1)) < 0:
            checkpoint = Path(self.args.model).expanduser().resolve()
            completed = completed_epochs(checkpoint.parent.parent)
            if completed is None:
                raise ValueError(
                    f"{self.args.model} has no saved epoch or usable results.csv; "
                    "cannot determine where to resume."
                )
            ckpt["epoch"] = completed - 1
        if ckpt is not None and self.resume and ckpt.get("optimizer") is None:
            print("Checkpoint has no optimizer state; resuming its weights/epoch with a fresh optimizer.")
        super().resume_training(ckpt)


def _normalize_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        def key(item: Any) -> tuple[int, int | str]:
            try:
                return (0, int(item))
            except (TypeError, ValueError):
                return (1, str(item))
        return [str(names[item]) for item in sorted(names, key=key)]
    return [str(name) for name in names] if isinstance(names, (list, tuple)) else []


def _has_images(path: Path) -> bool:
    return path.is_dir() and any(item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES for item in path.iterdir())


def _is_dataset_dir(path: Path) -> bool:
    return path.is_dir() and path.name not in SKIP_DIR_NAMES and ((path / "data.yaml").is_file() or ((path / "images").is_dir() and (path / "labels").is_dir()))


def discover_datasets() -> list[Path]:
    return [path.resolve() for path in sorted(SCRIPT_DIR.iterdir(), key=lambda path: path.name.lower()) if _is_dataset_dir(path)]


def read_dataset_config(dataset_dir: Path) -> dict[str, Any]:
    config_path = dataset_dir / "data.yaml"
    config: dict[str, Any] = {}
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        if not isinstance(config, dict):
            raise ValueError(f"{config_path} must contain a YAML mapping")
    names = _normalize_names(config.get("names", []))
    nc = int(config.get("nc", len(names) or 0))
    if not names or nc != len(names):
        raise ValueError(f"{config_path} must declare matching nc and names")
    config["names"], config["nc"] = names, nc
    return config


def _existing_image_dir(dataset_dir: Path, value: str) -> Path | None:
    if value.lower().endswith(".txt"):
        return None
    path = Path(value)
    candidates = [path] if path.is_absolute() else [(dataset_dir / value).resolve()]
    if value.startswith("../"):
        candidates.append((dataset_dir / value[3:]).resolve())
    return next((candidate for candidate in candidates if _has_images(candidate)), None)


def resolve_split(dataset_dir: Path, config: dict[str, Any], split: str) -> Path:
    candidates: list[str] = []
    for key in (split, "valid" if split == "val" else split):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    candidates.extend(IMAGE_DIR_LAYOUTS[split])
    for value in candidates:
        found = _existing_image_dir(dataset_dir, value)
        if found:
            return found
    raise FileNotFoundError(f"Dataset split '{split}' was not found under {dataset_dir}")


def _has_usable_train(dataset_dir: Path) -> bool:
    try:
        resolve_split(dataset_dir, read_dataset_config(dataset_dir), "train")
        return True
    except (FileNotFoundError, ValueError):
        return False


def select_datasets(dataset_name: str) -> list[Path]:
    requested, datasets = dataset_name.strip(), discover_datasets()
    if requested.lower() == "all":
        selected = [path for path in datasets if _has_usable_train(path)]
        if selected:
            return selected
        raise FileNotFoundError(f"No Vietnamese training datasets were found in {SCRIPT_DIR}")
    selected = Path(requested)
    if not selected.is_absolute():
        selected = SCRIPT_DIR / requested
    if not selected.is_dir():
        matches = [path for path in datasets if path.name.lower() == requested.lower()]
        if len(matches) != 1:
            available = ", ".join(path.name for path in datasets) or "(none)"
            raise FileNotFoundError(f"Dataset '{dataset_name}' was not found. Available: {available}")
        selected = matches[0]
    if not _has_usable_train(selected):
        raise FileNotFoundError(f"Dataset '{selected}' has no usable train images")
    return [selected.resolve()]


def download_dataset() -> list[Path]:
    if not DOWNLOAD_SCRIPT.is_file() or not DOWNLOAD_SCRIPT.read_text(encoding="utf-8").strip() or not os.getenv("ROBOFLOW_API_KEY", "").strip():
        return []
    print("Vietnamese dataset not found locally; downloading from Roboflow...")
    result = subprocess.run([sys.executable, str(DOWNLOAD_SCRIPT)], cwd=SCRIPT_DIR, check=False)
    return discover_datasets() if result.returncode == 0 else []


def ensure_datasets(dataset_name: str) -> list[Path]:
    try:
        return select_datasets(dataset_name)
    except FileNotFoundError:
        if dataset_name.strip().lower() != "all" or not download_dataset():
            raise
    return select_datasets(dataset_name)


def repair_dataset_caches(datasets: list[Path]) -> None:
    """Move truncated Ultralytics label caches aside so they can be rebuilt.

    ``cache=False`` prevents image caching, but Ultralytics still reads its
    label ``*.cache`` files. A killed worker can leave an empty cache behind,
    which otherwise fails before the first training epoch with ``EOFError``.
    The original file is retained with a ``.corrupt`` suffix for recovery.
    """
    for dataset in datasets:
        for cache_path in dataset.rglob("*.cache"):
            try:
                if cache_path.stat().st_size == 0:
                    raise EOFError("empty cache")
                with cache_path.open("rb") as stream:
                    cached = np.load(stream, allow_pickle=True).item()
                if not isinstance(cached, dict):
                    raise ValueError("cache is not a mapping")
            except Exception as error:
                backup = cache_path.with_name(cache_path.name + ".corrupt")
                suffix = 1
                while backup.exists():
                    backup = cache_path.with_name(f"{cache_path.name}.corrupt-{suffix}")
                    suffix += 1
                os.replace(cache_path, backup)
                print(f"Moved invalid dataset cache {cache_path} -> {backup} ({error})")


def build_data_manifest(datasets: list[Path], output: Path = GENERATED_DATA) -> Path:
    train_paths: list[str] = []
    val_paths: list[str] = []
    test_paths: list[str] = []
    shared_names: list[str] | None = None
    for dataset in datasets:
        config = read_dataset_config(dataset)
        names = config["names"]
        if shared_names is None:
            shared_names = names
        elif names != shared_names:
            raise ValueError(f"{dataset} class names do not match the other selected datasets")
        train_paths.append(str(resolve_split(dataset, config, "train")).replace("\\", "/"))
        for split, paths in (("val", val_paths), ("test", test_paths)):
            try:
                paths.append(str(resolve_split(dataset, config, split)).replace("\\", "/"))
            except FileNotFoundError:
                pass
    if not train_paths or shared_names is None:
        raise FileNotFoundError("No Vietnamese training images were resolved")
    if not val_paths:
        val_paths = [train_paths.pop()] if len(train_paths) > 1 else list(train_paths)
    payload: dict[str, Any] = {"train": train_paths, "val": val_paths, "nc": len(shared_names), "names": shared_names}
    if test_paths:
        payload["test"] = test_paths
    with output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, allow_unicode=True, sort_keys=False)
    return output


def find_default_weights() -> Path:
    candidates = (
        TRAINING_DIR / "weights" / "best.pt",
        SCRIPT_DIR.parents[1] / "vietnam_license_plate" / "weights" / "best.pt",
        PROJECT_ROOT / "vietnam_license_plate" / "weights" / "best.pt",
        SCRIPT_DIR / "weights" / "best.pt",
    )
    return next((path for path in candidates if path.is_file()), Path("yolo11s.pt"))


def select_device(requested: str) -> str:
    if requested:
        return requested
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "0" if torch.cuda.is_available() else "cpu"


def checkpoint_epoch(checkpoint: Path) -> int:
    """Read checkpoint epoch, falling back to the run's epoch log if stripped."""

    model = YOLO(str(checkpoint))
    epoch = model.ckpt.get("epoch") if model.ckpt else None
    if isinstance(epoch, int) and epoch >= 0:
        if not model.ckpt.get("optimizer"):
            print("Checkpoint has no optimizer state; the model and epoch can resume with a fresh optimizer.")
        return epoch
    completed = completed_epochs(checkpoint.parent.parent)
    if completed is None:
        raise ValueError(
            f"{checkpoint} has no saved epoch and its run has no usable results.csv; "
            "use --checkpoint to start a fine-tuning run instead."
        )
    print(f"Checkpoint epoch was stripped; using results.csv progress ({completed} completed epochs).")
    return completed - 1


def completed_epochs(run_dir: Path) -> int | None:
    """Return the number of completed epochs recorded by Ultralytics."""

    results = run_dir / "results.csv"
    if not results.is_file():
        return None
    try:
        with results.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        epochs = [int(float(row["epoch"])) for row in rows if row.get("epoch")]
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return max(epochs) + 1 if epochs else None


def _run_number(path: Path, base_name: str) -> int | None:
    if path.name == base_name:
        return 0
    match = re.fullmatch(re.escape(base_name) + r"_(\d+)", path.name)
    return int(match.group(1)) if match else None


def latest_run(base_name: str) -> Path | None:
    candidates = []
    if RUNS_DIR.is_dir():
        for path in RUNS_DIR.iterdir():
            checkpoint = path / "weights" / "last.pt"
            if _run_number(path, base_name) is not None and checkpoint.is_file() and checkpoint.stat().st_size > 0:
                candidates.append(path)
    return max(candidates, key=lambda path: (_run_number(path, base_name) or 0, path.stat().st_mtime), default=None)


def next_run_name(base_name: str) -> str:
    if not (RUNS_DIR / base_name).exists():
        return base_name
    index = 1
    while (RUNS_DIR / f"{base_name}_{index}").exists():
        index += 1
    return f"{base_name}_{index}"


def resolve_resume(value: str, name: str) -> Path | None:
    if value == "never":
        return None
    if value == "auto":
        latest = latest_run(name)
        return latest / "weights" / "last.pt" if latest else None
    checkpoint = Path(value).expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"Resume checkpoint is missing or empty: {checkpoint}")
    return checkpoint


def default_resume_value() -> str:
    """Keep compatibility with the previous CAR_SCAN_TRAIN_RESUME boolean."""
    value = os.getenv("CAR_SCAN_TRAIN_RESUME", "").strip()
    if value.lower() in {"", "1", "true", "yes"}:
        checkpoint = os.getenv("CAR_SCAN_TRAIN_CHECKPOINT", "").strip()
        return checkpoint or "auto"
    if value.lower() in {"0", "false", "no"}:
        return "never"
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or resume the Vietnamese licence-plate character detector.")
    parser.add_argument("--dataset", default=os.getenv("VIETNAM_TRAIN_DATASET", "all").strip() or "all")
    parser.add_argument("--checkpoint", default="", help="Weights for a fresh fine-tuning run, without optimizer state.")
    parser.add_argument("--base-model", default="", help="YOLO model to use when no local weights are found.")
    parser.add_argument("--resume", nargs="?", const="auto", default=default_resume_value(), metavar="CHECKPOINT", help="Resume: auto (default), never, or a last.pt path.")
    parser.add_argument("--epochs", type=int, default=80, help="Total epoch target; must exceed saved progress when resuming.")
    parser.add_argument("--additional-epochs", type=int, default=0, help="Epochs to add after a resume checkpoint; takes precedence over --epochs.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="")
    parser.add_argument("--name", default="vietnam_plate_detect")
    parser.add_argument("--new-run", action="store_true", help="Start a fresh numbered run instead of resuming the latest run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.additional_epochs < 0:
        raise ValueError("--epochs must be positive and --additional-epochs cannot be negative")
    datasets = ensure_datasets(args.dataset)
    repair_dataset_caches(datasets)
    data_yaml = build_data_manifest(datasets)
    resume_value = str(args.resume).strip() or "auto"
    if args.new_run:
        resume_value = "never"
    resume_checkpoint = resolve_resume(resume_value, args.name)
    if resume_checkpoint:
        args.name = resume_checkpoint.parent.parent.name
        completed_epoch = checkpoint_epoch(resume_checkpoint)
        total_epochs = completed_epoch + 1 + args.additional_epochs if args.additional_epochs else args.epochs
        if total_epochs <= completed_epoch + 1:
            raise ValueError(f"{resume_checkpoint} already completed epoch {completed_epoch + 1}. Use --additional-epochs N, or set --epochs to a larger total.")
        checkpoint, resume = resume_checkpoint, str(resume_checkpoint)
    else:
        args.name = next_run_name(args.name)
        total_epochs = args.epochs
        checkpoint = Path(args.checkpoint or os.getenv("CAR_SCAN_TRAIN_CHECKPOINT", "") or find_default_weights())
        if not checkpoint.is_file():
            checkpoint = Path(args.base_model or "yolo11s.pt")
        resume = False
    device = select_device(args.device)
    print(f"Datasets: {', '.join(path.name for path in datasets)}")
    print(f"Data yaml: {data_yaml}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Resume: {bool(resume)}; total epochs: {total_epochs}; device: {device}")
    model = YOLO(str(checkpoint))
    model.train(
        trainer=ResumableDetectionTrainer, data=str(data_yaml), epochs=total_epochs,
        imgsz=args.imgsz, batch=args.batch, workers=args.workers, device=device,
        project=str(RUNS_DIR), name=args.name, exist_ok=bool(resume), resume=resume,
        pretrained=True, optimizer="AdamW", lr0=0.001, lrf=0.01, warmup_epochs=3,
        close_mosaic=10, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=5.0,
        translate=0.08, scale=0.35, shear=2.0, perspective=0.0004, fliplr=0.0,
        mosaic=0.8, mixup=0.05, copy_paste=0.05, erasing=0.2, patience=25,
        save=True, plots=True, cache=False, amp=device != "cpu",
    )


if __name__ == "__main__":
    main()
