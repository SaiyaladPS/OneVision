"""Fine-tune the licence-plate detector from the existing ``best.pt``.

By default this trains on the Roboflow ``Lpr-2`` dataset. Use ``--dataset all``
to combine every local YOLO dataset with the same one-class schema.

If an unfinished run with the same name is found, the script asks whether to
resume it from ``last.pt`` or start a fresh fine-tuning run from ``best.pt``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils import LOGGER as ULTRALYTICS_LOGGER


SCRIPT_DIR = Path(__file__).resolve().parent
DETECTOR_DIR = SCRIPT_DIR.parent
WEIGHTS = DETECTOR_DIR / "weights" / "best.pt"
RUNS_DIR = DETECTOR_DIR / "runs"
COMBINED_DATA = SCRIPT_DIR / "combined_data.generated.yaml"
DEFAULT_DATASET = "Lpr"


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


def discover_datasets() -> list[Path]:
    """Find downloaded YOLO datasets next to this training script."""

    datasets = sorted(
        path
        for path in SCRIPT_DIR.iterdir()
        if path.is_dir() and (path / "data.yaml").is_file()
    )
    if not datasets:
        raise FileNotFoundError(
            f"No YOLO datasets containing data.yaml were found in {SCRIPT_DIR}"
        )
    return datasets


def select_datasets(dataset_name: str) -> list[Path]:
    """Select one downloaded dataset, or all compatible local datasets."""

    datasets = discover_datasets()
    if dataset_name.lower() == "all":
        return datasets

    selected = SCRIPT_DIR / dataset_name
    if not selected.is_dir() or not (selected / "data.yaml").is_file():
        available = ", ".join(path.name for path in datasets)
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' was not found in {SCRIPT_DIR}. "
            f"Available datasets: {available}"
        )
    return [selected]


def _read_dataset_config(dataset_dir: Path) -> dict[str, Any]:
    config_path = dataset_dir / "data.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    names = config.get("names", [])
    if isinstance(names, dict):
        names = [names[index] for index in sorted(names)]
    if config.get("nc") != 1 or list(names) != ["License_Plate"]:
        raise ValueError(
            f"{config_path} must use the same one-class schema: "
            "names: ['License_Plate']"
        )
    return config


def _split_path(dataset_dir: Path, config: dict[str, Any], split: str) -> Path:
    value = config.get(split)
    if not isinstance(value, str):
        raise ValueError(f"{dataset_dir / 'data.yaml'} has no string '{split}' path")
    # The downloaded files use paths such as ../train/images relative to the
    # data.yaml location. Resolve them before writing the combined manifest.
    path = (dataset_dir / value).resolve()
    # Ultralytics accepts Roboflow's ``../train/images`` convention by
    # falling back to a path relative to the dataset directory itself.
    if not path.is_dir() and value.startswith("../"):
        path = (dataset_dir / value[3:]).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset split does not exist: {path}")
    return path


def build_combined_data(datasets: list[Path]) -> Path:
    """Create one YOLO data file whose splits contain every local dataset."""

    train_paths: list[str] = []
    val_paths: list[str] = []
    test_paths: list[str] = []
    for dataset in datasets:
        config = _read_dataset_config(dataset)
        train_paths.append(str(_split_path(dataset, config, "train")).replace("\\", "/"))
        val_paths.append(str(_split_path(dataset, config, "val")).replace("\\", "/"))
        if config.get("test"):
            test_paths.append(str(_split_path(dataset, config, "test")).replace("\\", "/"))

    combined = {
        "train": train_paths,
        "val": val_paths,
        "test": test_paths,
        "nc": 1,
        "names": ["License_Plate"],
    }
    with COMBINED_DATA.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(combined, stream, sort_keys=False, allow_unicode=True)
    return COMBINED_DATA


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
    """Find unfinished runs with a usable ``last.pt`` checkpoint.

    Ultralytics writes ``last.pt`` for both interrupted and completed runs, so
    the results file is also checked to avoid asking about a finished run.
    """

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
        if not isinstance(run_args, dict):
            continue
        if run_args.get("name") != name:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the detector from best.pt on a local YOLO dataset."
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv("CAR_SCAN_TRAIN_DATASET", DEFAULT_DATASET),
        help="Dataset directory name, or 'all' to combine every local dataset.",
    )
    parser.add_argument("--epochs", type=int, default=int(os.getenv("CAR_SCAN_TRAIN_EPOCHS", "50")))
    parser.add_argument("--batch", type=int, default=int(os.getenv("CAR_SCAN_TRAIN_BATCH", "16")))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu, 0, 1, or xpu:0")
    parser.add_argument("--name", default="detect_license_lpr2_finetune")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=RUNS_DIR / "training_logs",
        help="Directory for timestamped training console logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = training_log_path(args.log_dir, args.name)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        # Ultralytics creates its console logger during import, before stdout
        # is redirected below. Attach the same log file explicitly so epoch
        # metrics, validation output and its own error messages are retained.
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
                resume_mode = os.getenv("CAR_SCAN_TRAIN_RESUME", "ask").strip().lower()
                if resume_mode == "yes":
                    requested_checkpoint = Path(
                        os.getenv("CAR_SCAN_TRAIN_CHECKPOINT", "")
                    )
                    if requested_checkpoint.is_file() and requested_checkpoint.name == "last.pt":
                        resume_run = requested_checkpoint.parent.parent
                    elif incomplete_runs:
                        resume_run = incomplete_runs[0]
                    else:
                        raise FileNotFoundError("No unfinished run with last.pt is available to resume")
                elif resume_mode == "no":
                    resume_run = None
                else:
                    resume_run = choose_resume_run(incomplete_runs)
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
                    # A smaller learning rate protects the useful features already in
                    # best.pt while adapting them to the newly added images.
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
