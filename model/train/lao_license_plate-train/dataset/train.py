"""Fine-tune the Lao licence-plate character detector from best.pt."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]


def find_dataset_dir() -> Path:
    """Find the available Lao YOLO dataset across common export names."""

    requested = os.getenv("LAO_TRAIN_DATASET", "").strip()
    candidates = [
        SCRIPT_DIR / requested,
        SCRIPT_DIR / "lao-plate-detect-1",
        SCRIPT_DIR / "lao-plate-detect",
    ] if requested else [
        SCRIPT_DIR / "lao-plate-detect-1",
        SCRIPT_DIR / "lao-plate-detect",
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "data.yaml").is_file():
            return candidate

    discovered = sorted(
        candidate
        for candidate in SCRIPT_DIR.iterdir()
        if candidate.is_dir() and (candidate / "data.yaml").is_file()
    )
    if discovered:
        return discovered[0]
    # Keep a deterministic path for the final, actionable error message.
    return candidates[0]


DATASET_DIR = find_dataset_dir()
DATA_CONFIG = DATASET_DIR / "data.yaml"
DATA_MANIFEST = SCRIPT_DIR / "lao_plate_data.generated.yaml"
# Keep training outputs beside ``dataset/`` rather than inside the dataset.
RUNS_DIR = TRAINING_DIR / "runs"
BASE_MODEL = os.getenv("LAO_TRAIN_BASE_MODEL", "yolo11s.pt")


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


def find_default_weights() -> Path:
    """Find the deployed or training checkpoint in supported layouts."""

    candidates = (
        TRAINING_DIR / "weights" / "best.pt",
        SCRIPT_DIR.parents[1] / "lao_license_plate" / "weights" / "best.pt",
        PROJECT_ROOT / "lao_license_plate" / "weights" / "best.pt",
        PROJECT_ROOT.parent / "lao_license_plate" / "weights" / "best.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # Keep a useful path in argparse help and in the final error message when
    # the user has not downloaded/copied a checkpoint yet.
    return candidates[0]


WEIGHTS = find_default_weights()


def read_dataset_config() -> dict[str, Any]:
    if not DATA_CONFIG.is_file():
        raise FileNotFoundError(f"Dataset config not found: {DATA_CONFIG}")

    with DATA_CONFIG.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    names = config.get("names", [])
    if isinstance(names, dict):
        names = [names[index] for index in sorted(names)]
    if config.get("nc") != 50 or len(names) != 50:
        raise ValueError(
            f"{DATA_CONFIG} must contain exactly 50 character classes; "
            f"found nc={config.get('nc')}, classes={len(names)}"
        )
    return config


def resolve_split(split: str, config: dict[str, Any]) -> Path:
    value = config.get(split)
    if not isinstance(value, str):
        raise ValueError(f"{DATA_CONFIG} has no string '{split}' path")

    path = (DATASET_DIR / value).resolve()
    # Roboflow exports commonly use ../train/images relative to data.yaml.
    if not path.is_dir() and value.startswith("../"):
        path = (DATASET_DIR / value[3:]).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset split does not exist: {path}")
    return path


def build_data_manifest() -> Path:
    config = read_dataset_config()
    manifest = {
        "train": [str(resolve_split("train", config)).replace("\\", "/")],
        "val": [str(resolve_split("val", config)).replace("\\", "/")],
        "test": [str(resolve_split("test", config)).replace("\\", "/")],
        "nc": 50,
        "names": config["names"],
    }
    with DATA_MANIFEST.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False, allow_unicode=True)
    return DATA_MANIFEST


def select_device(requested: str | None) -> str | int:
    if requested:
        return int(requested) if requested.isdigit() else requested
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu:0"
    if torch.cuda.is_available():
        return 0
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the Lao character detector from best.pt."
    )
    parser.add_argument(
        "--dataset",
        default="all",
        help=(
            "Compatibility option for the shared training launcher. "
            "The Lao model always uses lao-plate-detect-1; 'all' is accepted "
            "but does not mix datasets from other models."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"Optional trained checkpoint; auto-detected default: {WEIGHTS}.",
    )
    parser.add_argument(
        "--base-model",
        default=BASE_MODEL,
        help="Model used for a first training run when no trained checkpoint exists.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.getenv("LAO_TRAIN_EPOCHS", "100")),
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=int(os.getenv("LAO_TRAIN_BATCH", "4")),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu, 0, 1, or xpu:0")
    parser.add_argument("--name", default="lao_license_plate_finetune")
    parser.add_argument("--new-run", action="store_true", help="Start a fresh numbered run instead of resuming the latest run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_dataset = str(args.dataset).strip().lower()
    if requested_dataset not in {"all", DATASET_DIR.name.lower()}:
        raise ValueError(
            f"The Lao training script uses only '{DATASET_DIR.name}'; "
            f"received dataset '{args.dataset}'."
        )
    if requested_dataset == "all":
        print(
            f"Dataset selector 'all' is accepted by the shared launcher; "
            f"using Lao dataset: {DATASET_DIR}"
        )
    resume_mode = os.getenv("CAR_SCAN_TRAIN_RESUME", "auto").strip().lower()
    start_new = args.new_run or resume_mode in {"no", "false", "0", "new"}
    resume_run = None if start_new else latest_run(args.name)
    if resume_run is not None:
        args.name = resume_run.name
        resume_mode = "yes"
    elif start_new:
        args.name = next_run_name(args.name)
        resume_mode = "no"
    else:
        # No resumable checkpoint exists; avoid colliding with an old/incomplete directory.
        args.name = next_run_name(args.name)
    resume_checkpoint = os.getenv("CAR_SCAN_TRAIN_CHECKPOINT", "").strip()
    if resume_mode == "yes":
        if not resume_checkpoint:
            raise FileNotFoundError("No unfinished checkpoint was supplied for resume")
        checkpoint = Path(resume_checkpoint)
        start_from_scratch = False
    else:
        checkpoint = args.checkpoint or WEIGHTS
        if not checkpoint.is_absolute():
            checkpoint = (Path.cwd() / checkpoint).resolve()
        start_from_scratch = not checkpoint.is_file()
        if start_from_scratch:
            print(
                f"No trained checkpoint found at {checkpoint}; "
                f"starting a new training run from {args.base_model}."
            )
    if resume_mode == "yes" and not checkpoint.is_absolute():
        checkpoint = (Path.cwd() / checkpoint).resolve()
    if not start_from_scratch and not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32:
        raise ValueError("epochs, batch, and imgsz must be positive (imgsz >= 32)")

    data_path = build_data_manifest()
    device = select_device(args.device)
    print(f"Dataset: {DATASET_DIR}")
    print(f"Training data manifest: {data_path}")
    model_source = args.base_model if start_from_scratch else checkpoint
    print(
        f"{'Resuming from checkpoint' if resume_mode == 'yes' else 'Starting model'}: "
        f"{model_source}"
    )
    print(f"Device: {device}")

    model = YOLO(str(model_source))
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        patience=30,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=device,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=False,
        save=True,
        save_period=5,
        close_mosaic=10,
        # Keep the useful features in best.pt while adapting to this dataset.
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=1.0,
        optimizer="AdamW",
        pretrained=start_from_scratch,
        resume=str(checkpoint) if resume_mode == "yes" else False,
    )


if __name__ == "__main__":
    main()
