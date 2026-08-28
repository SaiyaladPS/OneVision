import os
from pathlib import Path

import torch
from ultralytics import YOLO


RUN_DIR = Path("runs/thai_license_plate")
WEIGHTS_DIR = RUN_DIR / "weights"
TARGET_EPOCHS = 100


def find_resume_checkpoint() -> Path:
    """Find the newest checkpoint that still contains training state."""
    requested = os.getenv("CAR_SCAN_TRAIN_CHECKPOINT", "").strip()
    if requested:
        checkpoint = Path(requested)
        if checkpoint.is_file():
            return checkpoint
        raise FileNotFoundError(f"Requested resume checkpoint not found: {checkpoint}")

    checkpoints = sorted(
        WEIGHTS_DIR.glob("epoch*.pt"),
        key=lambda path: int(path.stem.removeprefix("epoch")),
        reverse=True,
    )

    for checkpoint in checkpoints:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            0 <= saved.get("epoch", -1) < TARGET_EPOCHS
            and saved.get("optimizer") is not None
        ):
            return checkpoint

    raise FileNotFoundError(
        "No resumable checkpoint with optimizer state was found in "
        f"{WEIGHTS_DIR}. Use an epochXX.pt checkpoint instead of last.pt."
    )


def main() -> None:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu:0"
    elif torch.cuda.is_available():
        device = 0
    else:
        device = "cpu"

    checkpoint = find_resume_checkpoint()
    print(f"Resuming from: {checkpoint}")
    print(f"Device: {device}")

    model = YOLO(str(checkpoint))
    model.train(
        data="Thai-License-Plate-Character-Recognition-1/data.yaml",
        epochs=TARGET_EPOCHS,
        patience=30,
        batch=4,
        imgsz=640,
        workers=0,
        save_period=5,
        close_mosaic=10,
        device=device,
        save_dir=str(RUN_DIR),
        resume=str(checkpoint),
    )


if __name__ == "__main__":
    main()
