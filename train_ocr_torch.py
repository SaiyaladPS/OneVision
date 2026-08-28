"""Train the local plate CRNN from the labelled crop filenames.

Example:
    python train_ocr_torch.py --epochs 100

Each ``Images/*_ocr_crop.jpg`` file is labelled from the second underscore
component of its filename, e.g. ``00003_ບກ2031_...`` -> ``ບກ2031``.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from plate_ocr_torch import PlateCRNN, decode_logits, prepare_image


ROOT = Path(__file__).resolve().parent


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def label_for(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) < 2 or not parts[1]:
        raise ValueError(f"Cannot extract plate label from filename: {path.name}")
    return parts[1]


def load_characters(path: Path) -> list[str]:
    chars = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and line.strip() != "-"]
    if len(chars) != len(set(chars)):
        raise ValueError(f"Duplicate characters in dictionary: {path}")
    return chars


class PlateDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, paths: list[Path], char_to_index: dict[str, int], augment: bool) -> None:
        self.paths = paths
        self.char_to_index = char_to_index
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = read_image(self.paths[index])
        if self.augment:
            if random.random() < 0.7:
                alpha = random.uniform(0.75, 1.3)
                beta = random.randint(-25, 25)
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            if random.random() < 0.35:
                image = cv2.GaussianBlur(image, (3, 3), random.uniform(0.2, 0.8))
            if random.random() < 0.3:
                noise = np.random.normal(0, random.uniform(1.0, 5.0), image.shape).astype(np.float32)
                image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        label = torch.tensor([self.char_to_index[char] for char in label_for(self.paths[index])], dtype=torch.long)
        return prepare_image(image), label


def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = torch.stack([item[0] for item in batch])
    labels = torch.cat([item[1] for item in batch])
    lengths = torch.tensor([len(item[1]) for item in batch], dtype=torch.long)
    return images, labels, lengths


def run_epoch(
    model: PlateCRNN,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer | None,
    blank: int,
    chars: list[str],
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    exact = 0
    character_total = 0
    character_correct = 0
    for images, labels, lengths in loader:
        logits = model(images)
        time_steps, batch_size, _ = logits.shape
        input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long)
        loss = criterion(logits.log_softmax(2), labels, input_lengths, lengths)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total_loss += float(loss.item()) * batch_size
        offset = 0
        for row in range(batch_size):
            predicted, _ = decode_logits(logits[:, row : row + 1], chars)
            expected = "".join(chars[index] for index in labels[offset : offset + int(lengths[row])].tolist())
            exact += int(predicted == expected)
            character_correct += sum(a == b for a, b in zip(predicted, expected))
            character_total += len(expected)
            offset += int(lengths[row])
    count = len(loader.dataset)
    return total_loss / max(1, count), exact / max(1, count), character_correct / max(1, character_total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Lao/Thai plate CRNN-CTC model.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Images")
    parser.add_argument("--dictionary", type=Path, default=ROOT / "plate_dict.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "ocr" / "plate_crnn.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--numeric-only",
        action="store_true",
        help="Train a Thai registration-number recogniser using only six-digit labels.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or not 0 < args.validation_ratio < 0.5:
        raise ValueError("Invalid epochs or validation ratio")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))

    paths = sorted(args.data_dir.glob("*_ocr_crop.jpg"))
    if args.numeric_only:
        paths = [path for path in paths if label_for(path).isdigit()]
    if len(paths) < 20:
        raise RuntimeError(f"Need at least 20 labelled crops, found {len(paths)}")
    chars = list("0123456789") if args.numeric_only else load_characters(args.dictionary)
    char_to_index = {char: index for index, char in enumerate(chars)}
    unknown = sorted({char for path in paths for char in label_for(path) if char not in char_to_index})
    if unknown:
        raise ValueError(f"Labels contain characters missing from dictionary: {unknown}")

    random.Random(args.seed).shuffle(paths)
    validation_count = max(1, int(len(paths) * args.validation_ratio))
    validation_paths = paths[:validation_count]
    training_paths = paths[validation_count:]
    train_loader = DataLoader(PlateDataset(training_paths, char_to_index, True), batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    validation_loader = DataLoader(PlateDataset(validation_paths, char_to_index, False), batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    model = PlateCRNN(len(chars) + 1)
    criterion = nn.CTCLoss(blank=len(chars), zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_loss = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training crops: {len(training_paths)}; validation crops: {len(validation_paths)}; classes: {len(chars)}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_exact, train_char = run_epoch(model, train_loader, criterion, optimizer, len(chars), chars)
        with torch.no_grad():
            val_loss, val_exact, val_char = run_epoch(model, validation_loader, criterion, None, len(chars), chars)
        scheduler.step()
        print(f"epoch {epoch:03d}/{args.epochs}: train_loss={train_loss:.4f} train_exact={train_exact:.3f} train_char={train_char:.3f} val_loss={val_loss:.4f} val_exact={val_exact:.3f} val_char={val_char:.3f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model": model.state_dict(), "chars": chars, "metrics": {"val_loss": val_loss, "val_exact": val_exact, "val_char": val_char}}, args.output)
    print(f"Saved best checkpoint: {args.output}")


if __name__ == "__main__":
    main()
