"""Small CRNN-CTC recogniser used when PaddleOCR is unavailable.

The model is deliberately independent from PaddlePaddle so it can be trained
and used in the current Python environment.  Labels are the six-character
plate strings encoded in the ``Images/*_ocr_crop.jpg`` filenames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn


IMAGE_HEIGHT = 32
IMAGE_WIDTH = 160


class PlateCRNN(nn.Module):
    def __init__(self, class_count: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.sequence = nn.LSTM(512, 128, bidirectional=True, batch_first=False)
        self.classifier = nn.Linear(256, class_count)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        # [batch, channels, height, width] -> [time, batch, channels * height]
        sequence = features.permute(3, 0, 1, 2).contiguous()
        sequence = sequence.reshape(sequence.shape[0], sequence.shape[1], -1)
        sequence, _ = self.sequence(sequence)
        return self.classifier(sequence)


def prepare_image(image: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
    return (tensor - 0.5) / 0.5


def decode_logits(logits: torch.Tensor, chars: list[str]) -> tuple[str, float]:
    if logits.dim() == 3:
        if logits.shape[1] != 1:
            raise ValueError("decode_logits expects one sequence at a time")
        logits = logits[:, 0, :]
    probabilities = logits.softmax(dim=-1)
    values, indices = probabilities.max(dim=-1)
    blank = len(chars)
    previous = blank
    output: list[str] = []
    confidence: list[float] = []
    for value, index in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
        if index != blank and index != previous:
            output.append(chars[index])
            confidence.append(float(value))
        previous = index
    return "".join(output), float(np.mean(confidence)) if confidence else 0.0


class TorchPlateOCR:
    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint
        payload: dict[str, Any] = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        self.chars = list(payload["chars"])
        self.model = PlateCRNN(len(self.chars) + 1)
        self.model.load_state_dict(payload["model"])
        self.model.eval()

    def recognise(self, image: np.ndarray) -> tuple[str, float]:
        with torch.inference_mode():
            logits = self.model(prepare_image(image).unsqueeze(0))
        return decode_logits(logits, self.chars)
