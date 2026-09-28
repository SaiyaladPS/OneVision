"""Choose whether plate scanning runs on GPU, CPU, or both together."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

COMPUTE_MODES = ("auto", "gpu", "cpu", "hybrid")
_ALIASES = {
    "cuda": "gpu",
    "gpu+cpu": "hybrid",
    "gpu_cpu": "hybrid",
    "cpu+gpu": "hybrid",
    "cpu_gpu": "hybrid",
    "both": "hybrid",
    "shared": "hybrid",
}


def normalise_compute_mode(value: str | None) -> str:
    raw = str(value or "auto").strip().lower().replace(" ", "")
    raw = _ALIASES.get(raw, raw)
    return raw if raw in COMPUTE_MODES else "auto"


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def cuda_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return ""


def compute_mode_path(output_dir: Path | None) -> Path | None:
    if output_dir is None:
        return None
    return Path(output_dir) / ".compute_mode"


def load_saved_compute_mode(output_dir: Path | None) -> str:
    path = compute_mode_path(output_dir)
    if path is None or not path.is_file():
        return ""
    try:
        return normalise_compute_mode(path.read_text(encoding="utf-8"))
    except Exception:
        return ""


def save_compute_mode(output_dir: Path | None, mode: str) -> None:
    path = compute_mode_path(output_dir)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normalise_compute_mode(mode) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class ComputePlan:
    """Where YOLO and OCR should run for the current machine and mode."""

    requested: str
    mode: str
    yolo_device: str
    ocr_device: str
    hybrid_cpu_yolo: bool
    cuda: bool
    gpu_name: str
    fallback: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "mode": self.mode,
            "yolo_device": self.yolo_device,
            "ocr_device": self.ocr_device,
            "hybrid": self.hybrid_cpu_yolo,
            "cuda": self.cuda,
            "gpu": self.gpu_name or "CPU",
            "fallback": self.fallback,
        }


def resolve_compute(mode: str | None = None, *, has_cuda: bool | None = None) -> ComputePlan:
    """Map a user compute mode onto YOLO/OCR devices.

    ``auto`` uses CUDA for YOLO when present and keeps OCR on CPU so the two
    do not fight for the same GPU memory. ``hybrid`` also infers idle CCTV
    lanes on a second CPU YOLO copy while the GPU stays on live plates.
    """

    requested = normalise_compute_mode(mode)
    cuda = cuda_available() if has_cuda is None else bool(has_cuda)
    name = cuda_name() if has_cuda is None else ("GPU" if cuda else "")
    if requested == "cpu" or not cuda:
        fallback = requested in ("gpu", "hybrid") and not cuda
        effective = "cpu" if fallback else requested
        if requested == "auto":
            effective = "cpu"
        return ComputePlan(
            requested=requested,
            mode=effective,
            yolo_device="cpu",
            ocr_device="cpu",
            hybrid_cpu_yolo=False,
            cuda=cuda,
            gpu_name=name,
            fallback=fallback,
        )
    if requested == "hybrid":
        return ComputePlan(
            requested=requested,
            mode="hybrid",
            yolo_device="cuda:0",
            ocr_device="cpu",
            hybrid_cpu_yolo=True,
            cuda=True,
            gpu_name=name,
            fallback=False,
        )
    if requested == "gpu":
        return ComputePlan(
            requested=requested,
            mode="gpu",
            yolo_device="cuda:0",
            ocr_device="gpu:0",
            hybrid_cpu_yolo=False,
            cuda=True,
            gpu_name=name,
            fallback=False,
        )
    return ComputePlan(
        requested=requested,
        mode="auto",
        yolo_device="cuda:0",
        ocr_device="cpu",
        hybrid_cpu_yolo=False,
        cuda=True,
        gpu_name=name,
        fallback=False,
    )


def bind_yolo_device(model: Any, device: str | None) -> None:
    if model is None or not device:
        return
    try:
        model.to(device)
    except Exception:
        LOGGER.debug("could not move YOLO model to %s", device, exc_info=True)


def yolo_predict(model: Any, source: Any, *, device: str | None = None, **kwargs: Any) -> Any:
    """Call Ultralytics predict, pinning the device when the model allows it."""

    if device:
        try:
            return model.predict(source, device=device, **kwargs)
        except TypeError:
            pass
    return model.predict(source, **kwargs)
