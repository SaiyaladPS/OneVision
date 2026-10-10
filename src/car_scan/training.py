"""Manage the repository's model-training scripts through a narrow API."""

from __future__ import annotations

import os
import random
import re
import csv
import json
import stat
import signal
import shutil
import shlex
import subprocess
import sys
import tempfile
import threading
import uuid
import zipfile
import tarfile
from io import BytesIO
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import yaml


_TRAIN_ROOT = Path(__file__).resolve().parents[2] / "model" / "train"
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_SKIP_DATASET_DIRS = {".venv", ".git", "__pycache__", "runs", "images", "labels"}
_DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_WINDOWS_RESERVED_PATH = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?$", re.IGNORECASE)
_DATASET_FILE_SUFFIXES = _IMAGE_SUFFIXES | {".txt", ".yaml", ".yml", ".json"}
_MAX_DATASET_ARCHIVE_BYTES = 256 * 1024 * 1024
_MAX_DATASET_UNPACKED_BYTES = 4 * 1024 * 1024 * 1024
_MAX_DATASET_ARCHIVE_FILES = 50_000
_WORKER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
_WORKER_HOST_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$")
_WORKER_USER_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_DOCKER_IMAGE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,200}$")
_REMOTE_JOB_ROOT = "/tmp/onevision-training"


class TrainingRunManager:
    """Start at most one allow-listed training run and retain a bounded log."""

    def __init__(self, train_root: Path = _TRAIN_ROOT) -> None:
        self.train_root = train_root.resolve()
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._run: dict[str, Any] | None = None
        self._logs: deque[str] = deque(maxlen=1200)
        self._dataset_cache: dict[str, Any] | None = None
        self._dataset_cache_at = 0.0
        self._results_cache: dict[str, Any] | None = None
        self._results_cache_at = 0.0

    def _targets(self) -> list[dict[str, Any]]:
        targets = []
        if not self.train_root.is_dir():
            return targets
        for folder in sorted(self.train_root.glob("*-train")):
            script = folder / "dataset" / "train.py"
            name = folder.name.removesuffix("-train")
            if script.is_file() and _ID_PATTERN.fullmatch(name):
                runs_dir = folder / "runs"
                targets.append({
                    "id": name,
                    "name": name,
                    "directory": folder.name,
                    "resumeAvailable": runs_dir.is_dir() and any(runs_dir.glob("**/weights/last.pt")),
                })
        return targets

    def workers(self) -> list[dict[str, Any]]:
        """Expose local and configured SSH/Docker workers without secrets."""
        workers: list[dict[str, Any]] = [{
            "id": "local", "name": "This server", "kind": "local", "device": "auto",
            "available": bool(self._targets()),
        }]
        raw = str(os.getenv("CAR_SCAN_TRAIN_WORKERS") or "").strip()
        if not raw:
            return workers
        try:
            configured = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("CAR_SCAN_TRAIN_WORKERS must be a valid JSON array") from error
        if not isinstance(configured, list):
            raise ValueError("CAR_SCAN_TRAIN_WORKERS must be a JSON array")
        seen = {"local"}
        for item in configured:
            if not isinstance(item, dict):
                continue
            worker_id = str(item.get("id") or "").strip()
            name = str(item.get("name") or worker_id).strip()
            host = str(item.get("host") or "").strip()
            user = str(item.get("user") or "").strip()
            image = str(item.get("dockerImage") or "").strip()
            if (not _WORKER_ID_PATTERN.fullmatch(worker_id) or worker_id in seen
                    or not name or not _WORKER_HOST_PATTERN.fullmatch(host)
                    or not _WORKER_USER_PATTERN.fullmatch(user)
                    or not _DOCKER_IMAGE_PATTERN.fullmatch(image)):
                continue
            device = str(item.get("device") or "cpu").strip().lower()
            if device != "cpu" and not re.fullmatch(r"\d+", device):
                continue
            cpus = item.get("cpus", 4)
            try:
                cpus = max(1, min(64, int(cpus)))
            except (TypeError, ValueError):
                continue
            memory = str(item.get("memory") or "6g").strip().lower()
            if not re.fullmatch(r"[1-9][0-9]{0,2}[mg]", memory):
                continue
            seen.add(worker_id)
            workers.append({
                "id": worker_id, "name": name, "kind": "ssh-docker", "device": device,
                "available": True, "cpus": cpus, "memory": memory,
            })
        return workers

    def _worker_config(self, worker_id: str) -> dict[str, Any] | None:
        if worker_id == "local":
            return None
        for public in self.workers():
            if public["id"] == worker_id and public["kind"] == "ssh-docker":
                raw = json.loads(str(os.getenv("CAR_SCAN_TRAIN_WORKERS") or "[]"))
                item = next(item for item in raw if isinstance(item, dict) and item.get("id") == worker_id)
                return {**item, "cpus": public["cpus"], "memory": public["memory"], "device": public["device"]}
        raise ValueError("unknown training server")

    @staticmethod
    def _ssh_args(worker: dict[str, Any], *, scp: bool = False) -> list[str]:
        ssh = shutil.which("scp" if scp else "ssh")
        if not ssh:
            raise FileNotFoundError("OpenSSH client (ssh/scp) is required for remote training")
        args = [ssh, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10"]
        known_hosts = str(os.getenv("CAR_SCAN_TRAIN_SSH_KNOWN_HOSTS") or "").strip()
        key_path = str(os.getenv("CAR_SCAN_TRAIN_SSH_KEY") or "").strip()
        if known_hosts:
            args.extend(["-o", f"UserKnownHostsFile={known_hosts}"])
        if key_path:
            args.extend(["-i", key_path])
        if not scp:
            args.append(f"{worker['user']}@{worker['host']}")
        return args

    def _read_output(self, process: subprocess.Popen[str]) -> None:
        stream = process.stdout
        if stream is None:
            return
        for line in stream:
            with self._lock:
                self._logs.append(line.rstrip("\r\n"))

    def status(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            is_remote = bool(self._run and self._run.get("workerId") != "local")
            if self._run is not None and process is not None and not is_remote:
                code = process.poll()
                if code is not None and self._run["status"] in {"starting", "running", "stopping"}:
                    self._run["status"] = "completed" if code == 0 else "failed"
                    self._run["exitCode"] = code
                    self._run["finishedAt"] = datetime.now(timezone.utc).isoformat()
            return {
                "available": bool(self._targets()),
                "models": self._targets(),
                "workers": self.workers(),
                "run": dict(self._run) if self._run else None,
                "logs": list(self._logs),
            }

    @staticmethod
    def _count_images(directory: Path | None) -> int:
        if directory is None or not directory.is_dir():
            return 0
        count = 0
        for current, child_dirs, files in os.walk(directory, followlinks=False):
            child_dirs[:] = [name for name in child_dirs if name not in _SKIP_DATASET_DIRS and not name.startswith(".")]
            count += sum(Path(name).suffix.lower() in _IMAGE_SUFFIXES for name in files)
        return count

    @staticmethod
    def _within(root: Path, candidate: Path) -> bool:
        try:
            candidate.resolve().relative_to(root.resolve())
            return True
        except (OSError, ValueError):
            return False

    def _dataset_roots(self, dataset_root: Path) -> list[Path]:
        if not dataset_root.is_dir():
            return []
        result: set[Path] = set()
        base = dataset_root.resolve()
        for current, child_dirs, files in os.walk(base, followlinks=False):
            here = Path(current)
            depth = len(here.relative_to(base).parts)
            child_dirs[:] = [
                name for name in child_dirs
                if name not in _SKIP_DATASET_DIRS and not name.startswith(".") and depth < 4
            ]
            if "data.yaml" in files or ((here / "images").is_dir() and (here / "labels").is_dir()):
                result.add(here)
                child_dirs[:] = []
        return sorted(result, key=lambda item: item.relative_to(base).as_posix().lower())

    def _split_directory(self, dataset_dir: Path, dataset_root: Path, config: dict[str, Any], split: str) -> Path | None:
        value = config.get(split)
        if split == "val" and not value:
            value = config.get("valid")
        candidates: list[Path] = []
        if isinstance(value, str) and value.strip() and not value.lower().endswith(".txt"):
            raw = Path(value.strip())
            if raw.is_absolute():
                candidates.append(raw)
            else:
                configured_root = config.get("path")
                base = Path(str(configured_root)) if configured_root else dataset_dir
                if not base.is_absolute():
                    base = dataset_dir / base
                candidates.extend((base / raw, dataset_dir / raw))
                # Some dataset exporters write paths relative to a parent
                # directory (for example ``../valid/images``), even though
                # the ZIP contains that split beneath its own root. Retry that
                # path relative to the extracted dataset, without ever
                # accepting a candidate that escapes dataset_root.
                parts = raw.parts
                while parts and parts[0] == "..":
                    parts = parts[1:]
                if len(parts) < len(raw.parts):
                    candidates.append(dataset_dir.joinpath(*parts))
        aliases = {"train": ("train", "Train"), "val": ("val", "valid", "Validation", "Valid"), "test": ("test", "Test")}[split]
        candidates.extend(dataset_dir / "images" / name for name in aliases)
        candidates.extend(dataset_dir / name / "images" for name in aliases)
        for candidate in candidates:
            if self._within(dataset_root, candidate) and candidate.is_dir():
                return candidate.resolve()
        return None

    def _dataset_info(self, model: str, dataset_root: Path, dataset_dir: Path) -> dict[str, Any]:
        relative = dataset_dir.relative_to(dataset_root).as_posix()
        config_path = dataset_dir / "data.yaml"
        config: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        if config_path.is_file():
            try:
                with config_path.open("r", encoding="utf-8") as stream:
                    loaded = yaml.safe_load(stream) or {}
                if not isinstance(loaded, dict):
                    raise ValueError("data.yaml must contain a mapping")
                config = loaded
            except (OSError, UnicodeError, yaml.YAMLError, ValueError) as error:
                errors.append(f"data.yaml: {error}")
        else:
            warnings.append("data.yaml is missing")

        splits = {
            name: self._count_images(self._split_directory(dataset_dir, dataset_root, config, name))
            for name in ("train", "val", "test")
        }
        if not splits["train"]:
            errors.append("train split has no images")
        if not splits["val"]:
            warnings.append("validation split has no images")
        if not splits["test"]:
            warnings.append("test split has no images")

        names = config.get("names", [])
        if isinstance(names, dict):
            classes = [str(names[key]) for key in sorted(names, key=lambda value: int(value) if str(value).isdigit() else str(value))]
        elif isinstance(names, list):
            classes = [str(name) for name in names]
        else:
            classes = []
        try:
            class_count = int(config.get("nc", len(classes)))
        except (TypeError, ValueError):
            class_count = 0
        if not classes and model == "detect_license" and (dataset_dir / "images" / "Train").is_dir():
            classes, class_count = ["License_Plate"], 1
            warnings.append("class metadata inferred from the native detect_license folder layout")
        if config_path.is_file() and (not classes or class_count != len(classes)):
            if not (model == "detect_license" and classes == ["License_Plate"]):
                errors.append("class count and names do not match")

        return {
            "id": relative,
            "name": dataset_dir.name,
            "model": model,
            "path": f"{model}-train/dataset/{relative}",
            "ready": not errors,
            "splits": splits,
            "imageCount": sum(splits.values()),
            "classCount": class_count,
            "classes": classes[:40],
            "errors": errors,
            "warnings": warnings,
        }

    def datasets(self, *, force_refresh: bool = False) -> dict[str, Any]:
        """Return a read-only inventory and basic validation of local YOLO datasets."""
        now = datetime.now().timestamp()
        with self._lock:
            if not force_refresh and self._dataset_cache is not None and now - self._dataset_cache_at < 30:
                return self._dataset_cache
            models = []
            for target in self._targets():
                dataset_root = self.train_root / target["directory"] / "dataset"
                items = [self._dataset_info(target["id"], dataset_root, directory) for directory in self._dataset_roots(dataset_root)]
                models.append({"id": target["id"], "name": target["name"], "datasets": items})
            self._dataset_cache = {"models": models, "datasetCount": sum(len(item["datasets"]) for item in models)}
            self._dataset_cache_at = now
            return self._dataset_cache

    @staticmethod
    def _last_result_row(results_path: Path) -> tuple[int | None, dict[str, float]]:
        if not results_path.is_file():
            return None, {}
        try:
            with results_path.open("r", encoding="utf-8-sig", newline="") as stream:
                last_row = None
                for row in csv.DictReader(stream):
                    last_row = row
            if not last_row:
                return None, {}
            epoch = int(float(last_row.get("epoch") or 0)) + 1
            metrics = {}
            for name, value in last_row.items():
                lowered = name.lower()
                if "metrics/" not in lowered or not any(metric in lowered for metric in ("precision", "recall", "map50")):
                    continue
                try:
                    metrics[name] = float(value)
                except (TypeError, ValueError):
                    continue
            return epoch, metrics
        except (OSError, csv.Error, TypeError, ValueError):
            return None, {}

    def results(self, *, force_refresh: bool = False) -> dict[str, Any]:
        """List recent artifacts only from each model's dedicated runs directory."""
        now = datetime.now().timestamp()
        with self._lock:
            if not force_refresh and self._results_cache is not None and now - self._results_cache_at < 30:
                return self._results_cache
            models = []
            for target in self._targets():
                runs_root = self.train_root / target["directory"] / "runs"
                entries = []
                if runs_root.is_dir():
                    for run_dir in runs_root.iterdir():
                        if not run_dir.is_dir() or run_dir.name in {"training_logs", "__pycache__"}:
                            continue
                        weights = run_dir / "weights"
                        best = weights / "best.pt"
                        last = weights / "last.pt"
                        results_path = run_dir / "results.csv"
                        args_path = run_dir / "args.yaml"
                        if not (best.is_file() or last.is_file() or results_path.is_file() or args_path.is_file()):
                            continue
                        epochs, metrics = self._last_result_row(results_path)
                        try:
                            updated_at = datetime.fromtimestamp(run_dir.stat().st_mtime, timezone.utc).isoformat()
                        except OSError:
                            updated_at = None
                        entries.append({
                            "id": run_dir.name,
                            "name": run_dir.name,
                            "updatedAt": updated_at,
                            "epochsCompleted": epochs,
                            "hasResults": results_path.is_file(),
                            "hasBestWeights": best.is_file(),
                            "hasLastWeights": last.is_file(),
                            "metrics": metrics,
                        })
                entries.sort(key=lambda item: item["updatedAt"] or "", reverse=True)
                models.append({"id": target["id"], "name": target["name"], "runs": entries[:30]})
            self._results_cache = {"models": models, "runCount": sum(len(item["runs"]) for item in models)}
            self._results_cache_at = now
            return self._results_cache

    def _validate_uploaded_dataset(self, model: str, dataset_dir: Path, *, allow_split_layout: bool = False) -> dict[str, Any]:
        config_path = dataset_dir / "data.yaml"
        if not config_path.is_file():
            raise ValueError("ZIP must contain data.yaml at its dataset root")
        try:
            with config_path.open("r", encoding="utf-8") as stream:
                config = yaml.safe_load(stream)
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise ValueError(f"data.yaml could not be parsed: {error}") from error
        if not isinstance(config, dict):
            raise ValueError("data.yaml must contain a YAML mapping")

        # CVAT's Ultralytics YOLO Detection 1.0 export uses a root data.yaml,
        # a train.txt manifest, images/train, and labels/train. Require this
        # signature so unrelated ZIP layouts are not accepted by accident.
        train_manifest_value = config.get("train")
        supported_train_values = {"train.txt", "images/train"} if allow_split_layout else {"train.txt"}
        if not isinstance(train_manifest_value, str) or train_manifest_value not in supported_train_values:
            raise ValueError("Only CVAT Ultralytics YOLO Detection 1.0 ZIP exports with a root train.txt are supported")
        if config.get("path") not in (None, "."):
            raise ValueError("CVAT data.yaml path must be '.'")
        train_manifest = dataset_dir / "train.txt"
        if not train_manifest.is_file():
            raise ValueError("CVAT export is missing its root train.txt manifest")
        train_image_dir = dataset_dir / "images" / "train"
        train_label_dir = dataset_dir / "labels" / "train"
        if not train_image_dir.is_dir() or not train_label_dir.is_dir():
            raise ValueError("CVAT export must contain images/train and labels/train folders")

        names = config.get("names", [])
        if isinstance(names, dict):
            try:
                classes = [str(names[key]) for key in sorted(names, key=lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value)))]
            except (TypeError, ValueError) as error:
                raise ValueError("data.yaml names must be a list or indexed mapping") from error
        elif isinstance(names, list):
            classes = [str(name).strip() for name in names]
        else:
            raise ValueError("data.yaml names must be a list or indexed mapping")
        try:
            class_count = int(config.get("nc", len(classes)))
        except (TypeError, ValueError) as error:
            raise ValueError("data.yaml nc must be a positive integer") from error
        if not classes or class_count < 1 or class_count != len(classes) or any(not name for name in classes):
            raise ValueError("data.yaml nc must match the non-empty names list")

        split_stats: dict[str, dict[str, int]] = {}
        for split in ("train", "val", "test"):
            image_dir = self._split_directory(dataset_dir, dataset_dir, config, split)
            image_count = self._count_images(image_dir)
            if split == "train" and (image_count == 0 or image_dir is None):
                raise ValueError("train split must point to a folder containing images")
            if split != "train" and image_count == 0:
                split_stats[split] = {"images": 0, "labelFiles": 0}
                continue
            from PIL import Image, UnidentifiedImageError

            image_paths = [path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES]
            for image_path in image_paths:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except (OSError, ValueError, SyntaxError, UnidentifiedImageError) as error:
                    raise ValueError(f"invalid image file in {split} split: {image_path.name}") from error
            parts = list(image_dir.relative_to(dataset_dir).parts)
            if "images" in parts:
                parts[parts.index("images")] = "labels"
                labels_dir = dataset_dir.joinpath(*parts)
            else:
                split_name = "valid" if split == "val" and (dataset_dir / "labels" / "valid").is_dir() else split
                labels_dir = dataset_dir / "labels" / split_name
            if not self._within(dataset_dir, labels_dir) or not labels_dir.is_dir():
                raise ValueError(f"{split} split is missing its matching labels folder")
            image_files = {path.relative_to(image_dir).with_suffix(".txt").as_posix() for path in image_paths}
            label_files = [path for path in labels_dir.rglob("*.txt") if path.is_file()]
            if not label_files:
                raise ValueError(f"{split} split has no YOLO label files")
            label_names = {path.relative_to(labels_dir).as_posix() for path in label_files}
            unmatched = label_names - image_files
            if unmatched:
                raise ValueError(f"{split} labels contain files without matching images")
            for label_path in label_files:
                try:
                    lines = label_path.read_text(encoding="utf-8-sig").splitlines()
                except (OSError, UnicodeError) as error:
                    raise ValueError(f"could not read label file {label_path.relative_to(dataset_dir)}") from error
                for line_number, line in enumerate(lines, start=1):
                    if not line.strip():
                        continue
                    fields = line.split()
                    if len(fields) != 5:
                        raise ValueError(f"YOLO detection labels must have 5 values ({label_path.name}:{line_number})")
                    try:
                        class_id = int(fields[0])
                        x, y, width, height = map(float, fields[1:])
                    except ValueError as error:
                        raise ValueError(f"label contains a non-numeric value ({label_path.name}:{line_number})") from error
                    if not 0 <= class_id < class_count:
                        raise ValueError(f"label class id is outside data.yaml names ({label_path.name}:{line_number})")
                    if not all(0 <= value <= 1 for value in (x, y, width, height)) or width <= 0 or height <= 0:
                        raise ValueError(f"label coordinates must be normalized to 0..1 with positive size ({label_path.name}:{line_number})")
            split_stats[split] = {"images": image_count, "labelFiles": len(label_files)}

        try:
            manifest_lines = [line.strip() for line in train_manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        except (OSError, UnicodeError) as error:
            raise ValueError("Could not read CVAT train.txt manifest") from error
        if not manifest_lines:
            raise ValueError("CVAT train.txt manifest must list training images")
        seen_manifest_images: set[str] = set()
        for line in manifest_lines:
            if "\\" in line or ":" in line:
                raise ValueError("CVAT train.txt contains an unsafe image path")
            relative = PurePosixPath(line)
            parts = relative.parts
            if relative.is_absolute() or any(part in {"", ".", ".."} for part in parts):
                raise ValueError("CVAT train.txt contains an unsafe image path")
            # CVAT exports may include a leading `data/` in manifest entries
            # even though the ZIP places images/ at its dataset root.
            if parts and parts[0] == "data":
                parts = parts[1:]
            image_path = dataset_dir.joinpath(*parts)
            if not self._within(dataset_dir, image_path) or not image_path.is_file() or image_path.suffix.lower() not in _IMAGE_SUFFIXES:
                raise ValueError(f"CVAT train.txt references a missing image: {line}")
            if not self._within(train_image_dir, image_path):
                raise ValueError("CVAT train.txt may only reference images/train files")
            key = image_path.relative_to(train_image_dir).as_posix().casefold()
            if key in seen_manifest_images:
                raise ValueError("CVAT train.txt contains duplicate image paths")
            seen_manifest_images.add(key)
        if seen_manifest_images != {
            image.relative_to(train_image_dir).as_posix().casefold()
            for image in train_image_dir.rglob("*")
            if image.is_file() and image.suffix.lower() in _IMAGE_SUFFIXES
        }:
            raise ValueError("CVAT train.txt must list every image in images/train exactly once")

        return {"classes": classes, "classCount": class_count, "splits": split_stats}

    def delete_dataset(self, *, model: str, dataset_id: str) -> dict[str, Any]:
        target = next((item for item in self._targets() if item["id"] == model), None)
        if target is None:
            raise ValueError("unknown model or training script is unavailable")
        relative = PurePosixPath(str(dataset_id or ""))
        if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("invalid dataset id")
        dataset_root = self.train_root / target["directory"] / "dataset"
        destination = dataset_root.joinpath(*relative.parts)
        ancestors = [dataset_root.joinpath(*relative.parts[:index]) for index in range(1, len(relative.parts) + 1)]
        if any(path.is_symlink() for path in ancestors) or not self._within(dataset_root, destination) or not destination.is_dir():
            raise FileNotFoundError("dataset not found")
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("cannot delete a dataset while training is running")
            shutil.rmtree(destination)
            self._dataset_cache = None
        return {"deleted": str(relative)}

    def _split_dataset_contents(
        self,
        dataset_dir: Path,
        config: dict[str, Any],
        validation: dict[str, Any],
        percentages: dict[str, int],
        seed: str,
    ) -> dict[str, dict[str, int]]:
        split_names = ("train", "val", "test")
        sources: list[tuple[str, Path, Path]] = []
        for split in split_names:
            image_dir = self._split_directory(dataset_dir, dataset_dir, config, split)
            if image_dir is None or not self._count_images(image_dir):
                continue
            image_parts = list(image_dir.relative_to(dataset_dir).parts)
            image_parts[image_parts.index("images")] = "labels"
            sources.append((split, image_dir, dataset_dir.joinpath(*image_parts)))

        samples: list[tuple[str, Path, Path | None, Path]] = []
        for source_split, image_dir, label_dir in sources:
            for image_path in image_dir.rglob("*"):
                if not image_path.is_file() or image_path.suffix.lower() not in _IMAGE_SUFFIXES:
                    continue
                relative = image_path.relative_to(image_dir)
                label_path = label_dir / relative.with_suffix(".txt")
                samples.append((source_split, image_path, label_path if label_path.is_file() else None, relative))
        if not samples:
            raise ValueError("train split must point to a folder containing images")

        active_splits = [name for name in split_names if percentages[name] > 0]
        if len(samples) < len(active_splits):
            raise ValueError("Dataset needs at least one image for every split with a non-zero percentage")
        exact = {name: len(samples) * percentages[name] / 100 for name in split_names}
        counts = {name: int(exact[name]) for name in split_names}
        order = sorted(split_names, key=lambda name: (-(exact[name] - counts[name]), split_names.index(name)))
        for name in order[:len(samples) - sum(counts.values())]:
            counts[name] += 1
        for name in active_splits:
            if counts[name] == 0:
                donor = max((item for item in split_names if counts[item] > 1), key=lambda item: counts[item])
                counts[donor] -= 1
                counts[name] = 1

        random.Random(seed).shuffle(samples)
        assignments: dict[str, list[tuple[str, Path, Path | None, Path]]] = {name: [] for name in split_names}
        cursor = 0
        for name in split_names:
            assignments[name] = samples[cursor:cursor + counts[name]]
            cursor += counts[name]

        staging = Path(tempfile.mkdtemp(prefix=".split-staging-", dir=dataset_dir))
        staged_images = staging / "images"
        staged_labels = staging / "labels"
        staged_images.mkdir(parents=True)
        staged_labels.mkdir(parents=True)
        manifest_lines: dict[str, list[str]] = {name: [] for name in split_names}
        multiple_sources = len(sources) > 1
        for split in split_names:
            for source_split, image_path, label_path, relative in assignments[split]:
                output_relative = Path(source_split) / relative if multiple_sources else relative
                image_destination = staged_images / split / output_relative
                image_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(image_path), str(image_destination))
                if label_path is not None:
                    label_destination = staged_labels / split / output_relative.with_suffix(".txt")
                    label_destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(label_path), str(label_destination))
                manifest_lines[split].append(PurePosixPath("data", "images", split, *output_relative.parts).as_posix())

        shutil.rmtree(dataset_dir / "images")
        if (dataset_dir / "labels").exists():
            shutil.rmtree(dataset_dir / "labels")
        os.replace(staged_images, dataset_dir / "images")
        os.replace(staged_labels, dataset_dir / "labels")
        config["path"] = "."
        config["train"] = "images/train"
        config["nc"] = int(validation["classCount"])
        for split in ("val", "test"):
            if percentages[split] > 0:
                config[split] = f"images/{split}"
            else:
                config.pop(split, None)
        (dataset_dir / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        for split in split_names:
            if manifest_lines[split]:
                (dataset_dir / f"{split}.txt").write_text("\n".join(manifest_lines[split]) + "\n", encoding="utf-8")
        shutil.rmtree(staging, ignore_errors=True)
        return {
            split: {"images": len(assignments[split]), "labelFiles": sum(sample[2] is not None for sample in assignments[split])}
            for split in split_names
        }

    def upload_dataset(
        self,
        *,
        model: str,
        dataset_name: str,
        archive: bytes,
    ) -> dict[str, Any]:
        if not isinstance(archive, bytes) or not archive:
            raise ValueError("Choose a non-empty ZIP dataset archive")
        if len(archive) > _MAX_DATASET_ARCHIVE_BYTES:
            raise ValueError("Dataset ZIP exceeds the 256 MB upload limit")
        if not _DATASET_NAME_PATTERN.fullmatch(dataset_name or ""):
            raise ValueError("Dataset name must use 1-64 letters, numbers, underscores, or hyphens")
        target = next((item for item in self._targets() if item["id"] == model), None)
        if target is None:
            raise ValueError("unknown model or training script is unavailable")

        dataset_root = self.train_root / target["directory"] / "dataset"
        if not self._within(self.train_root, dataset_root):
            raise ValueError("Model dataset folder must remain inside model/train")
        dataset_root.mkdir(parents=True, exist_ok=True)
        destination = dataset_root / dataset_name
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("A dataset with this name already exists; choose another name")

        try:
            with zipfile.ZipFile(BytesIO(archive)) as zipped:
                entries = zipped.infolist()
                files = [entry for entry in entries if not entry.is_dir()]
                if not files or len(entries) > _MAX_DATASET_ARCHIVE_FILES:
                    raise ValueError("ZIP must contain files and stay within the 50,000-file limit")
                total_size = sum(entry.file_size for entry in files)
                if total_size > _MAX_DATASET_UNPACKED_BYTES:
                    raise ValueError("Unpacked dataset exceeds the 4 GB safety limit")
                seen: set[str] = set()
                ignored_paths: set[str] = set()
                for entry in entries:
                    raw_name = entry.filename
                    if "\\" in raw_name or ":" in raw_name:
                        raise ValueError("ZIP contains an unsafe path")
                    normalized = raw_name.rstrip("/")
                    if not normalized:
                        continue
                    if normalized.startswith("__MACOSX/") or Path(normalized).name in {".DS_Store", "Thumbs.db"}:
                        ignored_paths.add(normalized.casefold())
                        continue
                    parts = normalized.split("/")
                    if normalized.startswith("/") or any(part in {"", ".", ".."} for part in parts):
                        raise ValueError("ZIP contains an unsafe path")
                    if any(part.startswith(".") for part in parts):
                        raise ValueError("ZIP must not contain hidden files or folders")
                    if any(part.endswith((".", " ")) or _WINDOWS_RESERVED_PATH.fullmatch(part) for part in parts):
                        raise ValueError("ZIP contains a Windows-reserved path")
                    if "\x00" in normalized:
                        raise ValueError("ZIP contains an invalid path")
                    key = normalized.casefold()
                    if key in seen:
                        raise ValueError("ZIP contains duplicate paths")
                    seen.add(key)
                    mode = (entry.external_attr >> 16) & 0o170000
                    if mode == stat.S_IFLNK:
                        raise ValueError("ZIP must not contain symbolic links")
                    if entry.flag_bits & 0x1:
                        raise ValueError("Encrypted ZIP files are not supported")
                    if entry.is_dir():
                        continue
                    if Path(parts[-1]).suffix.lower() not in _DATASET_FILE_SUFFIXES:
                        raise ValueError(f"Unsupported file type in ZIP: {parts[-1]}")
                    if entry.file_size and (entry.compress_size == 0 or entry.file_size / max(entry.compress_size, 1) > 500):
                        raise ValueError("ZIP compression ratio exceeds the safety limit")

                with tempfile.TemporaryDirectory(prefix=".dataset-upload-", dir=dataset_root) as temporary:
                    staging = Path(temporary) / "payload"
                    staging.mkdir()
                    extracted_total = 0
                    for entry in files:
                        if entry.filename.rstrip("/").casefold() in ignored_paths:
                            continue
                        target_path = staging.joinpath(*entry.filename.split("/"))
                        if not self._within(staging, target_path):
                            raise ValueError("ZIP contains an unsafe path")
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        entry_size = 0
                        with zipped.open(entry) as source, target_path.open("xb") as output:
                            while chunk := source.read(1024 * 1024):
                                entry_size += len(chunk)
                                extracted_total += len(chunk)
                                if extracted_total > _MAX_DATASET_UNPACKED_BYTES:
                                    raise ValueError("Unpacked dataset exceeds the 4 GB safety limit")
                                output.write(chunk)
                        if entry_size != entry.file_size:
                            raise ValueError("ZIP entry size does not match its directory metadata")

                    children = list(staging.iterdir())
                    dataset_dir = staging
                    if len(children) == 1 and children[0].is_dir() and not (staging / "data.yaml").exists():
                        dataset_dir = children[0]
                    validation = self._validate_uploaded_dataset(model, dataset_dir)
                    dataset_dir.rename(destination)
        except zipfile.BadZipFile as error:
            raise ValueError("The uploaded file is not a valid ZIP archive") from error

        self._dataset_cache = None
        self._results_cache = None
        dataset_info = self._dataset_info(model, dataset_root, destination)
        return {"dataset": dataset_info, "validation": validation}

    def split_dataset(self, *, model: str, dataset_id: str, percentages: dict[str, int]) -> dict[str, Any]:
        target = next((item for item in self._targets() if item["id"] == model), None)
        if target is None:
            raise ValueError("unknown model or training script is unavailable")
        try:
            split_percentages = {name: percentages[name] for name in ("train", "val", "test")}
        except (KeyError, TypeError) as error:
            raise ValueError("Provide train, validation, and test split percentages") from error
        if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100 for value in split_percentages.values()):
            raise ValueError("Split percentages must be whole numbers from 0 to 100")
        if split_percentages["train"] < 1 or sum(split_percentages.values()) != 100:
            raise ValueError("Split percentages must total 100 and train must be greater than 0")
        relative = PurePosixPath(str(dataset_id or ""))
        if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("invalid dataset id")
        dataset_root = self.train_root / target["directory"] / "dataset"
        dataset_dir = dataset_root.joinpath(*relative.parts)
        ancestors = [dataset_root.joinpath(*relative.parts[:index]) for index in range(1, len(relative.parts) + 1)]
        if any(path.is_symlink() for path in ancestors) or not self._within(dataset_root, dataset_dir) or not dataset_dir.is_dir():
            raise FileNotFoundError("dataset not found")
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("cannot split a dataset while training is running")
            validation = self._validate_uploaded_dataset(model, dataset_dir, allow_split_layout=True)
            with (dataset_dir / "data.yaml").open("r", encoding="utf-8") as stream:
                config = yaml.safe_load(stream)
            validation["splits"] = self._split_dataset_contents(
                dataset_dir,
                config,
                validation,
                split_percentages,
                f"{model}:{relative.as_posix()}",
            )
            self._dataset_cache = None
            dataset_info = self._dataset_info(model, dataset_root, dataset_dir)
        return {"dataset": dataset_info, "validation": validation, "splitPercentages": split_percentages}

    def _create_worker_bundle(
        self, *, model: str, directory: str, dataset: str, resume: bool, destination: Path,
    ) -> Path:
        """Package only the selected model's scripts, weights and dataset for one transient job."""
        bundle_root = destination / "bundle"
        remote_train_root = bundle_root / "model" / "train"
        remote_train_root.mkdir(parents=True)
        launcher = self.train_root / "train.py"
        if not launcher.is_file():
            raise FileNotFoundError("model/train/train.py is missing")
        shutil.copy2(launcher, remote_train_root / "train.py")
        source_target = self.train_root / directory
        target_copy = remote_train_root / directory
        selected_parts = PurePosixPath(dataset).parts if dataset != "all" else ()
        source_dataset_root = source_target / "dataset"

        def ignored(source: str, names: list[str]) -> set[str]:
            current = Path(source).resolve()
            blocked = {name for name in names if name in {".venv", ".git", "__pycache__", ".ultralytics_view"}}
            if current == source_target.resolve() and not resume and "runs" in names:
                blocked.add("runs")
            if dataset != "all" and dataset != "." and self._within(source_dataset_root, current):
                try:
                    relative_parts = current.relative_to(source_dataset_root.resolve()).parts
                except ValueError:
                    relative_parts = ()
                if len(relative_parts) < len(selected_parts):
                    selected_name = selected_parts[len(relative_parts)]
                    blocked.update(
                        name for name in names
                        if (current / name).is_dir() and name != selected_name
                    )
            return blocked

        shutil.copytree(source_target, target_copy, ignore=ignored)
        source_model = self.train_root.parent / model
        if source_model.is_dir():
            shutil.copytree(
                source_model,
                bundle_root / "model" / model,
                ignore=shutil.ignore_patterns("__pycache__", ".git", ".venv"),
            )
        archive_path = destination / "job.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path in bundle_root.rglob("*"):
                if path.is_file() and not path.is_symlink():
                    archive.write(path, path.relative_to(bundle_root).as_posix())
        return archive_path

    def _worker_command(self, config: dict[str, Any], remote_command: str, *, scp: bool = False) -> list[str]:
        args = self._ssh_args(config, scp=scp)
        if scp:
            return args
        args.append(remote_command)
        return args

    def _remote_start(self, run_id: str, config: dict[str, Any], settings: dict[str, Any]) -> None:
        remote_root = f"{_REMOTE_JOB_ROOT}/{run_id}"
        remote_archive = f"{remote_root}/job.zip"
        remote_job = f"{remote_root}/job"
        remote_results = f"{remote_root}/results.tgz"
        container = f"onevision-train-{run_id}"
        try:
            with tempfile.TemporaryDirectory(prefix="onevision_training_") as temporary:
                archive_path = self._create_worker_bundle(
                    model=str(settings["model"]),
                    directory=str(settings["modelDirectory"]),
                    dataset=str(settings["dataset"]),
                    resume=bool(settings["resume"]),
                    destination=Path(temporary),
                )
                remote = f"{config['user']}@{config['host']}:{remote_archive}"
                mkdir_command = f"mkdir -p -- {shlex.quote(remote_root)}"
                subprocess.run(
                    self._worker_command(config, mkdir_command), check=True,
                    capture_output=True, text=True, timeout=30,
                )
                subprocess.run(
                    self._ssh_args(config, scp=True) + [str(archive_path), remote],
                    check=True, capture_output=True, text=True, timeout=600,
                )

            runner_args = [
                "python", "train.py", "--model", str(settings["model"]), "--yes",
                "--resume", "yes" if settings["resume"] else "no",
                "--dataset", str(settings["dataset"]), "--epochs", str(settings["epochs"]),
                "--batch", str(settings["batch"]), "--imgsz", str(settings["imageSize"]),
                "--workers", str(settings["workers"]), "--device", str(settings["device"]),
            ]
            image = str(config["dockerImage"])
            cpu_limit = str(config.get("cpus", 4))
            memory_limit = str(config.get("memory", "6g"))
            docker_args = [
                "docker", "run", "--rm", "--name", container,
                "--cpus", cpu_limit, "--memory", memory_limit, "--shm-size", "1g",
                "--network", "none", "--volume", f"{remote_job}:/job:rw",
                "--workdir", "/job/model/train", image, *runner_args,
            ]
            docker_command = " ".join(shlex.quote(part) for part in docker_args)
            result_tar = (
                f"if [ -d {shlex.quote(remote_job + '/model/train/' + str(settings['modelDirectory']) + '/runs')} ]; then "
                f"tar -czf {shlex.quote(remote_results)} -C "
                f"{shlex.quote(remote_job + '/model/train/' + str(settings['modelDirectory']))} runs; fi"
            )
            shell_command = (
                f"python3 -m zipfile -e {shlex.quote(remote_archive)} {shlex.quote(remote_job)} && "
                f"{docker_command}; code=$?; {result_tar}; exit $code"
            )
            process = subprocess.Popen(
                self._worker_command(config, "sh -lc " + shlex.quote(shell_command)),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                start_new_session=os.name != "nt",
            )
            with self._lock:
                if self._run and self._run["id"] == run_id:
                    if self._run["status"] == "stopped":
                        process.terminate()
                        return
                    self._process = process
                    self._run["status"] = "running"
                    self._reader = threading.current_thread()
            self._read_output(process)
        except Exception as error:
            with self._lock:
                if self._run and self._run["id"] == run_id:
                    self._logs.append(f"Remote training setup failed: {error}")
                    self._run["status"] = "failed"
                    self._run["exitCode"] = -1
                    self._run["finishedAt"] = datetime.now(timezone.utc).isoformat()
        finally:
            self._collect_worker_results(run_id, config, remote_results, remote_job, container)

    def _collect_worker_results(
        self, run_id: str, config: dict[str, Any], remote_results: str, remote_job: str, container: str,
    ) -> None:
        with self._lock:
            run = dict(self._run) if self._run and self._run["id"] == run_id else None
        if run:
            try:
                with tempfile.TemporaryDirectory(prefix="onevision_results_") as temporary:
                    local_archive = Path(temporary) / "results.tgz"
                    remote = f"{config['user']}@{config['host']}:{remote_results}"
                    subprocess.run(
                        self._ssh_args(config, scp=True) + [remote, str(local_archive)],
                        check=True, capture_output=True, text=True, timeout=180,
                    )
                    destination = self.train_root / str(run["modelDirectory"])
                    if destination.is_symlink():
                        raise ValueError("training results destination is unsafe")
                    destination.mkdir(parents=True, exist_ok=True)
                    with tarfile.open(local_archive, "r:gz") as archive:
                        for member in archive.getmembers():
                            path = PurePosixPath(member.name)
                            if (not path.parts or path.is_absolute()
                                    or any(part in {"", ".", ".."} for part in path.parts)
                                    or path.parts[0] != "runs" or not (member.isfile() or member.isdir())):
                                raise ValueError("unsafe training result path from worker")
                        for member in archive.getmembers():
                            output_path = destination.joinpath(*PurePosixPath(member.name).parts)
                            if output_path.is_symlink() or not self._within(destination, output_path):
                                raise ValueError("training result would overwrite an unsafe path")
                            parent = output_path.parent
                            while parent != destination:
                                if parent.is_symlink():
                                    raise ValueError("training result path contains a symbolic link")
                                parent = parent.parent
                            if member.isdir():
                                output_path.mkdir(parents=True, exist_ok=True)
                            else:
                                output_path.parent.mkdir(parents=True, exist_ok=True)
                                source = archive.extractfile(member)
                                if source is None:
                                    raise ValueError("training result archive contains an unreadable file")
                                with source, output_path.open("wb") as target:
                                    shutil.copyfileobj(source, target)
                with self._lock:
                    self._results_cache = None
            except subprocess.CalledProcessError as error:
                with self._lock:
                    self._logs.append(f"Could not retrieve worker results: {error.stderr or error}")
            except (OSError, tarfile.TarError, ValueError) as error:
                with self._lock:
                    self._logs.append(f"Could not unpack worker results: {error}")
        cleanup = f"docker rm -f {shlex.quote(container)} >/dev/null 2>&1 || true; rm -rf -- {shlex.quote(remote_job.rsplit('/job', 1)[0])}"
        try:
            subprocess.run(
                self._worker_command(config, cleanup), check=False,
                capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as error:
            with self._lock:
                if self._run and self._run["id"] == run_id:
                    self._logs.append(f"Worker cleanup will need retry: {error}")
        with self._lock:
            if self._run and self._run["id"] == run_id:
                code = self._process.poll() if self._process else self._run.get("exitCode")
                if self._run["status"] not in {"stopped", "failed"}:
                    self._run["status"] = "completed" if code == 0 else "failed"
                self._run["exitCode"] = code
                self._run["finishedAt"] = datetime.now(timezone.utc).isoformat()
                self._process = None

    def start(
        self,
        *,
        model: str,
        epochs: int = 50,
        batch: int = 16,
        image_size: int = 640,
        workers: int = 0,
        device: str = "auto",
        resume: bool = False,
        dataset: str = "all",
        server_id: str = "local",
    ) -> dict[str, Any]:
        if not 1 <= epochs <= 2000 or not 1 <= batch <= 512:
            raise ValueError("epochs or batch is outside the supported range")
        if not 32 <= image_size <= 4096 or not 0 <= workers <= 64:
            raise ValueError("image size or worker count is outside the supported range")
        device = str(device or "auto").strip().lower()
        if device not in {"auto", "cpu"} and not re.fullmatch(r"\d+|xpu:\d+", device):
            raise ValueError("device must be auto, cpu, a GPU index, or xpu:<index>")

        worker = self._worker_config(str(server_id or "local"))
        if worker:
            worker_device = str(worker.get("device") or "cpu").strip().lower()
            if worker_device == "cpu":
                if device not in {"auto", "cpu"}:
                    raise ValueError("selected training server has CPU resources only")
                device = "cpu"

        targets = {item["id"] for item in self._targets()}
        if model not in targets:
            raise ValueError("unknown model or training script is unavailable")
        selected_target = next(item for item in self._targets() if item["id"] == model)
        known_datasets = {item["id"] for item in next(
            item for item in self.datasets()["models"] if item["id"] == model
        )["datasets"]}
        if dataset != "all" and dataset not in known_datasets:
            raise ValueError("unknown dataset for the selected model")
        if resume and not selected_target["resumeAvailable"]:
            raise ValueError("no resumable last.pt checkpoint exists for this model")
        runner = self.train_root / "train.py"
        if not runner.is_file():
            raise FileNotFoundError("model/train/train.py is missing")

        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("another training run is already active")
            if self._run and self._run.get("status") in {"starting", "running", "stopping"}:
                raise RuntimeError("another training run is already active")
            self._logs.clear()
            run_id = uuid.uuid4().hex
            settings = {
                "epochs": epochs,
                "batch": batch,
                "imageSize": image_size,
                "workers": workers,
                "device": device,
                "resume": resume,
            }
            self._run = {
                "id": run_id,
                "model": model,
                "modelDirectory": selected_target["directory"],
                "workerId": str(server_id or "local"),
                "status": "starting" if worker else "running",
                "startedAt": datetime.now(timezone.utc).isoformat(),
                "finishedAt": None,
                "exitCode": None,
                "settings": settings,
            }
            if worker:
                remote_settings = {
                    **settings,
                    "model": model,
                    "modelDirectory": selected_target["directory"],
                    "dataset": dataset,
                }
                thread = threading.Thread(
                    target=self._remote_start,
                    args=(run_id, worker, remote_settings),
                    name=f"remote-model-training-{run_id[:8]}",
                    daemon=True,
                )
                thread.start()
                return self.status()
            command = [
                sys.executable,
                str(runner),
                "--model",
                model,
                "--yes",
                "--resume",
                "yes" if resume else "no",
                "--dataset",
                dataset,
                "--epochs",
                str(epochs),
                "--batch",
                str(batch),
                "--imgsz",
                str(image_size),
                "--workers",
                str(workers),
            ]
            if device != "auto":
                command.extend(["--device", device])
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            self._process = subprocess.Popen(
                command,
                cwd=str(self.train_root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=flags,
                start_new_session=os.name != "nt",
            )
            self._run["status"] = "running"
            self._reader = threading.Thread(
                target=self._read_output,
                args=(self._process,),
                name="model-training-log-reader",
                daemon=True,
            )
            self._reader.start()
            return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            run = dict(self._run) if self._run else None
            if process is None and run and run.get("workerId") != "local" and run.get("status") == "starting":
                self._run["status"] = "stopped"
                self._run["finishedAt"] = datetime.now(timezone.utc).isoformat()
                return self.status()
            if process is None or process.poll() is not None:
                return self.status()
            if self._run is not None:
                self._run["status"] = "stopping"
        if run and run.get("workerId") != "local":
            try:
                config = self._worker_config(str(run["workerId"]))
                if config:
                    command = f"docker stop --time 10 {shlex.quote('onevision-train-' + str(run['id']))} >/dev/null 2>&1 || true"
                    subprocess.run(
                        self._worker_command(config, command), check=False,
                        capture_output=True, text=True, timeout=30,
                    )
            except (OSError, subprocess.SubprocessError, ValueError):
                with self._lock:
                    self._logs.append("Could not send stop command to remote worker; waiting for training process")
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            else:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            if self._run is not None and self._run.get("status") != "failed":
                self._run["status"] = "stopped"
                self._run["exitCode"] = process.poll()
                self._run["finishedAt"] = datetime.now(timezone.utc).isoformat()
            return self.status()


training_runs = TrainingRunManager()
