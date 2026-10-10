"""Interactive launcher for every trainable model in this repository.

The launcher discovers ``*-train/dataset/train.py`` folders instead of keeping
a hard-coded model list.  It therefore follows the real folder names under
``model/train`` and can report unfinished Ultralytics runs before training.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

os.environ["MPLBACKEND"] = "Agg"

TRAIN_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = TRAIN_ROOT.parent


@dataclass(frozen=True)
class TrainingTarget:
    """A model whose local training entry point can be launched."""

    name: str
    training_dir: Path
    script: Path
    deployed_model_dir: Path


@dataclass(frozen=True)
class ResumeCandidate:
    checkpoint: Path
    completed_epochs: int | None
    target_epochs: int | None

    @property
    def unfinished(self) -> bool:
        return (
            self.completed_epochs is None
            or self.target_epochs is None
            or self.completed_epochs < self.target_epochs
        )


def discover_targets() -> list[TrainingTarget]:
    """Return trainable models using their actual ``*-train`` folder names."""

    targets: list[TrainingTarget] = []
    for training_dir in sorted(TRAIN_ROOT.glob("*-train")):
        script = training_dir / "dataset" / "train.py"
        if not script.is_file():
            continue
        name = training_dir.name.removesuffix("-train")
        targets.append(TrainingTarget(name, training_dir, script, MODEL_ROOT / name))
    return targets


def completed_epochs(run_dir: Path) -> int | None:
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


def resume_candidates(target: TrainingTarget) -> list[ResumeCandidate]:
    """Find newest ``last.pt`` checkpoints and identify unfinished runs."""

    candidates: list[ResumeCandidate] = []
    runs_dir = target.training_dir / "runs"
    if not runs_dir.is_dir():
        return candidates
    for checkpoint in runs_dir.glob("**/weights/last.pt"):
        run_dir = checkpoint.parent.parent
        args_file = run_dir / "args.yaml"
        target_epochs: int | None = None
        if args_file.is_file():
            try:
                with args_file.open("r", encoding="utf-8") as stream:
                    configured = yaml.safe_load(stream) or {}
                target_epochs = int(configured.get("epochs"))
            except (OSError, TypeError, ValueError, yaml.YAMLError):
                pass
        candidates.append(ResumeCandidate(checkpoint, completed_epochs(run_dir), target_epochs))
    return sorted(candidates, key=lambda item: item.checkpoint.stat().st_mtime, reverse=True)


def choose_target(targets: list[TrainingTarget], requested: str | None) -> TrainingTarget:
    if requested:
        selected = next((target for target in targets if target.name == requested), None)
        if selected is None:
            available = ", ".join(target.name for target in targets) or "(none)"
            raise ValueError(f"Unknown model '{requested}'. Available models: {available}")
        return selected

    print("Models available for training:")
    for index, target in enumerate(targets, start=1):
        location = target.deployed_model_dir if target.deployed_model_dir.is_dir() else target.training_dir
        print(f"  {index}. {target.name} ({location})")
    while True:
        answer = input("Choose model number (or q to cancel): ").strip().lower()
        if answer in {"q", "quit", "cancel", ""}:
            raise SystemExit("Training cancelled.")
        if answer.isdigit() and 1 <= int(answer) <= len(targets):
            return targets[int(answer) - 1]
        print("Please enter a number from the list.")


def ask_yes_no(question: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{question} {suffix}: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes", "ใช่", "ตกลง"}:
            return True
        if answer in {"n", "no", "ไม่"}:
            return False
        print("Please answer y or n.")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Choose and train a local car-scan model.")
    parser.add_argument("--model", help="Model folder name, for example detect_license.")
    parser.add_argument("--yes", action="store_true", help="Start without the final confirmation prompt.")
    parser.add_argument("--resume", choices=("ask", "yes", "no"), default="ask")
    parser.add_argument("--list", action="store_true", help="List discovered trainable models and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Show the selected command without starting training.")
    return parser.parse_known_args()


def print_resume_status(candidates: Iterable[ResumeCandidate]) -> ResumeCandidate | None:
    unfinished = next((candidate for candidate in candidates if candidate.unfinished), None)
    if unfinished:
        print("Unfinished training checkpoint found:")
        print(f"  {unfinished.checkpoint}")
        print(f"  completed epochs: {unfinished.completed_epochs or 0}/{unfinished.target_epochs or '?'}")
    return unfinished


def main() -> int:
    args, forwarded_args = parse_args()
    targets = discover_targets()
    if not targets:
        raise FileNotFoundError(f"No dataset/train.py scripts found under {TRAIN_ROOT}")
    if args.list:
        for target in targets:
            print(target.name)
        return 0

    target = choose_target(targets, args.model)
    unfinished = print_resume_status(resume_candidates(target))
    resume = False
    if unfinished and args.resume != "no":
        resume = args.resume == "yes" or ask_yes_no("Resume this interrupted run?", default=True)

    if not args.yes and not ask_yes_no(
        f"Train model '{target.name}'{' by resuming its previous run' if resume else ' from its existing weights'}?",
        default=True,
    ):
        print("Training cancelled.")
        return 0

    command = [sys.executable, str(target.script), *forwarded_args]
    # The target scripts keep their own model-specific training settings.
    # This environment contract lets scripts opt into a non-interactive resume
    # selection while retaining compatibility with their standalone usage.
    environment = os.environ.copy()
    environment["CAR_SCAN_TRAIN_RESUME"] = "yes" if resume else "no"
    environment["CAR_SCAN_TRAIN_CHECKPOINT"] = str(unfinished.checkpoint) if resume and unfinished else ""
    print(f"Launching: {' '.join(command)}")
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=target.script.parent, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
