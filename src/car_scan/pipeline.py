"""Modular classification, association, routing, and validation helpers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .compute import bind_yolo_device, yolo_predict


UNKNOWN_CLASS = {
    "raw_class": "unknown",
    "display_name": "Unknown",
    "normalized_class": "unknown",
    "confidence": 0.0,
    "box": None,
}


def load_pipeline_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Pipeline config must contain a mapping: {path}")
    return value


def _normalise_name(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return value or "unknown"


class ClassMapper:
    """Map display/normalised names without changing a model's raw class."""

    def __init__(self, mapping: dict[str, Any] | None = None) -> None:
        self.mapping = mapping or {}

    def map(
        self,
        raw_class: str,
        confidence: float = 0.0,
        box: list[int] | None = None,
        class_id: int | None = None,
    ) -> dict[str, Any]:
        raw_class = str(raw_class)
        configured = self.mapping.get(raw_class, {})
        if isinstance(configured, str):
            configured = {"normalized_class": configured}
        if not isinstance(configured, dict):
            configured = {}
        result = {
            "raw_class": raw_class,
            "display_name": str(configured.get("display_name") or raw_class),
            "normalized_class": str(
                configured.get("normalized_class")
                or configured.get("vehicle_type")
                or configured.get("plate_type")
                or _normalise_name(raw_class)
            ),
            "confidence": round(float(confidence), 4),
            "box": box,
        }
        if class_id is not None:
            result["class_id"] = int(class_id)
        return result


class YoloRegionClassifier:
    """Generic YOLO adapter whose class names always come from the model."""

    def __init__(
        self,
        model_path: Path | None,
        mapping: dict[str, Any] | None = None,
        device: str | None = None,
    ) -> None:
        self.mapper = ClassMapper(mapping)
        self.model = None
        self.model_path = model_path
        self.device = device or "cpu"
        if model_path is None:
            return
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Missing dependency 'ultralytics'") from error
        self.model = YOLO(str(model_path))
        bind_yolo_device(self.model, self.device)

    def _name(self, class_id: int) -> str:
        if self.model is None:
            return str(class_id)
        names = self.model.names
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        return str(names[class_id])

    def detect(self, image: Any, confidence: float, imgsz: int) -> list[dict[str, Any]]:
        if self.model is None:
            return []
        prediction = yolo_predict(
            self.model, image, device=self.device, conf=confidence, imgsz=imgsz, verbose=False
        )[0]
        output: list[dict[str, Any]] = []
        if getattr(prediction, "boxes", None) is not None:
            for xyxy, score, class_id in zip(
                prediction.boxes.xyxy.cpu().tolist(),
                prediction.boxes.conf.cpu().tolist(),
                prediction.boxes.cls.cpu().tolist(),
            ):
                class_index = int(class_id)
                output.append(
                    self.mapper.map(
                        self._name(class_index),
                        float(score),
                        [int(value) for value in xyxy],
                        class_index,
                    )
                )
        elif getattr(prediction, "probs", None) is not None:
            class_index = int(prediction.probs.top1)
            top_confidence = prediction.probs.top1conf
            if hasattr(top_confidence, "item"):
                top_confidence = top_confidence.item()
            output.append(
                self.mapper.map(
                    self._name(class_index),
                    float(top_confidence),
                    None,
                    class_index,
                )
            )
        return output

    def classify(self, image: Any, confidence: float, imgsz: int) -> dict[str, Any]:
        detections = self.detect(image, confidence, imgsz)
        return max(detections, key=lambda item: float(item["confidence"])) if detections else dict(UNKNOWN_CLASS)


class CountryClassifier:
    """Choose country independently from vehicle and plate classifications."""

    def __init__(self, unknown_score: float = 0.0) -> None:
        self.unknown_score = float(unknown_score)

    def classify(self, readings: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if not readings:
            return {"country": "unknown", "confidence": 0.0, "margin": 0.0, "raw_country": "unknown"}
        ranked = sorted(readings, key=lambda key: float(readings[key].get("score", 0.0)), reverse=True)
        winner = ranked[0]
        winner_score = float(readings[winner].get("score", 0.0))
        runner_score = float(readings[ranked[1]].get("score", 0.0)) if len(ranked) > 1 else 0.0
        margin = winner_score - runner_score
        country = winner if winner_score > self.unknown_score else "unknown"
        confidence = max(0.0, min(1.0, 0.50 + margin * 0.50)) if country != "unknown" else 0.0
        return {
            "country": country,
            "confidence": round(confidence, 4),
            "margin": round(margin, 4),
            "raw_country": winner,
        }


def box_iou(first: list[int], second: list[int]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


class VehiclePlateAssociator:
    """Associate each plate with the best vehicle using containment, IoU, and distance."""

    @staticmethod
    def _score(plate_box: list[int], vehicle_box: list[int]) -> float:
        px1, py1, px2, py2 = plate_box
        vx1, vy1, vx2, vy2 = vehicle_box
        center_x, center_y = (px1 + px2) / 2, (py1 + py2) / 2
        contained = vx1 <= center_x <= vx2 and vy1 <= center_y <= vy2
        overlap = box_iou(plate_box, vehicle_box)
        vehicle_center = ((vx1 + vx2) / 2, (vy1 + vy2) / 2)
        distance = math.hypot(center_x - vehicle_center[0], center_y - vehicle_center[1])
        diagonal = max(1.0, math.hypot(vx2 - vx1, vy2 - vy1))
        distance_score = max(0.0, 1.0 - distance / diagonal)
        return (1.0 if contained else 0.0) + overlap * 0.75 + distance_score * 0.25

    def associate(self, plate_box: list[int], vehicles: list[dict[str, Any]]) -> dict[str, Any]:
        candidates = [vehicle for vehicle in vehicles if isinstance(vehicle.get("box"), list)]
        if not candidates:
            return {**UNKNOWN_CLASS, "vehicle_id": None, "tracking_id": None, "association_score": 0.0}
        best = max(candidates, key=lambda vehicle: self._score(plate_box, vehicle["box"]))
        score = self._score(plate_box, best["box"])
        # Do not attach a plate to a distant unrelated vehicle.
        if score < 0.15:
            return {**UNKNOWN_CLASS, "vehicle_id": None, "tracking_id": None, "association_score": 0.0}
        result = dict(best)
        result["association_score"] = round(score, 4)
        return result


@dataclass(frozen=True)
class ProcessingContext:
    country: str
    vehicle_type: str
    plate_type: str


class ContextRouter:
    """Resolve preprocessing and OCR profiles using context with common fallbacks."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.preprocessing = config.get("preprocessing_profiles", {})
        self.ocr = config.get("ocr_routes", {})

    @staticmethod
    def _lookup(routes: dict[str, Any], context: ProcessingContext, default: str) -> str:
        keys = (
            f"{context.country}|{context.vehicle_type}|{context.plate_type}",
            f"{context.country}|{context.vehicle_type}|*",
            f"{context.country}|*|{context.plate_type}",
            f"{context.country}|*|*",
            "*|*|*",
        )
        for key in keys:
            value = routes.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                return str(value.get("profile") or value.get("model") or default)
        return default

    def preprocessing_profile(self, context: ProcessingContext) -> str:
        return self._lookup(self.preprocessing, context, "common")

    def ocr_model(self, context: ProcessingContext) -> str:
        return self._lookup(self.ocr, context, "common_ocr")


class ContextValidator:
    """Context-aware validation with country rules and configurable overrides."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.rules = config.get("validation", {})

    def _rule(self, country: str, vehicle_type: str, plate_type: str) -> tuple[str, dict[str, Any]]:
        for key in (
            f"{country}|{vehicle_type}|{plate_type}",
            f"{country}|{vehicle_type}|*",
            f"{country}|*|{plate_type}",
            f"{country}|*|*",
            "*|*|*",
        ):
            value = self.rules.get(key)
            if isinstance(value, dict):
                return key, value
        return f"{country}|{vehicle_type}|{plate_type}", {}

    def validate(
        self,
        country: str,
        vehicle_type: str,
        plate_type: str,
        prefix: str,
        number: str,
        province: str = "",
        layout: str = "",
    ) -> dict[str, Any]:
        reasons: list[str] = []
        if country not in ("thai", "lao"):
            reasons.append("country_unknown")
        if country == "thai":
            thai_letters = 1 <= len(prefix) <= 3 and all("\u0e01" <= character <= "\u0e2e" for character in prefix)
            thai_digits = prefix.isdigit() and len(prefix) in {2, 3}
            motorcycle = (
                len(number) == 4
                and number.isdigit()
                and bool(province)
                and (vehicle_type == "motorcycle" or not prefix)
            )
            if not (thai_letters or thai_digits or motorcycle):
                reasons.append("thai_prefix_must_have_2_digits")
            if not (len(number) == 4 and number.isdigit()):
                reasons.append("thai_number_must_have_4_digits")
        elif country == "lao":
            if not (
                len(prefix) == 2
                and all("\u0e80" <= character <= "\u0eff" for character in prefix)
            ):
                reasons.append("lao_prefix_must_have_2_characters")
            if not (len(number) == 4 and number.isdigit()):
                reasons.append("lao_number_must_have_4_digits")

        rule_key, configured = self._rule(country, vehicle_type, plate_type)
        text = f"{prefix}{number}"
        allowed = configured.get("allowed_pattern")
        if allowed and not re.fullmatch(str(allowed), text):
            reasons.append("configured_pattern_mismatch")
        prefix_pattern = configured.get("allowed_prefix_pattern")
        if prefix_pattern and not re.fullmatch(str(prefix_pattern), prefix):
            reasons.append("prefix_pattern_mismatch")
        number_pattern = configured.get("allowed_number_pattern")
        if number_pattern and not re.fullmatch(str(number_pattern), number):
            reasons.append("number_pattern_mismatch")
        allowed_provinces = configured.get("allowed_provinces")
        if isinstance(allowed_provinces, list) and province not in allowed_provinces:
            reasons.append("province_not_allowed")
        allowed_layouts = configured.get("allowed_layouts")
        if isinstance(allowed_layouts, list) and layout not in allowed_layouts:
            reasons.append("layout_not_allowed")
        return {"valid": not reasons, "reasons": reasons, "rule": rule_key}
