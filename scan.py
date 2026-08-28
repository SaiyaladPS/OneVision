"""Scan Thai and Lao vehicle licence plates from an image.

Pipeline
--------
1. ``type_car_license`` identifies the vehicle type in the source image.
2. ``detect_license`` finds each plate in the source image.
3. Each plate crop is read by the Thai and Lao YOLO character models.
   The model with the strongest character detections identifies the country.
4. The selected crop is sent to the custom PaddleOCR recogniser in
   ``model/paddleocr_train2`` when available, then to language-aware
   PaddleOCR/Tesseract fallbacks before using the YOLO character reading.

Example:
    python scan.py path/to/car.jpg
    python scan.py path/to/car.jpg --ocr-weights runs/ocr/plate_rec_digits/best_accuracy

The OCR training source and the locally fine-tuned checkpoint are included in
the project when training has completed.  If a checkpoint is unavailable, the
script falls back to the other configured recognisers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

try:
    from car_scan.pipeline import (
        ContextRouter,
        ContextValidator,
        CountryClassifier,
        ProcessingContext,
        VehiclePlateAssociator,
        YoloRegionClassifier,
        load_pipeline_config,
    )
except ModuleNotFoundError:  # Running directly from the repository root.
    from src.car_scan.pipeline import (
        ContextRouter,
        ContextValidator,
        CountryClassifier,
        ProcessingContext,
        VehiclePlateAssociator,
        YoloRegionClassifier,
        load_pipeline_config,
    )


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DEFAULT_DETECTOR = MODEL_DIR / "detect_license" / "weights" / "best.pt"
DEFAULT_VEHICLE_TYPE = MODEL_DIR / "type_car_license" / "iruvd_run1" / "weights" / "best.pt"
DEFAULT_THAI = MODEL_DIR / "thai_license_plate" / "weights" / "best.pt"
DEFAULT_LAO = MODEL_DIR / "lao_license_plate" / "weights" / "best.pt"
DEFAULT_OCR_SOURCE = MODEL_DIR / "paddleocr_train2"
DEFAULT_OCR_CONFIG = DEFAULT_OCR_SOURCE / "configs" / "rec" / "plate_rec_small.yml"
DEFAULT_FINETUNED_OCR_WEIGHTS = ROOT / "runs" / "ocr" / "plate_rec_digits" / "best_accuracy"
DEFAULT_OCR_WEIGHTS = (
    DEFAULT_FINETUNED_OCR_WEIGHTS
    if DEFAULT_FINETUNED_OCR_WEIGHTS.with_suffix(".pdparams").is_file()
    else DEFAULT_OCR_SOURCE / "output" / "plate_rec_small" / "best_accuracy"
)
DEFAULT_TORCH_OCR = ROOT / "runs" / "ocr" / "plate_crnn.pt"
DEFAULT_PIPELINE_CONFIG = ROOT / "pipeline_config.yaml"

# The Thai detector was trained with three-letter province codes.  These are
# labels, not English text printed on the plate; expose their Thai names in
# the OCR result instead of returning e.g. ``NBI`` to the caller.
THAI_PROVINCE_NAMES = {
    "ACR": "อำนาจเจริญ", "ATG": "อ่างทอง", "AYA": "พระนครศรีอยุธยา",
    "BKK": "กรุงเทพมหานคร", "BKN": "บึงกาฬ", "BRM": "บุรีรัมย์",
    "CBI": "ชลบุรี", "CCO": "ฉะเชิงเทรา", "CMI": "เชียงใหม่",
    "CNT": "ชัยนาท", "CPM": "ชัยภูมิ", "CPN": "ชุมพร",
    "CRI": "เชียงราย", "CTI": "จันทบุรี", "KBI": "กระบี่",
    "KKN": "ขอนแก่น", "KPT": "กำแพงเพชร", "KRI": "กาญจนบุรี",
    "KSN": "กาฬสินธุ์", "LEI": "เลย", "LPG": "ลำปาง",
    "LPN": "ลำพูน", "LRI": "ลพบุรี", "MDH": "มุกดาหาร",
    "MKM": "มหาสารคาม", "NAN": "น่าน", "NBI": "นนทบุรี",
    "NBP": "หนองบัวลำภู", "NKI": "หนองคาย", "NMA": "นครราชสีมา",
    "NPM": "นครพนม", "NPT": "นครปฐม", "NRT": "นครศรีธรรมราช", "NST": "นครศรีธรรมราช",
    "NSN": "นครสวรรค์", "NWT": "นราธิวาส", "NYK": "นครนายก",
    "PBI": "เพชรบุรี", "PCT": "พิจิตร", "PKN": "ประจวบคีรีขันธ์",
    "PKT": "ภูเก็ต", "PLG": "พัทลุง", "PLK": "พิษณุโลก",
    "PNA": "พังงา", "PNB": "เพชรบูรณ์", "PRE": "แพร่",
    "PRI": "ปราจีนบุรี", "PTE": "ปทุมธานี", "PTN": "ปัตตานี",
    "PYO": "พะเยา", "RBR": "ราชบุรี", "RET": "ร้อยเอ็ด",
    "RNG": "ระนอง", "RYG": "ระยอง", "SBR": "สิงห์บุรี",
    "SKA": "สงขลา", "SKM": "สมุทรสงคราม", "SKN": "สมุทรสาคร",
    "SKW": "สระแก้ว", "SNI": "สุราษฎร์ธานี", "SNK": "สกลนคร",
    "SPB": "สุพรรณบุรี", "SPK": "สมุทรปราการ", "SRI": "สระบุรี",
    "SRN": "สุรินทร์", "SSK": "ศรีสะเกษ", "STI": "สุโขทัย",
    "STN": "สตูล", "TAK": "ตาก", "TRG": "ตรัง", "TRT": "ตราด",
    "UBN": "อุบลราชธานี", "UDN": "อุดรธานี", "UTI": "อุทัยธานี", "UTT": "อุตรดิตถ์",
    "YLA": "ยะลา", "YST": "ยโสธร",
}

LAO_PROVINCE_NAMES = {
    "PSL": "ຜົ້ງສາລີ", "LNT": "ຫຼວງນ້ຳທາ", "ODX": "ອຸດົມໄຊ",
    "BOK": "ບໍ່ແກ້ວ", "LPB": "ຫຼວງພະບາງ", "HPN": "ຫົວພັນ",
    "XKH": "ຊຽງຂວາງ", "XYL": "ໄຊຍະບູລີ", "XSB": "ໄຊສົມບູນ",
    "VTE": "ກຳແພງນະຄອນ", "VTE2": "ນະຄອນຫຼວງວຽງຈັນ",
    "VTP": "ວຽງຈັນ", "BLK": "ບໍລິຄຳໄຊ", "KHM": "ຄຳມ່ວນ",
    "SVK": "ສະຫວັນນະເຂດ", "SLV": "ສາລະວັນ", "XEK": "ເຊກອງ",
    "CPS": "ຈຳປາສັກ", "ATP": "ອັດຕະປື",
}

# Character-code mappings are maintained with OCR.md.  Preserve the model
# codes in the raw tokens, while exposing their Unicode glyphs in structured
# OCR fields.
THAI_CHARACTER_NAMES = {
    "A01": "ก", "A02": "ข", "A03": "ฃ", "A04": "ค", "A05": "ฅ",
    "A06": "ฆ", "A07": "ง", "A08": "จ", "A09": "ฉ", "A10": "ช",
    "A11": "ซ", "A12": "ฌ", "A13": "ญ", "A14": "ฎ", "A15": "ฏ",
    "A16": "ฐ", "A17": "ฑ", "A18": "ฒ", "A19": "ณ", "A20": "ด",
    "A21": "ต", "A22": "ถ", "A23": "ท", "A24": "ธ", "A25": "น",
    "A26": "บ", "A27": "ป", "A28": "ผ", "A29": "ฝ", "A30": "พ",
    "A31": "ฟ", "A32": "ภ", "A33": "ม", "A34": "ย", "A35": "ร",
    "A36": "ล", "A37": "ว", "A38": "ศ", "A39": "ษ", "A40": "ส",
    "A41": "ห", "A42": "ฬ", "A43": "อ", "A44": "ฮ",
}

LAO_CHARACTER_NAMES = {
    "A": "ກ",
    "AA": "ຮ",
    "B": "ຂ",
    "C": "ຄ",
    "D": "ງ", "E": "ຈ", "F": "ສ", "H": "ຍ", "I": "ດ",
    "J": "ຕ", "K": "ຖ", "L": "ທ", "M": "ນ", "N": "ບ",
    "O": "ປ", "P": "ຜ", "Q": "ຝ", "R": "ພ", "S": "ຟ",
    "T": "ມ", "U": "ຢ", "V": "ຣ", "W": "ລ", "X": "ວ",
    "Y": "ຫ", "Z": "ອ",
}


@dataclass(frozen=True)
class CharacterDetection:
    text: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


def read_image(path: Path) -> np.ndarray:
    """Read an image even when its Windows path contains Thai/Lao characters."""

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise ValueError(f"Cannot encode output image: {path}")
    encoded.tofile(str(path))


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    """Return quadrilateral points in top-left, top-right, bottom-right, bottom-left order."""

    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def _rectify_plate_perspective(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    """Conservatively rectify a visible plate border inside a detector crop."""

    if not bool(parameters.get("perspective", False)):
        return image
    height, width = image.shape[:2]
    if min(height, width) < 24:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, width // 35), max(3, height // 35)))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = float(parameters.get("perspective_min_area", 0.35)) * width * height
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < minimum_area or area > width * height * 0.98:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        points = _order_quad_points(polygon.reshape(4, 2).astype(np.float32))
        top = np.linalg.norm(points[1] - points[0])
        bottom = np.linalg.norm(points[2] - points[3])
        left = np.linalg.norm(points[3] - points[0])
        right = np.linalg.norm(points[2] - points[1])
        target_width = max(top, bottom)
        target_height = max(left, right)
        if target_height <= 1 or target_width / target_height < 1.35:
            continue
        # Prefer the largest plausible plate-shaped contour.
        candidates.append((area * min(1.0, target_width / max(target_height, 1.0) / 8.0), points))
    if not candidates:
        return image
    _, points = max(candidates, key=lambda item: item[0])
    target_width = int(max(np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[3])))
    target_height = int(max(np.linalg.norm(points[3] - points[0]), np.linalg.norm(points[2] - points[1])))
    target_width = max(target_width, 64)
    target_height = max(target_height, 24)
    destination = np.array(
        [[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(points, destination)
    return cv2.warpPerspective(image, matrix, (target_width, target_height), borderMode=cv2.BORDER_REPLICATE)


def preprocess_plate_crop(
    image: np.ndarray,
    parameters: dict[str, Any] | None = None,
) -> np.ndarray:
    """Enhance a detected plate crop before it is sent to any OCR engine.

    This function deliberately accepts a crop only.  It normalises lighting,
    removes small compression noise, corrects a conservative horizontal skew,
    and enlarges the plate while preserving the original glyph shapes.
    """

    if image is None or image.size == 0:
        raise ValueError("Cannot preprocess an empty licence-plate crop")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    parameters = parameters or {}
    image = _rectify_plate_perspective(image, parameters)
    height, width = image.shape[:2]
    target_width = float(parameters.get("target_width", 720.0))
    max_scale = float(parameters.get("max_scale", 5.0))
    scale = min(max_scale, max(1.0, target_width / max(width, 1)))
    enlarged = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    bilateral_d = int(parameters.get("bilateral_d", 7))
    sigma_color = float(parameters.get("sigma_color", 55))
    sigma_space = float(parameters.get("sigma_space", 55))
    denoised = cv2.bilateralFilter(
        enlarged,
        d=bilateral_d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(30, int(enlarged.shape[1] * 0.16)),
        minLineLength=max(35, int(enlarged.shape[1] * 0.30)),
        maxLineGap=max(12, int(enlarged.shape[1] * 0.04)),
    )
    angles: list[float] = []
    if lines is not None:
        # OpenCV may return either (N, 1, 4) or (N, 4), depending on the
        # version/build. Flatten the optional middle dimension before
        # unpacking so a single line never becomes a numpy.int32 scalar.
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if abs(angle) <= 10:
                angles.append(angle)
    if angles:
        angle = float(np.median(angles))
        if 0.4 <= abs(angle) <= 8.0:
            center = (enlarged.shape[1] / 2, enlarged.shape[0] / 2)
            matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
            enlarged = cv2.warpAffine(
                enlarged,
                matrix,
                (enlarged.shape[1], enlarged.shape[0]),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            denoised = cv2.bilateralFilter(
                enlarged,
                d=bilateral_d,
                sigmaColor=sigma_color,
                sigmaSpace=sigma_space,
            )

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    lightness, green, red = cv2.split(lab)
    clahe_clip = float(parameters.get("clahe_clip", 2.2))
    lightness = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge((lightness, green, red)), cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
    sharpen_strength = float(parameters.get("sharpen_strength", 0.35))
    sharpened = cv2.addWeighted(enhanced, 1.0 + sharpen_strength, blurred, -sharpen_strength, 0)
    border_ratio = float(parameters.get("border_ratio", 0.025))
    border = max(4, int(min(sharpened.shape[:2]) * border_ratio))
    return cv2.copyMakeBorder(sharpened, border, border, border, border, cv2.BORDER_REPLICATE)


def enhance_character_crop(image: np.ndarray) -> np.ndarray:
    """Create a conservative contrast pass for a failed character detection."""

    if image is None or image.size == 0:
        raise ValueError("Cannot enhance an empty licence-plate crop")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    low, high = np.percentile(gray, (2.0, 98.0))
    if high - low >= 12:
        normalised = np.clip((gray.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    else:
        normalised = gray
    contrast = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(4, 4)).apply(normalised)
    return cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)


def preprocess_character_model_crop(image: np.ndarray) -> np.ndarray:
    """Enhance a plate for character YOLO while preserving its geometry."""

    if image is None or image.size == 0:
        raise ValueError("Cannot preprocess an empty licence-plate crop")
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness, green, red = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(4, 4)).apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge((lightness, green, red)), cv2.COLOR_LAB2BGR)
    denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=45, sigmaSpace=45)
    blurred = cv2.GaussianBlur(denoised, (0, 0), 0.9)
    return cv2.addWeighted(denoised, 1.35, blurred, -0.35, 0)


def clamp_crop(image: np.ndarray, xyxy: Iterable[float], padding: float) -> tuple[np.ndarray, list[int]]:
    """Crop a bounding box with relative padding and return its integer bounds."""

    height, width = image.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in xyxy)
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    left = max(0, int(np.floor(x1 - pad_x)))
    top = max(0, int(np.floor(y1 - pad_y)))
    right = min(width, int(np.ceil(x2 + pad_x)))
    bottom = min(height, int(np.ceil(y2 + pad_y)))
    if right <= left or bottom <= top:
        raise ValueError("Detector returned an invalid licence-plate box")
    return image[top:bottom, left:right].copy(), [left, top, right, bottom]


def group_characters_into_lines(items: list[CharacterDetection]) -> list[list[CharacterDetection]]:
    """Group character boxes into plate lines, then order each line left-to-right."""

    if not items:
        return []
    ordered = sorted(items, key=lambda item: (item.y1 + item.y2) / 2)
    typical_height = float(np.median([item.y2 - item.y1 for item in ordered]))
    threshold = max(8.0, typical_height * 0.65)
    lines: list[list[CharacterDetection]] = []
    centers: list[float] = []
    for item in ordered:
        center = (item.y1 + item.y2) / 2
        line_index = next(
            (index for index, line_center in enumerate(centers) if abs(center - line_center) <= threshold),
            None,
        )
        if line_index is None:
            lines.append([item])
            centers.append(center)
        else:
            lines[line_index].append(item)
            centers[line_index] = float(np.mean([(entry.y1 + entry.y2) / 2 for entry in lines[line_index]]))
    return [sorted(line, key=lambda item: item.x1) for _, line in sorted(zip(centers, lines), key=lambda pair: pair[0])]


def serialise_characters(items: list[CharacterDetection]) -> dict[str, Any]:
    lines = group_characters_into_lines(items)
    text_lines = ["".join(character.text for character in line) for line in lines]
    return {
        "text": " ".join(text_lines),
        "lines": text_lines,
        "confidence": round(float(np.mean([item.confidence for item in items])), 4) if items else 0.0,
        "character_count": len(items),
    }


def draw_character_boxes(
    image: np.ndarray,
    reading: dict[str, Any] | None,
    offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Draw structured YOLO character detections on a plate image.

    Character coordinates are relative to the tight plate crop used by the
    character model. ``offset`` places those boxes on a larger vehicle image.
    The returned image is the same array, which keeps this helper convenient
    for both the full annotated frame and the plate-only training preview.
    """

    if image is None or image.size == 0 or not isinstance(reading, dict):
        return image
    tokens = reading.get("tokens", [])
    if not isinstance(tokens, list):
        return image
    offset_x, offset_y = (int(offset[0]), int(offset[1]))
    colours = {
        "digit": (0, 210, 0),
        "character": (255, 120, 0),
        "province": (0, 165, 255),
        "other": (190, 190, 190),
    }
    height, width = image.shape[:2]
    for token in tokens:
        if not isinstance(token, dict) or len(token.get("box", [])) != 4:
            continue
        try:
            left, top, right, bottom = [float(value) for value in token["box"]]
        except (TypeError, ValueError):
            continue
        x1 = max(0, min(width - 1, int(round(left + offset_x))))
        y1 = max(0, min(height - 1, int(round(top + offset_y))))
        x2 = max(x1 + 1, min(width, int(round(right + offset_x))))
        y2 = max(y1 + 1, min(height, int(round(bottom + offset_y))))
        kind = str(token.get("kind") or "other")
        colour = colours.get(kind, colours["other"])
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 1)
        label = f"{token.get('label', '?')} {float(token.get('confidence', 0.0)):.2f}"
        text_y = max(12, y1 - 3)
        cv2.putText(
            image,
            label[:18],
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            colour,
            1,
            cv2.LINE_AA,
        )
    return image


def character_iou(first: CharacterDetection, second: CharacterDetection) -> float:
    """Return box IoU for suppressing duplicate character predictions."""

    left, top = max(first.x1, second.x1), max(first.y1, second.y1)
    right, bottom = min(first.x2, second.x2), min(first.y2, second.y2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first.x2 - first.x1) * max(0.0, first.y2 - first.y1)
    second_area = max(0.0, second.x2 - second.x1) * max(0.0, second.y2 - second.y1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def character_kind(item: CharacterDetection, country: str) -> str:
    if item.text.isdigit() and len(item.text) == 1:
        return "digit"
    provinces = THAI_PROVINCE_NAMES if country == "thai" else LAO_PROVINCE_NAMES
    if item.text in provinces:
        return "province"
    if country == "thai" and re.fullmatch(r"A\d{2}", item.text):
        return "character"
    if country == "lao" and item.text in LAO_CHARACTER_NAMES:
        return "character"
    return "other"


def character_glyph(label: str, country: str) -> str:
    """Return the Unicode glyph represented by a character-model label."""

    mapping = THAI_CHARACTER_NAMES if country == "thai" else LAO_CHARACTER_NAMES
    return mapping.get(label, "")


def deduplicate_characters(items: list[CharacterDetection], country: str) -> list[CharacterDetection]:
    """Remove same-glyph/strongly-overlapping boxes while keeping confidence."""

    kept: list[CharacterDetection] = []
    for item in sorted(items, key=lambda value: value.confidence, reverse=True):
        kind = character_kind(item, country)
        duplicate = any(
            (item.text == existing.text and character_iou(item, existing) >= 0.30)
            or (
                kind == character_kind(existing, country)
                and kind in {"digit", "character", "province"}
                and character_iou(item, existing) >= 0.72
            )
            for existing in kept
        )
        if not duplicate:
            kept.append(item)
    return kept


def sequence_quality(items: Iterable[CharacterDetection]) -> float:
    """Score whether character boxes form one clean horizontal plate line."""

    values = sorted(items, key=lambda item: item.x1)
    if not values:
        return -10.0
    heights = np.asarray([max(1.0, item.y2 - item.y1) for item in values], dtype=float)
    centers_y = np.asarray([(item.y1 + item.y2) / 2 for item in values], dtype=float)
    centers_x = np.asarray([(item.x1 + item.x2) / 2 for item in values], dtype=float)
    typical_height = max(1.0, float(np.median(heights)))
    alignment_penalty = float(np.std(centers_y)) / typical_height
    height_penalty = float(np.std(heights)) / typical_height
    gap_penalty = 0.0
    if len(centers_x) > 2:
        gaps = np.diff(centers_x)
        positive_gaps = gaps[gaps > 0]
        if len(positive_gaps):
            gap_penalty = min(1.0, float(np.std(positive_gaps)) / max(1.0, float(np.mean(positive_gaps))))
    return (
        float(np.mean([item.confidence for item in values]))
        - alignment_penalty * 0.35
        - height_penalty * 0.12
        - gap_penalty * 0.06
    )


def select_digit_sequence(
    items: list[CharacterDetection], target_length: int
) -> tuple[list[CharacterDetection], int]:
    """Select the most plate-like digit line and report its untrimmed size."""

    digits = [item for item in items if item.text.isdigit() and len(item.text) == 1]
    if not digits:
        return [], 0
    lines = group_characters_into_lines(digits)
    best_line = max(
        lines,
        key=lambda line: (
            min(len(line), target_length),
            -abs(len(line) - target_length),
            sequence_quality(line),
        ),
    )
    source_count = len(best_line)
    if source_count <= target_length:
        return sorted(best_line, key=lambda item: item.x1), source_count

    # Extra detections are usually reflections or duplicate boxes. Select the
    # subset that has the cleanest baseline, size and spacing.
    pool = sorted(best_line, key=lambda item: item.confidence, reverse=True)[:12]
    selected = max(combinations(pool, target_length), key=sequence_quality)
    return sorted(selected, key=lambda item: item.x1), source_count


def select_lao_prefix(
    items: list[CharacterDetection], digits: list[CharacterDetection]
) -> tuple[str, list[CharacterDetection]]:
    """Select the two Lao prefix boxes on the same baseline as the number."""

    candidates = [item for item in items if item.text in LAO_CHARACTER_NAMES]
    if digits and candidates:
        digit_centres = [(item.y1 + item.y2) / 2 for item in digits]
        digit_heights = [max(1.0, item.y2 - item.y1) for item in digits]
        baseline = float(np.median(digit_centres))
        tolerance = float(np.median(digit_heights)) * 0.72
        first_digit_x = min((item.x1 + item.x2) / 2 for item in digits)
        aligned = [
            item
            for item in candidates
            if abs((item.y1 + item.y2) / 2 - baseline) <= tolerance
            and (item.x1 + item.x2) / 2 < first_digit_x
        ]
        if len(aligned) >= 2:
            candidates = aligned
    if len(candidates) < 2:
        return "", []
    if len(candidates) > 2:
        candidates = list(max(combinations(candidates, 2), key=sequence_quality))
    selected = sorted(candidates, key=lambda item: item.x1)
    return "".join(LAO_CHARACTER_NAMES[item.text] for item in selected), selected


PROVINCE_MIN_CONFIDENCE = 0.45
PROVINCE_MIN_MARGIN = 0.08
PROVINCE_ROI_MIN_CONFIDENCE = 0.12


def select_province_item(
    items: list[CharacterDetection],
    country: str,
    digits: list[CharacterDetection],
) -> tuple[CharacterDetection | None, str, float]:
    """Select a province box only when its confidence and layout are plausible.

    Lao province names are printed on the small line above the registration
    number.  A detector can still produce a high-confidence province box on
    the number line or on background noise, so confidence alone is not enough.
    Thai province labels use a different layout and therefore keep the
    confidence/margin checks without the Lao top-line restriction.
    """

    candidates = [
        item
        for item in items
        if character_kind(item, country) == "province"
        and item.confidence >= PROVINCE_MIN_CONFIDENCE
    ]
    if not candidates:
        return None, "below_confidence", 0.0

    if country == "lao":
        if not digits:
            return None, "missing_digit_reference", 0.0
        digit_centres = [(item.y1 + item.y2) / 2 for item in digits]
        digit_heights = [max(1.0, item.y2 - item.y1) for item in digits]
        digit_line = float(np.median(digit_centres))
        # Leave a visible gap from the large number line. This prevents a
        # province class detected on the registration line from being used.
        top_line_limit = digit_line - max(4.0, float(np.median(digit_heights)) * 0.35)
        candidates = [
            item
            for item in candidates
            if (item.y1 + item.y2) / 2 <= top_line_limit
        ]
        if not candidates:
            return None, "not_top_line", 0.0

    candidates.sort(key=lambda item: item.confidence, reverse=True)
    best = candidates[0]
    margin = best.confidence - candidates[1].confidence if len(candidates) > 1 else 1.0
    if len(candidates) > 1 and margin < PROVINCE_MIN_MARGIN:
        return None, "ambiguous", round(margin, 4)
    return best, "accepted", round(margin, 4)


def analyse_country_characters(items: list[CharacterDetection], country: str) -> dict[str, Any]:
    """Turn raw model boxes into structured registration fields."""

    items = deduplicate_characters(items, country)
    target_length = 6 if country == "thai" else 4
    digit_items, digit_detection_count = select_digit_sequence(items, target_length)
    digits = "".join(item.text for item in digit_items)
    digit_confidence = float(np.mean([item.confidence for item in digit_items])) if digit_items else 0.0

    province_item, province_status, province_margin = select_province_item(
        items, country, digit_items
    )
    province_names = THAI_PROVINCE_NAMES if country == "thai" else LAO_PROVINCE_NAMES
    province = province_names.get(province_item.text, "") if province_item else ""

    prefix_items: list[CharacterDetection] = []
    if country == "thai":
        prefix = digits[:2] if len(digits) == target_length else ""
        number = digits[2:] if len(digits) == target_length else digits
    else:
        prefix, prefix_items = select_lao_prefix(items, digit_items)
        number = digits

    # Keep the model's Latin aliases as well as the converted Lao text.  The
    # aliases are stable identifiers for filenames, exports, and audit logs.
    prefix_code = "_".join(item.text for item in prefix_items) if country == "lao" else ""

    raw = serialise_characters(items)
    complete = digit_detection_count == target_length and len(digits) == target_length
    if country == "lao":
        complete = complete and len(prefix) == 2
    registration = (
        f"{prefix}-{number}" if country == "thai" and prefix and number
        else " ".join(part for part in (prefix, number) if part)
    )
    reading = {
        "text": registration,
        "lines": [registration] if registration else [],
        "raw_text": raw["text"],
        "raw_lines": raw["lines"],
        "confidence": round(digit_confidence, 4),
        "digit_confidence": round(digit_confidence, 4),
        "character_count": len(items),
        "digit_detection_count": digit_detection_count,
        "expected_digit_count": target_length,
        "complete": complete,
        "plate_prefix": prefix,
        "plate_prefix_code": prefix_code,
        "plate_number": number,
        "province": province,
        "province_code": province_item.text if province_item else "",
        "province_confidence": round(province_item.confidence, 4) if province_item else 0.0,
        "province_status": province_status,
        "province_margin": province_margin,
        "prefix_confidence": round(float(np.mean([item.confidence for item in prefix_items])), 4) if prefix_items else 0.0,
        "detected_characters": [
            {
                "code": item.text,
                "text": character_glyph(item.text, country),
                "confidence": round(item.confidence, 4),
            }
            for item in sorted(items, key=lambda item: (item.y1, item.x1))
            if character_kind(item, country) == "character"
        ],
        "script": country,
        "tokens": [
            {
                "label": item.text,
                "kind": character_kind(item, country),
                "confidence": round(item.confidence, 4),
                "box": [round(item.x1, 2), round(item.y1, 2), round(item.x2, 2), round(item.y2, 2)],
            }
            for item in sorted(items, key=lambda item: (item.y1, item.x1))
        ],
    }
    return reading


def country_reading_score(reading: dict[str, Any], country: str) -> float:
    """Score a country reading using both model confidence and plate shape.

    The two character models can have very similar raw confidence on a small
    crop. Thai plates in this project normally contain six digits and Thai
    province text, while Lao plates contain four digits plus Lao script. These
    structural signals are more reliable than selecting the largest raw
    detector score alone.
    """

    expected = 6 if country == "thai" else 4
    count = int(reading.get("digit_detection_count", 0))
    completeness = min(count, expected) / expected
    score = float(reading.get("digit_confidence", 0.0)) * 0.25 + completeness * 0.35
    if count == expected:
        score += 0.35
    else:
        score -= min(0.55, abs(count - expected) * 0.15)
    score += float(reading.get("province_confidence", 0.0)) * 0.15
    if country == "lao":
        if len(str(reading.get("plate_prefix", ""))) == 2:
            score += 0.25
        else:
            score -= 0.18
    else:
        thai_tokens = sum(
            token.get("kind") == "character" for token in reading.get("tokens", [])
        )
        if thai_tokens:
            score += min(0.08, thai_tokens * 0.025)
    return round(score, 4)


def localise_thai_reading(reading: dict[str, Any]) -> dict[str, Any]:
    """Convert detector province-code tokens to Thai province names."""

    province = ""
    lines = [
        " ".join(
            THAI_PROVINCE_NAMES.get(token, token)
            for token in line.split()
        )
        for line in reading["lines"]
    ]
    for line in reading["lines"]:
        for token in line.split():
            if token in THAI_PROVINCE_NAMES:
                province = THAI_PROVINCE_NAMES[token]
    reading = dict(reading)
    reading["lines"] = lines
    reading["text"] = " ".join(lines)
    reading["script"] = detect_text_script(reading["text"])
    if province:
        reading["province"] = province
    return reading


def localise_lao_reading(reading: dict[str, Any]) -> dict[str, Any]:
    """Convert Lao detector province codes to full Lao province names."""

    provinces: list[str] = []
    digit_tokens: list[str] = []
    for line in reading["lines"]:
        for token in line.split():
            letters = "".join(character for character in token.upper() if "A" <= character <= "Z")
            digits = "".join(character for character in token if character.isdigit())
            replacements: list[str] = []
            # A detector can merge adjacent province boxes, e.g. SVK + KHM.
            # Decode known codes greedily and discard unknown Latin noise.
            while letters:
                code = next((candidate for candidate in sorted(LAO_PROVINCE_NAMES, key=len, reverse=True) if letters.startswith(candidate)), None)
                if code is None:
                    letters = letters[1:]
                    continue
                replacements.append(LAO_PROVINCE_NAMES[code])
                provinces.append(LAO_PROVINCE_NAMES[code])
                letters = letters[len(code):]
            if digits:
                digit_tokens.append(digits)
    # One plate has one province.  When the detector merges multiple nearby
    # province boxes, the rightmost/last candidate is the usable one.
    province = provinces[-1] if provinces else ""
    lines = [province] if province else []
    if digit_tokens:
        lines.append(" ".join(digit_tokens[-1:]))
    reading = dict(reading)
    reading["lines"] = lines
    reading["text"] = " ".join(lines)
    reading["script"] = detect_text_script(reading["text"])
    if province:
        reading["province"] = province
    return reading


def detect_text_script(text: str) -> str:
    """Identify the script in OCR output, including Lao/Thai Unicode text."""

    lao = sum("\u0e80" <= character <= "\u0eff" for character in text)
    thai = sum("\u0e00" <= character <= "\u0e7f" for character in text)
    latin = sum(character.isascii() and character.isalpha() for character in text)
    scripts = [(lao, "lao"), (thai, "thai"), (latin, "latin")]
    active = [name for count, name in scripts if count]
    if not active:
        return "unknown"
    return active[0] if len(active) == 1 else "mixed"


def script_matches(text: str, expected_language: str) -> bool:
    """Prefer the requested plate language without rejecting digits."""

    script = detect_text_script(text)
    if expected_language == "lao":
        return script in {"lao", "mixed"} and any("\u0e80" <= char <= "\u0eff" for char in text)
    if expected_language == "thai":
        # The Thai YOLO model uses Latin province abbreviations (for example
        # BKK), so both Thai Unicode and Latin text are valid here.
        return script in {"thai", "latin", "mixed", "unknown"} and bool(re.search(r"\d", text))
    return script in {"latin", "mixed", "unknown"}


def plate_shape_matches(text: str, expected_language: str) -> bool:
    """Reject recogniser outputs with an impossible plate shape."""

    compact = re.sub(r"\s+", "", text)
    digits = re.findall(r"\d", compact)
    lao_letters = [character for character in compact if "\u0e80" <= character <= "\u0eff" and unicodedata.category(character) == "Lo"]
    if expected_language == "lao":
        return len(lao_letters) >= 2 and len(digits) == 4
    if expected_language == "thai" and compact.isdigit():
        return len(digits) == 6
    return script_matches(text, expected_language)


def clean_ocr_line(line: str) -> str:
    """Keep letters/digits from an OCR line and discard punctuation noise."""

    return "".join(
        character for character in line
        if (("\u0e80" <= character <= "\u0eff") and unicodedata.category(character) == "Lo")
        or character.isdigit()
        or ("A" <= character <= "Z")
        or ("a" <= character <= "z")
    ).strip()


def split_plate_text(text: str, country: str) -> tuple[str, str]:
    """Split a recognised plate into its script prefix and numeric part."""

    if country == "lao":
        compact = re.sub(r"\s+", "", text)
        match = re.match(r"^(?P<prefix>[\u0e80-\u0eff]+?)(?P<number>\d{4})$", compact)
        if match:
            return match.group("prefix"), match.group("number")
    # Province/model labels may contain other digits (for example ``A23``).
    # Prefer the standalone six-digit registration token when present.
    six_digit_tokens = re.findall(r"(?<!\d)\d{6}(?!\d)", text)
    digits = six_digit_tokens[-1] if six_digit_tokens else "".join(re.findall(r"\d", text))
    # Thai private-vehicle plates in this project use two leading digits and
    # four registration digits, e.g. 68-3197.  Keep both pieces for the GUI
    # and database instead of flattening them into one value.
    if country == "thai" and len(digits) == 6:
        return digits[:2], digits[2:]
    return "", digits


def extract_ocr_province(text: str, country: str) -> tuple[str, str]:
    """Extract a known province code/name from a full OCR transcription.

    The Lao recognition dataset intentionally trains on ``PROVINCE_CODE``
    followed by the registration line (for example ``VTE2 ກຍ0412``).  Keep
    province parsing separate from prefix/number parsing so that this code is
    not accidentally treated as a registration prefix.
    """

    province_names = THAI_PROVINCE_NAMES if country == "thai" else LAO_PROVINCE_NAMES
    compact = re.sub(r"[^A-Za-z0-9\u0e00-\u0eff]", " ", str(text).upper())
    tokens = compact.split()
    for token in tokens:
        if token in province_names:
            return token, province_names[token]
    # OCR may omit spaces around the code. Prefer the longest known code to
    # avoid matching VTE inside VTE2.
    for code in sorted(province_names, key=len, reverse=True):
        if re.search(rf"(?<![A-Z0-9]){re.escape(code)}(?![A-Z0-9])", compact):
            return code, province_names[code]
    # A recognition model can emit the full Lao/Thai province name rather than
    # its training code. Match the longest Unicode value in that case.
    normalised = re.sub(r"\s+", "", str(text))
    for code, name in sorted(province_names.items(), key=lambda item: len(item[1]), reverse=True):
        if name and name in normalised:
            return code, name
    return "", ""


def resolve_plate_fields(
    country: str, reading: dict[str, Any], ocr: dict[str, Any]
) -> tuple[str, str, str]:
    """Fuse structured YOLO fields with OCR without mixing model labels."""

    model_prefix = str(reading.get("plate_prefix") or "")
    model_number = str(reading.get("plate_number") or "")
    ocr_prefix, ocr_number = split_plate_text(str(ocr.get("text") or ""), country)

    if country == "thai":
        if len(model_prefix) == 2 and model_prefix.isdigit() and len(model_number) == 4 and model_number.isdigit():
            return model_prefix, model_number, "structured-yolo"
        return ocr_prefix, ocr_number, "ocr"

    valid_model_prefix = len(model_prefix) == 2 and all("\u0e80" <= char <= "\u0eff" for char in model_prefix)
    valid_ocr_prefix = len(ocr_prefix) == 2 and all("\u0e80" <= char <= "\u0eff" for char in ocr_prefix)
    # Tesseract often reads two valid-looking but wrong Lao glyphs. A pair of
    # spatially aligned YOLO glyph boxes is more reliable; OCR fills the prefix
    # only when the model pair is missing or weak.
    if valid_model_prefix and float(reading.get("prefix_confidence", 0.0)) >= 0.65:
        prefix = model_prefix
    elif valid_ocr_prefix:
        prefix = ocr_prefix
    else:
        prefix = model_prefix

    valid_model_number = len(model_number) == 4 and model_number.isdigit()
    valid_ocr_number = len(ocr_number) == 4 and ocr_number.isdigit()
    if valid_model_number and valid_ocr_number and model_number == ocr_number:
        number, source = model_number, "yolo+ocr-agree"
    elif valid_model_number and float(reading.get("digit_confidence", 0.0)) >= 0.55:
        number, source = model_number, "structured-yolo"
    elif valid_ocr_number:
        number, source = ocr_number, "ocr"
    else:
        number, source = model_number or ocr_number, "partial"
    return prefix, number, source


def display_plate_number(plate: dict[str, Any]) -> str:
    """Format the human-readable registration number for the CMD summary."""

    country = plate.get("country", "")
    prefix = str(plate.get("plate_prefix", ""))
    number = str(plate.get("plate_number", ""))
    if country == "lao":
        return " ".join(part for part in (prefix, number) if part) or str(plate.get("ocr", {}).get("text", "-"))
    if country == "thai" and prefix and number:
        return f"{prefix}-{number}"
    return number or str(plate.get("ocr", {}).get("text", "-"))


def print_cmd_summary(input_path: Path, plates: list[dict[str, Any]]) -> None:
    """Print a compact country-specific summary before the machine JSON."""

    print("=" * 68)
    print(f"SCAN : {input_path.name}")
    print(f"พบป้ายทะเบียน: {len(plates)} รายการ")
    print("=" * 68)
    if not plates:
        print("ไม่พบป้ายทะเบียน")
        return
    for plate in plates:
        country = plate.get("country", "unknown")
        ocr = plate.get("ocr", {})
        province = plate.get("province", "") or plate.get("country_readings", {}).get(country, {}).get("province", "")
        confidence = float(ocr.get("confidence", 0.0)) * 100
        if country == "lao":
            print(f"[ป้ายที่ {plate['id']}] ລາວ / LAOS")
            print(f"  ແຂວງ      : {province or '-'}")
            print(f"  ຄຳນຳໜ້າ  : {plate.get('plate_prefix') or '-'}")
            print(f"  ເລກທະບຽນ: {plate.get('plate_number') or '-'}")
            print(f"  ທະບຽນ   : {display_plate_number(plate)}")
        else:
            print(f"[ป้ายที่ {plate['id']}] ไทย / THAILAND")
            print(f"  จังหวัด : {province or '-'}")
            print(f"  ทะเบียน : {display_plate_number(plate)}")
        print(f"  OCR     : {ocr.get('text') or '-'}")
        print(f"  ความมั่นใจ: {confidence:.2f}% | {ocr.get('method', '-')}")
        print("-" * 68)


class CustomPaddleOCR:
    """Run custom OCR and optional PaddleOCR with a safe local fallback.

    The repository contains the PaddleOCR training source, but the trained
    checkpoint is intentionally not committed.  Keeping the fallback here is
    important: the YOLO character models can still produce a useful plate
    reading when PaddlePaddle is not installed or the checkpoint is absent.
    """

    def __init__(
        self,
        source: Path,
        config: Path,
        weights: Path,
        torch_model: Path | None = None,
        timeout: int = 120,
        lao_config: Path | None = None,
        lao_weights: Path | None = None,
    ) -> None:
        self.source = source
        self.config = config
        self.weights = weights.with_suffix("") if weights.suffix == ".pdparams" else weights
        default_lao_config = source / "configs" / "rec" / "plate_rec_lao_full.yml"
        default_lao_weights = ROOT / "runs" / "ocr" / "plate_rec_lao_full" / "best_accuracy"
        self.lao_config = lao_config or default_lao_config
        selected_lao_weights = lao_weights or default_lao_weights
        self.lao_weights = (
            selected_lao_weights.with_suffix("")
            if selected_lao_weights.suffix == ".pdparams"
            else selected_lao_weights
        )
        self.timeout = timeout
        self.script = source / "tools" / "infer_rec.py"
        self.torch_model = torch_model
        self._paddle: dict[str, Any] = {}
        self._paddle_error: dict[str, str] = {}
        self._torch = None
        self._torch_error = ""

    @property
    def available(self) -> bool:
        return self.script.is_file() and self.config.is_file() and self.weights.with_suffix(".pdparams").is_file()

    @property
    def lao_available(self) -> bool:
        return (
            self.script.is_file()
            and self.lao_config.is_file()
            and self.lao_weights.with_suffix(".pdparams").is_file()
        )

    @staticmethod
    def _variants(image: np.ndarray) -> list[np.ndarray]:
        """Create OCR-friendly views without changing the original crop."""

        _, width = image.shape[:2]
        # PaddleOCR is much more reliable when a detected plate is not only a
        # few hundred pixels wide.  Do not enlarge very large crops further.
        scale = min(4.0, max(1.0, 640.0 / max(width, 1)))
        enlarged = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        sharpened = cv2.addWeighted(contrast, 1.5, cv2.GaussianBlur(contrast, (0, 0), 2), -0.5, 0)
        adaptive = cv2.adaptiveThreshold(
            contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
        )
        return [
            enlarged,
            cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR),
        ]

    @staticmethod
    def _paddle_result(result: Any) -> tuple[str, float]:
        """Read PaddleOCR 3.x result objects and dict-like test doubles."""

        if isinstance(result, dict):
            data = result.get("res", result)
        else:
            try:
                data = result["res"]
            except (KeyError, TypeError, IndexError):
                data = result
        if hasattr(data, "to_dict"):
            data = data.to_dict()
        if not hasattr(data, "get"):
            return "", 0.0
        texts = data.get("rec_texts", []) or []
        scores = data.get("rec_scores", []) or []
        text = " ".join(str(value).strip() for value in texts if str(value).strip())
        try:
            confidence = float(np.mean([float(value) for value in scores])) if scores else 0.0
        except (TypeError, ValueError):
            confidence = 0.0
        return text, confidence

    def _recognise_with_paddle(self, image: np.ndarray, expected_language: str) -> tuple[str, float, str]:
        """Use installed PaddleOCR lazily; importing it must not break scanning."""

        # PaddleOCR does not provide a reliable Lao recognition model in this
        # project.  Do not replace Lao script with an English transliteration.
        if expected_language == "lao":
            return "", 0.0, "Lao-script OCR requires the custom Lao checkpoint"

        paddle_language = "th" if expected_language == "thai" else "en"
        if paddle_language not in self._paddle and paddle_language not in self._paddle_error:
            try:
                from paddleocr import PaddleOCR

                self._paddle[paddle_language] = PaddleOCR(
                    lang=paddle_language,
                    ocr_version="PP-OCRv5",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    enable_mkldnn=False,
                    text_det_limit_side_len=960,
                    text_det_limit_type="max",
                    text_det_box_thresh=0.3,
                )
            except Exception as error:  # optional dependency/model download
                self._paddle_error[paddle_language] = f"PaddleOCR unavailable: {error}"
        if paddle_language not in self._paddle:
            return "", 0.0, self._paddle_error[paddle_language]

        best_text, best_confidence = "", 0.0
        try:
            for variant in self._variants(image):
                for result in self._paddle[paddle_language].predict(variant):
                    text, confidence = self._paddle_result(result)
                    if script_matches(text, expected_language) and self._candidate_score(text, confidence) > self._candidate_score(best_text, best_confidence):
                        best_text, best_confidence = text, confidence
        except Exception as error:
            return "", 0.0, f"PaddleOCR inference failed: {error}"
        if not best_text:
            return "", 0.0, f"PaddleOCR returned no {expected_language}-script text"
        return best_text, best_confidence, ""

    def _recognise_with_torch(self, image: np.ndarray, expected_language: str) -> tuple[str, float, str]:
        """Use the locally trained CRNN checkpoint when it is available."""

        # The available training set has too few Thai examples to use its
        # CRNN as a general Thai recogniser. Keep this optional model for Lao
        # script, where it is trained to supplement the YOLO character model.
        if expected_language != "lao":
            return "", 0.0, "Trained Torch OCR is enabled for Lao script plates"
        if self.torch_model is None or not self.torch_model.is_file():
            return "", 0.0, "Trained Torch OCR checkpoint unavailable"
        if self._torch is None and not self._torch_error:
            try:
                from plate_ocr_torch import TorchPlateOCR

                self._torch = TorchPlateOCR(self.torch_model)
            except Exception as error:
                self._torch_error = f"Trained Torch OCR unavailable: {error}"
        if self._torch is None:
            return "", 0.0, self._torch_error
        try:
            # A small ensemble of contrast/threshold variants is materially
            # more stable for Lao glyphs. Pick the most frequent valid string
            # instead of trusting one highest-confidence frame.
            candidates: list[tuple[str, float]] = []
            for variant in self._variants(image):
                text, confidence = self._torch.recognise(variant)
                if text and plate_shape_matches(text, expected_language):
                    candidates.append((text, confidence))
            if not candidates:
                return "", 0.0, f"Trained Torch OCR returned incompatible {expected_language} text"
            grouped: dict[str, list[float]] = {}
            for text, confidence in candidates:
                grouped.setdefault(text, []).append(confidence)
            text = max(grouped, key=lambda value: (len(grouped[value]), sum(grouped[value])))
            confidence = max(grouped[text])
        except Exception as error:
            return "", 0.0, f"Trained Torch OCR failed: {error}"
        return text, confidence, ""

    def _recognise_with_tesseract(self, image: np.ndarray, expected_language: str) -> tuple[str, float, str]:
        """Use the bundled Lao Tesseract model when PaddleOCR is unavailable."""

        if expected_language != "lao":
            return "", 0.0, ""
        candidates = [
            ROOT / "tools" / "tesseract" / "tesseract.exe",
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        ]
        detected = shutil.which("tesseract")
        if detected:
            candidates.insert(0, Path(detected))
        executable = next((path for path in candidates if path.is_file()), None)
        tessdata = ROOT / "tools" / "tesseract" / "tessdata"
        if executable is None or not (tessdata / "lao.traineddata").is_file():
            return "", 0.0, "Lao Tesseract model unavailable"

        best_text, best_score = "", float("-inf")
        environment = os.environ.copy()
        environment["TESSDATA_PREFIX"] = str(tessdata)
        try:
            with tempfile.TemporaryDirectory(prefix="plate_tess_") as directory:
                for index, variant in enumerate(self._variants(image)):
                    input_path = Path(directory) / f"variant_{index}.png"
                    write_image(input_path, variant)
                    for psm in (6, 7, 11, 13):
                        process = subprocess.run(
                            [str(executable), str(input_path), "stdout", "-l", "lao", "--psm", str(psm)],
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            env=environment,
                            timeout=20,
                            check=False,
                        )
                        if process.returncode != 0:
                            continue
                        for line in process.stdout.splitlines():
                            text = clean_ocr_line(line)
                            digits = len(re.findall(r"\d", text))
                            lao_count = sum("\u0e80" <= char <= "\u0eff" for char in text)
                            if lao_count + digits < 2:
                                continue
                            if digits == 4:
                                digit_score = 8.0
                            elif digits == 3:
                                digit_score = 2.0
                            elif digits > 4:
                                digit_score = -1.0
                            else:
                                digit_score = -2.0
                            score = digit_score + min(lao_count, 3) * 0.15 - max(0, len(text) - 10) * 0.08
                            if score > best_score:
                                best_text, best_score = text, score
        except subprocess.TimeoutExpired:
            return "", 0.0, "Lao Tesseract timed out"
        except OSError as error:
            # An installer or inaccessible executable must never terminate a
            # scan.  The caller will fall back to the character recogniser.
            return "", 0.0, f"Lao Tesseract could not start: {error}"
        if not best_text:
            return "", 0.0, "Lao Tesseract returned no usable text"
        return best_text, round(max(0.0, min(0.99, best_score / 10.0)), 4), ""

    @staticmethod
    def _candidate_score(text: str, confidence: float) -> float:
        digits = len(re.findall(r"\d", text))
        return confidence + (0.35 if digits >= 4 else 0.0) + min(len(text), 12) * 0.005

    def _recognise_custom(
        self,
        image: np.ndarray,
        config: Path | None = None,
        weights: Path | None = None,
        expected_language: str = "",
    ) -> dict[str, Any]:
        """Run the repository's trained recogniser when its checkpoint exists."""

        config = config or self.config
        weights = weights or self.weights

        with tempfile.TemporaryDirectory(prefix="plate_ocr_") as temporary_directory:
            temporary = Path(temporary_directory)
            environment = os.environ.copy()
            # Required before PaddlePaddle imports on some Windows CPU installs.
            environment.setdefault("FLAGS_enable_pir_api", "0")
            environment.setdefault("FLAGS_use_mkldnn", "0")
            environment["PYTHONUTF8"] = "1"
            # The bundled PaddleOCR source is stored as ``model/paddleocr_train2``.
            # Its scripts use imports rooted at ``model.paddleocr_train2``;
            # expose the project root so that namespace package is resolvable
            # both from a source checkout and from the PyInstaller bundle.
            source_root = str(self.source.parent.parent)
            source_parent = str(self.source.parent)
            python_path = environment.get("PYTHONPATH", "")
            environment["PYTHONPATH"] = os.pathsep.join(
                value for value in (source_root, source_parent, python_path) if value
            )
            paddle_python = os.environ.get("CAR_SCAN_PADDLE_PYTHON", "").strip()
            if not paddle_python:
                bundled_python = Path(__file__).resolve().parent / ".venv-paddle" / "Scripts" / "python.exe"
                paddle_python = str(bundled_python) if bundled_python.is_file() else sys.executable
            candidates: list[tuple[float, str, float]] = []
            last_message = "PaddleOCR returned no recognition result"
            # The first pass keeps the colour information; the other passes
            # help with blue/black glyphs, glare, compression and low contrast.
            for index, variant in enumerate(self._variants(image)[:3]):
                input_path = temporary / f"plate_{index}.jpg"
                result_path = temporary / f"result_{index}.txt"
                write_image(input_path, variant)
                command = [
                    paddle_python,
                    str(self.script),
                    "-c",
                    str(config),
                    "-o",
                    f"Global.pretrained_model={weights}",
                    f"Global.infer_img={input_path}",
                    f"Global.save_res_path={result_path}",
                ]
                try:
                    process = subprocess.run(
                        command,
                        cwd=str(self.source),
                        env=environment,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=self.timeout,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    last_message = "Custom PaddleOCR timed out"
                    continue
                if process.returncode != 0 or not result_path.is_file():
                    last_message = next(
                        (line.strip() for line in reversed(process.stderr.splitlines()) if line.strip()),
                        "PaddleOCR inference failed",
                    )
                    continue
                fields = result_path.read_text(encoding="utf-8").strip().split("\t")
                if len(fields) < 3:
                    continue
                try:
                    confidence = float(fields[2])
                except ValueError:
                    confidence = 0.0
                text = fields[1].strip()
                digits = len(re.findall(r"\d", text))
                lao_letters = sum("\u0e80" <= char <= "\u0eff" for char in text)
                if expected_language == "lao" and (digits != 4 or lao_letters < 2):
                    # Do not let a partial custom result hide the structured
                    # Lao detector, which can still recover the four digits.
                    continue
                if expected_language == "thai" and digits < 4:
                    continue
                score = self._candidate_score(text, confidence)
                if expected_language == "lao":
                    score += min(lao_letters, 3) * 0.08
                candidates.append((score, text, confidence))
            if not candidates:
                return {"status": "error", "text": "", "confidence": 0.0, "message": last_message}
            _, text, confidence = max(candidates, key=lambda item: item[0])
            return {"status": "ok", "text": text, "confidence": round(confidence, 4), "message": ""}

    def recognise(
        self,
        image: np.ndarray,
        fallback_text: str = "",
        fallback_confidence: float = 0.0,
        expected_language: str = "",
    ) -> dict[str, Any]:
        """Return OCR text, never an empty result solely because weights are absent."""

        use_lao_model = expected_language == "lao" and self.lao_available
        if use_lao_model or self.available:
            selected_config = self.lao_config if use_lao_model else self.config
            selected_weights = self.lao_weights if use_lao_model else self.weights
            result = self._recognise_custom(image, selected_config, selected_weights, expected_language)
            if result["text"]:
                result["method"] = "custom-paddleocr-lao-full" if use_lao_model else "custom-paddleocr"
                result["script"] = detect_text_script(result["text"])
                return result

        missing_weights = self.lao_weights if expected_language == "lao" else self.weights
        missing = f"Custom checkpoint unavailable; using fallback: {missing_weights.with_suffix('.pdparams')}"
        torch_text, torch_confidence, torch_message = self._recognise_with_torch(image, expected_language)
        if torch_text:
            return {
                "status": "ok",
                "method": "trained-torch-crnn",
                "text": torch_text,
                "confidence": round(torch_confidence, 4),
                "script": detect_text_script(torch_text),
                "message": "",
            }
        paddle_text, paddle_confidence, paddle_message = self._recognise_with_paddle(image, expected_language)
        if paddle_text:
            return {
                "status": "fallback",
                "method": "paddleocr-pretrained",
                "text": paddle_text,
                "confidence": round(paddle_confidence, 4),
                "script": detect_text_script(paddle_text),
                "message": missing,
            }
        tesseract_text, tesseract_confidence, tesseract_message = self._recognise_with_tesseract(image, expected_language)
        if tesseract_text:
            return {
                "status": "fallback",
                "method": "tesseract-lao",
                "text": tesseract_text,
                "confidence": tesseract_confidence,
                "script": detect_text_script(tesseract_text),
                "message": missing,
            }
        if fallback_text:
            fallback_script = detect_text_script(fallback_text)
            language_message = ""
            if expected_language == "lao" and fallback_script != "lao":
                language_message = "Lao script was not detected; fallback is a Latin transliteration"
            if expected_language != "lao":
                # For Thai/English the YOLO character model is the trained
                # recogniser used by this project, not an OCR failure.
                return {
                    "status": "ok",
                    "method": "yolo-character-reading",
                    "text": fallback_text,
                    "confidence": round(float(fallback_confidence), 4),
                    "script": fallback_script,
                    "message": "",
                }
            return {
                "status": "fallback",
                "method": "yolo-character-reading",
                "text": fallback_text,
                "confidence": round(float(fallback_confidence), 4),
                "script": fallback_script,
                "message": "; ".join(value for value in (missing, torch_message, paddle_message, tesseract_message, language_message) if value),
            }
        return {
            "status": "unavailable",
            "method": "none",
            "text": "",
            "confidence": 0.0,
            "script": "unknown",
            "message": "; ".join(value for value in (missing, torch_message, paddle_message, tesseract_message) if value),
        }


class LicensePlateScanner:
    def __init__(
        self,
        detector_path: Path,
        thai_path: Path,
        lao_path: Path,
        ocr: CustomPaddleOCR,
        vehicle_type_path: Path | None = None,
        plate_type_path: Path | None = None,
        pipeline_config_path: Path | None = None,
        pipeline_mode: str | None = None,
        scan_roi_override: dict[str, Any] | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Missing dependency 'ultralytics'. Install it with: pip install ultralytics") from error
        for path in (detector_path, thai_path, lao_path):
            if not path.is_file():
                raise FileNotFoundError(f"Model not found: {path}")
        self.pipeline_config = load_pipeline_config(pipeline_config_path)
        roi_config = (
            scan_roi_override
            if scan_roi_override is not None
            else self.pipeline_config.get("scan_roi")
        )
        self.scan_roi = self._normalise_scan_roi(roi_config)
        configured_mode = str(self.pipeline_config.get("pipeline_mode", "auto"))
        self.pipeline_mode = pipeline_mode or configured_mode
        if self.pipeline_mode not in ("auto", "full", "fast"):
            raise ValueError("pipeline_mode must be auto, full, or fast")
        self.detector = YOLO(str(detector_path))
        self.vehicle_classifier = YoloRegionClassifier(
            vehicle_type_path,
            self.pipeline_config.get("vehicle_class_mapping", {}),
        )
        self.plate_type_classifier = YoloRegionClassifier(
            plate_type_path,
            self.pipeline_config.get("plate_class_mapping", {}),
        )
        self.country_models = {"thai": YOLO(str(thai_path)), "lao": YOLO(str(lao_path))}
        self.country_classifier = CountryClassifier(
            self.pipeline_config.get("country", {}).get("unknown_score", 0.0)
        )
        self.associator = VehiclePlateAssociator()
        self.context_router = ContextRouter(self.pipeline_config)
        self.context_validator = ContextValidator(self.pipeline_config)
        self.ocr = ocr

    @staticmethod
    def _normalise_scan_roi(value: Any) -> dict[str, Any] | None:
        """Validate and normalise the optional plate scanning region."""

        if not isinstance(value, dict):
            return None
        enabled = value.get("enabled", False)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() in ("1", "true", "yes", "on")
        if not enabled:
            return None

        shape = str(value.get("shape", "rectangle")).strip().lower()
        shape_aliases = {"rect": "rectangle", "box": "rectangle", "oval": "ellipse"}
        shape = shape_aliases.get(shape, shape)
        if shape not in ("rectangle", "circle", "ellipse"):
            raise ValueError("scan_roi.shape must be rectangle, circle, or ellipse")
        unit = str(value.get("unit", "normalized")).strip().lower()
        if unit in ("normalised", "ratio", "relative"):
            unit = "normalized"
        if unit not in ("normalized", "pixels"):
            raise ValueError("scan_roi.unit must be normalized or pixels")

        try:
            x = float(value.get("x", 0.0))
            y = float(value.get("y", 0.0))
            width = float(value.get("width", 1.0))
            height = float(value.get("height", 1.0))
        except (TypeError, ValueError) as error:
            raise ValueError("scan_roi x, y, width, and height must be numbers") from error
        if unit == "normalized":
            if not (0.0 <= x < 1.0 and 0.0 <= y < 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                raise ValueError("normalized scan_roi x/y/width/height must be between 0 and 1")
            if x + width > 1.0 or y + height > 1.0:
                raise ValueError("normalized scan_roi must stay inside the image")
        elif width <= 0 or height <= 0 or x < 0 or y < 0:
            raise ValueError("pixel scan_roi x/y must be non-negative and width/height must be positive")

        colour = value.get("color", "#ff0000")
        if isinstance(colour, (list, tuple)) and len(colour) == 3:
            bgr = tuple(max(0, min(255, int(channel))) for channel in colour)
        else:
            colour_text = str(colour).strip().lstrip("#")
            if len(colour_text) != 6:
                raise ValueError("scan_roi.color must be #RRGGBB or a 3-item BGR list")
            try:
                # Config uses the familiar RGB hex notation; OpenCV draws BGR.
                bgr = (
                    int(colour_text[4:6], 16),
                    int(colour_text[2:4], 16),
                    int(colour_text[0:2], 16),
                )
            except ValueError as error:
                raise ValueError("scan_roi.color must be #RRGGBB") from error
        thickness = max(1, int(value.get("thickness", 3)))
        return {
            "enabled": True,
            "shape": shape,
            "unit": unit,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "color": bgr,
            "thickness": thickness,
        }

    def _scan_roi_bounds(self, image: np.ndarray) -> tuple[int, int, int, int] | None:
        roi = getattr(self, "scan_roi", None)
        if not roi:
            return None
        image_height, image_width = image.shape[:2]
        if roi["unit"] == "pixels":
            x = int(round(roi["x"]))
            y = int(round(roi["y"]))
            width = int(round(roi["width"]))
            height = int(round(roi["height"]))
        else:
            x = int(round(roi["x"] * image_width))
            y = int(round(roi["y"] * image_height))
            width = int(round(roi["width"] * image_width))
            height = int(round(roi["height"] * image_height))
        left = max(0, min(image_width - 1, x))
        top = max(0, min(image_height - 1, y))
        right = max(left + 1, min(image_width - 1, x + width))
        bottom = max(top + 1, min(image_height - 1, y + height))
        return left, top, right, bottom

    def _box_inside_scan_roi(self, box: Iterable[float], image: np.ndarray) -> bool:
        """Accept a detector box when its centre is inside the configured ROI."""

        bounds = self._scan_roi_bounds(image)
        if bounds is None:
            return True
        x1, y1, x2, y2 = [float(value) for value in box]
        centre_x = (x1 + x2) / 2.0
        centre_y = (y1 + y2) / 2.0
        left, top, right, bottom = bounds
        roi = self.scan_roi
        if roi["shape"] == "rectangle":
            return left <= centre_x <= right and top <= centre_y <= bottom
        radius_x = max(1.0, (right - left) / 2.0)
        radius_y = max(1.0, (bottom - top) / 2.0)
        if roi["shape"] == "circle":
            radius = min(radius_x, radius_y)
            radius_x = radius_y = radius
        normal_x = (centre_x - (left + right) / 2.0) / radius_x
        normal_y = (centre_y - (top + bottom) / 2.0) / radius_y
        return normal_x * normal_x + normal_y * normal_y <= 1.0

    def draw_scan_roi(self, image: np.ndarray) -> np.ndarray:
        """Draw the configured scan region on a preview frame."""

        annotated = image.copy()
        bounds = self._scan_roi_bounds(annotated)
        if bounds is None:
            return annotated
        left, top, right, bottom = bounds
        roi = self.scan_roi
        colour = tuple(int(channel) for channel in roi["color"])
        thickness = int(roi["thickness"])
        if roi["shape"] == "rectangle":
            cv2.rectangle(annotated, (left, top), (right, bottom), colour, thickness)
        else:
            centre = ((left + right) // 2, (top + bottom) // 2)
            radius_x = max(1, (right - left) // 2)
            radius_y = max(1, (bottom - top) // 2)
            if roi["shape"] == "circle":
                radius = min(radius_x, radius_y)
                cv2.circle(annotated, centre, radius, colour, thickness)
            else:
                cv2.ellipse(annotated, centre, (radius_x, radius_y), 0, 0, 360, colour, thickness)
        cv2.putText(
            annotated,
            "SCAN ROI",
            (left + 8, max(24, top + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            colour,
            max(1, min(2, thickness)),
            cv2.LINE_AA,
        )
        return annotated

    def _vehicle_types(
        self,
        image: np.ndarray,
        confidence: float,
        imgsz: int,
    ) -> list[dict[str, Any]]:
        """Detect vehicle types and keep their boxes for plate matching."""

        output = self.vehicle_classifier.detect(image, confidence, imgsz)
        for vehicle_id, item in enumerate(output, start=1):
            item["vehicle_id"] = vehicle_id
            item["tracking_id"] = None
            item["vehicle_type"] = item["normalized_class"]
            item["vehicle_type_confidence"] = item["confidence"]
            item["vehicle_type_box"] = item["box"]
        return output

    @staticmethod
    def _match_vehicle_type(
        plate_box: list[int] | tuple[int, int, int, int],
        vehicle_types: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Match a plate to the vehicle whose detection contains its centre."""

        matched = VehiclePlateAssociator().associate(list(plate_box), vehicle_types)
        matched["vehicle_type"] = matched.get("normalized_class", "unknown")
        matched["vehicle_type_confidence"] = matched.get("confidence", 0.0)
        matched["vehicle_type_box"] = matched.get("box")
        return matched

    @staticmethod
    def _draw_vehicle_types(image: np.ndarray, vehicle_types: list[dict[str, Any]]) -> None:
        """Draw vehicle detections before plate boxes are drawn on top."""

        colour = (255, 140, 0)
        for vehicle in vehicle_types:
            x1, y1, x2, y2 = vehicle["box"]
            label = f"{vehicle['display_name']} {vehicle['confidence']:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(
                image,
                label[:50],
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                colour,
                2,
                cv2.LINE_AA,
            )

    @staticmethod
    def _characters(result: Any, names: dict[int, str]) -> list[CharacterDetection]:
        if result.boxes is None:
            return []
        boxes = result.boxes
        output: list[CharacterDetection] = []
        for xyxy, confidence, class_id in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()):
            output.append(CharacterDetection(names[int(class_id)], float(confidence), *map(float, xyxy)))
        return output

    def _read_country_model(
        self,
        crop: np.ndarray,
        country: str,
        confidence: float,
        imgsz: int,
        fast_mode: bool = False,
    ) -> dict[str, Any]:
        model = self.country_models[country]
        expected = 6 if country == "thai" else 4

        def read_variant(image: np.ndarray, variant: str, variant_confidence: float) -> dict[str, Any]:
            prediction = model.predict(
                image,
                conf=variant_confidence,
                imgsz=max(960, imgsz),
                verbose=False,
            )[0]
            result = analyse_country_characters(self._characters(prediction, model.names), country)
            result["score"] = country_reading_score(result, country)
            result["inference_variant"] = variant
            return result

        # Use the original and two geometry-preserving enhanced variants for
        # every read. This favours accuracy over inference speed and keeps all
        # resulting boxes aligned with the saved crop/full-vehicle labels.
        reading = read_variant(crop, "original-high-resolution", confidence)
        def reading_rank(value: dict[str, Any]) -> tuple[bool, int, float, float]:
            count = int(value.get("digit_detection_count", 0))
            return (
                bool(value.get("complete")),
                -abs(count - expected),
                float(value.get("score", 0.0)),
                float(value.get("digit_confidence", 0.0)),
            )

        def reading_digits(value: dict[str, Any]) -> str:
            return f"{value.get('plate_prefix', '')}{value.get('plate_number', '')}"

        def accept_variant(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
            """Avoid replacing a complete reading with a speculative result."""

            if candidate.get("complete") and current.get("complete"):
                candidate_digits = reading_digits(candidate)
                current_digits = reading_digits(current)
                differences = sum(
                    first != second for first, second in zip(candidate_digits, current_digits)
                ) + abs(len(candidate_digits) - len(current_digits))
                return (
                    differences <= 1
                    and float(candidate.get("digit_confidence", 0.0))
                    >= float(current.get("digit_confidence", 0.0)) + 0.03
                )
            return reading_rank(candidate) > reading_rank(current)

        variants = () if fast_mode else (
            (preprocess_character_model_crop(crop), "colour-enhanced", confidence),
            (enhance_character_crop(crop), "contrast-enhanced", max(0.08, confidence * 0.60)),
        )
        for image, variant, variant_confidence in variants:
            candidate = read_variant(image, variant, variant_confidence)
            if accept_variant(candidate, reading):
                reading = candidate
        return self._attach_province_candidates(
            reading, crop, country, confidence, imgsz, fast_mode=fast_mode
        )

    def _attach_province_candidates(
        self,
        reading: dict[str, Any],
        crop: np.ndarray,
        country: str,
        confidence: float,
        imgsz: int,
        fast_mode: bool = False,
    ) -> dict[str, Any]:
        """Run a high-resolution province-only pass and retain low-score evidence.

        Province text occupies only a thin band of a number plate.  It needs a
        separate pass so it is not competing with the much larger registration
        digits. Low-confidence candidates are retained for temporal voting but
        are never accepted for a single still image.
        """

        # Province ROI inference uses three additional model calls. During
        # video/camera sampling the regular reading is sufficient for temporal
        # voting; the full province pass is retained for still-image/final OCR.
        if fast_mode:
            return reading
        height, width = crop.shape[:2]
        if height < 8 or width < 16:
            return reading
        top_ratio, bottom_ratio = (0.0, 0.55) if country == "lao" else (0.45, 1.0)
        y1 = max(0, min(height - 1, int(round(height * top_ratio))))
        y2 = max(y1 + 1, min(height, int(round(height * bottom_ratio))))
        roi = crop[y1:y2, :]
        model = self.country_models[country]
        province_names = THAI_PROVINCE_NAMES if country == "thai" else LAO_PROVINCE_NAMES
        candidates: dict[str, dict[str, Any]] = {}

        def add_candidate(
            code: object,
            score: object,
            box: list[float],
            source: str,
        ) -> None:
            code_text = str(code or "")
            confidence_value = float(score or 0.0)
            if code_text not in province_names or confidence_value < PROVINCE_ROI_MIN_CONFIDENCE:
                return
            candidate = {
                "code": code_text,
                "province": province_names[code_text],
                "confidence": round(confidence_value, 4),
                "box": [round(value, 2) for value in box],
                "source": source,
            }
            previous = candidates.get(code_text)
            if previous is None or candidate["confidence"] > previous["confidence"]:
                candidates[code_text] = candidate

        # Keep the original plate-level candidate as temporal evidence too.
        for token in reading.get("tokens", []):
            if not isinstance(token, dict) or token.get("kind") != "province":
                continue
            token_box = token.get("box", [])
            if isinstance(token_box, list) and len(token_box) == 4:
                add_candidate(token.get("label"), token.get("confidence"), token_box, "plate")

        for image, source, variant_confidence in (
            (roi, "province_roi_raw", max(0.05, confidence * 0.45)),
            (
                preprocess_character_model_crop(roi),
                "province_roi_colour",
                max(0.05, confidence * 0.35),
            ),
            (
                enhance_character_crop(roi),
                "province_roi_contrast",
                max(0.05, confidence * 0.25),
            ),
        ):
            prediction = model.predict(image, conf=variant_confidence, imgsz=max(1280, imgsz), verbose=False)[0]
            for item in self._characters(prediction, model.names):
                add_candidate(
                    item.text,
                    item.confidence,
                    [item.x1, item.y1 + y1, item.x2, item.y2 + y1],
                    source,
                )

        result = dict(reading)
        province_candidates = sorted(
            candidates.values(), key=lambda value: float(value["confidence"]), reverse=True
        )
        result["province_candidates"] = province_candidates
        if not province_candidates:
            return result

        best = province_candidates[0]
        runner_up = province_candidates[1] if len(province_candidates) > 1 else None
        margin = float(best["confidence"]) - float(runner_up["confidence"]) if runner_up else 1.0
        current_confidence = float(result.get("province_confidence", 0.0))
        if (
            float(best["confidence"]) >= PROVINCE_MIN_CONFIDENCE
            and (not result.get("province_code") or float(best["confidence"]) >= current_confidence + 0.03)
        ):
            result["province"] = best["province"]
            result["province_code"] = best["code"]
            result["province_confidence"] = best["confidence"]
            result["province_status"] = "province_roi_accepted"
            result["province_margin"] = round(margin, 4)
            tokens = list(result.get("tokens", []))
            if not any(
                isinstance(token, dict) and token.get("label") == best["code"]
                for token in tokens
            ):
                tokens.append(
                    {
                        "label": best["code"],
                        "kind": "province",
                        "confidence": best["confidence"],
                        "box": best["box"],
                    }
                )
                result["tokens"] = tokens
        return result

    def finalise_reading(
        self,
        ocr_ready_crop: np.ndarray,
        country: str,
        reading: dict[str, Any],
        run_ocr: bool = True,
    ) -> tuple[dict[str, Any], str, str]:
        """Fuse OCR once with a structured country-model reading."""

        if run_ocr:
            ocr = self.ocr.recognise(
                ocr_ready_crop,
                fallback_text=reading["text"],
                fallback_confidence=reading["confidence"],
                expected_language=country,
            )
        else:
            ocr = {
                "status": "deferred",
                "method": "structured-yolo",
                "text": reading["text"],
                "confidence": reading["confidence"],
                "script": country,
                "message": "Full OCR is deferred until temporal video voting selects the best crop",
            }

        original_ocr_text = str(ocr.get("text") or "")
        ocr_province_code, ocr_province = extract_ocr_province(original_ocr_text, country)
        if ocr_province_code:
            ocr_confidence = float(ocr.get("confidence", 0.0))
            existing_province_confidence = float(reading.get("province_confidence", 0.0))
            # A known province from the trained OCR transcription is preferred
            # over a weak/incorrect detector box. Keep a strong YOLO result if
            # OCR confidence is not sufficient to replace it.
            if (
                ocr_confidence >= 0.45
                and (
                    not reading.get("province_code")
                    or ocr_confidence >= existing_province_confidence + 0.03
                )
            ):
                reading["province"] = ocr_province
                reading["province_code"] = ocr_province_code
                reading["province_confidence"] = round(max(existing_province_confidence, ocr_confidence), 4)
                reading["province_status"] = "ocr_accepted"
                reading["province_source"] = "ocr"
        plate_prefix, plate_number, fusion_source = resolve_plate_fields(country, reading, ocr)
        display_text = (
            f"{plate_prefix}-{plate_number}"
            if country == "thai" and plate_prefix and plate_number
            else " ".join(part for part in (plate_prefix, plate_number) if part)
        )
        if original_ocr_text and original_ocr_text != display_text:
            ocr["raw_text"] = original_ocr_text
        ocr["text"] = display_text
        ocr["plate_prefix"] = plate_prefix
        ocr["plate_number"] = plate_number
        if ocr_province_code:
            ocr["province_code"] = ocr_province_code
            ocr["province"] = ocr_province
            ocr["province_source"] = "ocr"
        ocr["fusion_source"] = fusion_source
        return ocr, plate_prefix, plate_number

    def scan(
        self,
        image: np.ndarray,
        detector_confidence: float,
        character_confidence: float,
        padding: float,
        imgsz: int,
        run_ocr: bool = True,
        vehicle_type_confidence: float = 0.25,
        plate_type_confidence: float = 0.25,
        fast_mode: bool = False,
    ) -> tuple[list[dict[str, Any]], np.ndarray]:
        annotated = self.draw_scan_roi(image)
        vehicle_types = (
            self._vehicle_types(image, vehicle_type_confidence, imgsz)
            if self.pipeline_mode == "full"
            else []
        )
        detections = self.detector.predict(image, conf=detector_confidence, imgsz=imgsz, verbose=False)[0]
        if detections.boxes is None or len(detections.boxes) == 0:
            self._draw_vehicle_types(annotated, vehicle_types)
            return [], annotated
        if self.pipeline_mode != "full":
            vehicle_types = self._vehicle_types(image, vehicle_type_confidence, imgsz)
        self._draw_vehicle_types(annotated, vehicle_types)

        records: list[dict[str, Any]] = []
        detector_names = self.detector.names
        for index, (xyxy, confidence, raw_class_id) in enumerate(
            zip(
                detections.boxes.xyxy.cpu().tolist(),
                detections.boxes.conf.cpu().tolist(),
                detections.boxes.cls.cpu().tolist(),
            ),
            start=1,
        ):
            if not self._box_inside_scan_roi(xyxy, image):
                continue
            crop, crop_box = clamp_crop(image, xyxy, padding)
            # The detector box is intentionally tight, while the recogniser
            # was trained on crops containing a small plate border.  Keep the
            # displayed/saved box unchanged but give OCR a slightly wider crop.
            # Lao glyphs can lose their upper stroke when the detector box is
            # tight. A little extra border lets the variant ensemble recover
            # the prefix without materially including the surrounding car.
            ocr_crop, ocr_crop_box = clamp_crop(image, xyxy, max(0.25, padding))
            readings = {
                country: self._read_country_model(
                    crop,
                    country,
                    character_confidence,
                    imgsz,
                    fast_mode=fast_mode,
                )
                for country in ("thai", "lao")
            }
            country_result = self.country_classifier.classify(readings)
            country = str(country_result["country"])
            recognition_country = str(country_result["raw_country"])
            country_margin = float(country_result["margin"])
            vehicle = self._match_vehicle_type(crop_box, vehicle_types)
            plate_type = self.plate_type_classifier.classify(crop, plate_type_confidence, imgsz)
            context = ProcessingContext(
                country=country,
                vehicle_type=str(vehicle.get("normalized_class", "unknown")),
                plate_type=str(plate_type.get("normalized_class", "unknown")),
            )
            preprocessing_profile = self.context_router.preprocessing_profile(context)
            ocr_model = self.context_router.ocr_model(context)
            preprocessing_parameters = self.pipeline_config.get(
                "preprocessing_parameters", {}
            ).get(preprocessing_profile, {})
            if not isinstance(preprocessing_parameters, dict):
                preprocessing_parameters = {}
            ocr_ready_crop = preprocess_plate_crop(ocr_crop, preprocessing_parameters)
            ocr, plate_prefix, plate_number = self.finalise_reading(
                ocr_ready_crop,
                recognition_country,
                readings[recognition_country],
                run_ocr=run_ocr,
            )
            selected_reading = readings[recognition_country]
            reading_layout = (
                "two_line"
                if len(selected_reading.get("raw_lines", [])) > 1
                else "single_line"
            )
            validation = self.context_validator.validate(
                country,
                context.vehicle_type,
                context.plate_type,
                plate_prefix,
                plate_number,
                str(selected_reading.get("province", "")),
                reading_layout,
            )
            recognition_confidence = min(
                0.99,
                float(confidence) * 0.20
                + float(selected_reading.get("digit_confidence", 0.0)) * 0.43
                + float(selected_reading.get("prefix_confidence", 0.0)) * (0.12 if recognition_country == "lao" else 0.0)
                + max(0.0, min(1.0, country_margin)) * 0.15
                + (0.10 if selected_reading.get("complete") else 0.0)
                + float(selected_reading.get("province_confidence", 0.0)) * 0.05,
            )
            x1, y1, x2, y2 = crop_box
            raw_class_index = int(raw_class_id)
            raw_plate_class = str(
                detector_names.get(raw_class_index, raw_class_index)
                if isinstance(detector_names, dict)
                else detector_names[raw_class_index]
            )
            vehicle_summary = {
                "id": vehicle.get("vehicle_id"),
                "type": context.vehicle_type,
                "raw_class": vehicle.get("raw_class", "unknown"),
                "display_name": vehicle.get("display_name", "Unknown"),
                "confidence": vehicle.get("confidence", 0.0),
                "bbox": vehicle.get("box"),
                "tracking_id": vehicle.get("tracking_id"),
                "association_score": vehicle.get("association_score", 0.0),
            }
            record = {
                "id": index,
                "country": country,
                "country_confidence": country_result["confidence"],
                "vehicle_type": vehicle["vehicle_type"],
                "vehicle_type_confidence": vehicle["vehicle_type_confidence"],
                "vehicle_type_box": vehicle["vehicle_type_box"],
                "vehicle_id": vehicle.get("vehicle_id"),
                "vehicle": vehicle_summary,
                "plate_type": context.plate_type,
                "plate_type_raw_class": plate_type.get("raw_class", "unknown"),
                "plate_type_display_name": plate_type.get("display_name", "Unknown"),
                "plate_type_confidence": plate_type.get("confidence", 0.0),
                "raw_plate_class": raw_plate_class,
                "province": readings[recognition_country].get("province", ""),
                "province_code": readings[recognition_country].get("province_code", ""),
                "plate_prefix": plate_prefix,
                "plate_prefix_code": readings[recognition_country].get("plate_prefix_code", ""),
                "plate_number": plate_number,
                "detection_confidence": round(float(confidence), 4),
                "recognition_confidence": round(recognition_confidence, 4),
                "confirmed": bool(selected_reading.get("complete")) and bool(validation["valid"]),
                "country_confidence_margin": round(country_margin, 4),
                "box": crop_box,
                "ocr_crop_box": ocr_crop_box,
                "ocr_preprocessing": preprocessing_profile,
                "country_readings": readings,
                "ocr": ocr,
                "validation": validation,
                "processing": {
                    "pipeline_mode": self.pipeline_mode,
                    "preprocessing_profile": preprocessing_profile,
                    "preprocessing_parameters": preprocessing_parameters,
                    "ocr_model": ocr_model,
                },
                "plate": {
                    "bbox": crop_box,
                    "detection_confidence": round(float(confidence), 4),
                    "raw_class": raw_plate_class,
                    "country": country,
                    "country_confidence": country_result["confidence"],
                    "type": context.plate_type,
                    "plate_type": context.plate_type,
                    "plate_type_raw_class": plate_type.get("raw_class", "unknown"),
                    "plate_type_confidence": plate_type.get("confidence", 0.0),
                    "prefix": plate_prefix,
                    "number": plate_number,
                    "province": readings[recognition_country].get("province", ""),
                    "text": ocr.get("text", ""),
                    "ocr_confidence": ocr.get("confidence", 0.0),
                    "ocr": ocr,
                    "validation": validation,
                },
            }
            records.append(record)
            # Keep rejected detector candidates available for debug output, but
            # do not present them as licence plates. This prevents reflective
            # lamps and grilles from appearing as grey ``UNKNOWN`` boxes.
            if record["confirmed"]:
                draw_character_boxes(annotated, selected_reading, (x1, y1))
                colour = (0, 180, 0) if country == "thai" else (0, 100, 255)
                label = (
                    f"{raw_plate_class}->{context.plate_type} "
                    f"{country.upper()} {float(country_result['confidence']):.2f} "
                    f"OCR {float(ocr.get('confidence', 0.0)):.2f}: "
                    f"{ocr['text'] or selected_reading['text']}"
                )
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(annotated, label[:70], (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2, cv2.LINE_AA)
        return records, annotated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect, classify, and OCR Thai/Lao vehicle licence plates.")
    parser.add_argument("image", type=Path, help="Input vehicle image")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "scan", help="Directory for crop images, annotated image, and result JSON")
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--vehicle-type-model", type=Path, default=DEFAULT_VEHICLE_TYPE)
    parser.add_argument("--plate-type-model", type=Path, default=None)
    parser.add_argument("--thai-model", type=Path, default=DEFAULT_THAI)
    parser.add_argument("--lao-model", type=Path, default=DEFAULT_LAO)
    parser.add_argument("--pipeline-config", type=Path, default=DEFAULT_PIPELINE_CONFIG)
    parser.add_argument("--pipeline-mode", choices=("auto", "full", "fast"), default=None)
    parser.add_argument("--ocr-source", type=Path, default=DEFAULT_OCR_SOURCE)
    parser.add_argument("--ocr-config", type=Path, default=DEFAULT_OCR_CONFIG)
    parser.add_argument("--ocr-weights", type=Path, default=DEFAULT_OCR_WEIGHTS, help="PaddleOCR checkpoint path, with or without .pdparams")
    parser.add_argument("--ocr-torch-model", type=Path, default=DEFAULT_TORCH_OCR, help="Locally trained CRNN checkpoint")
    parser.add_argument("--detector-confidence", type=float, default=0.35)
    parser.add_argument("--vehicle-type-confidence", type=float, default=0.25)
    parser.add_argument("--plate-type-confidence", type=float, default=0.25)
    parser.add_argument("--character-confidence", type=float, default=0.20)
    parser.add_argument("--padding", type=float, default=0.03, help="Extra crop padding as a fraction of plate size")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--camera-id",
        default=os.getenv("CAR_SCAN_ARCHIVE_CAMERA_ID", "0"),
        help="Camera identifier used in archived full_vehicle/plate_crops names",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    image_path = arguments.image.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not 0 <= arguments.padding < 1:
        raise ValueError("--padding must be between 0 and 1")

    image = read_image(image_path)
    ocr = CustomPaddleOCR(arguments.ocr_source.resolve(), arguments.ocr_config.resolve(), arguments.ocr_weights.resolve(), arguments.ocr_torch_model.resolve())
    scanner = LicensePlateScanner(
        arguments.detector.resolve(),
        arguments.thai_model.resolve(),
        arguments.lao_model.resolve(),
        ocr,
        arguments.vehicle_type_model.resolve(),
        plate_type_path=arguments.plate_type_model.resolve() if arguments.plate_type_model else None,
        pipeline_config_path=arguments.pipeline_config.resolve(),
        pipeline_mode=arguments.pipeline_mode,
    )
    plates, annotated = scanner.scan(
        image,
        arguments.detector_confidence,
        arguments.character_confidence,
        arguments.padding,
        arguments.imgsz,
        vehicle_type_confidence=arguments.vehicle_type_confidence,
        plate_type_confidence=arguments.plate_type_confidence,
    )

    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from car_scan.config import Settings
        from car_scan.service import ScanService
    except ModuleNotFoundError:
        from src.car_scan.config import Settings
        from src.car_scan.service import ScanService

    archive_service = ScanService(
        Settings(
            root=ROOT,
            output_dir=output_dir,
            database_url="",
            archive_camera_id=str(arguments.camera_id),
        )
    )
    scan_output_dir = archive_service._dated_output_dir()
    annotated_path = scan_output_dir / f"{image_path.stem}_annotated.jpg"
    write_image(annotated_path, annotated)
    for plate in plates:
        if not archive_service._has_complete_registration(plate):
            continue
        x1, y1, x2, y2 = plate["box"]
        ox1, oy1, ox2, oy2 = plate["ocr_crop_box"]
        processing = plate.get("processing", {})
        parameters = (
            processing.get("preprocessing_parameters", {})
            if isinstance(processing, dict)
            else {}
        )
        full_path, crop_path, ready_path, character_path = archive_service._save_archive_images(
            plate,
            str(arguments.camera_id),
            image,
            image[y1:y2, x1:x2],
            preprocess_plate_crop(image[oy1:oy2, ox1:ox2], parameters),
            draw_character_boxes(
                image[y1:y2, x1:x2].copy(),
                archive_service._selected_character_reading(plate),
            ),
            scanner,
            write_image,
        )
        plate["full_vehicle_image"] = str(full_path)
        plate["crop_image"] = str(crop_path)
        plate["ocr_ready_image"] = str(ready_path)
        plate["character_annotated_image"] = str(character_path)
    result = {
        "input": str(image_path),
        "plate_count": len(plates),
        "plates": plates,
        "annotated_image": str(annotated_path),
        "output_dir": str(scan_output_dir),
    }
    result_path = scan_output_dir / f"{image_path.stem}_result.json"
    result["result_json"] = str(result_path)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print_cmd_summary(image_path, plates)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved results to: {result_path}")


if __name__ == "__main__":
    main()
