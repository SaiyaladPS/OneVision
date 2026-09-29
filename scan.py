"""Scan Thai and Lao vehicle licence plates from an image.

Pipeline
--------
1. ``type_car_license`` identifies the vehicle type in the source image.
2. ``detect_license`` finds each plate in the source image.
3. ``thai_license_plate`` and ``lao_license_plate`` run only on that detector
   crop, never on the full source image. If no plate is cropped, they are skipped.
   The model with the strongest character detections identifies the country.
4. ``thai_license_plate`` or ``lao_license_plate`` supplies every character.
   PaddleOCR is not part of this pipeline.

Example:
    python scan.py path/to/car.jpg

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
    from car_scan.compute import bind_yolo_device, yolo_predict
    from car_scan.config import latest_ocr_weights
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
    from src.car_scan.compute import bind_yolo_device, yolo_predict
    from src.car_scan.config import latest_ocr_weights
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
DEFAULT_OCR_SOURCE = MODEL_DIR
DEFAULT_OCR_CONFIG = DEFAULT_OCR_SOURCE / "configs" / "rec" / "plate_rec_small.yml"
DEFAULT_OCR_WEIGHTS = DEFAULT_OCR_SOURCE / "output" / "plate_rec_small" / "best_accuracy"
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
    "NPM": "นครพนม", "NPT": "นครปฐม", "NST": "นครศรีธรรมราช",
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
    # Alias present in the current thai_license_plate checkpoint.
    "NRT": "นราธิวาส",
}

THAI_PROVINCE_ALIASES = {
    "กทม": "BKK",
    "กรุงเทพ": "BKK",
    "กรุงเทพฯ": "BKK",
    "กรุงเทพมหานคร": "BKK",
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

LAO_PROVINCE_ALIASES = {
    "ນະຄອນຫຼວງ": "VTE2",
    "ນະຄອນຫຼວງວຽງຈັນ": "VTE2",
    "ກຳແພງນະຄອນ": "VTE",
    "ວຽງຈັນ": "VTP",
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
    "D": "ງ", "E": "ຈ", "F": "ສ", "G": "ຊ", "H": "ຍ", "I": "ດ",
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
    report: dict[str, Any] | None = None,
) -> np.ndarray:
    """Enhance a detected plate crop before it is sent to any OCR engine.

    This function deliberately accepts a crop only.  It normalises lighting,
    removes small compression noise, corrects a conservative horizontal skew,
    and enlarges the plate while preserving the original glyph shapes.
    ``report`` (optional) receives what was done: ``perspective_applied``,
    ``skew_angle`` (degrees, 0 when no rotation was applied) and ``scale``.
    """

    if image is None or image.size == 0:
        raise ValueError("Cannot preprocess an empty licence-plate crop")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    parameters = parameters or {}
    source_shape = image.shape[:2]
    image = _rectify_plate_perspective(image, parameters)
    if report is not None:
        report["perspective_applied"] = image.shape[:2] != source_shape
        report["skew_angle"] = 0.0
    height, width = image.shape[:2]
    target_width = float(parameters.get("target_width", 720.0))
    max_scale = float(parameters.get("max_scale", 5.0))
    scale = min(max_scale, max(1.0, target_width / max(width, 1)))
    enlarged = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if report is not None:
        report["scale"] = round(float(scale), 3)
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
            if report is not None:
                report["skew_angle"] = round(angle, 2)
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


# The Thai/Lao character models are fed a straightened, enhanced copy of the
# plate alongside the raw detector crop. On CCTV crops (50-130 px wide) a
# rectified copy at roughly twice the crop width measured best: rotating the
# tiny raw crop alone created 1/7 confusions, while the enhanced copy at
# 2x raised digit confidence and the number of complete readings.
RECOGNITION_RECTIFY_MIN_WIDTH = 256
RECOGNITION_RECTIFY_MAX_WIDTH = 640
RECOGNITION_RECTIFY_SCALE = 2.0


def rectify_plate_for_recognition(
    ocr_crop: np.ndarray,
    tight_width: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Return a straightened, enhanced plate copy for the character models.

    ``ocr_crop`` is the slightly padded detector crop; the extra border lets a
    perspective/rotation fix run without cutting glyphs. The result keeps the
    plate geometry (no cropping to text) so character boxes remain plausible.
    """

    report: dict[str, Any] = {"applied": False}
    if ocr_crop is None or getattr(ocr_crop, "size", 0) == 0:
        return None, report
    height, width = ocr_crop.shape[:2]
    if height < 8 or width < 16:
        return None, report
    target_width = float(
        min(
            RECOGNITION_RECTIFY_MAX_WIDTH,
            max(RECOGNITION_RECTIFY_MIN_WIDTH, int(round(tight_width * RECOGNITION_RECTIFY_SCALE))),
        )
    )
    parameters = {
        "perspective": True,
        "perspective_min_area": 0.35,
        "target_width": target_width,
        "max_scale": 4.0,
        "clahe_clip": 2.2,
        "sharpen_strength": 0.35,
        "border_ratio": 0.0,
    }
    try:
        rectified = preprocess_plate_crop(ocr_crop, parameters, report)
    except Exception:
        return None, {"applied": False}
    report.update(
        {
            "applied": True,
            "target_width": int(target_width),
            "output_size": [int(rectified.shape[1]), int(rectified.shape[0])],
        }
    )
    return rectified, report


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


def normalise_character_illumination(image: np.ndarray) -> np.ndarray:
    """Remove uneven glare while retaining the strokes needed by character YOLO.

    Reflective plates often contain a bright patch from headlights, sunlight,
    or the camera's own IR light.  A raw detector can interpret the broken
    stroke of a ``1`` as a ``6`` in that patch.  Estimate the slow-changing
    illumination field and divide it out, then keep the result greyscale so a
    coloured reflection cannot dominate the digit shape.
    """

    if image is None or image.size == 0:
        raise ValueError("Cannot normalise an empty licence-plate crop")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    height, width = gray.shape[:2]
    # The blur must be wider than a numeral; it models lighting, not glyphs.
    sigma = max(7.0, min(width, height) * 0.22)
    illumination = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    normalised = cv2.divide(gray, illumination, scale=150)
    normalised = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(normalised)
    sharpened = cv2.addWeighted(
        normalised,
        1.25,
        cv2.GaussianBlur(normalised, (0, 0), 0.8),
        -0.25,
        0,
    )
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


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


def is_usable_detector_crop(crop: np.ndarray | None) -> bool:
    """Return whether a detect_license crop is large enough for character models."""

    if crop is None or not hasattr(crop, "shape") or getattr(crop, "size", 0) == 0:
        return False
    height, width = crop.shape[:2]
    return height >= 8 and width >= 16


FRAME_CLIP_PIXELS = 2
FULL_PLATE_MIN_WIDTH = 40
FULL_PLATE_MIN_HEIGHT = 16
FULL_PLATE_MIN_ASPECT = 1.2
FULL_PLATE_TARGET_ASPECT = 2.4


def plate_box_clip_sides(box: Iterable[float], image_shape: tuple[int, ...]) -> set[str]:
    """Return which image edges a detector box is sitting on."""

    x1, y1, x2, y2 = (float(value) for value in box)
    height, width = int(image_shape[0]), int(image_shape[1])
    sides: set[str] = set()
    if x1 <= FRAME_CLIP_PIXELS:
        sides.add("left")
    if y1 <= FRAME_CLIP_PIXELS:
        sides.add("top")
    if x2 >= width - FRAME_CLIP_PIXELS:
        sides.add("right")
    if y2 >= height - FRAME_CLIP_PIXELS:
        sides.add("bottom")
    return sides


def full_plate_crop_status(box: Iterable[float], image_shape: tuple[int, ...]) -> str:
    """Classify whether a detector box is a complete plate sheet."""

    x1, y1, x2, y2 = (float(value) for value in box)
    width = x2 - x1
    height = y2 - y1
    if plate_box_clip_sides(box, image_shape) & {"left", "right"}:
        return "clipped_by_frame"
    if width < FULL_PLATE_MIN_WIDTH or height < FULL_PLATE_MIN_HEIGHT:
        return "too_small"
    if width / max(height, 1.0) < FULL_PLATE_MIN_ASPECT:
        return "too_narrow"
    return "full"


def expand_detector_box_to_full_plate(
    xyxy: Iterable[float], image_shape: tuple[int, ...]
) -> list[float]:
    """Grow a tight/narrow detector box toward a typical full-plate rectangle."""

    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = (float(value) for value in xyxy)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    aspect = width / height
    if aspect < FULL_PLATE_MIN_ASPECT or width < FULL_PLATE_MIN_WIDTH:
        target_width = max(FULL_PLATE_MIN_WIDTH, height * FULL_PLATE_TARGET_ASPECT)
        extra = max(0.0, (target_width - width) / 2.0)
        x1 -= extra
        x2 += extra
    pad_y = height * 0.14
    y1 -= pad_y
    y2 += pad_y
    return [
        max(0.0, x1),
        max(0.0, y1),
        min(float(image_width), x2),
        min(float(image_height), y2),
    ]


def complete_detector_crop(
    image: np.ndarray, xyxy: Iterable[float], padding: float
) -> tuple[np.ndarray, list[int], str] | tuple[None, None, str]:
    """Crop a full plate sheet, or explain why the detection is incomplete."""

    expanded = expand_detector_box_to_full_plate(xyxy, image.shape)
    status = full_plate_crop_status(expanded, image.shape)
    if status != "full":
        return None, None, status
    crop, crop_box = clamp_crop(image, expanded, padding)
    if not is_usable_detector_crop(crop):
        return None, None, "too_small"
    status = full_plate_crop_status(crop_box, image.shape)
    if status != "full":
        return None, None, status
    return crop, crop_box, "full"


def glyph_edge_expansion(
    crop_box: list[int],
    readings: dict[str, dict[str, Any]],
    crop_shape: tuple[int, ...],
    image_shape: tuple[int, ...],
) -> list[float] | None:
    """Widen a crop when character boxes are pressed against its left/right edge."""

    height, width = int(crop_shape[0]), int(crop_shape[1])
    if width < 8 or height < 8:
        return None
    if any(isinstance(item, dict) and item.get("complete") for item in readings.values()):
        return None
    margin_x = 2.5
    margin_y = 2.0
    left = right = top = bottom = False
    for reading in readings.values():
        if not isinstance(reading, dict):
            continue
        for token in reading.get("tokens") or []:
            box = token.get("box") if isinstance(token, dict) else None
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            tx1, ty1, tx2, ty2 = (float(value) for value in box)
            if tx1 <= margin_x:
                left = True
            if tx2 >= width - margin_x:
                right = True
            if ty1 <= margin_y:
                top = True
            if ty2 >= height - margin_y:
                bottom = True
    if not (left or right or top or bottom):
        return None
    x1, y1, x2, y2 = (float(value) for value in crop_box)
    if left:
        x1 -= width * 0.45
    if right:
        x2 += width * 0.45
    if top:
        y1 -= height * 0.28
    if bottom:
        y2 += height * 0.28
    expanded = [
        max(0.0, x1),
        max(0.0, y1),
        min(float(image_shape[1]), x2),
        min(float(image_shape[0]), y2),
    ]
    hugging = {side for side, flag in (("left", left), ("right", right)) if flag}
    if hugging and hugging <= plate_box_clip_sides(expanded, image_shape):
        return None
    if full_plate_crop_status(expanded, image_shape) != "full":
        return None
    return expanded


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


def select_thai_prefix(
    items: list[CharacterDetection], digits: list[CharacterDetection]
) -> tuple[str, list[CharacterDetection]]:
    """Select Thai series letters on the same baseline as the registration digits.

    Passenger plates print two consonants to the left of four digits.  Truck
    plates have no series letters, only six or seven digits, so a glyph on the
    province line must not be treated as a prefix.
    """

    candidates = [item for item in items if item.text in THAI_CHARACTER_NAMES]
    if digits and candidates:
        digit_centres = [(item.y1 + item.y2) / 2 for item in digits]
        digit_heights = [max(1.0, item.y2 - item.y1) for item in digits]
        baseline = float(np.median(digit_centres))
        tolerance = float(np.median(digit_heights)) * 0.72
        last_digit_x = max((item.x1 + item.x2) / 2 for item in digits)
        candidates = [
            item
            for item in candidates
            if abs((item.y1 + item.y2) / 2 - baseline) <= tolerance
            and (item.x1 + item.x2) / 2 < last_digit_x
        ]
    if not candidates:
        return "", []
    if len(candidates) >= 2:
        pair = list(max(combinations(candidates, 2), key=sequence_quality))
        selected = sorted(pair, key=lambda item: item.x1)
    else:
        selected = candidates
    leading = _thai_series_digit(digits, selected)
    glyphs = ([leading] if leading is not None else []) + selected
    prefix = "".join(
        item.text if item.text.isdigit() else THAI_CHARACTER_NAMES[item.text]
        for item in glyphs
    )
    return prefix, glyphs


def _thai_series_digit(
    digits: list[CharacterDetection],
    letters: list[CharacterDetection],
) -> CharacterDetection | None:
    """Return the single digit printed to the left of a two-letter Thai series."""

    if len(letters) != 2 or not digits:
        return None
    left_edge = min(item.x1 for item in letters)
    leading = [
        item
        for item in digits
        if (item.x1 + item.x2) / 2 < left_edge
    ]
    if len(leading) != 1:
        return None
    return leading[0]


def is_thai_letter_prefix(prefix: str) -> bool:
    """Thai series letters, including a newer plate such as ``4กธ``."""

    body = prefix[1:] if prefix[:1].isdigit() else prefix
    return (
        bool(body)
        and 1 <= len(body) <= 3
        and all("\u0e01" <= character <= "\u0e2e" for character in body)
        and (not prefix[:1].isdigit() or len(body) == 2)
    )


def is_thai_digit_prefix(prefix: str) -> bool:
    return prefix.isdigit() and len(prefix) in {2, 3}


PROVINCE_MIN_CONFIDENCE = 0.45
PROVINCE_MIN_MARGIN = 0.08
PROVINCE_ROI_MIN_CONFIDENCE = 0.12
PROVINCE_ROI_TARGET_WIDTH = 1280
PROVINCE_ROI_MAX_SCALE = 6.0


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
    target_length = 7 if country == "thai" else 4
    digit_items, digit_detection_count = select_digit_sequence(items, target_length)
    digits = "".join(item.text for item in digit_items)
    digit_confidence = float(np.mean([item.confidence for item in digit_items])) if digit_items else 0.0

    province_item, province_status, province_margin = select_province_item(
        items, country, digit_items
    )
    province_names = THAI_PROVINCE_NAMES if country == "thai" else LAO_PROVINCE_NAMES
    province = province_names.get(province_item.text, "") if province_item else ""
    province_code = province_item.text if province_item else ""
    if country == "thai" and not province_code:
        alias_code, alias_name = resolve_thai_province_text(
            topline_character_text(items, digit_items, country)
        )
        if alias_code:
            province_code, province = alias_code, alias_name
            province_status = "topline_alias"

    prefix_items: list[CharacterDetection] = []
    if country == "thai":
        letter_prefix, prefix_items = select_thai_prefix(items, digit_items)
        if letter_prefix and len(digits) >= 4:
            right_digits = digit_items
            if prefix_items:
                series_edge = max(item.x2 for item in prefix_items)
                beside_series = [
                    item
                    for item in digit_items
                    if (item.x1 + item.x2) / 2 > series_edge
                ]
                if len(beside_series) >= 4:
                    right_digits = beside_series
            number = "".join(item.text for item in right_digits[-4:])
            prefix = letter_prefix
        elif len(digits) == 7:
            prefix, number = digits[:3], digits[3:]
        elif len(digits) == 6:
            prefix, number = digits[:2], digits[2:]
        else:
            prefix, number = letter_prefix, digits
    else:
        prefix, prefix_items = select_lao_prefix(items, digit_items)
        number = digits

    # Keep the model's Latin aliases as well as the converted Lao text.  The
    # aliases are stable identifiers for filenames, exports, and audit logs.
    prefix_source = list(prefix_items)
    if country == "thai" and is_thai_digit_prefix(prefix):
        prefix_source = digit_items[: len(prefix)]
    if prefix_source and is_thai_digit_prefix(prefix):
        glyph_items = digit_items[: len(prefix) + len(number)]
    elif prefix_source:
        glyph_items = prefix_source + digit_items[-len(number) :]
    else:
        glyph_items = list(digit_items)
    glyph_confidences = [float(item.confidence) for item in glyph_items]
    registration_confidence = (
        float(np.mean(glyph_confidences)) if glyph_confidences else digit_confidence
    )
    weakest_glyph_confidence = min(glyph_confidences) if glyph_confidences else 0.0

    prefix_code = "_".join(item.text for item in prefix_items) if prefix_items else ""

    raw = serialise_characters(items)
    if country == "thai":
        motorcycle = (
            len(number) == 4
            and number.isdigit()
            and (len(digits) == 4 or digit_detection_count <= 5)
            and bool(province_code)
        )
        complete = (
            (is_thai_letter_prefix(prefix) or is_thai_digit_prefix(prefix) or motorcycle)
            and len(number) == 4
            and number.isdigit()
        )
    else:
        complete = digit_detection_count == target_length and len(digits) == target_length and len(prefix) == 2
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
        "character_confidence": round(
            float(np.mean([item.confidence for item in items])), 4
        ) if items else 0.0,
        "character_count": len(items),
        "digit_detection_count": digit_detection_count,
        "expected_digit_count": (
            4
            if country == "thai" and (is_thai_letter_prefix(prefix) or motorcycle)
            else len(digits) if country == "thai" and len(digits) in {6, 7} else target_length
        ),
        "complete": complete,
        "plate_prefix": prefix,
        "plate_prefix_code": prefix_code,
        "plate_number": number,
        "province": province,
        "province_code": province_code,
        "province_confidence": round(province_item.confidence, 4) if province_item else (0.55 if province_code else 0.0),
        "province_status": province_status,
        "province_margin": province_margin,
        "prefix_confidence": round(float(np.mean([item.confidence for item in prefix_source])), 4) if prefix_source else 0.0,
        "registration_confidence": round(registration_confidence, 4),
        "weakest_glyph_confidence": round(weakest_glyph_confidence, 4),
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


def resolve_thai_province_text(text: str) -> tuple[str, str]:
    """Map motorcycle/top-line Thai province text onto a province code."""

    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return "", ""
    if compact in THAI_PROVINCE_ALIASES:
        code = THAI_PROVINCE_ALIASES[compact]
        return code, THAI_PROVINCE_NAMES.get(code, "")
    for code, name in THAI_PROVINCE_NAMES.items():
        if compact == name or compact == name.replace(" ", ""):
            return code, name
    return "", ""


def topline_character_text(
    items: list[CharacterDetection],
    digits: list[CharacterDetection],
    country: str,
) -> str:
    """Join character glyphs printed above the registration digits."""

    if not digits:
        return ""
    digit_centres = [(item.y1 + item.y2) / 2 for item in digits]
    digit_heights = [max(1.0, item.y2 - item.y1) for item in digits]
    limit = float(np.median(digit_centres)) - max(4.0, float(np.median(digit_heights)) * 0.35)
    glyphs = [
        character_glyph(item.text, country)
        for item in sorted(items, key=lambda value: value.x1)
        if character_kind(item, country) == "character"
        and (item.y1 + item.y2) / 2 <= limit
        and character_glyph(item.text, country)
    ]
    return "".join(glyphs)


def is_thai_motorcycle_layout(reading: dict[str, Any]) -> bool:
    """Thai motorcycle plates: province on top, four digits, often no series letters."""

    number = str(reading.get("plate_number") or "")
    prefix = str(reading.get("plate_prefix") or "")
    province = str(reading.get("province_code") or "") in THAI_PROVINCE_NAMES
    digits = int(reading.get("digit_detection_count") or 0)
    return (
        number.isdigit()
        and len(number) == 4
        and digits <= 5
        and province
        and not is_thai_digit_prefix(prefix)
        and len(prefix) <= 3
    )


def country_script_cues(reading: dict[str, Any], country: str) -> dict[str, Any]:
    """Collect script/province evidence that is stronger than raw digit scores."""

    prefix = str(reading.get("plate_prefix") or "")
    province_code = str(reading.get("province_code") or "")
    tokens = reading.get("tokens") if isinstance(reading.get("tokens"), list) else []
    character_tokens = sum(token.get("kind") == "character" for token in tokens)
    province_tokens = sum(token.get("kind") == "province" for token in tokens)
    thai_province = country == "thai" and province_code in THAI_PROVINCE_NAMES
    lao_province = country == "lao" and province_code in LAO_PROVINCE_NAMES
    return {
        "thai_province": thai_province,
        "lao_province": lao_province,
        "thai_letters": country == "thai" and is_thai_letter_prefix(prefix),
        "lao_letters": country == "lao" and len(prefix) == 2 and all("\u0e80" <= ch <= "\u0eff" for ch in prefix),
        "character_tokens": character_tokens,
        "province_tokens": province_tokens,
        "motorcycle": country == "thai" and is_thai_motorcycle_layout(reading),
    }


def country_reading_score(reading: dict[str, Any], country: str) -> float:
    """Score a country reading using both model confidence and plate shape.

    Thai car/truck plates normally contain six or seven digits. Thai
    motorcycle plates contain four digits plus a จังหวัด line. Lao plates
    contain four digits plus two Lao prefix letters. Province/script evidence
    outweighs a raw digit score from the other country's model.
    """

    prefix = str(reading.get("plate_prefix") or "")
    cues = country_script_cues(reading, country)
    thai_letters = bool(cues["thai_letters"])
    motorcycle = bool(cues["motorcycle"])
    expected = 4 if country == "lao" or thai_letters or motorcycle else 6
    count = int(reading.get("digit_detection_count", 0))
    if country == "thai" and not thai_letters and not motorcycle and count == 7:
        expected = 7
    completeness = min(count, expected) / expected
    score = float(reading.get("digit_confidence", 0.0)) * 0.25 + completeness * 0.35
    if (thai_letters or motorcycle) and count >= 4:
        score += 0.35
    elif country == "thai" and count in {6, 7}:
        score += 0.35
    elif count == expected:
        score += 0.35
    else:
        score -= min(0.55, abs(count - expected) * 0.15)
    score += float(reading.get("province_confidence", 0.0)) * 0.15
    if cues["thai_province"]:
        score += 0.35
    if cues["lao_province"]:
        score += 0.35
    if country == "lao":
        if cues["lao_letters"]:
            score += 0.25
        else:
            score -= 0.22
        if cues["thai_province"]:
            score -= 0.30
    elif thai_letters:
        score += 0.25
        score += min(0.08, int(cues["character_tokens"]) * 0.025)
    else:
        score += min(0.08, int(cues["character_tokens"]) * 0.025)
        if motorcycle:
            score += 0.20
    return round(score, 4)


def empty_country_reading(country: str) -> dict[str, Any]:
    """Structured empty reading used when Thai/Lao models must not run."""

    result = analyse_country_characters([], country)
    result["score"] = country_reading_score(result, country)
    result["inference_variant"] = "skipped-no-detector-crop"
    result["province_candidates"] = []
    return result


def has_complete_country_layout(reading: dict[str, Any], country: str) -> bool:
    """Return whether a reading has the registration layout for ``country``.

    This is deliberately stricter than ``complete``.  It is used to stop a
    character model for the other country from taking ownership of a plate
    merely because one glare-affected digit received a higher detector score.
    """

    prefix = str(reading.get("plate_prefix") or "")
    number = str(reading.get("plate_number") or "")
    if country == "lao":
        return (
            len(prefix) == 2
            and all("\u0e80" <= character <= "\u0eff" for character in prefix)
            and len(number) == 4
            and number.isdigit()
        )
    if is_thai_motorcycle_layout(reading):
        return True
    return (
        (is_thai_letter_prefix(prefix) or is_thai_digit_prefix(prefix))
        and number.isdigit()
        and len(number) == 4
    )


def apply_country_layout_priority(
    country_result: dict[str, Any], readings: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Keep a complete Lao layout authoritative over Thai digit lookalikes.

    The Thai model is still run for every crop and is useful evidence for
    digit shape.  It must not, however, turn a valid Lao registration into a
    Thai result just because a reflected ``1`` looks like ``6`` to that model.
    """

    result = dict(country_result)
    lao = readings.get("lao", {})
    thai = readings.get("thai", {})
    thai_cues = country_script_cues(thai, "thai")
    lao_cues = country_script_cues(lao, "lao")
    thai_province = bool(thai_cues["thai_province"])
    thai_prefix = str(thai.get("plate_prefix") or "")
    thai_truck = is_thai_digit_prefix(thai_prefix)
    thai_identity = (
        bool(thai.get("complete"))
        and
        has_complete_country_layout(thai, "thai")
        and (thai_province or thai_truck or thai_cues["thai_letters"] or thai_cues["motorcycle"])
    )
    lao_identity = (
        bool(lao.get("complete"))
        and has_complete_country_layout(lao, "lao")
        and (lao_cues["lao_letters"] or lao_cues["lao_province"])
    )
    # A Lao plate can make the Thai model emit lookalike letters. That only
    # overrides a real Thai series when the Lao letters are more confident
    # and a Lao province is present. A numbered series such as 4กธ is Thai.
    numbered_series = (
        len(thai_prefix) == 3
        and thai_prefix[:1].isdigit()
        and is_thai_letter_prefix(thai_prefix)
    )
    thai_prefix_conf = float(thai.get("prefix_confidence") or 0.0)
    lao_prefix_conf = float(lao.get("prefix_confidence") or 0.0)
    thai_province_conf = float(thai.get("province_confidence") or 0.0)
    lao_province_conf = float(lao.get("province_confidence") or 0.0)
    thai_score = float(thai.get("score", 0.0))
    lao_score = float(lao.get("score", 0.0))
    # Lao and Thai share several visually similar glyphs (for example ບ/บ).
    # A Thai character model can therefore be very confident about a
    # lookalike even when the Lao model has read both the Lao prefix and a
    # real Lao province from the same crop.  Treat that two-part Lao evidence
    # as provenance, rather than letting a small prefix-confidence difference
    # route the crop through the Thai pipeline.  A similarly strong, known
    # Thai province remains authoritative for genuine Thai registrations.
    strong_lao_provenance = (
        lao_identity
        and lao_cues["lao_letters"]
        and lao_cues["lao_province"]
        and lao_prefix_conf >= 0.62
        and lao_province_conf >= 0.55
        and lao_score >= thai_score - 0.08
    )
    strong_thai_provenance = (
        thai_identity
        and thai_province
        and (
            # A numbered Thai series plus a known Thai province has a very
            # distinctive layout; do not let an unrelated Lao lookalike
            # displace it merely because its province score is close.
            (numbered_series and thai_province_conf >= 0.72)
            or (
                thai_province_conf >= 0.55
                and thai_province_conf >= lao_province_conf + 0.12
                and thai_prefix_conf >= lao_prefix_conf - 0.05
            )
        )
    )
    if thai_identity and strong_lao_provenance and not strong_thai_provenance:
        margin = lao_score - thai_score
        result.update(
            {
                "country": "lao",
                "raw_country": "lao",
                "margin": round(margin, 4),
                "confidence": round(max(0.0, min(1.0, 0.50 + max(margin, 0.12) * 0.50)), 4),
                "selection_reason": "lao_script_priority",
            }
        )
        return result
    if (
        thai_identity
        and lao_identity
        and not thai_truck
        and not numbered_series
        and is_thai_letter_prefix(thai_prefix)
        and lao_cues["lao_letters"]
        and lao_cues["lao_province"]
        and lao_prefix_conf > thai_prefix_conf
        and not (
            thai_province
            and thai_province_conf >= lao_province_conf + 0.12
            and thai_prefix_conf >= lao_prefix_conf + 0.05
        )
    ):
        margin = lao_score - thai_score
        result.update(
            {
                "country": "lao",
                "raw_country": "lao",
                "margin": round(margin, 4),
                "confidence": round(max(0.0, min(1.0, 0.50 + max(margin, 0.12) * 0.50)), 4),
                "selection_reason": "lao_script_priority",
            }
        )
        return result
    if thai_identity and (
        thai_province
        or thai_cues["thai_letters"]
        or numbered_series
        or thai_truck
        or not lao_identity
    ):
        margin = thai_score - lao_score
        result.update(
            {
                "country": "thai",
                "raw_country": "thai",
                "margin": round(margin, 4),
                "confidence": round(max(0.0, min(1.0, 0.50 + max(margin, 0.12) * 0.50)), 4),
                "selection_reason": "thai_layout_priority",
            }
        )
        return result
    if lao_identity and not thai_identity:
        margin = lao_score - thai_score
        result.update(
            {
                "country": "lao",
                "raw_country": "lao",
                "margin": round(margin, 4),
                "confidence": round(max(0.0, min(1.0, 0.50 + max(margin, 0.12) * 0.50)), 4),
                "selection_reason": "lao_layout_priority",
            }
        )
        return result
    if thai_province and not lao_identity and not lao_cues["lao_province"]:
        margin = thai_score - lao_score
        result.update(
            {
                "country": "thai",
                "raw_country": "thai",
                "margin": round(margin, 4),
                "confidence": round(max(0.0, min(1.0, 0.50 + max(margin, 0.12) * 0.50)), 4),
                "selection_reason": "thai_province_priority",
            }
        )
        return result
    if lao_cues["lao_province"] and not thai_identity and not thai_province:
        margin = lao_score - thai_score
        result.update(
            {
                "country": "lao",
                "raw_country": "lao",
                "margin": round(margin, 4),
                "confidence": round(max(0.0, min(1.0, 0.50 + max(margin, 0.12) * 0.50)), 4),
                "selection_reason": "lao_province_priority",
            }
        )
        return result
    result.setdefault("selection_reason", "country_model_score")
    return result


def cross_model_digit_evidence(
    readings: dict[str, dict[str, Any]], primary_country: str
) -> dict[str, Any]:
    """Describe whether the other country model confirms the four digits.

    This helper is intentionally evidence-only: it never chooses a number.
    The primary country/layout model remains the source of the plate text;
    disagreement is retained for temporal voting and audit JSON.
    """

    assistant_country = "thai" if primary_country == "lao" else "lao"
    primary = readings.get(primary_country, {})
    assistant = readings.get(assistant_country, {})
    primary_number = str(primary.get("plate_number") or "")
    assistant_number = str(assistant.get("plate_number") or "")
    valid_primary = len(primary_number) == 4 and primary_number.isdigit()
    valid_assistant = len(assistant_number) == 4 and assistant_number.isdigit()
    evidence: dict[str, Any] = {
        "primary_country": primary_country,
        "assistant_country": assistant_country,
        "primary_number": primary_number,
        "assistant_number": assistant_number,
        "primary_digit_confidence": round(float(primary.get("digit_confidence", 0.0)), 4),
        "assistant_digit_confidence": round(float(assistant.get("digit_confidence", 0.0)), 4),
        "assistant_is_confirmation_only": True,
        "resolution": "primary-retained",
    }
    if not valid_primary:
        evidence["status"] = "primary_incomplete"
        evidence["different_positions"] = []
    elif not valid_assistant:
        evidence["status"] = "assistant_unavailable"
        evidence["different_positions"] = []
    elif primary_number == assistant_number:
        evidence["status"] = "agree"
        evidence["different_positions"] = []
    else:
        evidence["status"] = "conflict"
        evidence["different_positions"] = [
            index
            for index, (primary_digit, assistant_digit) in enumerate(zip(primary_number, assistant_number))
            if primary_digit != assistant_digit
        ]
    return evidence


def published_plate_confidence(
    reading: dict[str, Any],
    digit_evidence: dict[str, Any] | None = None,
) -> float:
    """Confidence of the glyphs that actually form the published plate.

    Detector score, country margin, and a complete-layout bonus are not added
    on top. A weak required glyph pulls the result down, and a cross-model
    digit conflict is a penalty.
    """

    value = float(
        reading.get("registration_confidence", reading.get("digit_confidence", 0.0)) or 0.0
    )
    weakest = reading.get("weakest_glyph_confidence")
    if weakest is not None and float(weakest) < 0.50:
        value = min(value, (value + float(weakest)) / 2.0)
    if reading.get("complete") is False:
        value = min(value, 0.74)
    status = str((digit_evidence or {}).get("status") or "")
    if status == "conflict":
        value = max(0.0, value - 0.08)
    elif status == "agree":
        value = min(value + 0.02, 0.99)
    return round(max(0.0, min(0.99, value)), 4)


def contested_script_confidence(
    confidence: float,
    readings: dict[str, dict[str, Any]],
    country: str,
) -> float:
    """Stop a Thai lookalike of a Lao plate from publishing a high confidence.

    Same digits with a different script are not agreement. A Thai letter
    prefix that competes with a complete Lao prefix and province stays below
    the level the live panel treats as a confident read.
    """

    thai = readings.get("thai") or {}
    lao = readings.get("lao") or {}
    if not (
        has_complete_country_layout(thai, "thai")
        and has_complete_country_layout(lao, "lao")
    ):
        return round(max(0.0, min(0.99, float(confidence))), 4)
    thai_prefix = str(thai.get("plate_prefix") or "")
    if is_thai_digit_prefix(thai_prefix) or country != "thai":
        return round(max(0.0, min(0.99, float(confidence))), 4)
    if not (
        is_thai_letter_prefix(thai_prefix)
        and country_script_cues(lao, "lao").get("lao_province")
    ):
        return round(max(0.0, min(0.99, float(confidence))), 4)
    return round(min(max(0.0, float(confidence)), 0.62), 4)


def build_plate_quality(
    *,
    country: str,
    primary_country: str,
    readings: dict[str, dict[str, Any]],
    plate_prefix: str,
    plate_number: str,
    ocr: dict[str, Any],
    validation: dict[str, Any],
    detection_confidence: float,
    country_confidence: float,
    digit_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Create the production QC contract after model/OCR fusion.

    The secondary country model is corroborating evidence only.  QC therefore
    judges completeness from the selected primary model, not from whether the
    other model can read a different country's plate layout.
    """

    primary = readings.get(primary_country, {})
    secondary_country = "thai" if primary_country == "lao" else "lao"
    secondary = readings.get(secondary_country, {})
    expected_digits = int(
        primary.get(
            "expected_digit_count",
            6 if primary_country == "thai" else 4,
        )
    )
    if expected_digits <= 0:
        expected_digits = 6 if primary_country == "thai" else 4
    detected_digits = int(primary.get("digit_detection_count", 0))
    primary_prefix = str(plate_prefix or primary.get("plate_prefix") or "")
    if primary_country == "lao":
        prefix_complete = (
            len(primary_prefix) == 2
            and all("\u0e80" <= character <= "\u0eff" for character in primary_prefix)
        )
        prefix_missing = max(0, 2 - len(primary_prefix))
    else:
        thai_motorcycle = is_thai_motorcycle_layout(
            {
                **primary,
                "plate_prefix": primary_prefix,
                "plate_number": plate_number,
            }
        )
        prefix_complete = (
            is_thai_letter_prefix(primary_prefix)
            or is_thai_digit_prefix(primary_prefix)
            or thai_motorcycle
        )
        # Thai's leading registration digits are already included in its six
        # digit count; do not count them as missing twice.
        prefix_missing = 0
    digit_complete = len(str(plate_number or "")) == 4 and str(plate_number or "").isdigit()
    province_complete = bool(primary.get("province_code") or primary.get("province"))
    unknown = any(
        str(token.get("label", "")).strip().lower() in {"unknown", "unk", "?"}
        for token in primary.get("tokens", [])
        if isinstance(token, dict)
    )
    missing = max(0, expected_digits - detected_digits) + prefix_missing
    reasons: list[str] = []
    if not digit_complete:
        reasons.append("missing_digit")
    if not prefix_complete:
        reasons.append("prefix_missing")
    if not province_complete:
        reasons.append("province_missing")
    if unknown:
        reasons.append("unknown_character")

    cross_status = str(digit_evidence.get("status") or "")
    if cross_status == "conflict":
        reasons.append("cross_model_digit_conflict")

    layout_complete = digit_complete and prefix_complete
    if not layout_complete and (not prefix_complete or detected_digits <= expected_digits // 2):
        character_status = "REJECT"
    elif reasons:
        character_status = "REVIEW"
    else:
        character_status = "PASS"

    ocr_confidence = float(ocr.get("confidence", 0.0))
    overall = published_plate_confidence(primary, digit_evidence)
    weakest = primary.get("weakest_glyph_confidence")
    if weakest is not None and float(weakest) < 0.70:
        overall = min(overall, 0.89)
    if not validation.get("valid") or not layout_complete:
        overall = min(overall, 0.79)
    if not province_complete:
        overall = min(overall, 0.89)
    overall = round(max(0.0, min(0.99, overall)), 4)
    confidence_level = "HIGH" if overall >= 0.95 else "MEDIUM" if overall >= 0.80 else "LOW"
    qc_reasons = list(reasons)
    if character_status == "PASS":
        if overall < 0.80:
            qc_status = "REJECT"
            qc_reasons.append("overall_confidence_below_review_threshold")
        elif overall < 0.95:
            qc_status = "REVIEW"
            qc_reasons.append("overall_confidence_below_pass_threshold")
        else:
            qc_status = "PASS"
    else:
        qc_status = character_status
    character_validation = {
        "country": country,
        "primary_model": primary_country,
        "secondary_model": secondary_country,
        "primary_complete": has_complete_country_layout(primary, primary_country),
        "secondary_complete": has_complete_country_layout(secondary, secondary_country),
        "primary_expected_digit_count": expected_digits,
        "primary_detected_digit_count": detected_digits,
        "secondary_expected_digit_count": int(
            secondary.get(
                "expected_digit_count",
                6 if secondary_country == "thai" else 4,
            )
        ),
        "secondary_detected_digit_count": int(secondary.get("digit_detection_count", 0)),
        "digit_complete": digit_complete,
        "prefix_complete": prefix_complete,
        "province_complete": province_complete,
        "duplicate": False,
        "unknown": unknown,
        "missing": missing,
        "status": character_status,
        "reason": reasons,
    }
    return {
        "overall_confidence": round(overall, 4),
        "confidence_level": confidence_level,
        "qc": {"status": qc_status, "reason": qc_reasons, "need_review": qc_status != "PASS"},
        "detection": {"confidence": round(float(detection_confidence), 4), "model": "YOLO11"},
        "country_confidence": round(max(0.0, min(1.0, float(country_confidence or 0.0))), 4),
        "character_validation": character_validation,
        "cfd": {
            "ocr": round(ocr_confidence, 4),
            "detect": round(float(detection_confidence), 4),
            "fusion": round(overall, 4),
        },
        "dataset": {
            "split": "unassigned",
            "country": "lao" if primary_country == "lao" else "thai",
            "qc": qc_status,
            "exported": False,
            # The legacy dataset tree contains PASS/REJECT only. REVIEW is
            # intentionally exported to the manual-review (REJECT) folder.
            "export_status": "PASS" if qc_status == "PASS" else "REJECT",
        },
    }


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


def country_hint_from_text(text: str) -> str:
    """Guess Thai vs Lao from OCR text, including จังหวัด / ແຂວງ names."""

    raw = str(text or "")
    if not raw.strip():
        return ""
    if "THAILAND" in raw.upper() or "กรุงเทพ" in raw:
        return "thai"
    thai_code, _ = extract_ocr_province(raw, "thai")
    lao_code, _ = extract_ocr_province(raw, "lao")
    thai_chars = sum("\u0e00" <= character <= "\u0e7f" for character in raw)
    lao_chars = sum("\u0e80" <= character <= "\u0eff" for character in raw)
    if thai_code and not lao_code:
        return "thai"
    if lao_code and not thai_code:
        return "lao"
    if thai_chars >= 2 and thai_chars > lao_chars:
        return "thai"
    if lao_chars >= 2 and lao_chars > thai_chars:
        return "lao"
    return ""


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


TESSERACT_LANGUAGES = {
    "lao": {
        "code": "lao",
        "model": "lao.traineddata",
        "method": "tesseract-lao",
        "start": 0x0E80,
        "end": 0x0EFF,
        "length_penalty_after": 10,
    },
    "thai": {
        "code": "tha",
        "model": "tha.traineddata",
        "method": "tesseract-thai",
        "start": 0x0E00,
        "end": 0x0E7F,
        "length_penalty_after": 24,
    },
}


def clean_ocr_line(line: str) -> str:
    """Keep letters/digits from an OCR line and discard punctuation noise."""

    return "".join(
        character for character in line
        if character.isdigit()
        or ("A" <= character <= "Z")
        or ("a" <= character <= "z")
        or (
            "\u0e00" <= character <= "\u0e7f"
            and unicodedata.category(character)[0] in "LM"
        )
        or (
            "\u0e80" <= character <= "\u0eff"
            and unicodedata.category(character) == "Lo"
        )
    ).strip()


def _tesseract_executable() -> Path | None:
    candidates = [
        ROOT / "tools" / "tesseract" / "tesseract.exe",
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]
    detected = shutil.which("tesseract")
    if detected:
        candidates.insert(0, Path(detected))
    return next((path for path in candidates if path.is_file()), None)


def _tesseract_model_dir(model_file: str) -> Path | None:
    prefix = os.environ.get("TESSDATA_PREFIX", "").strip()
    candidates = [
        ROOT / "tools" / "tesseract" / "tessdata",
        Path(prefix) if prefix else None,
        Path(r"C:\Program Files\Tesseract-OCR\tessdata"),
        Path("/usr/share/tesseract-ocr/5/tessdata"),
        Path("/usr/share/tesseract-ocr/4.00/tessdata"),
        Path("/usr/share/tessdata"),
    ]
    for folder in candidates:
        if folder and (folder / model_file).is_file():
            return folder
    return None


def tesseract_language_available(expected_language: str) -> bool:
    spec = TESSERACT_LANGUAGES.get(expected_language)
    return bool(spec and _tesseract_executable() and _tesseract_model_dir(str(spec["model"])))


def split_plate_text(text: str, country: str) -> tuple[str, str]:
    """Split a recognised plate into its script prefix and numeric part."""

    if country == "lao":
        remainder = str(text)
        for name in sorted((value for value in LAO_PROVINCE_NAMES.values() if value), key=len, reverse=True):
            remainder = remainder.replace(name, " ")
        for alias in sorted(LAO_PROVINCE_ALIASES, key=len, reverse=True):
            remainder = remainder.replace(alias, " ")
        compact = re.sub(r"\s+", "", remainder)
        match = re.search(r"([\u0e80-\u0eff]{1,3})(\d{4})", compact)
        if match:
            return match.group(1), match.group(2)
        match = re.match(r"^(?P<prefix>[\u0e80-\u0eff]+?)(?P<number>\d{4})$", compact)
        if match:
            return match.group("prefix"), match.group("number")
    if country == "thai":
        remainder = str(text)
        for name in sorted((value for value in THAI_PROVINCE_NAMES.values() if value), key=len, reverse=True):
            remainder = remainder.replace(name, " ")
        for alias in sorted(THAI_PROVINCE_ALIASES, key=len, reverse=True):
            remainder = remainder.replace(alias, " ")
        compact = re.sub(r"[\s\-]+", "", remainder)
        letter_match = re.search(r"([\u0e01-\u0e2e]{1,3})(\d{3,6})", compact)
        if letter_match:
            return letter_match.group(1), letter_match.group(2)
        digits = "".join(re.findall(r"\d", remainder))
        if len(digits) == 7:
            return digits[:3], digits[3:]
        if len(digits) == 6:
            return digits[:2], digits[2:]
        return "", digits
    # Province/model labels may contain other digits (for example ``A23``).
    # Prefer the standalone six-digit registration token when present.
    six_digit_tokens = re.findall(r"(?<!\d)\d{6}(?!\d)", text)
    digits = six_digit_tokens[-1] if six_digit_tokens else "".join(re.findall(r"\d", text))
    return "", digits


def extract_ocr_province(text: str, country: str) -> tuple[str, str]:
    """Extract a known province code/name from a full OCR transcription.

    Trained Thai/Lao recognisers emit the human-readable province or ແຂວງ
    name first (for example ``ระยอง ผค-5939`` or ``ກຳແພງນະຄອນ ບກ0507``).
    YOLO still uses ASCII codes such as ``RYG`` / ``VTE2``.  Keep province
    parsing separate from prefix/number parsing so a code is not treated as
    a registration prefix.
    """

    province_names = THAI_PROVINCE_NAMES if country == "thai" else LAO_PROVINCE_NAMES
    aliases = THAI_PROVINCE_ALIASES if country == "thai" else LAO_PROVINCE_ALIASES
    compact = re.sub(r"[^A-Za-z0-9\u0e00-\u0eff]", " ", str(text).upper())
    tokens = compact.split()
    for token in tokens:
        if token in province_names:
            return token, province_names[token]
    for code in sorted(province_names, key=len, reverse=True):
        if re.search(rf"(?<![A-Z0-9]){re.escape(code)}(?![A-Z0-9])", compact):
            return code, province_names[code]
    normalised = re.sub(r"\s+", "", str(text))
    for code, name in sorted(province_names.items(), key=lambda item: len(item[1]), reverse=True):
        if name and name in normalised:
            return code, name
    for alias, code in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if alias and alias in normalised and code in province_names:
            return code, province_names[code]
    first = str(text).split()[0] if str(text).split() else ""
    if first in province_names:
        return first, province_names[first]
    if first in aliases and aliases[first] in province_names:
        return aliases[first], province_names[aliases[first]]
    return "", ""


def _glyph_kind(char: str) -> str:
    if char.isdigit():
        return "digit"
    if "\u0e01" <= char <= "\u0e2e":
        return "thai"
    if "\u0e80" <= char <= "\u0eff":
        return "lao"
    return "other"


def _yolo_registration_tokens(reading: dict[str, Any], country: str) -> list[dict[str, Any]]:
    tokens = reading.get("tokens") if isinstance(reading.get("tokens"), list) else []
    glyphs: list[dict[str, Any]] = []
    for token in tokens:
        if not isinstance(token, dict) or token.get("kind") not in {"digit", "character"}:
            continue
        label = str(token.get("label") or "")
        char = label if token.get("kind") == "digit" else character_glyph(label, country)
        if not char:
            continue
        box = token.get("box") if isinstance(token.get("box"), list) else [0, 0, 0, 0]
        glyphs.append(
            {
                "char": char,
                "kind": "digit" if char.isdigit() else "character",
                "confidence": float(token.get("confidence") or 0.0),
                "x": float(box[0]) if box else 0.0,
                "x2": float(box[2]) if len(box) >= 3 else (float(box[0]) if box else 0.0),
            }
        )
    glyphs.sort(key=lambda item: item["x"])
    return glyphs


def _confidences_matching(text: str, tokens: list[dict[str, Any]], fallback: float) -> list[float]:
    remaining = list(tokens)
    confs: list[float] = []
    for char in text:
        index = next((i for i, item in enumerate(remaining) if item["char"] == char), None)
        if index is None:
            index = next(
                (i for i, item in enumerate(remaining) if item["char"].isdigit() == char.isdigit()),
                None,
            )
        if index is None:
            confs.append(fallback)
            continue
        confs.append(float(remaining.pop(index)["confidence"]))
    return confs


def _horizontal_overlap(first: dict[str, Any], second: dict[str, Any]) -> float:
    """Return how much of the narrower glyph sits inside the other glyph."""

    left = max(float(first.get("x") or 0.0), float(second.get("x") or 0.0))
    right = min(float(first.get("x2") or 0.0), float(second.get("x2") or 0.0))
    overlap = max(0.0, right - left)
    narrower = min(
        max(1.0, float(first.get("x2") or 0.0) - float(first.get("x") or 0.0)),
        max(1.0, float(second.get("x2") or 0.0) - float(second.get("x") or 0.0)),
    )
    return overlap / narrower


def _number_digit_slots(
    number: str,
    tokens: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the digit boxes that form the registration number, left to right."""

    digits = [item for item in tokens if item.get("kind") == "digit"]
    if number and len(digits) >= len(number):
        suffix = digits[-len(number) :]
        if "".join(str(item.get("char") or "") for item in suffix) == number:
            return suffix
    return []


def _number_token_confidences(
    number: str,
    tokens: list[dict[str, Any]],
    fallback: float,
) -> list[float]:
    """Match the registration-number suffix to ordered YOLO digit boxes."""

    digits = [item for item in tokens if item.get("kind") == "digit"]
    if number and len(digits) >= len(number):
        suffix = digits[-len(number) :]
        if "".join(str(item.get("char") or "") for item in suffix) == number:
            return [float(item.get("confidence") or fallback) for item in suffix]
    return _confidences_matching(number, digits, fallback)


def _ocr_field_confidences(
    ocr: dict[str, Any],
    field_text: str,
    fallback: float,
) -> list[float]:
    """Extract CTC confidences for a prefix/number from the full OCR string."""

    if not field_text:
        return []
    raw_text = str(ocr.get("text") or "")
    raw_confidences = ocr.get("character_confidences")
    if not isinstance(raw_confidences, list) or len(raw_confidences) != len(raw_text):
        return [fallback] * len(field_text)
    compact: list[tuple[str, float]] = [
        (char, max(0.0, min(1.0, float(confidence))))
        for char, confidence in zip(raw_text, raw_confidences)
        if not char.isspace() and char != "-"
    ]
    compact_text = "".join(char for char, _ in compact)
    start = (
        compact_text.rfind(field_text)
        if field_text.isdigit()
        else compact_text.find(field_text)
    )
    if start < 0:
        return [fallback] * len(field_text)
    return [
        confidence
        for _, confidence in compact[start : start + len(field_text)]
    ]


def fuse_cross_model_digits(
    readings: dict[str, dict[str, Any]],
    primary_country: str,
) -> dict[str, Any]:
    """Use the other YOLO model only as per-digit corroborating evidence."""

    if primary_country not in {"thai", "lao"}:
        return {}
    assistant_country = "lao" if primary_country == "thai" else "thai"
    primary = readings.get(primary_country)
    assistant = readings.get(assistant_country)
    if not isinstance(primary, dict) or not isinstance(assistant, dict):
        return {}
    primary_number = str(primary.get("plate_number") or "")
    assistant_number = str(assistant.get("plate_number") or "")
    if len(assistant_number) != 4 or not assistant_number.isdigit():
        return {}

    primary_tokens = [
        item
        for item in _yolo_registration_tokens(primary, primary_country)
        if item["kind"] == "digit"
    ]
    assistant_tokens = [
        item
        for item in _yolo_registration_tokens(assistant, assistant_country)
        if item["kind"] == "digit"
    ]
    primary_fallback = float(primary.get("digit_confidence") or 0.0)
    assistant_fallback = float(assistant.get("digit_confidence") or 0.0)
    assistant_confs = _number_token_confidences(
        assistant_number, assistant_tokens, assistant_fallback
    )

    details: list[dict[str, Any]] = []
    if len(primary_number) == 4 and primary_number.isdigit():
        primary_slots = _number_digit_slots(primary_number, primary_tokens)
        assistant_slots = _number_digit_slots(assistant_number, assistant_tokens)
        if len(primary_slots) != 4:
            primary_confs = _number_token_confidences(
                primary_number, primary_tokens, primary_fallback
            )
            primary_slots = [
                {"char": char, "confidence": primary_confs[index], "x": 0.0, "x2": 0.0}
                for index, char in enumerate(primary_number)
            ]
        used_assistant: set[int] = set()
        fused: list[str] = []
        for index, primary_slot in enumerate(primary_slots):
            primary_char = str(primary_slot.get("char") or primary_number[index])
            primary_conf = float(primary_slot.get("confidence") or primary_fallback)
            match_index = None
            match_overlap = 0.0
            for assistant_index, assistant_slot in enumerate(assistant_slots):
                if assistant_index in used_assistant:
                    continue
                overlap = _horizontal_overlap(primary_slot, assistant_slot)
                if overlap > match_overlap:
                    match_index, match_overlap = assistant_index, overlap
            assistant_char = ""
            assistant_conf = 0.0
            if match_index is not None and match_overlap >= 0.50:
                used_assistant.add(match_index)
                assistant_char = str(assistant_slots[match_index].get("char") or "")
                assistant_conf = float(
                    assistant_slots[match_index].get("confidence") or 0.0
                )
            use_assistant = (
                bool(assistant_char)
                and primary_char != assistant_char
                and assistant_conf > primary_conf
            )
            char = assistant_char if use_assistant else primary_char
            confidence = assistant_conf if use_assistant else primary_conf
            source = assistant_country if use_assistant else primary_country
            if assistant_char and primary_char == assistant_char:
                confidence = max(primary_conf, assistant_conf)
                source = "agree"
            fused.append(char)
            details.append(
                {
                    "position": index,
                    "char": char,
                    "source": source,
                    f"{primary_country}_confidence": round(primary_conf, 4),
                    f"{assistant_country}_confidence": round(assistant_conf, 4),
                    "confidence": round(confidence, 4),
                }
            )
        fused_number = "".join(fused)
    elif assistant_fallback >= max(0.45, primary_fallback - 0.05):
        fused_number = assistant_number
        details = [
            {
                "position": index,
                "char": char,
                "source": assistant_country,
                f"{primary_country}_confidence": round(primary_fallback, 4),
                f"{assistant_country}_confidence": round(
                    assistant_confs[index], 4
                ),
                "confidence": round(assistant_confs[index], 4),
            }
            for index, char in enumerate(assistant_number)
        ]
    else:
        return {}

    primary["plate_number"] = fused_number
    confidences = [float(item["confidence"]) for item in details]
    if confidences:
        primary["digit_confidence"] = round(sum(confidences) / len(confidences), 4)
        primary["confidence"] = primary["digit_confidence"]
    prefix = str(primary.get("plate_prefix") or "")
    primary["text"] = (
        f"{prefix}-{fused_number}"
        if primary_country == "thai" and prefix
        else " ".join(part for part in (prefix, fused_number) if part)
    )
    primary["lines"] = [primary["text"]] if primary["text"] else []
    primary["complete"] = has_complete_country_layout(primary, primary_country)
    primary["digit_fusion"] = {
        "primary_model": f"{primary_country}_license_plate",
        "assistant_model": f"{assistant_country}_license_plate",
        "characters": details,
    }
    glyph_values = [float(item["confidence"]) for item in details]
    prefix_value = float(primary.get("prefix_confidence") or 0.0)
    if prefix and prefix_value:
        glyph_values = [prefix_value] * len(prefix) + glyph_values
    if glyph_values:
        primary["registration_confidence"] = round(sum(glyph_values) / len(glyph_values), 4)
        primary["weakest_glyph_confidence"] = round(min(glyph_values), 4)
    primary["score"] = country_reading_score(primary, primary_country)
    return primary["digit_fusion"]


def fuse_characters_by_confidence(
    yolo_text: str,
    yolo_confs: list[float],
    ocr_text: str,
    ocr_conf: float,
    ocr_confs: list[float] | None = None,
    *,
    align: str = "left",
) -> tuple[str, list[dict[str, Any]]]:
    """Keep YOLO glyphs unless OCR is more confident on the same slot."""

    ocr_conf = max(0.0, min(1.0, float(ocr_conf or 0.0)))
    ocr_confs = [
        max(0.0, min(1.0, float(value)))
        for value in (ocr_confs or [])
    ]
    if len(ocr_confs) != len(ocr_text):
        ocr_confs = [ocr_conf] * len(ocr_text)
    if not yolo_text:
        details = [
            {
                "char": char,
                "source": "ocr",
                "yolo_confidence": 0.0,
                "ocr_confidence": ocr_confs[index],
                "confidence": ocr_confs[index],
            }
            for index, char in enumerate(ocr_text)
        ]
        return ocr_text, details
    if not ocr_text:
        details = []
        for index, char in enumerate(yolo_text):
            yolo_conf = float(yolo_confs[index]) if index < len(yolo_confs) else 0.0
            details.append(
                {
                    "char": char,
                    "source": "yolo",
                    "yolo_confidence": yolo_conf,
                    "ocr_confidence": 0.0,
                    "confidence": yolo_conf,
                }
            )
        return yolo_text, details

    if align == "right":
        ocr_slots = ocr_text[-len(yolo_text) :] if len(ocr_text) >= len(yolo_text) else ocr_text.rjust(len(yolo_text))
        ocr_conf_slots = (
            ocr_confs[-len(yolo_text) :]
            if len(ocr_confs) >= len(yolo_text)
            else [0.0] * (len(yolo_text) - len(ocr_confs)) + ocr_confs
        )
    else:
        ocr_slots = ocr_text[: len(yolo_text)] if len(ocr_text) >= len(yolo_text) else ocr_text.ljust(len(yolo_text))
        ocr_conf_slots = (
            ocr_confs[: len(yolo_text)]
            if len(ocr_confs) >= len(yolo_text)
            else ocr_confs + [0.0] * (len(yolo_text) - len(ocr_confs))
        )

    fused: list[str] = []
    details: list[dict[str, Any]] = []
    for index, yolo_char in enumerate(yolo_text):
        yolo_conf = float(yolo_confs[index]) if index < len(yolo_confs) else 0.0
        ocr_char = ocr_slots[index] if index < len(ocr_slots) else " "
        slot_ocr_conf = (
            ocr_conf_slots[index] if index < len(ocr_conf_slots) else ocr_conf
        )
        if ocr_char in {"", " "}:
            fused.append(yolo_char)
            details.append(
                {
                    "char": yolo_char,
                    "source": "yolo",
                    "yolo_confidence": yolo_conf,
                    "ocr_confidence": 0.0,
                    "confidence": yolo_conf,
                }
            )
            continue
        if ocr_char == yolo_char:
            chosen = max(yolo_conf, slot_ocr_conf)
            fused.append(yolo_char)
            details.append(
                {
                    "char": yolo_char,
                    "source": "agree",
                    "yolo_confidence": yolo_conf,
                    "ocr_confidence": slot_ocr_conf,
                    "confidence": chosen,
                }
            )
        elif _glyph_kind(ocr_char) == _glyph_kind(yolo_char) and slot_ocr_conf > yolo_conf:
            fused.append(ocr_char)
            details.append(
                {
                    "char": ocr_char,
                    "source": "ocr",
                    "yolo_confidence": yolo_conf,
                    "ocr_confidence": slot_ocr_conf,
                    "confidence": slot_ocr_conf,
                }
            )
        else:
            fused.append(yolo_char)
            details.append(
                {
                    "char": yolo_char,
                    "source": "yolo",
                    "yolo_confidence": yolo_conf,
                    "ocr_confidence": slot_ocr_conf,
                    "confidence": yolo_conf,
                }
            )
    return "".join(fused), details


def _fusion_source(details: list[dict[str, Any]], prefix_source: str = "") -> str:
    sources = {str(item.get("source") or "") for item in details}
    sources.discard("")
    if prefix_source:
        sources.add(prefix_source)
    if sources <= {"yolo"}:
        return "structured-yolo"
    if sources <= {"ocr"}:
        return "ocr"
    if sources <= {"agree"} or sources <= {"agree", "yolo"}:
        return "yolo+ocr-agree"
    if "ocr" in sources and "agree" in sources:
        return "ocr+yolo-agree"
    if "ocr" in sources:
        return "yolo+ocr"
    return "structured-yolo"


def resolve_plate_fields(
    country: str, reading: dict[str, Any], ocr: dict[str, Any]
) -> tuple[str, str, str]:
    """Fuse YOLO characters with OCR as a per-glyph supplement.

    ``thai_license_plate`` / ``lao_license_plate`` remain the primary reading.
    OCR may fill a missing field or replace one glyph when its confidence is
    higher than that glyph's YOLO score.
    """

    model_prefix = str(reading.get("plate_prefix") or "")
    model_number = str(reading.get("plate_number") or "")
    original_ocr_confidence = float(ocr.get("confidence") or 0.0)
    independent_ocr = str(ocr.get("method") or "") != "structured-yolo"
    ocr_prefix, ocr_number = (
        split_plate_text(str(ocr.get("text") or ""), country)
        if independent_ocr
        else ("", "")
    )
    ocr_conf = original_ocr_confidence if independent_ocr else 0.0
    ocr_prefix_confs = _ocr_field_confidences(ocr, ocr_prefix, ocr_conf)
    ocr_number_confs = _ocr_field_confidences(ocr, ocr_number, ocr_conf)
    tokens = _yolo_registration_tokens(reading, country)
    prefix_fallback = float(reading.get("prefix_confidence") or reading.get("digit_confidence") or 0.0)
    number_fallback = float(reading.get("digit_confidence") or 0.0)
    prefix_confs = _confidences_matching(model_prefix, tokens, prefix_fallback)
    number_confs = _number_token_confidences(
        model_number, [item for item in tokens if item["kind"] == "digit"], number_fallback
    )

    prefix_source = ""
    prefix_details: list[dict[str, Any]] = []
    thai_ocr_prefix = bool(ocr_prefix) and all("\u0e01" <= char <= "\u0e2e" for char in ocr_prefix)
    lao_ocr_prefix = bool(ocr_prefix) and all("\u0e80" <= char <= "\u0eff" for char in ocr_prefix)
    yolo_letter_prefix = is_thai_letter_prefix(model_prefix) or (
        country == "lao" and len(model_prefix) == 2 and all("\u0e80" <= char <= "\u0eff" for char in model_prefix)
    )
    ocr_letter_prefix = thai_ocr_prefix or lao_ocr_prefix

    if ocr_letter_prefix and not yolo_letter_prefix:
        model_prefix_conf = (
            sum(prefix_confs) / len(prefix_confs)
            if prefix_confs
            else prefix_fallback
        )
        ocr_prefix_conf = (
            sum(ocr_prefix_confs) / len(ocr_prefix_confs)
            if ocr_prefix_confs
            else ocr_conf
        )
        if not model_prefix or ocr_prefix_conf > model_prefix_conf:
            prefix = ocr_prefix
            prefix_source = "ocr"
            prefix_details = [
                {
                    "char": char,
                    "source": "ocr",
                    "yolo_confidence": prefix_fallback,
                    "ocr_confidence": (
                        ocr_prefix_confs[index]
                        if index < len(ocr_prefix_confs)
                        else ocr_conf
                    ),
                    "confidence": (
                        ocr_prefix_confs[index]
                        if index < len(ocr_prefix_confs)
                        else ocr_conf
                    ),
                }
                for index, char in enumerate(prefix)
            ]
        else:
            prefix, prefix_details = fuse_characters_by_confidence(
                model_prefix,
                prefix_confs,
                ocr_prefix,
                ocr_conf,
                ocr_prefix_confs,
                align="left",
            )
    else:
        prefix, prefix_details = fuse_characters_by_confidence(
            model_prefix,
            prefix_confs,
            ocr_prefix,
            ocr_conf,
            ocr_prefix_confs,
            align="left",
        )

    number, number_details = fuse_characters_by_confidence(
        model_number,
        number_confs,
        ocr_number,
        ocr_conf,
        ocr_number_confs,
        align="right",
    )
    details = [
        {**item, "field": "prefix"} for item in prefix_details
    ] + [
        {**item, "field": "number"} for item in number_details
    ]
    source = _fusion_source(details, prefix_source)
    if isinstance(ocr, dict):
        fused_confs = [float(item.get("confidence") or 0.0) for item in details]
        ocr["sequence_confidence"] = round(original_ocr_confidence, 4)
        ocr["character_fusion"] = details
        ocr["model_primary"] = (
            "thai_license_plate" if country == "thai" else "lao_license_plate"
        )
        ocr["ocr_supplemental"] = independent_ocr
        if fused_confs:
            ocr["confidence"] = round(sum(fused_confs) / len(fused_confs), 4)
            ocr["fusion_confidence"] = ocr["confidence"]
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
        thai_config: Path | None = None,
        thai_weights: Path | None = None,
        paddle_device: str = "cpu",
    ) -> None:
        self.source = source
        self.config = config
        self.weights = weights.with_suffix("") if weights.suffix == ".pdparams" else weights
        default_lao_config = source / "configs" / "rec" / "plate_rec_lao_full.yml"
        default_lao_weights = latest_ocr_weights(ROOT, "lao-full", "plate_rec_lao_full") or (
            ROOT / "runs" / "ocr" / "plate_rec_lao_full" / "best_accuracy"
        )
        self.lao_config = lao_config or default_lao_config
        selected_lao_weights = lao_weights or default_lao_weights
        self.lao_weights = (
            selected_lao_weights.with_suffix("")
            if selected_lao_weights.suffix == ".pdparams"
            else selected_lao_weights
        )
        default_thai_config = source / "configs" / "rec" / "plate_rec_thai_full.yml"
        default_thai_weights = latest_ocr_weights(ROOT, "thai-full", "plate_rec_thai_full") or (
            ROOT / "runs" / "ocr" / "plate_rec_thai_full" / "best_accuracy"
        )
        self.thai_config = thai_config or default_thai_config
        selected_thai_weights = thai_weights or default_thai_weights
        self.thai_weights = (
            selected_thai_weights.with_suffix("")
            if selected_thai_weights.suffix == ".pdparams"
            else selected_thai_weights
        )
        self.timeout = timeout
        self.script = source / "tools" / "infer_rec.py"
        self.torch_model = torch_model
        self._paddle: dict[str, Any] = {}
        self._paddle_error: dict[str, str] = {}
        self._torch = None
        self._torch_error = ""
        self.paddle_device = str(paddle_device or "cpu")

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

    @property
    def thai_available(self) -> bool:
        return (
            self.script.is_file()
            and self.thai_config.is_file()
            and self.thai_weights.with_suffix(".pdparams").is_file()
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

                use_gpu = self.paddle_device.startswith("gpu") or self.paddle_device.startswith("cuda")
                options = dict(
                    lang=paddle_language,
                    ocr_version="PP-OCRv5",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    enable_mkldnn=not use_gpu,
                    text_det_limit_side_len=960,
                    text_det_limit_type="max",
                    text_det_box_thresh=0.3,
                )
                try:
                    self._paddle[paddle_language] = PaddleOCR(
                        **options, device=self.paddle_device
                    )
                except TypeError:
                    options["use_gpu"] = use_gpu
                    self._paddle[paddle_language] = PaddleOCR(**options)
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
        """Use bundled Thai/Lao Tesseract models when PaddleOCR is unavailable."""

        spec = TESSERACT_LANGUAGES.get(expected_language)
        if spec is None:
            return "", 0.0, ""
        label = expected_language.title()
        executable = _tesseract_executable()
        tessdata = _tesseract_model_dir(str(spec["model"]))
        if executable is None or tessdata is None:
            return "", 0.0, f"{label} Tesseract model unavailable"

        best_text, best_score = "", float("-inf")
        environment = os.environ.copy()
        environment["TESSDATA_PREFIX"] = str(tessdata)
        script_start = int(spec["start"])
        script_end = int(spec["end"])
        length_penalty_after = int(spec["length_penalty_after"])
        try:
            with tempfile.TemporaryDirectory(prefix="plate_tess_") as directory:
                for index, variant in enumerate(self._variants(image)):
                    input_path = Path(directory) / f"variant_{index}.png"
                    write_image(input_path, variant)
                    for psm in (6, 7, 11, 13):
                        process = subprocess.run(
                            [
                                str(executable),
                                str(input_path),
                                "stdout",
                                "-l",
                                str(spec["code"]),
                                "--psm",
                                str(psm),
                            ],
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
                            script_count = sum(script_start <= ord(char) <= script_end for char in text)
                            if script_count + digits < 2:
                                continue
                            if digits == 4:
                                digit_score = 8.0
                            elif expected_language == "thai" and digits in {6, 7}:
                                digit_score = 6.0
                            elif digits == 5:
                                digit_score = 3.0
                            elif digits == 3:
                                digit_score = 2.0
                            elif digits > 4:
                                digit_score = -1.0
                            else:
                                digit_score = -2.0
                            score = (
                                digit_score
                                + min(script_count, 4) * 0.15
                                - max(0, len(text) - length_penalty_after) * 0.08
                            )
                            if score > best_score:
                                best_text, best_score = text, score
        except subprocess.TimeoutExpired:
            return "", 0.0, f"{label} Tesseract timed out"
        except OSError as error:
            # An installer or inaccessible executable must never terminate a
            # scan.  The caller will fall back to the character recogniser.
            return "", 0.0, f"{label} Tesseract could not start: {error}"
        if not best_text:
            return "", 0.0, f"{label} Tesseract returned no usable text"
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
            candidates: list[tuple[float, str, float, list[float]]] = []
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
                    (
                        "Global.use_gpu=true"
                        if self.paddle_device.startswith("gpu") or self.paddle_device.startswith("cuda")
                        else "Global.use_gpu=false"
                    ),
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
                character_confidences: list[float] = []
                if len(fields) >= 4:
                    try:
                        values = json.loads(fields[3])
                        if isinstance(values, list):
                            character_confidences = [
                                max(0.0, min(1.0, float(value)))
                                for value in values
                            ]
                    except (TypeError, ValueError, json.JSONDecodeError):
                        character_confidences = []
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
                if expected_language == "thai":
                    thai_letters = sum("\u0e00" <= char <= "\u0e7f" for char in text)
                    score += min(thai_letters, 4) * 0.08
                candidates.append((score, text, confidence, character_confidences))
            if not candidates:
                return {"status": "error", "text": "", "confidence": 0.0, "message": last_message}
            _, text, confidence, character_confidences = max(
                candidates, key=lambda item: item[0]
            )
            return {
                "status": "ok",
                "text": text,
                "confidence": round(confidence, 4),
                "character_confidences": character_confidences,
                "confidence_mode": (
                    "per-character-ctc"
                    if len(character_confidences) == len(text)
                    else "sequence"
                ),
                "message": "",
            }

    def recognise(
        self,
        image: np.ndarray,
        fallback_text: str = "",
        fallback_confidence: float = 0.0,
        expected_language: str = "",
    ) -> dict[str, Any]:
        """Return the YOLO character reading. PaddleOCR is no longer used."""

        model_name = {
            "thai": "thai_license_plate",
            "lao": "lao_license_plate",
        }.get(expected_language, "license-plate")
        return {
            "status": "ok",
            "method": "structured-yolo",
            "model": model_name,
            "text": fallback_text,
            "confidence": round(float(fallback_confidence or 0.0), 4),
            "script": expected_language or detect_text_script(fallback_text),
            "message": f"Read by {model_name}",
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
        yolo_device: str | None = None,
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
        self.yolo_device = yolo_device or "cpu"
        bind_yolo_device(self.detector, self.yolo_device)
        self.vehicle_classifier = YoloRegionClassifier(
            vehicle_type_path,
            self.pipeline_config.get("vehicle_class_mapping", {}),
            device=self.yolo_device,
        )
        self.plate_type_classifier = YoloRegionClassifier(
            plate_type_path,
            self.pipeline_config.get("plate_class_mapping", {}),
            device=self.yolo_device,
        )
        self.country_models = {"thai": YOLO(str(thai_path)), "lao": YOLO(str(lao_path))}
        for model in self.country_models.values():
            bind_yolo_device(model, self.yolo_device)
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

    def _predict_character_variants(
        self,
        model: Any,
        variants: list[tuple[np.ndarray, str, float]],
        imgsz: int,
    ) -> list[tuple[Any, str, float]]:
        """Infer compatible crop variants in one model call when possible.

        Character recognition intentionally compares multiple representations
        of the *same* crop.  Sending those representations as one YOLO batch
        removes repeated model setup and GPU synchronisation while preserving
        the individual confidence threshold of every representation.  A small
        fallback keeps this compatible with older Ultralytics releases and
        lightweight model doubles used by integrations.
        """

        device = getattr(self, "yolo_device", None)
        if not variants:
            return []
        if len(variants) == 1:
            image, label, confidence = variants[0]
            prediction = yolo_predict(
                model, image, device=device, conf=confidence, imgsz=imgsz, verbose=False
            )[0]
            return [(prediction, label, confidence)]

        minimum_confidence = min(confidence for _, _, confidence in variants)
        try:
            predictions = yolo_predict(
                model,
                [image for image, _, _ in variants],
                device=device,
                conf=minimum_confidence,
                imgsz=imgsz,
                verbose=False,
            )
            if len(predictions) != len(variants):
                raise ValueError("YOLO returned an unexpected number of batch predictions")
            return [
                (prediction, label, confidence)
                for prediction, (_, label, confidence) in zip(predictions, variants)
            ]
        except Exception:
            # Some providers only accept a single ndarray.  Retain the exact
            # previous inference path instead of sacrificing recognition.
            return [
                (
                    yolo_predict(
                        model, image, device=device, conf=confidence, imgsz=imgsz, verbose=False
                    )[0],
                    label,
                    confidence,
                )
                for image, label, confidence in variants
            ]

    def _read_country_models_from_detector_crop(
        self,
        crop: np.ndarray,
        character_confidence: float,
        imgsz: int,
        fast_mode: bool = False,
        rectified: np.ndarray | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Run Thai/Lao models only after detect_license has produced a crop."""

        if not is_usable_detector_crop(crop):
            return {
                "thai": empty_country_reading("thai"),
                "lao": empty_country_reading("lao"),
            }
        return {
            country: self._read_country_model(
                crop,
                country,
                character_confidence,
                imgsz,
                fast_mode=fast_mode,
                rectified=rectified,
            )
            for country in ("thai", "lao")
        }

    @staticmethod
    def _country_decision_is_committed(
        country_result: dict[str, Any], readings: dict[str, dict[str, Any]]
    ) -> bool:
        """Return whether Thai vs Lao is clear enough to run only that pipeline."""

        country = str(country_result.get("country") or "")
        if country not in {"thai", "lao"}:
            return False
        if str(country_result.get("selection_reason") or "") in {
            "thai_layout_priority",
            "lao_layout_priority",
            "thai_province_priority",
            "lao_province_priority",
        }:
            return True
        reading = readings.get(country, {})
        if isinstance(reading, dict) and has_complete_country_layout(reading, country):
            return True
        return float(country_result.get("margin") or 0.0) >= 0.12

    def _classify_and_read_plate(
        self,
        crop: np.ndarray,
        character_confidence: float,
        imgsz: int,
        *,
        fast_mode: bool = False,
        rectified: np.ndarray | None = None,
        preferred_country: str = "",
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """Decide Thai vs Lao first, then run that country's full plate pipeline."""

        classifier = getattr(self, "country_classifier", None) or CountryClassifier()

        def decide(readings: dict[str, dict[str, Any]]) -> dict[str, Any]:
            result = apply_country_layout_priority(classifier.classify(readings), readings)
            thai = readings.get("thai") or {}
            lao = readings.get("lao") or {}
            result["checks"] = {
                "thai_layout": has_complete_country_layout(thai, "thai"),
                "lao_layout": has_complete_country_layout(lao, "lao"),
                "thai_province": bool(country_script_cues(thai, "thai").get("thai_province")),
                "lao_province": bool(country_script_cues(lao, "lao").get("lao_province")),
                "thai_score": round(float(thai.get("score") or 0.0), 4),
                "lao_score": round(float(lao.get("score") or 0.0), 4),
            }
            return result

        def probe_is_sufficient(reading: dict[str, Any], selected: str) -> bool:
            prefix = str(reading.get("plate_prefix") or "")
            if not fast_mode or not reading.get("complete"):
                return False
            if not has_complete_country_layout(reading, selected):
                return False
            if float(reading.get("digit_confidence") or 0.0) < 0.80:
                return False
            if float(reading.get("weakest_glyph_confidence") or 0.0) < 0.55:
                return False
            if is_thai_digit_prefix(prefix) or is_thai_letter_prefix(prefix) or selected == "lao":
                return float(reading.get("prefix_confidence") or 0.0) >= 0.70
            return bool(reading.get("province_code"))

        if preferred_country in {"thai", "lao"}:
            other = "lao" if preferred_country == "thai" else "thai"
            readings = {
                preferred_country: self._read_country_model(
                    crop,
                    preferred_country,
                    character_confidence,
                    imgsz,
                    fast_mode=fast_mode,
                    rectified=rectified,
                ),
                other: empty_country_reading(other),
            }
            country_result = decide(readings)
            country_result["committed"] = True
            country_result["pipeline"] = preferred_country
            country_result["selection_stage"] = "preferred"
            return readings, country_result

        probe = {
            country: self._read_country_model(
                crop, country, character_confidence, imgsz, probe_mode=True
            )
            for country in ("thai", "lao")
        }
        country_result = decide(probe)
        country = str(country_result.get("country") or "")
        committed = self._country_decision_is_committed(country_result, probe)
        if not committed or country not in {"thai", "lao"}:
            readings = self._read_country_models_from_detector_crop(
                crop,
                character_confidence,
                imgsz,
                fast_mode=fast_mode,
                rectified=rectified,
            )
            country_result = decide(readings)
            chosen = str(country_result.get("country") or "")
            country_result["committed"] = chosen in {"thai", "lao"}
            country_result["pipeline"] = chosen or "unknown"
            country_result["selection_stage"] = "full-both"
            return readings, country_result

        other = "lao" if country == "thai" else "thai"
        if probe_is_sufficient(probe[country], country):
            readings = {country: probe[country], other: probe[other]}
            country_result = decide(readings)
            country_result["committed"] = True
            country_result["pipeline"] = country
            country_result["selection_stage"] = "probe-accepted"
            return readings, country_result

        readings = {
            country: self._read_country_model(
                crop,
                country,
                character_confidence,
                imgsz,
                fast_mode=fast_mode,
                rectified=rectified,
            ),
            other: probe[other],
        }
        if not has_complete_country_layout(readings[country], country):
            readings[other] = self._read_country_model(
                crop,
                other,
                character_confidence,
                imgsz,
                fast_mode=fast_mode,
                rectified=rectified,
            )
            country_result = decide(readings)
            country_result["selection_stage"] = "full-fallback"
        else:
            country_result = decide(readings)
            country_result["selection_stage"] = "full-selected"
        chosen = str(country_result.get("country") or country)
        country_result["committed"] = chosen in {"thai", "lao"}
        country_result["pipeline"] = chosen
        return readings, country_result

    def _read_country_model(
        self,
        crop: np.ndarray,
        country: str,
        confidence: float,
        imgsz: int,
        fast_mode: bool = False,
        rectified: np.ndarray | None = None,
        probe_mode: bool = False,
    ) -> dict[str, Any]:
        """Read characters from a detect_license crop, never from a full frame."""

        if not is_usable_detector_crop(crop):
            return empty_country_reading(country)
        model = self.country_models[country]
        expected = 6 if country == "thai" else 4

        def read_variant(
            prediction: Any, variant: str, variant_confidence: float
        ) -> dict[str, Any]:
            characters = [
                item
                for item in self._characters(prediction, model.names)
                if item.confidence >= variant_confidence
            ]
            result = analyse_country_characters(characters, country)
            result["score"] = country_reading_score(result, country)
            result["inference_variant"] = variant
            return result

        if probe_mode:
            probe = self._predict_character_variants(
                model, [(crop, "country-probe", confidence)], max(640, min(imgsz, 960))
            )
            reading = read_variant(*probe[0]) if probe else empty_country_reading(country)
            reading["score"] = country_reading_score(reading, country)
            reading["inference_variant"] = "country-probe"
            return reading

        # Always compare the raw crop with an illumination-normalised crop.
        # Camera/video mode used to skip every enhancement for speed, exactly
        # where glare varies most between frames.  The normalised pass removes
        # uneven reflection without changing the plate geometry.
        variants = [
            (crop, "original-high-resolution", confidence),
            (
                normalise_character_illumination(crop),
                "illumination-normalised",
                max(0.08, confidence * 0.70),
            ),
        ]
        if not fast_mode:
            variants.extend(
                (
                    (preprocess_character_model_crop(crop), "colour-enhanced", confidence),
                    (enhance_character_crop(crop), "contrast-enhanced", max(0.08, confidence * 0.60)),
                )
            )
        readings = [
            read_variant(prediction, variant, variant_confidence)
            for prediction, variant, variant_confidence in self._predict_character_variants(
                model, variants, max(960, imgsz)
            )
        ]
        # A straightened/enhanced copy of the padded crop (see
        # rectify_plate_for_recognition) competes with the raw crop; the
        # selection below only lets it win when it reads at least as well.
        # It has a different pixel size, so it is inferred in its own call:
        # mixing shapes in one batch disables the rectangular letterbox and
        # roughly triples the cost of every variant.
        if rectified is not None and is_usable_detector_crop(rectified):
            readings.extend(
                read_variant(prediction, variant, variant_confidence)
                for prediction, variant, variant_confidence in self._predict_character_variants(
                    model, [(rectified, "rectified-enhanced", confidence)], max(960, imgsz)
                )
            )
        reading = readings[0]
        def reading_rank(value: dict[str, Any]) -> tuple[bool, bool, int, float, float]:
            count = int(value.get("digit_detection_count", 0))
            prefix = str(value.get("plate_prefix") or "")
            return (
                bool(value.get("complete")),
                is_thai_letter_prefix(prefix) if country == "thai" else False,
                -abs(count - expected),
                float(value.get("score", 0.0)),
                float(value.get("digit_confidence", 0.0)),
            )

        def reading_digits(value: dict[str, Any]) -> str:
            return f"{value.get('plate_prefix', '')}{value.get('plate_number', '')}"

        def accept_variant(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
            """Avoid replacing a complete reading with a speculative result."""

            if candidate.get("complete") and current.get("complete"):
                candidate_prefix = str(candidate.get("plate_prefix") or "")
                current_prefix = str(current.get("plate_prefix") or "")
                if country == "thai" and is_thai_letter_prefix(candidate_prefix) and not is_thai_letter_prefix(current_prefix):
                    return True
                if country == "thai" and is_thai_letter_prefix(current_prefix) and not is_thai_letter_prefix(candidate_prefix):
                    return False
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

        for candidate in readings[1:]:
            if accept_variant(candidate, reading):
                reading = candidate

        # A repeated reading from independent lighting/contrast passes is
        # stronger evidence than one deceptively confident classification.
        # Apply this only with a clear majority, otherwise retain the existing
        # conservative confidence-based selection.
        complete_readings = [item for item in readings if item.get("complete")]
        letter_readings = [
            item
            for item in complete_readings
            if country == "thai" and is_thai_letter_prefix(str(item.get("plate_prefix") or ""))
        ]
        ranked_pool = letter_readings or complete_readings
        votes: dict[str, list[dict[str, Any]]] = {}
        for candidate in ranked_pool:
            votes.setdefault(reading_digits(candidate), []).append(candidate)
        if votes:
            _, supported = max(
                votes.items(),
                key=lambda item: (
                    len(item[1]),
                    float(np.mean([value.get("score", 0.0) for value in item[1]])),
                ),
            )
            if len(supported) > len(ranked_pool) / 2:
                reading = max(
                    supported,
                    key=lambda value: (
                        float(value.get("score", 0.0)),
                        float(value.get("digit_confidence", 0.0)),
                    ),
                )
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

        # The province/region line is usually only a few dozen pixels high in
        # a detector crop. Upscale it before inference so the model receives
        # enough glyph detail. Keep the scale explicit so every returned box
        # can be mapped back to the original plate crop.
        roi_scale = min(
            PROVINCE_ROI_MAX_SCALE,
            max(1.0, PROVINCE_ROI_TARGET_WIDTH / max(width, 1)),
        )

        def upscale(image: np.ndarray) -> np.ndarray:
            if roi_scale <= 1.0:
                return image
            return cv2.resize(
                image,
                None,
                fx=roi_scale,
                fy=roi_scale,
                interpolation=cv2.INTER_CUBIC,
            )

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
                "support_count": 1,
            }
            previous = candidates.get(code_text)
            if previous is None:
                candidates[code_text] = candidate
            else:
                previous["support_count"] = int(previous.get("support_count", 1)) + 1
                if candidate["confidence"] > previous["confidence"]:
                    candidate["support_count"] = previous["support_count"]
                    candidates[code_text] = candidate

        # Keep the original plate-level candidate as temporal evidence too.
        for token in reading.get("tokens", []):
            if not isinstance(token, dict) or token.get("kind") != "province":
                continue
            token_box = token.get("box", [])
            if isinstance(token_box, list) and len(token_box) == 4:
                add_candidate(token.get("label"), token.get("confidence"), token_box, "plate")

        variants = [
            (upscale(roi), "province_roi_upscaled", max(0.05, confidence * 0.45)),
            (
                preprocess_character_model_crop(upscale(roi)),
                "province_roi_colour",
                max(0.05, confidence * 0.35),
            ),
            (
                enhance_character_crop(upscale(roi)),
                "province_roi_contrast",
                max(0.05, confidence * 0.25),
            ),
        ]
        for prediction, source, variant_confidence in self._predict_character_variants(
            model, variants, max(1280, imgsz)
        ):
            for item in self._characters(prediction, model.names):
                if item.confidence < variant_confidence:
                    continue
                add_candidate(
                    item.text,
                    item.confidence,
                    [
                        item.x1 / roi_scale,
                        item.y1 / roi_scale + y1,
                        item.x2 / roi_scale,
                        item.y2 / roi_scale + y1,
                    ],
                    source,
                )

        result = dict(reading)
        province_candidates = sorted(
            candidates.values(),
            key=lambda value: (
                int(value.get("support_count", 1)),
                float(value["confidence"]),
            ),
            reverse=True,
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
            existing_province = str(reading.get("province") or "")
            existing_province_code = str(reading.get("province_code") or "")
            existing_province_confidence = float(reading.get("province_confidence", 0.0))
            missing = not existing_province and not existing_province_code
            # A known จังหวัด / ແຂວງ string from OCR fills a blank detector
            # result even at modest softmax confidence. Replace a YOLO guess
            # only when OCR is competitive.
            if missing or ocr_confidence >= 0.28 or ocr_confidence + 0.05 >= existing_province_confidence:
                reading["province"] = ocr_province
                reading["province_code"] = ocr_province_code
                reading["province_confidence"] = round(
                    max(existing_province_confidence, ocr_confidence, 0.45 if missing else 0.0),
                    4,
                )
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

    def finalise_country_reading(
        self,
        ocr_ready_crop: np.ndarray,
        country: str,
        readings: dict[str, Any],
        run_ocr: bool = True,
    ) -> tuple[str, dict[str, Any], str, str]:
        """Run OCR for ``country``, then retry the other language if the text disagrees."""

        reading = readings.get(country) if isinstance(readings, dict) else None
        if not isinstance(reading, dict):
            reading = {}
        ocr, prefix, number = self.finalise_reading(
            ocr_ready_crop, country, reading, run_ocr=run_ocr
        )
        if not run_ocr:
            return country, ocr, prefix, number
        text = str(ocr.get("raw_text") or ocr.get("text") or "")
        hint = country_hint_from_text(text)
        thai = readings.get("thai") if isinstance(readings, dict) else None
        lao = readings.get("lao") if isinstance(readings, dict) else None
        if not hint and isinstance(thai, dict) and str(thai.get("province_code") or "") in THAI_PROVINCE_NAMES:
            hint = "thai"
        if not hint and isinstance(lao, dict) and str(lao.get("province_code") or "") in LAO_PROVINCE_NAMES:
            hint = "lao"
        if hint and hint != country and isinstance(readings, dict) and isinstance(readings.get(hint), dict):
            ocr2, prefix2, number2 = self.finalise_reading(
                ocr_ready_crop, hint, readings[hint], run_ocr=True
            )
            text2 = str(ocr2.get("raw_text") or ocr2.get("text") or "")
            if country_hint_from_text(text2) == hint or extract_ocr_province(text2, hint)[0]:
                return hint, ocr2, prefix2, number2
        return country, ocr, prefix, number

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
        vehicle_types_override: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], np.ndarray]:
        annotated = self.draw_scan_roi(image)
        if vehicle_types_override is not None:
            # Streaming callers may reuse a recent vehicle classification.
            # Plate detection/recognition still runs on the current frame.
            vehicle_types = [dict(item) for item in vehicle_types_override]
        else:
            vehicle_types = (
                self._vehicle_types(image, vehicle_type_confidence, imgsz)
                if self.pipeline_mode == "full"
                else []
            )
        # detect_license is trained on full vehicle frames. Cropping to the
        # user ROI first shrinks/enlarges plates away from that distribution
        # and misses boxes; run YOLO on the whole image then keep ROI hits.
        detections = yolo_predict(
            self.detector,
            image,
            device=getattr(self, "yolo_device", None),
            conf=detector_confidence,
            imgsz=imgsz,
            verbose=False,
        )[0]
        self.last_plate_detection_count = (
            len(detections.boxes) if detections.boxes is not None else 0
        )
        if detections.boxes is None or len(detections.boxes) == 0:
            self.last_vehicle_types = vehicle_types
            self._draw_vehicle_types(annotated, vehicle_types)
            return [], annotated
        if self.pipeline_mode != "full" and vehicle_types_override is None:
            vehicle_types = self._vehicle_types(image, vehicle_type_confidence, imgsz)
        self.last_vehicle_types = vehicle_types
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
            xyxy = [float(value) for value in xyxy]
            if not self._box_inside_scan_roi(xyxy, image):
                continue
            crop, crop_box, crop_status = complete_detector_crop(image, xyxy, padding)
            if crop is None or crop_box is None or crop_status != "full":
                continue
            # Thai/Lao character models are trained on plate crops. Never send
            # the full vehicle frame, and skip this detection if the crop is
            # empty or too small to be a usable plate.
            if not is_usable_detector_crop(crop):
                continue
            # The detector box is intentionally tight, while the recogniser
            # was trained on crops containing a small plate border.  Keep the
            # displayed/saved box unchanged but give OCR a slightly wider crop.
            # Lao glyphs can lose their upper stroke when the detector box is
            # tight. A little extra border lets the variant ensemble recover
            # the prefix without materially including the surrounding car.
            ocr_crop, ocr_crop_box = clamp_crop(image, crop_box, max(0.25, padding))
            # CCTV keeps the raw crop for the first decision. Perspective
            # correction is an extra model call, so it runs for still images
            # and only for a CCTV plate that is still incomplete.
            if fast_mode:
                rectified_crop, crop_rectification = None, {
                    "applied": False,
                    "reason": "deferred-live",
                }
            else:
                rectified_crop, crop_rectification = rectify_plate_for_recognition(
                    ocr_crop, crop_box[2] - crop_box[0]
                )
            readings, country_result = self._classify_and_read_plate(
                crop,
                character_confidence,
                imgsz,
                fast_mode=fast_mode,
                rectified=rectified_crop,
            )
            grown = glyph_edge_expansion(crop_box, readings, crop.shape, image.shape)
            selected_now = readings.get(str(country_result.get("country") or ""))
            live_complete = fast_mode and isinstance(selected_now, dict) and bool(selected_now.get("complete"))
            if grown is not None and not live_complete:
                retry_crop, retry_box, retry_status = complete_detector_crop(
                    image, grown, padding
                )
                if retry_crop is not None and retry_box is not None and retry_status == "full":
                    crop, crop_box = retry_crop, retry_box
                    ocr_crop, ocr_crop_box = clamp_crop(image, crop_box, max(0.25, padding))
                    rectified_crop, crop_rectification = rectify_plate_for_recognition(
                        ocr_crop, crop_box[2] - crop_box[0]
                    )
                    preferred = (
                        str(country_result.get("pipeline") or "")
                        if country_result.get("committed")
                        else ""
                    )
                    readings, country_result = self._classify_and_read_plate(
                        crop,
                        character_confidence,
                        imgsz,
                        fast_mode=fast_mode,
                        rectified=rectified_crop,
                        preferred_country=preferred,
                    )
            country = str(country_result["country"])
            recognition_country = str(country_result["raw_country"])
            country_margin = float(country_result["margin"])
            cross_model_character_fusion = fuse_cross_model_digits(
                readings, recognition_country
            )
            vehicle = self._match_vehicle_type(crop_box, vehicle_types)
            plate_type = (
                {
                    "raw_class": "unknown",
                    "display_name": "Unknown",
                    "normalized_class": "unknown",
                    "confidence": 0.0,
                    "box": None,
                }
                if fast_mode
                else self.plate_type_classifier.classify(crop, plate_type_confidence, imgsz)
            )
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
            country, ocr, plate_prefix, plate_number = self.finalise_country_reading(
                ocr_ready_crop,
                recognition_country,
                readings,
                run_ocr=run_ocr,
            )
            recognition_country = country
            country_result["country"] = country
            country_result["raw_country"] = country
            digit_evidence = cross_model_digit_evidence(readings, recognition_country)
            selected_reading = readings.get(recognition_country) or {}
            if not isinstance(selected_reading, dict):
                selected_reading = {}
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
            quality = build_plate_quality(
                country=country,
                primary_country=recognition_country,
                readings=readings,
                plate_prefix=plate_prefix,
                plate_number=plate_number,
                ocr=ocr,
                validation=validation,
                detection_confidence=float(confidence),
                country_confidence=float(country_result["confidence"]),
                digit_evidence=digit_evidence,
            )
            glyph_confidence = published_plate_confidence(selected_reading, digit_evidence)
            recognition_confidence = contested_script_confidence(
                glyph_confidence,
                readings,
                recognition_country,
            )
            quality["overall_confidence"] = contested_script_confidence(
                quality["overall_confidence"],
                readings,
                recognition_country,
            )
            if recognition_confidence + 0.01 < glyph_confidence:
                reasons = list(quality["qc"].get("reason") or [])
                if "contested_country_script" not in reasons:
                    reasons.append("contested_country_script")
                quality["qc"] = {
                    "status": "REVIEW",
                    "reason": reasons,
                    "need_review": True,
                }
            if quality["overall_confidence"] < 0.80:
                quality["confidence_level"] = "LOW"
            elif quality["overall_confidence"] < 0.95:
                quality["confidence_level"] = "MEDIUM"
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
                "overall_confidence": quality["overall_confidence"],
                "confidence_level": quality["confidence_level"],
                "qc": quality["qc"],
                "detection": quality["detection"],
                "character_validation": quality["character_validation"],
                "cfd": quality["cfd"],
                "dataset": quality["dataset"],
                "confirmed": bool(selected_reading.get("complete")) and bool(validation["valid"]),
                "country_confidence_margin": round(country_margin, 4),
                "country_selection_reason": country_result["selection_reason"],
                "box": crop_box,
                "ocr_crop_box": ocr_crop_box,
                "crop_status": "full",
                "ocr_preprocessing": preprocessing_profile,
                "country_readings": readings,
                "cross_model_digit_evidence": digit_evidence,
                "ocr": ocr,
                "validation": validation,
                "processing": {
                    "pipeline_mode": self.pipeline_mode,
                    "preprocessing_profile": preprocessing_profile,
                    "preprocessing_parameters": preprocessing_parameters,
                    "ocr_model": ocr_model,
                    "cross_model_digit_evidence": digit_evidence,
                    "cross_model_character_fusion": cross_model_character_fusion,
                    "crop_rectification": crop_rectification,
                    "country_decision": {
                        "country": country,
                        "pipeline": str(country_result.get("pipeline") or country),
                        "committed": bool(country_result.get("committed")),
                        "reason": str(country_result.get("selection_reason") or ""),
                        "stage": str(country_result.get("selection_stage") or ""),
                        "margin": round(country_margin, 4),
                    },
                    "recognition_variant": {
                        country_name: reading.get("inference_variant", "")
                        for country_name, reading in readings.items()
                        if isinstance(reading, dict)
                    },
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
                    "overall_confidence": quality["overall_confidence"],
                    "qc": quality["qc"],
                    "cfd": quality["cfd"],
                    "cross_model_digit_evidence": digit_evidence,
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "scan" / "data", help="Directory for dated scan evidence and result JSON")
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
        help="Camera identifier used in archived -full_vehicle/-plate_crops filenames",
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
    annotated_path = scan_output_dir / "log" / f"{image_path.stem}_annotated.jpg"
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
        plate["character_annotated_image"] = ""
    result = {
        "media_type": "image",
        "input": str(image_path),
        "plate_count": len(plates),
        "plates": plates,
        "annotated_image": str(annotated_path),
        "output_dir": str(scan_output_dir),
    }
    result = archive_service._write_result_manifest(
        result,
        scan_output_dir / "json" / f"{image_path.stem}_result.json",
    )
    result_path = Path(result["result_json"])
    print_cmd_summary(image_path, plates)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved results to: {result_path}")


if __name__ == "__main__":
    main()
