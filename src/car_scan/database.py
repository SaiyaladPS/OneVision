"""PostgreSQL repository implemented with Prisma Client Python."""

from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .realtime import publish_event

LAOS_TIMEZONE = timezone(timedelta(hours=7), "Asia/Vientiane")
# Compatibility alias for existing report helpers and callers.
BANGKOK = LAOS_TIMEZONE
LOGGER = logging.getLogger(__name__)

_SKIP_RECORD_KEYS = {
    "best_crop",
    "best_ready_crop",
    "best_vehicle_frame",
    "crop",
    "frame",
    "image",
    "annotated",
    "preview",
    "ready_crop",
    "full_vehicle",
    "processing",
}


def jsonable_value(value: Any, *, depth: int = 0) -> Any:
    """Drop numpy frames and other non-JSON values before Prisma/API encode."""

    if depth > 12:
        return None
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return None
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name in _SKIP_RECORD_KEYS:
                continue
            encoded = jsonable_value(item, depth=depth + 1)
            if encoded is not None:
                compact[name] = encoded
        return compact
    if isinstance(value, (list, tuple)):
        if len(value) > 64:
            sample = value[0] if value else None
            if isinstance(sample, (list, tuple)) or _is_ndarray(sample):
                return None
        return [jsonable_value(item, depth=depth + 1) for item in value]
    if _is_ndarray(value):
        ndim = int(getattr(value, "ndim", 1) or 1)
        size = int(getattr(value, "size", 0) or 0)
        if ndim == 0 or size == 1:
            try:
                return value.item()
            except Exception:
                return None
        if ndim == 1 and size <= 16:
            return [jsonable_value(item, depth=depth + 1) for item in value.tolist()]
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def json_ready(value: Any) -> Any:
    """Coerce a payload into plain JSON types Prisma can encode."""

    return json.loads(json.dumps(jsonable_value(value), ensure_ascii=False, default=str))


def wrap_prisma_json_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Prisma Json columns must be wrapped in prisma.Json, not a plain dict."""

    payload = dict(data)
    if "rawResult" in payload:
        payload["rawResult"] = prisma_json(payload["rawResult"])
    plates = payload.get("plates")
    if isinstance(plates, dict) and isinstance(plates.get("create"), list):
        wrapped_plates = []
        for plate in plates["create"]:
            row = dict(plate)
            if "rawPlate" in row:
                row["rawPlate"] = prisma_json(row["rawPlate"])
            wrapped_plates.append(row)
        payload["plates"] = {"create": wrapped_plates}
    return payload


def prisma_json(value: Any) -> Any:
    from prisma import Json

    return Json(json_ready(value))


def _is_ndarray(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "tolist")


def stored_output_path(raw: Any, output_dir: Path | str | None = None) -> str | None:
    """Store image/video paths as posix paths relative to CAR_SCAN_OUTPUT_DIR."""

    if raw in (None, ""):
        return None
    path = Path(str(raw))
    if output_dir is not None:
        try:
            relative = path.expanduser().resolve().relative_to(Path(output_dir).expanduser().resolve())
            return relative.as_posix()
        except (OSError, ValueError):
            pass
    return path.as_posix()


def resolve_stored_path(raw: str | None, output_dir: Path | str | None = None) -> Path | None:
    """Resolve a stored relative or absolute evidence path against output_dir."""

    if raw in (None, ""):
        return None
    text = str(raw).strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        if output_dir is None:
            return None
        path = Path(output_dir) / path
    try:
        return path.expanduser().resolve()
    except OSError:
        return None


def _text(value: Any, limit: int | None = None) -> str | None:
    if isinstance(value, dict):
        for key in ("normalized_class", "type", "label", "name", "class_name"):
            if value.get(key) not in (None, ""):
                value = value.get(key)
                break
        else:
            return None
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    if limit is not None:
        return text[:limit]
    return text


def _float_value(value: Any, default: float = 0.0) -> float:
    if _is_ndarray(value):
        try:
            if int(getattr(value, "size", 1) or 1) != 1:
                return default
            value = value.item()
        except Exception:
            return default
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _coord_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if _is_ndarray(value):
        value = value.tolist()
    try:
        items = list(value)
    except TypeError:
        return None
    if len(items) < 4:
        return None
    try:
        return [int(float(item)) for item in items[:4]]
    except (TypeError, ValueError):
        return None


def _vehicle_type(plate: dict[str, Any]) -> Any:
    vehicle = plate.get("vehicle")
    if isinstance(vehicle, dict):
        return vehicle.get("normalized_class") or vehicle.get("type") or plate.get("vehicle_type")
    return plate.get("vehicle_type")


def compact_plate_record(plate: dict[str, Any], output_dir: Path | str | None = None) -> dict[str, Any]:
    """Keep gate-useful plate fields and relative image paths only."""

    ocr = plate.get("ocr") if isinstance(plate.get("ocr"), dict) else {}
    payload: dict[str, Any] = {
        "id": plate.get("id"),
        "country": plate.get("country"),
        "country_confidence": _float_value(plate.get("country_confidence")),
        "province": plate.get("province"),
        "province_code": plate.get("province_code"),
        "plate_prefix": plate.get("plate_prefix"),
        "plate_prefix_code": plate.get("plate_prefix_code"),
        "plate_number": plate.get("plate_number"),
        "camera_label": plate.get("camera_label"),
        "vehicle_type": _text(_vehicle_type(plate)),
        "vehicle_type_confidence": _float_value(
            plate.get("vehicle_type_confidence")
            or (plate.get("vehicle", {}).get("confidence") if isinstance(plate.get("vehicle"), dict) else None)
        ),
        "plate_type": _text(plate.get("plate_type")),
        "ocr": jsonable_value(ocr),
        "detection_confidence": _float_value(plate.get("detection_confidence")),
        "recognition_confidence": _float_value(plate.get("recognition_confidence") or ocr.get("confidence")),
        "overall_confidence": _float_value(plate.get("overall_confidence")),
        "confidence_level": plate.get("confidence_level"),
        "box": _coord_list(plate.get("box")),
        "ocr_crop_box": _coord_list(plate.get("ocr_crop_box")),
        "full_vehicle_image": stored_output_path(plate.get("full_vehicle_image"), output_dir),
        "crop_image": stored_output_path(plate.get("crop_image"), output_dir),
        "ocr_ready_image": stored_output_path(
            plate.get("ocr_ready_image") or plate.get("ocr_ready_archive_image"),
            output_dir,
        ),
        "archive_filename": plate.get("archive_filename"),
        "archive_sequence": plate.get("archive_sequence"),
        "live_result_key": plate.get("live_result_key"),
        "track_id": plate.get("track_id") or plate.get("video_track_id") or plate.get("camera_track_id"),
        "video_first_frame": plate.get("video_first_frame"),
        "video_confirmed_frame": plate.get("video_confirmed_frame"),
        "video_last_frame": plate.get("video_last_frame"),
        "video_occurrences": plate.get("video_occurrences"),
        "camera_first_frame": plate.get("camera_first_frame"),
        "camera_confirmed_frame": plate.get("camera_confirmed_frame"),
        "camera_occurrences": plate.get("camera_occurrences"),
        "first_seen_frame": plate.get("first_seen_frame"),
        "last_seen_frame": plate.get("last_seen_frame"),
        "sighting_count": plate.get("sighting_count"),
        "validation": jsonable_value(plate.get("validation")) if isinstance(plate.get("validation"), dict) else None,
    }
    return {key: value for key, value in payload.items() if value not in (None, "")}


def plate_registration_key(plate: dict[str, Any]) -> str:
    prefix = str(plate.get("plate_prefix") or "").strip()
    number = str(plate.get("plate_number") or "").strip()
    if not prefix and not number:
        return ""
    country = str(plate.get("country") or "").strip()
    return f"reg:{country}|{prefix}|{number}"


def plate_text_key(plate: dict[str, Any]) -> str:
    """Build a country-independent identity for a complete live reading."""

    def compact(value: Any) -> str:
        return re.sub(r"[\s\-_.]+", "", str(value or "").strip().lower())

    prefix = compact(plate.get("plate_prefix"))
    number = compact(plate.get("plate_number"))
    if not prefix and not number:
        return ""
    return f"reg-text:{prefix}|{number}"


def live_plate_keys(plate: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    live = str(plate.get("live_result_key") or "").strip()
    if live:
        keys.append(f"live:{live}")
    registration = plate_registration_key(plate)
    if registration and registration not in keys:
        keys.append(registration)
    return keys


def plate_create_data(plate: dict[str, Any], output_dir: Path | str | None = None) -> dict[str, Any]:
    """Map one confirmed plate onto Prisma Plate create/update input."""

    ocr = plate.get("ocr") if isinstance(plate.get("ocr"), dict) else {}
    return {
        "plateIndex": int(plate.get("id") or 0),
        "country": str(plate.get("country") or "unknown")[:16],
        "province": _text(plate.get("province")),
        "provinceCode": _text(plate.get("province_code")),
        "platePrefix": _text(plate.get("plate_prefix")),
        "platePrefixCode": _text(plate.get("plate_prefix_code")),
        "plateNumber": _text(plate.get("plate_number")),
        "ocrText": _text(ocr.get("text")),
        "ocrConfidence": _float_value(ocr.get("confidence")),
        "detectionConfidence": _float_value(plate.get("detection_confidence")),
        "recognitionConfidence": _float_value(plate.get("recognition_confidence") or ocr.get("confidence")),
        "overallConfidence": _float_value(plate.get("overall_confidence")),
        "confidenceLevel": _text(plate.get("confidence_level")),
        "vehicleType": _text(_vehicle_type(plate)),
        "vehicleTypeConfidence": _float_value(plate.get("vehicle_type_confidence")),
        "plateType": _text(plate.get("plate_type")),
        "vehicleImage": stored_output_path(plate.get("full_vehicle_image"), output_dir),
        "cropImage": stored_output_path(
            plate.get("crop_image") or plate.get("ocr_ready_archive_image"),
            output_dir,
        ),
        "ocrReadyImage": stored_output_path(
            plate.get("ocr_ready_image") or plate.get("ocr_ready_archive_image"),
            output_dir,
        ),
        "rawPlate": json_ready(compact_plate_record(plate, output_dir)),
    }


def compact_scan_record(result: dict[str, Any], output_dir: Path | str | None = None) -> dict[str, Any]:
    plates = [
        compact_plate_record(plate, output_dir)
        for plate in result.get("plates") or []
        if isinstance(plate, dict)
    ]
    rejected = [
        compact_plate_record(plate, output_dir)
        for plate in result.get("rejected_plates") or []
        if isinstance(plate, dict)
    ]
    payload: dict[str, Any] = {
        "media_type": result.get("media_type"),
        "input": stored_output_path(result.get("input"), output_dir),
        "plate_count": int(result.get("plate_count") or len(plates)),
        "rejected_plate_count": int(result.get("rejected_plate_count") or len(rejected)),
        "annotated_image": stored_output_path(result.get("annotated_image"), output_dir),
        "output_video": stored_output_path(result.get("output_video"), output_dir),
        "sampled_frames": result.get("sampled_frames"),
        "cancelled": result.get("cancelled"),
        "operator_id": result.get("operator_id"),
        "operator_name": result.get("operator_name"),
        "operator_username": result.get("operator_username"),
        "plates": plates,
    }
    if rejected:
        payload["rejected_plates"] = rejected
    return {key: value for key, value in payload.items() if value not in (None, "")}


def scan_create_data(
    result: dict[str, Any],
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Map a scan JSON payload onto Prisma ScanRun/Plate create input."""

    plates: list[dict[str, Any]] = []
    for plate in result.get("plates") or []:
        if not isinstance(plate, dict):
            continue
        plates.append(plate_create_data(plate, output_dir))
    data: dict[str, Any] = {
        "sourcePath": stored_output_path(result.get("input"), output_dir) or "",
        "annotatedImage": stored_output_path(result.get("annotated_image"), output_dir),
        "outputVideo": stored_output_path(result.get("output_video"), output_dir),
        "mediaType": _text(result.get("media_type")),
        "plateCount": int(result.get("plate_count") or len(plates)),
        "rawResult": json_ready(compact_scan_record(result, output_dir)),
    }
    operator_id = result.get("operator_id")
    if operator_id not in (None, ""):
        data["operatorId"] = int(operator_id)
    name = str(result.get("operator_name") or "").strip()
    username = str(result.get("operator_username") or "").strip()
    if name:
        data["operatorName"] = name
    if username:
        data["operatorUsername"] = username
    if plates:
        data["plates"] = {"create": plates}
    return data


def _queue_report_plate_images(
    plate_id: int,
    plate: dict[str, Any],
    output_dir: Path | str | None = None,
) -> None:
    """Upload locally archived plate images after their database row exists."""

    try:
        from .report_storage import enqueue_plate_images

        enqueue_plate_images(
            plate_id,
            output_dir=Path(output_dir) if output_dir else None,
            full=plate.get("full_vehicle_image"),
            crop=plate.get("crop_image") or plate.get("ocr_ready_archive_image"),
            ocr=plate.get("ocr_ready_image") or plate.get("ocr_ready_archive_image"),
        )
    except Exception:
        # Image sync must not make a successful PostgreSQL scan save fail.
        LOGGER.exception("Could not queue plate images for Report upload: %s", plate_id)


def plate_query_terms(raw: str) -> list[str]:
    """Split a gate lookup into prefix/number fragments officers actually type."""

    text = (raw or "").strip()
    if not text:
        return []
    compact = re.sub(r"[\s\-_.]+", "", text)
    terms: list[str] = []
    for item in (text, compact):
        if item and item not in terms:
            terms.append(item)
    digits = re.sub(r"\D+", "", compact)
    if len(digits) >= 3 and digits not in terms:
        terms.append(digits)
    return terms[:6]


def bangkok_day_bounds(day: date) -> tuple[datetime, datetime]:
    start = datetime(day.year, day.month, day.day, tzinfo=BANGKOK)
    return start, start + timedelta(days=1)


def serialize_scan(run: Any) -> dict[str, Any]:
    scanned = run.scannedAt
    if scanned.tzinfo is None:
        scanned = scanned.replace(tzinfo=timezone.utc)
    local = scanned.astimezone(BANGKOK)
    plates = [_serialize_plate(plate) for plate in getattr(run, "plates", None) or []]
    raw = run.rawResult if isinstance(getattr(run, "rawResult", None), dict) else {}
    source_path = run.sourcePath or raw.get("input") or ""
    return {
        "id": int(run.id),
        "scanned_at": scanned.isoformat(),
        "scanned_time": local.strftime("%H:%M:%S"),
        "date": local.strftime("%Y-%m-%d"),
        "media_type": getattr(run, "mediaType", None) or raw.get("media_type") or "",
        "plate_count": int(run.plateCount or len(plates)),
        "source_path": source_path,
        "source_name": Path(str(source_path)).name,
        "annotated_image": getattr(run, "annotatedImage", None) or raw.get("annotated_image") or "",
        "output_video": getattr(run, "outputVideo", None) or raw.get("output_video") or "",
        "operator_id": run.operatorId,
        "operator_name": run.operatorName or "",
        "operator_username": run.operatorUsername or "",
        "plates": plates,
    }


def _serialize_plate(plate: Any) -> dict[str, Any]:
    raw = plate.rawPlate if isinstance(getattr(plate, "rawPlate", None), dict) else {}
    ocr = raw.get("ocr") if isinstance(raw.get("ocr"), dict) else {}
    crop = (
        getattr(plate, "cropImage", None)
        or raw.get("crop_image")
        or getattr(plate, "ocrReadyImage", None)
        or raw.get("ocr_ready_archive_image")
        or raw.get("ocr_ready_image")
        or ""
    )
    return {
        "id": int(plate.id),
        "country": plate.country or raw.get("country") or "",
        "province": plate.province or raw.get("province") or "",
        "province_code": getattr(plate, "provinceCode", None) or raw.get("province_code") or "",
        "plate_prefix": plate.platePrefix or raw.get("plate_prefix") or "",
        "plate_prefix_code": getattr(plate, "platePrefixCode", None) or raw.get("plate_prefix_code") or "",
        "plate_number": plate.plateNumber or raw.get("plate_number") or "",
        "ocr_text": plate.ocrText or ocr.get("text") or "",
        "vehicle_type": getattr(plate, "vehicleType", None) or raw.get("vehicle_type") or "",
        "vehicle_type_confidence": _float_value(
            getattr(plate, "vehicleTypeConfidence", None) or raw.get("vehicle_type_confidence")
        ),
        "plate_type": getattr(plate, "plateType", None) or raw.get("plate_type") or "",
        "full_vehicle_image": str(getattr(plate, "vehicleImage", None) or raw.get("full_vehicle_image") or ""),
        "crop_image": str(crop or ""),
        "ocr_ready_image": str(getattr(plate, "ocrReadyImage", None) or raw.get("ocr_ready_image") or ""),
        "confidence": _float_value(
            getattr(plate, "recognitionConfidence", None)
            or plate.ocrConfidence
            or raw.get("recognition_confidence")
            or ocr.get("confidence")
        ),
        "detection_confidence": _float_value(
            plate.detectionConfidence or raw.get("detection_confidence")
        ),
        "overall_confidence": _float_value(
            getattr(plate, "overallConfidence", None) or raw.get("overall_confidence")
        ),
        "confidence_level": getattr(plate, "confidenceLevel", None) or raw.get("confidence_level") or "",
    }


def plate_matches_query(plate: dict[str, Any], query: str) -> bool:
    terms = plate_query_terms(query)
    if not terms:
        return True
    blob = "".join(
        str(plate.get(key) or "")
        for key in ("plate_prefix", "plate_number", "ocr_text", "province")
    )
    compact = re.sub(r"[\s\-_.]+", "", blob)
    return any(term in blob or term in compact for term in terms)


def flatten_scans_to_passages(scans: list[dict[str, Any]], query: str = "") -> list[dict[str, Any]]:
    """One ledger row per vehicle/plate, not per video job."""

    passages: list[dict[str, Any]] = []
    for scan in scans:
        plates = list(scan.get("plates") or [])
        if query:
            plates = [plate for plate in plates if plate_matches_query(plate, query)]
            if not plates:
                continue
        elif not plates:
            plates = [{}]
        for plate in plates:
            item = {
                key: value
                for key, value in scan.items()
                if key not in {"plates", "annotated_image", "source_path"}
            }
            item.update(
                {
                    "scan_id": scan.get("id"),
                    "plate_id": plate.get("id"),
                    "plates": [plate] if plate else [],
                    "plate_count": 1 if plate else 0,
                    "country": plate.get("country") or "",
                    "province": plate.get("province") or "",
                    "province_code": plate.get("province_code") or "",
                    "vehicle_type": plate.get("vehicle_type") or "",
                    "plate_type": plate.get("plate_type") or "",
                    "plate_prefix": plate.get("plate_prefix") or "",
                    "plate_prefix_code": plate.get("plate_prefix_code") or "",
                    "plate_number": plate.get("plate_number") or "",
                    "ocr_text": plate.get("ocr_text") or "",
                    "confidence": plate.get("confidence") or 0,
                    "confidence_level": plate.get("confidence_level") or "",
                    "has_vehicle": bool(plate.get("has_vehicle") or scan.get("has_image")),
                    "vehicle_url": plate.get("vehicle_url") or scan.get("image_url"),
                    "crop_url": plate.get("crop_url"),
                    "source_name": (str(scan.get("source_path") or "").replace("\\", "/").rsplit("/", 1)[-1]),
                }
            )
            passages.append(item)
    return passages


def _label_or_unknown(value: Any) -> str:
    text = str(value or "").strip()
    return text or "unknown"


def _hour_from_passage(passage: dict[str, Any]) -> int:
    text = str(passage.get("scanned_time") or "00:00:00")
    try:
        return max(0, min(23, int(text.split(":", 1)[0])))
    except ValueError:
        return 0


def _confidence_band(passage: dict[str, Any]) -> str:
    level = str(passage.get("confidence_level") or "").strip().upper()
    if level in {"HIGH", "MEDIUM", "LOW"}:
        return level.lower()
    confidence = _float_value(passage.get("confidence"))
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.5:
        return "medium"
    return "low"


def _share_rows(counts: dict[str, int], total: int) -> list[dict[str, Any]]:
    rows = [
        {
            "label": label,
            "count": count,
            "share": round((count / total) * 100, 1) if total else 0.0,
        }
        for label, count in counts.items()
        if count
    ]
    rows.sort(key=lambda row: (-int(row["count"]), str(row["label"])))
    return rows


def build_scan_report(scans: list[dict[str, Any]]) -> dict[str, Any]:
    """Duty briefing totals from saved scan runs: one vehicle per plate."""

    passages = flatten_scans_to_passages(scans)
    total = len(passages)
    by_day: dict[str, int] = {}
    by_hour = [0] * 24
    by_operator: dict[tuple[Any, str, str], int] = {}
    by_country: dict[str, int] = {}
    by_vehicle: dict[str, int] = {}
    by_province: dict[str, int] = {}
    by_media: dict[str, int] = {}
    by_confidence = {"high": 0, "medium": 0, "low": 0}
    for item in passages:
        day = str(item.get("date") or "")
        by_day[day] = by_day.get(day, 0) + 1
        by_hour[_hour_from_passage(item)] += 1
        operator_key = (
            item.get("operator_id"),
            str(item.get("operator_username") or ""),
            str(item.get("operator_name") or ""),
        )
        by_operator[operator_key] = by_operator.get(operator_key, 0) + 1
        country = _label_or_unknown(item.get("country"))
        vehicle = _label_or_unknown(item.get("vehicle_type"))
        province = _label_or_unknown(item.get("province"))
        media = _label_or_unknown(item.get("media_type"))
        by_country[country] = by_country.get(country, 0) + 1
        by_vehicle[vehicle] = by_vehicle.get(vehicle, 0) + 1
        by_province[province] = by_province.get(province, 0) + 1
        by_media[media] = by_media.get(media, 0) + 1
        by_confidence[_confidence_band(item)] += 1
    operators = [
        {
            "operator_id": key[0],
            "operator_username": key[1],
            "operator_name": key[2],
            "count": count,
            "share": round((count / total) * 100, 1) if total else 0.0,
        }
        for key, count in by_operator.items()
    ]
    operators.sort(key=lambda row: (-int(row["count"]), str(row["operator_username"])))
    days = [{"date": day, "count": count} for day, count in sorted(by_day.items())]
    return {
        "vehicle_count": total,
        "scan_count": len(scans),
        "operator_count": len(by_operator),
        "days": days,
        "hours": [{"hour": hour, "count": by_hour[hour]} for hour in range(24)],
        "operators": operators,
        "countries": _share_rows(by_country, total),
        "vehicles": _share_rows(by_vehicle, total),
        "provinces": _share_rows(by_province, total),
        "media": _share_rows(by_media, total),
        "confidence": _share_rows(by_confidence, total),
        "truncated": total > 120,
        "passages": [
            {
                "date": item.get("date"),
                "scanned_time": item.get("scanned_time"),
                "operator_name": item.get("operator_name") or "",
                "operator_username": item.get("operator_username") or "",
                "country": item.get("country") or "",
                "vehicle_type": item.get("vehicle_type") or "",
                "plate_prefix": item.get("plate_prefix") or "",
                "plate_number": item.get("plate_number") or "",
                "province": item.get("province") or "",
                "media_type": item.get("media_type") or "",
                "confidence": _float_value(item.get("confidence")),
                "scan_id": item.get("scan_id"),
                "plate_id": item.get("plate_id"),
            }
            for item in passages[:120]
        ],
    }


def group_scans_by_operator(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: OrderedDict[tuple[Any, str, str], dict[str, Any]] = OrderedDict()
    for row in rows:
        key = (row.get("operator_id"), str(row.get("operator_username") or ""), str(row.get("operator_name") or ""))
        if key not in groups:
            groups[key] = {
                "operator_id": row.get("operator_id"),
                "operator_username": key[1],
                "operator_name": key[2],
                "scan_count": 0,
                "plate_count": 0,
                "scans": [],
            }
        group = groups[key]
        group["scans"].append(row)
        group["scan_count"] += 1
        group["plate_count"] += int(row.get("plate_count") or 0)
    return list(groups.values())


def merge_history_operators(
    scan_operators: list[dict[str, Any]],
    roster: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Named operators first, then the duty roster, then unassigned scans."""

    merged: list[dict[str, Any]] = []
    seen: set[tuple[Any, str]] = set()

    def add(item: dict[str, Any]) -> None:
        key = (item.get("id"), str(item.get("username") or ""))
        if key in seen:
            return
        seen.add(key)
        merged.append(
            {
                "id": item.get("id"),
                "username": str(item.get("username") or ""),
                "display_name": str(item.get("display_name") or item.get("operator_name") or ""),
            }
        )

    unassigned: dict[str, Any] | None = None
    for operator in scan_operators:
        if operator.get("id") is None and not operator.get("username"):
            unassigned = operator
            continue
        add(operator)
    for user in roster or []:
        if user.get("active") is False:
            continue
        add(
            {
                "id": user.get("id"),
                "username": user.get("username") or "",
                "display_name": user.get("display_name") or user.get("username") or "",
            }
        )
    if unassigned is not None:
        add(unassigned)
    return merged


class DatabaseRepository:
    """Reuse one Prisma client per process; safe to call from GUI worker threads."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url.strip()
        if not self.database_url:
            raise ValueError("CAR_SCAN_DATABASE_URL is not configured")

    def _client(self) -> Any:
        from .prisma_db import get_client

        return get_client(self.database_url)

    def initialize_schema(self) -> None:
        from .prisma_db import ensure_schema

        ensure_schema(self.database_url)

    def ensure_camera_storage(self) -> None:
        """Add camera-only fields to an existing database without replacing its schema."""

        self._client().execute_raw(
            "ALTER TABLE cameras ADD COLUMN IF NOT EXISTS roi_json TEXT"
        )
        self._client().execute_raw(
            "ALTER TABLE cameras ADD COLUMN IF NOT EXISTS direction TEXT NOT NULL DEFAULT 'UNASSIGNED'"
        )

    def ping(self) -> None:
        self._client().query_raw("SELECT 1")

    def replace_plate_image_reference(self, local_path: str, storage_uri: str) -> int:
        """Replace matching local plate-image paths with a portable storage URI."""

        source = str(local_path or "").replace("\\", "/").lstrip("./")
        target = str(storage_uri or "").strip()
        if not source or not target.startswith("storage://"):
            return 0
        return int(self._client().execute_raw(
            """
            UPDATE plates SET
                vehicle_image = CASE
                    WHEN vehicle_image = $1 OR raw_plate->>'full_vehicle_image' = $1 THEN $2
                    ELSE vehicle_image END,
                crop_image = CASE
                    WHEN crop_image = $1 OR raw_plate->>'crop_image' = $1 THEN $2
                    ELSE crop_image END,
                ocr_ready_image = CASE
                    WHEN ocr_ready_image = $1 OR raw_plate->>'ocr_ready_image' = $1 THEN $2
                    ELSE ocr_ready_image END,
                raw_plate = raw_plate || jsonb_strip_nulls(jsonb_build_object(
                    'full_vehicle_image', CASE WHEN raw_plate->>'full_vehicle_image' = $1 THEN $2 END,
                    'crop_image', CASE WHEN raw_plate->>'crop_image' = $1 THEN $2 END,
                    'ocr_ready_image', CASE WHEN raw_plate->>'ocr_ready_image' = $1 THEN $2 END
                ))
            WHERE vehicle_image = $1 OR crop_image = $1 OR ocr_ready_image = $1
               OR raw_plate->>'full_vehicle_image' = $1
               OR raw_plate->>'crop_image' = $1
               OR raw_plate->>'ocr_ready_image' = $1
            """,
            source,
            target,
        ))

    def list_cameras(self, *, enabled_only: bool = True) -> list[dict[str, Any]]:
        """Return CCTV settings without exposing credentials through the API."""

        where = "WHERE enabled = TRUE" if enabled_only else ""
        rows = self._client().query_raw(
            """
            SELECT id, host, stream_url, label, kind, device_index, enabled, roi_json, direction,
                   created_at, updated_at
            FROM cameras
            """ + where + " ORDER BY id ASC"
        )
        return [
            {
                "id": int(row["id"]),
                "host": str(row.get("host") or ""),
                "stream_url": str(row.get("stream_url") or ""),
                "label": str(row.get("label") or ""),
                "kind": str(row.get("kind") or "ip"),
                "device_index": row.get("device_index"),
                "enabled": bool(row.get("enabled", True)),
                "roi": self._decode_camera_roi(row.get("roi_json")),
                "direction": str(row.get("direction") or "UNASSIGNED"),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
            }
            for row in rows
        ]

    def list_camera_access(self, user_id: int) -> dict[str, dict[str, bool]]:
        """Return the shared OneVision camera permissions keyed by camera host."""

        rows = self._client().query_raw(
            """
            SELECT c.host, ca.can_view, ca.can_scan
            FROM camera_access AS ca
            INNER JOIN cameras AS c ON c.id = ca.camera_id
            WHERE ca.user_id = $1 AND c.enabled = TRUE
            """,
            int(user_id),
        )
        return {
            str(row.get("host") or "").strip().lower(): {
                "can_view": bool(row.get("can_view", False)),
                "can_scan": bool(row.get("can_scan", False)),
            }
            for row in rows
            if str(row.get("host") or "").strip()
        }

    @staticmethod
    def _decode_camera_roi(value: Any) -> dict[str, Any] | None:
        if value is None or value == "":
            return None
        if isinstance(value, dict):
            return value
        try:
            decoded = json.loads(str(value))
        except (TypeError, ValueError):
            LOGGER.warning("Ignoring invalid saved camera ROI")
            return None
        return decoded if isinstance(decoded, dict) else None

    def list_camera_rois(self) -> dict[str, dict[str, Any]]:
        """Return saved normalized ROI values keyed by camera host."""

        rows = self._client().query_raw(
            "SELECT host, roi_json FROM cameras WHERE roi_json IS NOT NULL"
        )
        rois: dict[str, dict[str, Any]] = {}
        for row in rows:
            host = str(row.get("host") or "").strip()
            roi = self._decode_camera_roi(row.get("roi_json"))
            if host and roi is not None:
                rois[host] = roi
        return rois

    def set_camera_roi(
        self,
        host: str,
        roi: dict[str, Any] | None,
        *,
        stream_url: str = "",
    ) -> None:
        """Persist one camera ROI without overwriting its stream settings."""

        wanted = str(host or "").strip()
        if not wanted:
            raise ValueError("camera host is required")
        encoded = json.dumps(roi, ensure_ascii=False, separators=(",", ":")) if roi is not None else None
        self._client().execute_raw(
            """
            INSERT INTO cameras (host, stream_url, roi_json, updated_at)
            VALUES ($1, NULLIF($2, ''), $3, CURRENT_TIMESTAMP)
            ON CONFLICT (host) DO UPDATE SET
                stream_url = COALESCE(NULLIF(EXCLUDED.stream_url, ''), cameras.stream_url),
                roi_json = EXCLUDED.roi_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            wanted,
            str(stream_url or ""),
            encoded,
        )

    def upsert_camera(
        self,
        host: str,
        stream_url: str = "",
        label: str | None = None,
        *,
        kind: str = "ip",
        device_index: int | None = None,
        enabled: bool = True,
    ) -> dict[str, Any]:
        """Create or update one CCTV setting in PostgreSQL."""

        wanted = str(host or "").strip()
        if not wanted:
            raise ValueError("camera host is required")
        self._client().execute_raw(
            """
            INSERT INTO cameras (host, stream_url, label, kind, device_index, enabled, updated_at)
            VALUES ($1, NULLIF($2, ''), NULLIF($3, ''), $4, $5, $6, CURRENT_TIMESTAMP)
            ON CONFLICT (host) DO UPDATE SET
                stream_url = COALESCE(NULLIF(EXCLUDED.stream_url, ''), cameras.stream_url),
                label = CASE WHEN $3 IS NULL THEN cameras.label ELSE NULLIF($3, '') END,
                kind = COALESCE(NULLIF($4, ''), cameras.kind),
                device_index = COALESCE($5, cameras.device_index),
                enabled = $6,
                updated_at = CURRENT_TIMESTAMP
            """,
            wanted,
            str(stream_url or ""),
            label,
            str(kind or "ip"),
            device_index,
            bool(enabled),
        )
        rows = self.list_cameras(enabled_only=False)
        return next(item for item in rows if item["host"].lower() == wanted.lower())

    def delete_camera(self, host: str) -> bool:
        """Delete one non-built-in CCTV setting from PostgreSQL."""

        deleted = self._client().execute_raw("DELETE FROM cameras WHERE LOWER(host) = LOWER($1)", str(host or "").strip())
        return bool(deleted)

    def count_scans(self) -> int:
        return int(self._client().scanrun.count())

    def save_scan(self, result: dict[str, Any], output_dir: Path | str | None = None) -> int:
        """Persist one scan and all detected plates in one Prisma nested create."""

        data = wrap_prisma_json_fields(scan_create_data(result, output_dir=output_dir))
        try:
            created = self._client().scanrun.create(data=data)
        except Exception:
            LOGGER.exception("Failed to insert scan_runs row")
            raise
        scan_id = int(created.id)
        source_plates = [plate for plate in result.get("plates") or [] if isinstance(plate, dict)]
        if source_plates:
            created_plates = self._client().plate.find_many(
                where={"scanId": scan_id},
                order_by={"id": "asc"},
            )
            for source_plate, created_plate in zip(source_plates, created_plates):
                _queue_report_plate_images(int(created_plate.id), source_plate, output_dir)
        publish_event(
            "SCAN_SAVED",
            scan_id=scan_id,
            result=result,
            message="บันทึกผลการตรวจสอบแล้ว",
        )
        return scan_id

    def upsert_live_plate(
        self,
        scan_id: int,
        plate: dict[str, Any],
        output_dir: Path | str | None = None,
        plate_pk: int | None = None,
    ) -> int:
        """Insert or refresh one confirmed plate on an open scan_runs row."""

        row = plate_create_data(plate, output_dir)
        row["rawPlate"] = prisma_json(row["rawPlate"])
        client = self._client()
        if plate_pk is not None:
            client.plate.update(where={"id": plate_pk}, data=row)
            _queue_report_plate_images(int(plate_pk), plate, output_dir)
            publish_event(
                "PLATE_SAVED",
                scan_id=scan_id,
                plate={**plate, "id": plate_pk},
                message="บันทึกผลทะเบียนที่ตรวจพบแล้ว",
            )
            return int(plate_pk)
        created = client.plate.create(data={**row, "scanId": scan_id})
        _queue_report_plate_images(int(created.id), plate, output_dir)
        count = int(client.plate.count(where={"scanId": scan_id}))
        client.scanrun.update(where={"id": scan_id}, data={"plateCount": count})
        publish_event(
            "PLATE_SAVED",
            scan_id=scan_id,
            plate={**plate, "id": int(created.id)},
            result={"plate_count": count},
            message="บันทึกผลทะเบียนที่ตรวจพบแล้ว",
        )
        return int(created.id)

    def update_scan_run(
        self,
        scan_id: int,
        result: dict[str, Any],
        output_dir: Path | str | None = None,
    ) -> None:
        """Refresh run metadata after a video/camera session ends."""

        wrapped = wrap_prisma_json_fields(scan_create_data(result, output_dir=output_dir))
        payload: dict[str, Any] = {}
        for key in (
            "sourcePath",
            "annotatedImage",
            "outputVideo",
            "mediaType",
            "rawResult",
            "operatorId",
            "operatorName",
            "operatorUsername",
        ):
            if key in wrapped and wrapped[key] is not None:
                payload[key] = wrapped[key]
        payload["plateCount"] = int(self._client().plate.count(where={"scanId": scan_id}))
        self._client().scanrun.update(where={"id": scan_id}, data=payload)
        publish_event(
            "REPORT_READY",
            scan_id=scan_id,
            result=result,
            message="รายงานจาก OneVision พร้อมแสดงผลแล้ว",
        )

    def list_scans_for_day(
        self,
        day: date,
        operator_id: int | None = None,
        *,
        unassigned_only: bool = False,
    ) -> list[dict[str, Any]]:
        start, end = bangkok_day_bounds(day)
        where: dict[str, Any] = {"scannedAt": {"gte": start, "lt": end}}
        if unassigned_only:
            where["operatorId"] = None
        elif operator_id is not None:
            where["operatorId"] = operator_id
        rows = self._client().scanrun.find_many(
            where=where,
            order={"scannedAt": "desc"},
            include={"plates": True},
        )
        return [serialize_scan(row) for row in rows]

    def list_scans_in_range(
        self,
        start: date,
        end: date,
        operator_id: int | None = None,
        *,
        unassigned_only: bool = False,
        limit: int = 4000,
    ) -> list[dict[str, Any]]:
        begin, _ = bangkok_day_bounds(start)
        _, stop = bangkok_day_bounds(end)
        where: dict[str, Any] = {"scannedAt": {"gte": begin, "lt": stop}}
        if unassigned_only:
            where["operatorId"] = None
        elif operator_id is not None:
            where["operatorId"] = operator_id
        rows = self._client().scanrun.find_many(
            where=where,
            order={"scannedAt": "desc"},
            include={"plates": True},
            take=max(1, min(limit, 8000)),
        )
        return [serialize_scan(row) for row in rows]

    def get_scan(self, scan_id: int) -> dict[str, Any] | None:
        row = self._client().scanrun.find_unique(where={"id": scan_id}, include={"plates": True})
        if row is None:
            return None
        return serialize_scan(row)

    def search_scans(
        self,
        query: str,
        operator_id: int | None = None,
        *,
        unassigned_only: bool = False,
        limit: int = 80,
    ) -> list[dict[str, Any]]:
        terms = plate_query_terms(query)
        if not terms:
            return []
        clauses: list[dict[str, Any]] = []
        for term in terms:
            clauses.extend(
                [
                    {"plateNumber": {"contains": term}},
                    {"platePrefix": {"contains": term}},
                    {"ocrText": {"contains": term}},
                ]
            )
        plate_where: dict[str, Any] = {"OR": clauses}
        scan_filter: dict[str, Any] = {}
        if unassigned_only:
            scan_filter["operatorId"] = None
        elif operator_id is not None:
            scan_filter["operatorId"] = operator_id
        if scan_filter:
            plate_where["scan"] = {"is": scan_filter}
        plates = self._client().plate.find_many(
            where=plate_where,
            include={"scan": True},
            order={"id": "desc"},
            take=min(max(limit, 1) * 4, 400),
        )
        seen: OrderedDict[int, Any] = OrderedDict()
        for plate in plates:
            scan = getattr(plate, "scan", None)
            if scan is None:
                continue
            scan_id = int(scan.id)
            if scan_id in seen:
                continue
            seen[scan_id] = scan
            if len(seen) >= limit:
                break
        if not seen:
            return []
        rows = self._client().scanrun.find_many(
            where={"id": {"in": list(seen.keys())}},
            include={"plates": True},
            order={"scannedAt": "desc"},
        )
        return [serialize_scan(row) for row in rows]

    def list_scan_days(self, limit: int = 45) -> list[dict[str, Any]]:
        rows = self._client().scanrun.find_many(
            order={"scannedAt": "desc"},
            take=max(1, min(limit * 80, 4000)),
        )
        counts: OrderedDict[str, int] = OrderedDict()
        for row in rows:
            scanned = row.scannedAt
            if scanned.tzinfo is None:
                scanned = scanned.replace(tzinfo=timezone.utc)
            key = scanned.astimezone(BANGKOK).strftime("%Y-%m-%d")
            counts[key] = counts.get(key, 0) + 1
            if len(counts) >= limit:
                break
        return [{"date": day, "scan_count": count} for day, count in counts.items()]

    def list_scan_operators(self) -> list[dict[str, Any]]:
        rows = self._client().scanrun.find_many(
            order={"scannedAt": "desc"},
            take=2000,
        )
        seen: OrderedDict[tuple[Any, str], dict[str, Any]] = OrderedDict()
        for row in rows:
            username = row.operatorUsername or ""
            name = row.operatorName or ""
            key = (row.operatorId, username)
            if row.operatorId is None and not username and not name:
                key = (None, "")
            if key in seen:
                continue
            seen[key] = {
                "id": row.operatorId,
                "username": username,
                "display_name": name or username,
            }
        return list(seen.values())


class LiveScanPersister:
    """Write each confirmed video/camera plate while the stream is still running."""

    def __init__(
        self,
        *,
        database_url: str,
        output_dir: Path | str | None,
        media_type: str,
        source: str,
        operator_id: int | None = None,
        operator_name: str = "",
        operator_username: str = "",
    ) -> None:
        self.database_url = (database_url or "").strip()
        self.output_dir = output_dir
        self.media_type = media_type
        self.source = source
        self.operator_id = operator_id
        self.operator_name = operator_name
        self.operator_username = operator_username
        self.repository: DatabaseRepository | None = None
        self.scan_id: int | None = None
        self.saved_count = 0
        self.error: str | None = None
        self._keys: dict[str, int] = {}

    def _seed(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "media_type": self.media_type,
            "input": self.source,
            "plates": [],
            "plate_count": 0,
            "operator_id": self.operator_id,
            "operator_name": self.operator_name,
            "operator_username": self.operator_username,
        }
        if extra:
            payload.update(extra)
        return payload

    def _repository(self) -> DatabaseRepository | None:
        if not self.database_url:
            return None
        if self.repository is None:
            self.repository = DatabaseRepository(self.database_url)
            self.repository.initialize_schema()
        return self.repository

    def save_plate(self, plate: dict[str, Any]) -> int | None:
        """Persist one temporally confirmed vehicle as soon as archive files exist."""

        if not isinstance(plate, dict):
            return None
        repository = self._repository()
        if repository is None:
            return None
        try:
            if self.scan_id is None:
                self.scan_id = repository.save_scan(self._seed(), output_dir=self.output_dir)
            keys = live_plate_keys(plate)
            text_key = plate_text_key(plate)
            if text_key and text_key not in keys:
                keys.append(text_key)
            existing = None
            for key in keys:
                if key in self._keys:
                    existing = self._keys[key]
                    break
            plate_pk = repository.upsert_live_plate(
                self.scan_id,
                plate,
                output_dir=self.output_dir,
                plate_pk=existing,
            )
            for key in keys:
                self._keys[key] = plate_pk
            if existing is None:
                self.saved_count += 1
            return plate_pk
        except Exception as error:
            LOGGER.exception("Live plate save failed")
            self.error = str(error)
            return None

    def finish(self, result: dict[str, Any]) -> dict[str, Any]:
        repository = self._repository()
        if repository is None:
            result["database_status"] = "ยังไม่ได้ตั้งค่า PostgreSQL"
            return result
        try:
            for plate in result.get("plates") or []:
                if isinstance(plate, dict):
                    self.save_plate(plate)
            if self.scan_id is None:
                if self.media_type == "camera" and not (result.get("plates") or []):
                    result["database_status"] = "ไม่มีทะเบียนที่ยืนยันสำหรับบันทึก"
                    return result
                self.scan_id = repository.save_scan(self._seed(result), output_dir=self.output_dir)
            else:
                repository.update_scan_run(self.scan_id, self._seed(result), output_dir=self.output_dir)
            result["database_id"] = self.scan_id
            if self.error:
                result["database_status"] = f"PostgreSQL บันทึกไม่สำเร็จ: {self.error}"
            else:
                result["database_status"] = f"บันทึก PostgreSQL แล้ว ({self.saved_count} คัน)"
        except Exception as error:
            LOGGER.exception("Live scan finalize failed")
            result["database_status"] = f"PostgreSQL บันทึกไม่สำเร็จ: {error}"
        return result
