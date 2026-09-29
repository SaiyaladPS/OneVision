"""FastAPI web interface that reuses the existing ScanService pipeline."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile
from starlette.middleware.body_limit import RequestBodyLimitMiddleware
from starlette.concurrency import run_in_threadpool

from .auth import (
    AuthBackend,
    AuthStore,
    COOKIE_NAME,
    clear_session_cookie,
    current_user,
    canonical_role,
    has_permission,
    public_user,
    require_permission,
    require_user,
    scan_permission,
    set_session_cookie,
    verify_password,
)
from .config import (
    Settings,
    compose_camera_url,
    is_camera_stream_url,
    load_extra_cameras,
    local_camera_url,
    parse_local_camera,
    public_cameras,
    redact_stream_url,
    replace_camera_userinfo,
    remove_extra_camera_host,
    resolve_camera_url,
    save_extra_camera_host,
)
from .database import (
    BANGKOK,
    DatabaseRepository,
    LiveScanPersister,
    build_scan_report,
    flatten_scans_to_passages,
    group_scans_by_operator,
    jsonable_value,
    merge_history_operators,
    resolve_stored_path,
)
from .service import ScanService
from .worker import WorkerPool
from .realtime import event_hub, publish_roi_event, serve_websocket

STATIC_DIR = Path(__file__).resolve().parent / "web_static"
ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}
IMAGE_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}
DEFAULT_MAX_UPLOAD_MB = 4096
LOGGER = logging.getLogger(__name__)


def _max_upload_bytes() -> int:
    raw = os.getenv("CAR_SCAN_WEB_MAX_UPLOAD_MB", str(DEFAULT_MAX_UPLOAD_MB)).strip()
    try:
        megabytes = max(1, int(float(raw)))
    except ValueError:
        megabytes = DEFAULT_MAX_UPLOAD_MB
    return megabytes * 1024 * 1024


def _max_upload_label() -> str:
    megabytes = max(1, _max_upload_bytes() // (1024 * 1024))
    if megabytes >= 1024:
        return f"{megabytes / 1024:g}GB"
    return f"{megabytes}MB"

_JOBS: dict[str, "ScanJob"] = {}
_JOBS_LOCK = threading.Lock()
_SCAN_LOCK = threading.Lock()
_WORKER: WorkerPool | None = None
_WORKER_LOCK = threading.Lock()


def _jsonable(value: Any) -> Any:
    return jsonable_value(value)


def _parse_roi(raw: str | None) -> dict[str, Any] | None:
    if not raw or not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail=f"ROI JSON ไม่ถูกต้อง: {error}") from error
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="ROI ต้องเป็น object")
    enabled_raw = data.get("enabled", True)
    if isinstance(enabled_raw, str):
        enabled = enabled_raw.strip().lower() in ("1", "true", "yes", "on")
    else:
        enabled = bool(enabled_raw)
    x = float(data.get("x", 0.05))
    y = float(data.get("y", 0.10))
    width = float(data.get("width", 0.90))
    height = float(data.get("height", 0.80))
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    width = max(0.01, min(1.0 - x, width))
    height = max(0.01, min(1.0 - y, height))
    return {
        "enabled": enabled,
        "shape": str(data.get("shape") or "rectangle").lower(),
        "unit": "normalized",
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "color": "#f25c05",
        "thickness": 3,
    }


def _coerce_roi(value: Any) -> dict[str, Any] | None:
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        return _parse_roi(json.dumps(value))
    return _parse_roi(str(value))


def _save_database(settings: Settings, result: dict[str, Any], job: ScanJob | None = None) -> dict[str, Any]:
    if job is not None and job.owner_id is not None:
        result.setdefault("operator_id", job.owner_id)
        result.setdefault("operator_name", job.owner_name)
        result.setdefault("operator_username", job.owner_username)
    if settings.database_url:
        try:
            repository = DatabaseRepository(settings.database_url)
            repository.initialize_schema()
            result["database_id"] = repository.save_scan(result, output_dir=settings.output_dir)
            result["database_status"] = "บันทึก PostgreSQL แล้ว"
        except Exception as error:
            LOGGER.exception("PostgreSQL scan save failed")
            result["database_status"] = f"PostgreSQL บันทึกไม่สำเร็จ: {error}"
    else:
        result["database_status"] = "ยังไม่ได้ตั้งค่า PostgreSQL"
    return result


def _parse_report_range(raw_from: str | None, raw_to: str | None) -> tuple[date, date]:
    today = datetime.now(BANGKOK).date()
    start = _parse_scan_day(raw_from) if raw_from and str(raw_from).strip() else today - timedelta(days=6)
    end = _parse_scan_day(raw_to) if raw_to and str(raw_to).strip() else today
    if end < start:
        raise HTTPException(status_code=400, detail="วันสิ้นสุดต้องไม่ก่อนวันเริ่มต้น")
    if (end - start).days > 92:
        raise HTTPException(status_code=400, detail="ช่วงรายงานไม่เกิน 93 วัน")
    return start, end


def _report_payload(
    request: Request,
    raw_from: str | None,
    raw_to: str | None,
    operator_id: str | None,
) -> dict[str, Any]:
    start, end = _parse_report_range(raw_from, raw_to)
    selected_operator, unassigned_only = _parse_operator_filter(operator_id)
    settings = Settings.from_env()
    empty = {
        "from": start.isoformat(),
        "to": end.isoformat(),
        "vehicle_count": 0,
        "scan_count": 0,
        "operator_count": 0,
        "days": [],
        "hours": [{"hour": hour, "count": 0} for hour in range(24)],
        "operators": [],
        "countries": [],
        "vehicles": [],
        "provinces": [],
        "media": [],
        "confidence": [],
        "passages": [],
        "truncated": False,
        "filter_operators": [],
        "database": "missing" if not settings.database_url else "ok",
    }
    if not settings.database_url:
        return empty
    try:
        repository = DatabaseRepository(settings.database_url)
        rows = repository.list_scans_in_range(
            start,
            end,
            selected_operator,
            unassigned_only=unassigned_only,
        )
        payload = build_scan_report(rows)
        roster = [public_user(item) for item in request.app.state.auth.list_users()]
        payload["from"] = start.isoformat()
        payload["to"] = end.isoformat()
        payload["filter_operators"] = merge_history_operators(repository.list_scan_operators(), roster)
        payload["database"] = "ok"
        payload["_rows"] = rows
        return payload
    except HTTPException:
        raise
    except Exception as error:
        empty["database"] = f"error: {error}"
        return empty


def _parse_scan_day(raw: str | None) -> date:
    if not raw or not str(raw).strip():
        return datetime.now(BANGKOK).date()
    try:
        return date.fromisoformat(str(raw).strip())
    except ValueError as error:
        raise HTTPException(status_code=400, detail="วันที่ต้องเป็น YYYY-MM-DD") from error


def _parse_operator_filter(raw: str | None) -> tuple[int | None, bool]:
    if not raw or not str(raw).strip():
        return None, False
    value = str(raw).strip()
    if value in {"unassigned", "none"}:
        return None, True
    try:
        return int(value), False
    except ValueError as error:
        raise HTTPException(status_code=400, detail="operator_id ไม่ถูกต้อง") from error


def _safe_scan_image(settings: Settings, raw: str | None) -> Path | None:
    path = resolve_stored_path(raw, settings.output_dir)
    if path is None:
        return None
    try:
        path.relative_to(settings.output_dir.resolve())
    except (OSError, ValueError):
        return None
    if path.suffix.lower() not in IMAGE_TYPES or not path.is_file():
        return None
    return path


def _public_saved_plate(settings: Settings, scan_id: int, plate: dict[str, Any]) -> dict[str, Any]:
    vehicle = _safe_scan_image(settings, plate.get("full_vehicle_image"))
    crop = _safe_scan_image(settings, plate.get("crop_image"))
    payload = {
        key: value
        for key, value in plate.items()
        if key not in {"full_vehicle_image", "crop_image"}
    }
    plate_id = plate.get("id")
    payload["has_vehicle"] = vehicle is not None
    payload["has_crop"] = crop is not None
    payload["vehicle_url"] = (
        f"/api/scans/{scan_id}/plates/{plate_id}/vehicle" if plate_id is not None and vehicle is not None else None
    )
    payload["crop_url"] = (
        f"/api/scans/{scan_id}/plates/{plate_id}/crop" if plate_id is not None and crop is not None else None
    )
    return payload


def _public_saved_scan(settings: Settings, row: dict[str, Any]) -> dict[str, Any]:
    scan_id = int(row["id"])
    payload = dict(row)
    image = _safe_scan_image(settings, row.get("annotated_image"))
    payload["has_image"] = image is not None
    payload["image_url"] = f"/api/scans/{scan_id}/image" if image is not None else None
    payload["plates"] = [_public_saved_plate(settings, scan_id, plate) for plate in row.get("plates") or []]
    payload.pop("annotated_image", None)
    return payload


def _plate_from_scan(row: dict[str, Any] | None, plate_id: int) -> dict[str, Any] | None:
    if not row:
        return None
    for plate in row.get("plates") or []:
        if plate.get("id") == plate_id:
            return plate
    return None


def _scan_repository(settings: Settings) -> DatabaseRepository:
    repository = DatabaseRepository(settings.database_url)
    repository.initialize_schema()
    return repository


def _serve_plate_image(request: Request, scan_id: int, plate_id: int, kind: str) -> FileResponse:
    require_user(request)
    settings = Settings.from_env()
    if not settings.database_url:
        raise HTTPException(status_code=404, detail="ยังไม่ได้ตั้งค่าฐานข้อมูล")
    try:
        row = _scan_repository(settings).get_scan(scan_id)
    except Exception as error:
        raise HTTPException(status_code=404, detail=f"ไม่พบภาพ: {error}") from error
    plate = _plate_from_scan(row, plate_id)
    if plate is None:
        raise HTTPException(status_code=404, detail="ไม่พบป้ายในรายการนี้")
    path = _safe_scan_image(settings, plate.get("full_vehicle_image" if kind == "vehicle" else "crop_image"))
    if path is None and kind == "vehicle":
        path = _safe_scan_image(settings, (row or {}).get("annotated_image"))
    if path is None:
        raise HTTPException(status_code=404, detail="ไม่มีรูปรถของรายการนี้" if kind == "vehicle" else "ไม่มีรูปป้ายของรายการนี้")
    return FileResponse(path, media_type=IMAGE_TYPES[path.suffix.lower()], filename=path.name)


def _serve_worker_plate_image(request: Request, host: str, plate_id: int, kind: str) -> FileResponse:
    user = require_user(request)
    settings = Settings.from_env()
    _require_camera_view_or_scan(settings, user, host)
    raw = _worker().plate_image(host, plate_id, kind)
    path = _safe_scan_image(settings, raw)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail="ไม่มีรูปรถของป้ายนี้" if kind == "vehicle" else "ไม่มีรูปครอปของป้ายนี้",
        )
    return FileResponse(path, media_type=IMAGE_TYPES[path.suffix.lower()], filename=path.name)


@dataclass
class ScanJob:
    id: str
    media_type: str
    path: Path
    settings: Settings
    filename: str
    owner_id: int | None = None
    owner_name: str = ""
    owner_username: str = ""
    source_url: str = ""
    status: str = "queued"
    message: str = ""
    frame_index: int = 0
    frame_total: int = 0
    preview_jpeg: bytes | None = None
    plates: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    public_result: dict[str, Any] | None = None
    files: dict[str, Path] = field(default_factory=dict)
    error: str | None = None
    stop: threading.Event = field(default_factory=threading.Event)
    updated: threading.Event = field(default_factory=threading.Event)
    seq: int = 0

    def bump(self, message: str | None = None) -> None:
        if message is not None:
            self.message = message
        self.seq += 1
        self.updated.set()

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "media_type": self.media_type,
            "filename": self.filename,
            "owner_id": self.owner_id,
            "status": self.status,
            "message": self.message,
            "frame_index": self.frame_index,
            "frame_total": self.frame_total,
            "seq": self.seq,
            "error": self.error,
            "has_preview": self.preview_jpeg is not None,
            "plates": self.plates,
            "result": self.public_result,
        }


def _register_file(job: ScanJob, path: Any) -> str | None:
    file_path = Path(str(path or ""))
    if not file_path.is_file():
        return None
    file_id = uuid.uuid4().hex
    job.files[file_id] = file_path
    suffix = quote(file_path.suffix.lstrip("."))
    return f"/api/jobs/{job.id}/files/{file_id}?ext={suffix}"


def _public_plate(job: ScanJob, plate: dict[str, Any]) -> dict[str, Any]:
    ocr = plate.get("ocr", {}) if isinstance(plate.get("ocr"), dict) else {}
    crop = (
        plate.get("character_annotated_image")
        or plate.get("ocr_ready_image")
        or plate.get("crop_image")
        or ""
    )
    return {
        "id": plate.get("id"),
        "country": plate.get("country"),
        "province": plate.get("province"),
        "plate_prefix": plate.get("plate_prefix"),
        "plate_prefix_code": plate.get("plate_prefix_code"),
        "plate_number": plate.get("plate_number"),
        "province_code": plate.get("province_code"),
        "vehicle_type": plate.get("vehicle_type"),
        "recognition_confidence": float(plate.get("recognition_confidence", ocr.get("confidence", 0.0)) or 0.0),
        "ocr": {
            "text": ocr.get("text"),
            "method": ocr.get("method"),
            "confidence": ocr.get("confidence"),
        },
        "crop_url": _register_file(job, crop),
        "full_vehicle_url": _register_file(job, plate.get("full_vehicle_image")),
    }


def _public_result(job: ScanJob, result: dict[str, Any]) -> dict[str, Any]:
    plates = [_public_plate(job, plate) for plate in result.get("plates", []) if isinstance(plate, dict)]
    job.plates = plates
    return {
        "media_type": result.get("media_type"),
        "plate_count": result.get("plate_count", len(plates)),
        "rejected_plate_count": result.get("rejected_plate_count", 0),
        "sampled_frames": result.get("sampled_frames"),
        "cancelled": result.get("cancelled"),
        "database_status": result.get("database_status"),
        "annotated_image": _register_file(job, result.get("annotated_image")),
        "output_video": _register_file(job, result.get("output_video")),
        "plates": plates,
    }


def _encode_preview(frame: Any, settings: Settings) -> bytes | None:
    try:
        import cv2
    except Exception:
        return None
    preview = ScanService(settings)._prepare_preview_frame(frame)
    ok, buffer = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        return None
    return bytes(buffer)


def _worker() -> WorkerPool:
    global _WORKER
    with _WORKER_LOCK:
        settings = Settings.from_env()
        if _WORKER is None:
            _WORKER = WorkerPool(settings, busy=None)
        else:
            _WORKER.configure(settings)
        return _WORKER


def reset_runtime_state() -> None:
    global _WORKER
    with _WORKER_LOCK:
        worker = _WORKER
        _WORKER = None
    if worker is not None:
        worker.stop_all()


def _shutdown_runtime_state() -> None:
    """Flush camera persistence before Uvicorn/Prisma shuts down."""

    global _WORKER
    with _WORKER_LOCK:
        worker = _WORKER
        _WORKER = None
    if worker is not None:
        worker.persist_rois()
        worker.stop_all(wait=True)
    try:
        from .prisma_db import disconnect

        disconnect()
    except Exception:
        LOGGER.exception("Unable to close Prisma client cleanly")


def _resolve_camera(settings: Settings, raw: str) -> tuple[str, str]:
    cameras = _camera_urls(settings)
    camera_url = resolve_camera_url(raw, cameras)
    local = parse_local_camera(raw) or parse_local_camera(camera_url)
    if local is not None:
        return local_camera_url(local[1]), local[0]
    if not is_camera_stream_url(camera_url):
        raise HTTPException(status_code=400, detail="เลือกกล้องจากรายการ หรือใส่ IP/RTSP หรือกล้องคอมนี้")
    host = urlsplit(camera_url).hostname or raw
    return camera_url, host


def _operator_fields(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "operator_id": int(user["id"]),
        "operator_name": str(user.get("display_name") or user.get("username") or ""),
        "operator_username": str(user.get("username") or ""),
    }


def _camera_pairs(settings: Settings) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for url in _camera_urls(settings):
        local = parse_local_camera(url)
        if local is not None:
            pairs.append((url, local[0]))
            continue
        host = urlsplit(url).hostname or ""
        if host:
            pairs.append((url, host))
    return pairs


def _start_lane(
    url: str,
    host: str,
    user: dict[str, Any],
    scan: bool = True,
    index: int | None = None,
    roi: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if index is None:
        listed = next((item for item in _camera_payload(Settings.from_env()) if item["host"] == host), None)
        if listed and isinstance(listed.get("index"), int):
            index = int(listed["index"])
    try:
        return _worker().start_camera(
            url,
            host,
            scan=scan,
            index=index,
            roi=roi,
            **_operator_fields(user),
        )
    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error) or (
                "ไม่สามารถเปิดกล้องคอมพิวเตอร์ได้ กรุณาอนุญาตกล้องของเครื่องนี้"
                if parse_local_camera(url or host)
                else "ไม่สามารถเปิดกล้อง IP ได้ กรุณาตรวจ IP และบัญชีกล้อง"
            ),
        ) from error


def _open_lane(host: str):
    lane = _worker().get(host)
    if lane is None or not (lane.opened or lane.jpeg):
        raise HTTPException(status_code=404, detail="ยังไม่ได้เปิดกล้อง")
    return lane


def _live_lane(host: str):
    worker = _worker()
    lane = worker.get(host) if host else None
    if lane is None and not host and len(worker.hub.lanes) == 1:
        lane = next(iter(worker.hub.lanes.values()))
    if lane is None or not (lane.opened or lane.jpeg):
        raise HTTPException(status_code=404, detail="ยังไม่ได้เปิดกล้อง")
    return lane


def _iter_mjpeg(lane: Any):
    last = -1
    misses = 0
    while not lane.stop.is_set():
        with lane.lock:
            seq = lane.seq
            jpeg = lane.jpeg
            opened = lane.opened
        if last != -1 and not opened:
            break
        if jpeg and seq != last:
            last = seq
            misses = 0
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                + jpeg
                + b"\r\n"
            )
        else:
            misses += 1
            if misses > 250:
                break
            time.sleep(0.04)


def _builtin_hosts(settings: Settings) -> set[str]:
    return {
        (urlsplit(url).hostname or "").lower()
        for url in settings.ip_camera_urls
        if urlsplit(url).hostname
    }


def _camera_host(url: str, fallback: str = "") -> str:
    local = parse_local_camera(url) or parse_local_camera(fallback)
    if local is not None:
        return local[0]
    return urlsplit(url).hostname or str(fallback or "").strip()


def _camera_records(settings: Settings) -> list[dict[str, Any]]:
    """Read camera settings from PostgreSQL and migrate the legacy JSON store."""

    legacy = load_extra_cameras(settings.output_dir)
    if not settings.database_url:
        return [
            {
                "host": item.get("host", ""),
                "stream_url": item.get("url", ""),
                "label": item.get("label", ""),
                "kind": "local" if parse_local_camera(item.get("url", "") or item.get("host", "")) else "ip",
                "device_index": (parse_local_camera(item.get("url", "") or item.get("host", "")) or ("", None))[1],
                "enabled": True,
            }
            for item in legacy
        ]
    try:
        repository = DatabaseRepository(settings.database_url)
        repository.ensure_camera_storage()
        rows = repository.list_cameras(enabled_only=False)
        known = {str(item["host"]).lower() for item in rows}
        first_database_boot = not rows

        # Persist server-configured cameras and old file-based entries once.
        seeds: list[tuple[str, str, str, str, int | None]] = []
        for url in settings.ip_camera_urls:
            local = parse_local_camera(url)
            seeds.append(
                (
                    _camera_host(url),
                    url,
                    "local" if local else "ip",
                    "",
                    local[1] if local else None,
                )
            )
        if first_database_boot:
            for item in legacy:
                raw_url = str(item.get("url") or "")
                raw_host = str(item.get("host") or "")
                local = parse_local_camera(raw_url or raw_host)
                seeds.append(
                    (
                        raw_host,
                        raw_url,
                        "local" if local else "ip",
                        str(item.get("label") or ""),
                        local[1] if local else None,
                    )
                )
        for host, url, kind, label, device_index in seeds:
            if not host or host.lower() in known:
                continue
            repository.upsert_camera(
                host,
                url,
                label if label else None,
                kind=kind,
                device_index=device_index,
            )
            known.add(host.lower())
        return repository.list_cameras(enabled_only=True)
    except Exception:
        LOGGER.exception("Unable to read CCTV settings from PostgreSQL; using legacy file")
        return [
            {
                "host": item.get("host", ""),
                "stream_url": item.get("url", ""),
                "label": item.get("label", ""),
                "kind": "ip",
                "device_index": None,
                "enabled": True,
            }
            for item in legacy
        ]


def _camera_urls(settings: Settings) -> tuple[str, ...]:
    urls = list(settings.ip_camera_urls)
    records = _camera_records(settings)
    by_host = {_camera_host(url).lower(): index for index, url in enumerate(urls) if _camera_host(url)}
    for item in records:
        host = str(item.get("host") or "").strip()
        stored = str(item.get("stream_url") or item.get("url") or "").strip()
        local = parse_local_camera(stored) or parse_local_camera(host)
        candidate = local_camera_url(local[1]) if local is not None else stored
        if not candidate or not (is_camera_stream_url(candidate) or parse_local_camera(candidate)):
            continue
        key = host.lower() or _camera_host(candidate).lower()
        if key in by_host:
            urls[by_host[key]] = candidate
        else:
            by_host[key] = len(urls)
            urls.append(candidate)
    return tuple(urls)


def _save_camera_record(
    settings: Settings,
    host: str,
    url: str = "",
    label: str | None = None,
) -> None:
    local = parse_local_camera(url) or parse_local_camera(host)
    if settings.database_url:
        try:
            repository = DatabaseRepository(settings.database_url)
            repository.ensure_camera_storage()
            repository.upsert_camera(
                host,
                url,
                label,
                kind="local" if local else "ip",
                device_index=local[1] if local else None,
            )
            return
        except Exception:
            LOGGER.exception("Unable to save CCTV setting to PostgreSQL")
            raise HTTPException(status_code=503, detail="ไม่สามารถบันทึกการตั้งค่ากล้องลง PostgreSQL ได้")
    save_extra_camera_host(settings.output_dir, host, url, label=label)


def _delete_camera_record(settings: Settings, host: str) -> None:
    if settings.database_url:
        try:
            repository = DatabaseRepository(settings.database_url)
            repository.ensure_camera_storage()
            repository.delete_camera(host)
            return
        except Exception:
            LOGGER.exception("Unable to delete CCTV setting from PostgreSQL")
            raise HTTPException(status_code=503, detail="ไม่สามารถลบการตั้งค่ากล้องจาก PostgreSQL ได้")
    remove_extra_camera_host(settings.output_dir, host)


def _camera_permissions(settings: Settings, user: dict[str, Any]) -> dict[str, dict[str, bool]] | None:
    """Read shared camera permissions; administrators retain full access."""

    if canonical_role(user.get("role")) in {"admin", "superuser"}:
        return None
    if not settings.database_url:
        # A standalone/file-backed deployment has no shared access table. Keep
        # the existing role-level camera permission for operators, but never
        # grant camera access to viewer accounts implicitly.
        return None if has_permission(user, "scan.camera") else {}
    try:
        return DatabaseRepository(settings.database_url).list_camera_access(int(user["id"]))
    except Exception:
        LOGGER.exception("Unable to read shared CCTV permissions")
        return {}


def _require_camera_access(
    settings: Settings,
    user: dict[str, Any],
    host: str,
    capability: str,
) -> dict[str, bool] | None:
    permissions = _camera_permissions(settings, user)
    if permissions is None:
        return None
    permission = permissions.get(str(host or "").strip().lower())
    if not permission or not permission.get(capability, False):
        action = "ดูภาพ" if capability == "can_view" else "สแกน"
        raise HTTPException(status_code=403, detail=f"บัญชีนี้ไม่มีสิทธิ์{action}กล้อง {host}")
    return permission


def _camera_has_access(settings: Settings, user: dict[str, Any], host: str, capability: str) -> bool:
    permissions = _camera_permissions(settings, user)
    return permissions is None or bool(permissions.get(str(host or "").strip().lower(), {}).get(capability, False))


def _require_camera_view_or_scan(settings: Settings, user: dict[str, Any], host: str) -> None:
    permissions = _camera_permissions(settings, user)
    if permissions is None:
        return
    permission = permissions.get(str(host or "").strip().lower())
    if not permission or not (permission.get("can_view") or permission.get("can_scan")):
        raise HTTPException(status_code=403, detail=f"บัญชีนี้ไม่มีสิทธิ์เข้าถึงกล้อง {host}")


def _require_camera_stop_access(settings: Settings, user: dict[str, Any], host: str) -> None:
    if has_permission(user, "scan.camera"):
        return
    _require_camera_access(settings, user, host, "can_view")
    lane = _worker().get(host)
    if lane is None or int(getattr(lane, "operator_id", -1) or -1) != int(user["id"]):
        raise HTTPException(status_code=403, detail=f"บัญชีนี้ไม่ใช่ผู้เปิด preview ของกล้อง {host}")


def _camera_payload(settings: Settings, user: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cameras = public_cameras(_camera_urls(settings), _builtin_hosts(settings))
    records = _camera_records(settings)
    labels = {str(item["host"]).lower(): item.get("label", "") for item in records}
    rois = {str(item["host"]).lower(): item.get("roi") for item in records if item.get("roi")}
    for camera in cameras:
        custom_label = labels.get(camera["host"].lower()) or ""
        camera["label"] = custom_label or camera["label"]
        camera["custom_label"] = bool(custom_label)
        saved_roi = rois.get(camera["host"].lower())
        if isinstance(saved_roi, dict):
            camera["roi"] = saved_roi
    if user is None:
        return cameras
    permissions = _camera_permissions(settings, user)
    for camera in cameras:
        permission = permissions.get(str(camera["host"]).lower(), {}) if permissions is not None else {}
        camera["can_view"] = permissions is None or bool(permission.get("can_view", False))
        camera["can_scan"] = permissions is None or bool(permission.get("can_scan", False))
    return [camera for camera in cameras if camera.get("can_view")]


def _filter_worker_snapshot(
    snapshot: dict[str, Any],
    settings: Settings,
    user: dict[str, Any],
) -> dict[str, Any]:
    """Remove worker lanes and plates outside the current user's camera scope."""

    visible = {str(item["host"]).lower() for item in _camera_payload(settings, user)}
    result = dict(snapshot)
    cameras = [
        item for item in snapshot.get("cameras") or []
        if str(item.get("host") or "").lower() in visible
    ]
    result["cameras"] = cameras
    result["plates"] = [
        item for item in snapshot.get("plates") or []
        if str(item.get("camera_host") or "").lower() in visible
    ]
    return result


def _camera_response(settings: Settings, cameras: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cameras": cameras,
        "storage": "database" if settings.database_url else "file",
    }


def _run_job(job: ScanJob) -> None:
    if not _SCAN_LOCK.acquire(blocking=False):
        job.status = "error"
        job.error = "มีงานสแกนอื่นกำลังทำงานอยู่ กรุณารอให้เสร็จก่อน"
        job.bump(job.error)
        return
    try:
        job.status = "running"
        job.bump("กำลังโหลดโมเดลและสแกน...")
        service = ScanService(job.settings)
        persister = LiveScanPersister(
            database_url=job.settings.database_url,
            output_dir=job.settings.output_dir,
            media_type=job.media_type,
            source=redact_stream_url(job.source_url) if job.media_type == "camera" else str(job.path),
            operator_id=job.owner_id,
            operator_name=job.owner_name,
            operator_username=job.owner_username,
        )

        def on_preview(frame: Any, index: int, total: int = 0) -> None:
            job.frame_index = int(index)
            job.frame_total = int(total)
            encoded = _encode_preview(frame, job.settings)
            if encoded:
                job.preview_jpeg = encoded
            job.bump(f"กำลังสแกนเฟรม {index}" + (f"/{total}" if total else ""))

        def on_plate(plate: dict[str, Any]) -> None:
            saved = persister.save_plate(plate) if job.media_type in {"video", "camera"} else None
            public = _public_plate(job, _jsonable(plate) if isinstance(plate, dict) else {})
            existing = next(
                (
                    item
                    for item in job.plates
                    if item.get("country") == public.get("country")
                    and item.get("plate_prefix") == public.get("plate_prefix")
                    and item.get("plate_number") == public.get("plate_number")
                ),
                None,
            )
            if existing is None:
                job.plates.append(public)
            else:
                existing.update(public)
            label = f"{public.get('plate_prefix') or ''} {public.get('plate_number') or ''}".strip()
            if saved is not None:
                job.bump(f"บันทึกทะเบียน {label or 'ที่ยืนยันแล้ว'}")
            else:
                job.bump("พบทะเบียนที่ยืนยันแล้ว")

        if job.media_type == "video":
            result = service.scan_video(
                job.path,
                frame_callback=lambda frame, index, total: on_preview(frame, index, total),
                plate_callback=on_plate,
                stop_requested=job.stop.is_set,
            )
            result = persister.finish(result)
        else:
            result = service.scan_file(job.path)
            result = _save_database(job.settings, result, job)
        job.result = result
        job.public_result = _public_result(job, _jsonable(result))
        job.status = "done"
        job.bump(str(result.get("database_status") or "สแกนเสร็จ"))
    except Exception as error:
        job.status = "error"
        job.error = str(error)
        job.bump("สแกนไม่สำเร็จ")
    finally:
        _SCAN_LOCK.release()


def _get_job(job_id: str, user: dict[str, Any] | None = None) -> ScanJob:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ไม่พบงานสแกน")
    if user is not None and job.owner_id not in (None, int(user["id"])) and canonical_role(user.get("role")) not in {"admin", "superuser"}:
        raise HTTPException(status_code=403, detail="ไม่มีสิทธิ์ดูงานสแกนนี้")
    return job


def create_app() -> FastAPI:
    max_upload = _max_upload_bytes()
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await run_in_threadpool(_shutdown_runtime_state)

    app = FastAPI(title="Car Scan", version="0.1.0", lifespan=lifespan)
    app.state.auth = AuthStore.from_env()
    app.add_middleware(RequestBodyLimitMiddleware, max_body_size=max_upload)
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.exception_handler(413)
    async def content_too_large(_request: Request, _exc: Exception) -> JSONResponse:
        return JSONResponse(
            {"detail": f"ไฟล์ใหญ่เกิน {_max_upload_label()} ตั้ง CAR_SCAN_WEB_MAX_UPLOAD_MB หากต้องการรับไฟล์ใหญ่ขึ้น"},
            status_code=413,
        )

    @app.get("/", response_model=None)
    def index(request: Request) -> FileResponse | RedirectResponse:
        if current_user(request) is None:
            return RedirectResponse("/login", status_code=302)
        page = STATIC_DIR / "index.html"
        if not page.is_file():
            raise HTTPException(status_code=500, detail="ไม่พบหน้าเว็บ")
        return FileResponse(page)

    @app.get("/login", response_model=None)
    def login_page(request: Request) -> FileResponse | RedirectResponse:
        if current_user(request) is not None:
            return RedirectResponse("/", status_code=302)
        page = STATIC_DIR / "login.html"
        if not page.is_file():
            raise HTTPException(status_code=500, detail="ไม่พบหน้าเข้าสู่ระบบ")
        return FileResponse(page)

    @app.get("/api/health")
    def health(request: Request) -> dict[str, Any]:
        user = require_user(request)
        settings = Settings.from_env()
        cameras = _camera_payload(settings, user)
        database = "missing"
        if settings.database_url:
            try:
                DatabaseRepository(settings.database_url).ping()
                database = "ok"
            except Exception as error:
                database = f"error: {error}"
        worker = _filter_worker_snapshot(_worker().snapshot(), settings, user)
        gpu = worker.get("gpu") or {}
        return {
            "ok": True,
            "app": "car-scan-web",
            "database": database,
            "output_dir": str(settings.output_dir),
            "max_upload": _max_upload_label(),
            "ip_camera": bool(cameras),
            "ip_camera_host": cameras[0]["host"] if cameras else "",
            "ip_cameras": cameras,
            "camera_storage": "database" if settings.database_url else "file",
            "worker": worker.get("worker"),
            "gpu": gpu.get("gpu") or "CPU",
            "gpu_ready": bool(gpu.get("ready")),
            "gpu_loading": bool(gpu.get("loading")),
            "compute": gpu.get("requested") or gpu.get("mode") or "auto",
            "compute_yolo": gpu.get("yolo_device") or "",
            "compute_ocr": gpu.get("ocr_device") or "",
            "compute_hybrid": bool(gpu.get("hybrid")),
            "active_cameras": int(gpu.get("active_cameras") or 0),
        }

    @app.websocket("/ws")
    async def onevision_websocket(websocket: WebSocket):
        store: AuthBackend = websocket.app.state.auth
        token = websocket.cookies.get(COOKIE_NAME)
        user = store.read_session(token)
        session_id = store.session_id(token)
        if user is None or not session_id:
            await websocket.close(code=1008)
            return
        await serve_websocket(
            websocket,
            user_id=int(user["id"]),
            session_id=session_id,
            worker_snapshot_filter=lambda snapshot: _filter_worker_snapshot(
                snapshot,
                Settings.from_env(),
                user,
            ),
            camera_event_filter=lambda host: _camera_has_access(
                Settings.from_env(),
                user,
                host,
                "can_view",
            ),
        )

    @app.get("/api/cameras")
    def list_cameras(request: Request) -> dict[str, Any]:
        user = require_user(request)
        settings = Settings.from_env()
        return _camera_response(settings, _camera_payload(settings, user))

    @app.post("/api/cameras")
    async def add_camera(request: Request) -> JSONResponse:
        user = require_user(request)
        if not has_permission(user, "scan.camera"):
            raise HTTPException(status_code=403, detail="บัญชีนี้ไม่มีสิทธิ์เพิ่มกล้อง")
        try:
            payload = await request.json()
        except Exception as error:
            raise HTTPException(status_code=400, detail="ต้องส่ง host เป็น JSON") from error
        settings = Settings.from_env()
        raw = str(payload.get("host") or payload.get("url") or "").strip()
        username = str(payload.get("username") or payload.get("user") or "").strip()
        password = str(payload.get("password") or payload.get("pass") or "")
        path = str(payload.get("path") or payload.get("rtsp_path") or "").strip()
        label = str(payload.get("label") or "").strip()
        if len(label) > 80:
            raise HTTPException(status_code=400, detail="à¸Šà¸·à¹ˆà¸­à¸à¸¥à¹‰à¸­à¸‡à¸¢à¸²à¸§à¹„à¸› (à¸ªà¸¹à¸‡à¸ªà¸¸à¸” 80 à¸•à¸±à¸§à¸­à¸±à¸à¸©à¸£)")
        if not raw:
            raise HTTPException(status_code=400, detail="ใส่ IP ของกล้อง หรือกดเพิ่มกล้องคอมนี้")
        local = parse_local_camera(raw)
        if local is not None:
            host = local[0]
            _save_camera_record(settings, host, local_camera_url(local[1]), label or None)
            cameras = _camera_payload(settings)
            return JSONResponse({"host": host, **_camera_response(settings, cameras)})
        known = _camera_urls(settings) or settings.ip_camera_urls
        template = known[0] if known else ""
        resolved = compose_camera_url(
            raw,
            username=username,
            password=password,
            template=template,
            path=path,
        )
        if not is_camera_stream_url(resolved):
            raise HTTPException(
                status_code=400,
                detail="ใส่ IP เช่น 10.0.75.24, URL rtsp/http, พร้อม user/password ของกล้องนั้น",
            )
        host = urlsplit(resolved).hostname or ""
        _save_camera_record(settings, host, resolved, label or None)
        cameras = _camera_payload(settings)
        return JSONResponse({"host": host, **_camera_response(settings, cameras)})

    @app.put("/api/cameras")
    async def update_camera(request: Request, host: str = Query("")) -> JSONResponse:
        user = require_user(request)
        if not has_permission(user, "scan.camera"):
            raise HTTPException(status_code=403, detail="à¸šà¸±à¸à¸Šà¸µà¸™à¸µà¹‰à¹„à¸¡à¹ˆà¸¡à¸µà¸ªà¸´à¸—à¸˜à¸´à¹Œà¹à¸à¹‰à¹„à¸‚à¸à¸¥à¹‰à¸­à¸‡")
        try:
            payload = await request.json()
        except Exception as error:
            raise HTTPException(status_code=400, detail="à¸‚à¹‰à¸­à¸¡à¸¹à¸™à¸à¸¥à¹‰à¸­à¸‡à¹„à¸¡à¹ˆà¸–à¸¹à¸à¸•à¹‰à¸­à¸‡") from error
        settings = Settings.from_env()
        wanted = host.strip()
        camera = next((item for item in _camera_payload(settings) if item["host"].lower() == wanted.lower()), None)
        if not camera:
            raise HTTPException(status_code=404, detail="à¹„à¸¡à¹ˆà¸žà¸šà¸à¸¥à¹‰à¸­à¸‡à¸™à¸µà¹‰")
        label = str(payload.get("label") or "").strip()
        if len(label) > 80:
            raise HTTPException(status_code=400, detail="à¸Šà¸·à¹ˆà¸­à¸à¸¥à¹‰à¸­à¸‡à¸¢à¸²à¸§à¹„à¸› (à¸ªà¸¹à¸‡à¸ªà¸¸à¸” 80 à¸•à¸±à¸§à¸­à¸±à¸à¸©à¸£)")
        new_url = str(payload.get("stream_url") or "").strip()
        if new_url:
            current = next(
                (
                    url for url in _camera_urls(settings)
                    if ((urlsplit(url).hostname or "").lower() == wanted.lower())
                    or ((parse_local_camera(url) or ("", 0))[0].lower() == wanted.lower())
                ),
                "",
            )
            local = parse_local_camera(new_url)
            if local:
                if local[0].lower() != wanted.lower():
                    raise HTTPException(status_code=400, detail="à¸£à¸«à¸±à¸ªà¸à¸¥à¹‰à¸­à¸‡à¸„à¸­à¸¡à¸žà¸´à¸§à¹€à¸•à¸­à¸£à¹Œà¸•à¹‰à¸­à¸‡à¸„à¸‡à¹€à¸”à¸´à¸¡")
            elif not is_camera_stream_url(new_url) or (urlsplit(new_url).hostname or "").lower() != wanted.lower():
                raise HTTPException(status_code=400, detail="URL à¸•à¹‰à¸­à¸‡à¸Šà¸µà¹‰à¹„à¸›à¸¢à¸±à¸‡ host à¸‚à¸­à¸‡à¸à¸¥à¹‰à¸­à¸‡à¸™à¸µà¹‰")
            else:
                parsed_current = urlsplit(current)
                if parsed_current.username and not urlsplit(new_url).username:
                    new_url = replace_camera_userinfo(
                        new_url,
                        unquote(parsed_current.username),
                        unquote(parsed_current.password or ""),
                    )
            if new_url == current:
                new_url = ""
            else:
                _worker().stop_camera(wanted)
        _save_camera_record(settings, wanted, new_url, label)
        return JSONResponse({"host": wanted, **_camera_response(settings, _camera_payload(settings))})

    @app.delete("/api/cameras")
    def remove_camera(request: Request, host: str = Query("")) -> JSONResponse:
        user = require_user(request)
        if not has_permission(user, "scan.camera"):
            raise HTTPException(status_code=403, detail="บัญชีนี้ไม่มีสิทธิ์ลบกล้อง")
        host = host.strip()
        if not host:
            raise HTTPException(status_code=400, detail="ระบุ IP กล้องที่ต้องการลบ")
        settings = Settings.from_env()
        if host.lower() in _builtin_hosts(settings):
            raise HTTPException(status_code=400, detail="กล้องนี้ตั้งในเซิร์ฟเวอร์ ลบจากรายการเพิ่มไม่ได้")
        _worker().stop_camera(host)
        _delete_camera_record(settings, host)
        return JSONResponse({"host": host, **_camera_response(settings, _camera_payload(settings))})

    @app.get("/api/worker")
    def worker_status(request: Request) -> dict[str, Any]:
        user = require_user(request)
        settings = Settings.from_env()
        return _filter_worker_snapshot(_worker().snapshot(), settings, user)

    @app.post("/api/worker/compute")
    async def worker_set_compute(request: Request) -> JSONResponse:
        user = require_user(request)
        if not has_permission(user, "scan.camera"):
            raise HTTPException(status_code=403, detail="บัญชีนี้ไม่มีสิทธิ์เปลี่ยน GPU/CPU")
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        mode = str((payload or {}).get("mode") or (payload or {}).get("compute") or "").strip()
        if not mode:
            raise HTTPException(status_code=400, detail="เลือก auto, gpu, cpu หรือ hybrid")
        snapshot = await run_in_threadpool(_worker().set_compute, mode)
        return JSONResponse(_filter_worker_snapshot(snapshot, Settings.from_env(), user))

    @app.post("/api/worker/start")
    async def worker_start_camera(request: Request) -> JSONResponse:
        user = require_user(request)
        if not has_permission(user, "scan.camera"):
            raise HTTPException(status_code=403, detail="บัญชีนี้ไม่มีสิทธิ์เปิดกล้อง")
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        settings = Settings.from_env()
        raw = str((payload or {}).get("host") or (payload or {}).get("url") or "").strip()
        camera_url, host = _resolve_camera(settings, raw)
        _require_camera_access(settings, user, host, "can_scan")
        roi = _coerce_roi((payload or {}).get("roi"))
        status = await run_in_threadpool(lambda: _start_lane(camera_url, host, user, True, None, roi))
        return JSONResponse(status)

    @app.post("/api/worker/start-all")
    async def worker_start_all(request: Request) -> JSONResponse:
        user = require_user(request)
        if not has_permission(user, "scan.camera"):
            raise HTTPException(status_code=403, detail="บัญชีนี้ไม่มีสิทธิ์เปิดกล้อง")
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        settings = Settings.from_env()
        cameras = _camera_pairs(settings)
        permissions = _camera_permissions(settings, user)
        if permissions is not None:
            cameras = [
                (url, host)
                for url, host in cameras
                if permissions.get(host.lower(), {}).get("can_scan", False)
            ]
        requested_hosts = {
            str(item or "").strip().lower()
            for item in ((payload or {}).get("hosts") or [])
            if str(item or "").strip()
        }
        if requested_hosts:
            cameras = [(url, host) for url, host in cameras if host.lower() in requested_hosts]
        if not cameras:
            raise HTTPException(status_code=400, detail="ยังไม่มีกล้องในรายการ")
        fields = _operator_fields(user)
        roi = _coerce_roi((payload or {}).get("roi"))
        raw_rois = (payload or {}).get("rois") if isinstance(payload, dict) else None
        rois = None
        if isinstance(raw_rois, dict):
            rois = {str(key): _coerce_roi(value) for key, value in raw_rois.items() if _coerce_roi(value)}

        def boot() -> dict[str, Any]:
            return _worker().start_all(cameras, scan=True, roi=roi, rois=rois, **fields)

        snapshot = await run_in_threadpool(boot)
        return JSONResponse(_filter_worker_snapshot(snapshot, settings, user))

    @app.post("/api/worker/roi")
    async def worker_set_roi(request: Request) -> JSONResponse:
        user = require_user(request)
        if not has_permission(user, "scan.camera"):
            raise HTTPException(status_code=403, detail="บัญชีนี้ไม่มีสิทธิ์ตั้งกรอบสแกน")
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        host = str((payload or {}).get("host") or "").strip()
        if not host:
            raise HTTPException(status_code=400, detail="ระบุ IP กล้องที่ต้องการตั้งกรอบสแกน")
        _require_camera_access(Settings.from_env(), user, host, "can_scan")
        roi = _coerce_roi((payload or {}).get("roi"))
        worker = _worker()
        result = await run_in_threadpool(worker.set_roi, host, roi)
        publish_roi_event(host, roi, user=user)
        return JSONResponse(result)

    @app.post("/api/worker/stop")
    async def worker_stop_camera(request: Request, host: str = Query("")) -> dict[str, Any]:
        user = require_user(request)
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        target = str(host or (payload or {}).get("host") or "").strip()
        if not target:
            raise HTTPException(status_code=400, detail="ระบุ IP กล้องที่ต้องการปิด")
        _require_camera_stop_access(Settings.from_env(), user, target)
        return await run_in_threadpool(_worker().stop_camera, target)

    @app.post("/api/worker/stop-all")
    async def worker_stop_all(request: Request) -> dict[str, Any]:
        user = require_user(request)
        if not has_permission(user, "scan.camera"):
            raise HTTPException(status_code=403, detail="บัญชีนี้ไม่มีสิทธิ์ปิดการสแกนกล้องทั้งหมด")
        await run_in_threadpool(_worker().stop_all)
        return {"ok": True, "cameras": []}

    @app.get("/api/worker/cameras/{host}/plates/{plate_id}/crop")
    def worker_plate_crop(host: str, plate_id: int, request: Request) -> FileResponse:
        return _serve_worker_plate_image(request, host, plate_id, "crop")

    @app.get("/api/worker/cameras/{host}/plates/{plate_id}/vehicle")
    def worker_plate_vehicle(host: str, plate_id: int, request: Request) -> FileResponse:
        return _serve_worker_plate_image(request, host, plate_id, "vehicle")

    @app.get("/api/cameras/live")
    def live_status(request: Request, host: str = Query("")) -> dict[str, Any]:
        user = require_user(request)
        settings = Settings.from_env()
        if host:
            _require_camera_access(settings, user, host, "can_view")
        snapshot = _filter_worker_snapshot(_worker().snapshot(), settings, user)
        if host:
            lane = next((item for item in snapshot["cameras"] if item.get("host") == host), None)
            if lane is None:
                raise HTTPException(status_code=404, detail="ยังไม่ได้เปิดกล้อง")
            return lane
        if len(snapshot["cameras"]) == 1:
            return {**snapshot["cameras"][0], **snapshot}
        return snapshot

    @app.get("/api/cameras/live.jpg")
    def live_frame(request: Request, host: str = Query("")) -> Response:
        user = require_user(request)
        settings = Settings.from_env()
        lane = _live_lane(host)
        _require_camera_access(settings, user, lane.host or host, "can_view")
        jpeg = lane.jpeg
        if not jpeg:
            raise HTTPException(status_code=404, detail="ยังไม่ได้เปิดกล้อง")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/cameras/live.mjpeg")
    def live_mjpeg(request: Request, host: str = Query("")) -> StreamingResponse:
        user = require_user(request)
        settings = Settings.from_env()
        lane = _live_lane(host)
        _require_camera_access(settings, user, lane.host or host, "can_view")
        return StreamingResponse(
            _iter_mjpeg(lane),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/cameras/live")
    async def open_live(request: Request) -> JSONResponse:
        user = require_user(request)
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        settings = Settings.from_env()
        raw = str((payload or {}).get("host") or (payload or {}).get("url") or "").strip()
        camera_url, host = _resolve_camera(settings, raw)
        _require_camera_access(settings, user, host, "can_view")
        roi = _coerce_roi((payload or {}).get("roi"))
        status = await run_in_threadpool(lambda: _start_lane(camera_url, host, user, False, None, roi))
        return JSONResponse(status)

    @app.post("/api/cameras/live/stop")
    async def stop_live(request: Request, host: str = Query("")) -> dict[str, Any]:
        user = require_user(request)
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        target = str(host or (payload or {}).get("host") or "").strip()
        if target:
            _require_camera_stop_access(Settings.from_env(), user, target)
            return await run_in_threadpool(_worker().stop_camera, target)
        if not has_permission(user, "scan.camera"):
            raise HTTPException(status_code=403, detail="บัญชีนี้ไม่มีสิทธิ์ปิดกล้องทั้งหมด")
        await run_in_threadpool(_worker().stop_all)
        return {"ok": True, "opened": False}

    @app.get("/api/cameras/live/snapshot.jpg")
    def live_snapshot(request: Request, host: str = Query("")) -> Response:
        user = require_user(request)
        settings = Settings.from_env()
        lane = _live_lane(host)
        _require_camera_access(settings, user, lane.host or host, "can_view")
        jpeg = lane.snapshot_jpeg()
        if not jpeg:
            raise HTTPException(status_code=404, detail="ยังไม่มีภาพจากกล้อง")
        return Response(
            jpeg,
            media_type="image/jpeg",
            headers={"Content-Disposition": f'attachment; filename="{lane.host or host or "camera"}.jpg"'},
        )

    @app.post("/api/cameras/live/record")
    def start_live_record(request: Request, host: str = Query("")) -> dict[str, Any]:
        user = require_user(request)
        if not has_permission(user, "scan.camera"):
            raise HTTPException(status_code=403, detail="บัญชีนี้ไม่มีสิทธิ์อัดกล้อง")
        settings = Settings.from_env()
        lane = _open_lane(host) if host else None
        if lane is None:
            lanes = list(_worker().hub.lanes.values())
            lane = lanes[0] if len(lanes) == 1 else None
        if lane is None:
            raise HTTPException(status_code=404, detail="ยังไม่ได้เปิดกล้อง")
        _require_camera_access(settings, user, lane.host or host, "can_scan")
        try:
            path = lane.start_recording()
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"ok": True, "recording": True, "host": lane.host, "file": path.name}

    @app.post("/api/cameras/live/record/stop")
    def stop_live_record(request: Request, host: str = Query("")) -> FileResponse:
        user = require_user(request)
        settings = Settings.from_env()
        lane = _open_lane(host) if host else None
        if lane is None:
            lanes = [item for item in _worker().hub.lanes.values() if item.recording]
            lane = lanes[0] if lanes else None
        if lane is None:
            raise HTTPException(status_code=404, detail="ยังไม่ได้เปิดกล้อง")
        _require_camera_access(settings, user, lane.host or host, "can_scan")
        try:
            path = lane.stop_recording()
        except RuntimeError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return FileResponse(path, filename=path.name)

    @app.post("/api/auth/login")
    async def login(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception as error:
            raise HTTPException(status_code=400, detail="ต้องส่ง username และ password เป็น JSON") from error
        username = str(payload.get("username") or "")
        password = str(payload.get("password") or "")
        store: AuthBackend = request.app.state.auth
        user = store.authenticate(username, password)
        token = store.issue_session(user)
        session_id = store.session_id(token)
        if session_id:
            event_hub.close_user_sessions(int(user["id"]), keep_session_id=session_id)
        response = JSONResponse(public_user(user))
        return set_session_cookie(response, token)

    @app.post("/api/auth/logout")
    def logout(request: Request) -> JSONResponse:
        store: AuthBackend = request.app.state.auth
        token = request.cookies.get(COOKIE_NAME)
        user = store.read_session(token)
        store.revoke_session(token)
        if user is not None:
            event_hub.close_user_sessions(int(user["id"]))
        return clear_session_cookie(JSONResponse({"ok": True}))

    @app.get("/api/auth/me")
    def me(request: Request) -> dict[str, Any]:
        return public_user(require_user(request))

    @app.get("/api/auth/catalog")
    def auth_catalog(request: Request) -> dict[str, Any]:
        require_user(request)
        store: AuthBackend = request.app.state.auth
        return store.list_catalog()

    @app.post("/api/auth/password")
    async def change_password(request: Request) -> dict[str, Any]:
        user = require_user(request)
        try:
            payload = await request.json()
        except Exception as error:
            raise HTTPException(status_code=400, detail="ต้องส่งรหัสผ่านเป็น JSON") from error
        current = str(payload.get("current_password") or "")
        new_password = str(payload.get("new_password") or "")
        stored = request.app.state.auth.get_by_id(int(user["id"]))
        if stored is None or not verify_password(current, str(stored.get("password_hash") or "")):
            raise HTTPException(status_code=400, detail="รหัสผ่านปัจจุบันไม่ถูกต้อง")
        request.app.state.auth.update_user(int(user["id"]), password=new_password)
        return {"ok": True}

    @app.get("/api/users")
    def list_users(request: Request) -> dict[str, Any]:
        require_permission(request, "users.manage")
        store: AuthBackend = request.app.state.auth
        return {"users": [public_user(user) for user in store.list_users()]}

    @app.post("/api/users")
    async def create_user(request: Request) -> dict[str, Any]:
        require_permission(request, "users.manage")
        payload = await request.json()
        store: AuthBackend = request.app.state.auth
        user = store.create_user(
            str(payload.get("username") or ""),
            str(payload.get("display_name") or ""),
            str(payload.get("role") or "operator"),
            str(payload.get("password") or ""),
        )
        return public_user(user)

    @app.patch("/api/users/{user_id}")
    async def update_user(user_id: int, request: Request) -> dict[str, Any]:
        require_permission(request, "users.manage")
        payload = await request.json()
        store: AuthBackend = request.app.state.auth
        user = store.update_user(
            user_id,
            display_name=str(payload["display_name"]) if "display_name" in payload else None,
            role=str(payload["role"]) if "role" in payload else None,
            password=str(payload["password"]) if payload.get("password") else None,
            active=bool(payload["active"]) if "active" in payload else None,
        )
        if any(key in payload for key in ("role", "password", "active")):
            store.revoke_user_sessions(user_id)
            event_hub.close_user_sessions(user_id)
        return public_user(user)

    @app.delete("/api/users/{user_id}")
    def delete_user(user_id: int, request: Request) -> dict[str, Any]:
        actor = require_permission(request, "users.manage")
        if int(actor["id"]) == user_id:
            raise HTTPException(status_code=400, detail="ลบบัญชีของตนเองไม่ได้")
        store: AuthBackend = request.app.state.auth
        store.revoke_user_sessions(user_id)
        event_hub.close_user_sessions(user_id)
        store.delete_user(user_id)
        return {"ok": True}

    @app.get("/api/reports")
    def get_report(
        request: Request,
        from_: str | None = Query(None, alias="from"),
        to: str | None = None,
        operator_id: str | None = None,
    ) -> dict[str, Any]:
        require_user(request)
        payload = _report_payload(request, from_, to, operator_id)
        payload.pop("_rows", None)
        return payload

    @app.get("/api/reports.csv")
    def download_report_csv(
        request: Request,
        from_: str | None = Query(None, alias="from"),
        to: str | None = None,
        operator_id: str | None = None,
    ) -> Response:
        require_user(request)
        payload = _report_payload(request, from_, to, operator_id)
        rows = payload.pop("_rows", [])
        passages = flatten_scans_to_passages(rows)
        buffer = io.StringIO()
        buffer.write("\ufeff")
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "date",
                "time",
                "operator_name",
                "operator_username",
                "country",
                "vehicle_type",
                "plate_prefix",
                "plate_number",
                "province",
                "media_type",
                "confidence",
            ]
        )
        for item in passages:
            writer.writerow(
                [
                    item.get("date") or "",
                    item.get("scanned_time") or "",
                    item.get("operator_name") or "",
                    item.get("operator_username") or "",
                    item.get("country") or "",
                    item.get("vehicle_type") or "",
                    item.get("plate_prefix") or "",
                    item.get("plate_number") or "",
                    item.get("province") or "",
                    item.get("media_type") or "",
                    item.get("confidence") or 0,
                ]
            )
        filename = f"car-scan-report-{payload.get('from')}_{payload.get('to')}.csv"
        return Response(
            buffer.getvalue().encode("utf-8"),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/scans")
    def list_scans(
        request: Request,
        day: str | None = None,
        operator_id: str | None = None,
        q: str | None = None,
    ) -> dict[str, Any]:
        require_user(request)
        settings = Settings.from_env()
        query = str(q or "").strip()
        selected = None if query and not (day and str(day).strip()) else _parse_scan_day(day)
        selected_operator, unassigned_only = _parse_operator_filter(operator_id)
        empty = {
            "date": selected.isoformat() if selected else None,
            "query": query,
            "groups": [],
            "days": [],
            "operators": [],
            "total": 0,
            "database": "missing" if not settings.database_url else "ok",
        }
        if not settings.database_url:
            return empty
        try:
            repository = _scan_repository(settings)
            if query:
                rows = repository.search_scans(
                    query,
                    selected_operator,
                    unassigned_only=unassigned_only,
                )
            else:
                rows = repository.list_scans_for_day(
                    selected or _parse_scan_day(None),
                    selected_operator,
                    unassigned_only=unassigned_only,
                )
            public_rows = [_public_saved_scan(settings, row) for row in rows]
            passages = flatten_scans_to_passages(public_rows, query)
            roster = [public_user(item) for item in request.app.state.auth.list_users()]
            return {
                "date": selected.isoformat() if selected else None,
                "query": query,
                "groups": group_scans_by_operator(passages),
                "days": repository.list_scan_days(),
                "operators": merge_history_operators(repository.list_scan_operators(), roster),
                "total": len(passages),
                "database": "ok",
            }
        except Exception as error:
            empty["database"] = f"error: {error}"
            return empty

    @app.get("/api/scans/{scan_id}")
    def get_scan(scan_id: int, request: Request) -> dict[str, Any]:
        require_user(request)
        settings = Settings.from_env()
        if not settings.database_url:
            raise HTTPException(status_code=404, detail="ยังไม่ได้ตั้งค่าฐานข้อมูล")
        try:
            row = _scan_repository(settings).get_scan(scan_id)
        except Exception as error:
            raise HTTPException(status_code=404, detail=f"ไม่พบรายการสแกน: {error}") from error
        if row is None:
            raise HTTPException(status_code=404, detail="ไม่พบรายการสแกน")
        return _public_saved_scan(settings, row)

    @app.get("/api/scans/{scan_id}/plates/{plate_id}/vehicle")
    def get_plate_vehicle(scan_id: int, plate_id: int, request: Request) -> FileResponse:
        return _serve_plate_image(request, scan_id, plate_id, "vehicle")

    @app.get("/api/scans/{scan_id}/plates/{plate_id}/crop")
    def get_plate_crop(scan_id: int, plate_id: int, request: Request) -> FileResponse:
        return _serve_plate_image(request, scan_id, plate_id, "crop")

    @app.get("/api/scans/{scan_id}/image")
    def get_scan_image(scan_id: int, request: Request) -> FileResponse:
        require_user(request)
        settings = Settings.from_env()
        if not settings.database_url:
            raise HTTPException(status_code=404, detail="ยังไม่ได้ตั้งค่าฐานข้อมูล")
        try:
            row = _scan_repository(settings).get_scan(scan_id)
        except Exception as error:
            raise HTTPException(status_code=404, detail=f"ไม่พบภาพหลักฐาน: {error}") from error
        path = _safe_scan_image(settings, (row or {}).get("annotated_image"))
        if path is None:
            raise HTTPException(status_code=404, detail="ไม่มีภาพหลักฐานของรายการนี้")
        return FileResponse(path, media_type=IMAGE_TYPES[path.suffix.lower()], filename=path.name)

    @app.post("/api/jobs")
    async def create_job(request: Request) -> JSONResponse:
        user = require_user(request)
        form = await request.form(max_files=1, max_fields=20, max_part_size=_max_upload_bytes())
        upload = form.get("file")
        if isinstance(upload, (list, tuple)):
            upload = upload[0] if upload else None
        media_type = str(form.get("media_type") or "image").strip().lower()
        roi = str(form.get("roi") or "") or None
        if media_type not in {"image", "video", "camera"}:
            raise HTTPException(status_code=400, detail="media_type ต้องเป็น image, video หรือ camera")
        if not has_permission(user, scan_permission(media_type)):
            raise HTTPException(status_code=403, detail="บัญชีนี้ไม่มีสิทธิ์ทำรายการนี้")
        if media_type != "camera" and _SCAN_LOCK.locked():
            raise HTTPException(status_code=409, detail="มีงานสแกนอื่นกำลังทำงานอยู่")

        settings = Settings.from_env()
        roi_config = _parse_roi(roi)
        if roi_config is not None:
            settings = replace(settings, scan_roi_override=roi_config)
        job_id = uuid.uuid4().hex[:12]
        upload_dir = settings.output_dir / "web_uploads" / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        if media_type == "camera":
            camera_ref = str(form.get("camera_url") or form.get("camera_host") or "").strip()
            camera_url, host = _resolve_camera(settings, camera_ref)
            _require_camera_access(settings, user, host, "can_scan")
            status = await run_in_threadpool(lambda: _start_lane(camera_url, host, user, True, None, roi_config))
            snapshot = _filter_worker_snapshot(_worker().snapshot(), settings, user)
            return JSONResponse(
                {
                    "id": f"cam-{host}",
                    "media_type": "camera",
                    "status": "running",
                    "camera": status,
                    **snapshot,
                },
                status_code=202,
            )

        if not isinstance(upload, UploadFile):
            raise HTTPException(status_code=400, detail="ต้องแนบไฟล์ในฟิลด์ file")
        original = Path(upload.filename or "upload")
        suffix = original.suffix.lower()
        allowed = ALLOWED_IMAGE if media_type == "image" else ALLOWED_VIDEO
        if suffix not in allowed:
            raise HTTPException(status_code=400, detail=f"ชนิดไฟล์ไม่รองรับ: {suffix or '(ไม่มีนามสกุล)'}")
        stored = upload_dir / f"input{suffix}"
        size = 0
        limit = _max_upload_bytes()
        with stored.open("wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    handle.close()
                    stored.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"ไฟล์ใหญ่เกิน {_max_upload_label()} ตั้ง CAR_SCAN_WEB_MAX_UPLOAD_MB หากต้องการรับไฟล์ใหญ่ขึ้น",
                    )
                handle.write(chunk)
        if size == 0:
            stored.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="ไฟล์ว่าง")
        await upload.close()

        job = ScanJob(
            id=job_id,
            media_type=media_type,
            path=stored,
            settings=settings,
            filename=original.name,
            owner_id=int(user["id"]),
            owner_name=str(user.get("display_name") or user.get("username") or ""),
            owner_username=str(user.get("username") or ""),
        )
        with _JOBS_LOCK:
            _JOBS[job_id] = job
            if len(_JOBS) > 30:
                oldest = next(iter(_JOBS))
                if oldest != job_id:
                    _JOBS.pop(oldest, None)
        threading.Thread(target=_run_job, args=(job,), daemon=True, name=f"scan-{job_id}").start()
        return JSONResponse(job.snapshot(), status_code=202)

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str, request: Request) -> dict[str, Any]:
        return _get_job(job_id, require_user(request)).snapshot()

    @app.post("/api/jobs/{job_id}/stop")
    def stop_job(job_id: str, request: Request) -> dict[str, Any]:
        user = require_user(request)
        job = _get_job(job_id, user)
        permission = scan_permission(job.media_type)
        if not has_permission(user, permission):
            raise HTTPException(status_code=403, detail="บัญชีนี้ไม่มีสิทธิ์หยุดงานสแกน")
        if job.status == "running":
            job.stop.set()
            job.status = "stopping"
            job.bump("กำลังหยุดสแกน...")
        return job.snapshot()

    @app.get("/api/jobs/{job_id}/preview")
    def job_preview(job_id: str, request: Request) -> Response:
        job = _get_job(job_id, require_user(request))
        if not job.preview_jpeg:
            raise HTTPException(status_code=404, detail="ยังไม่มีภาพพรีวิว")
        return Response(job.preview_jpeg, media_type="image/jpeg")

    @app.get("/api/jobs/{job_id}/files/{file_id}")
    def job_file(job_id: str, file_id: str, request: Request) -> FileResponse:
        job = _get_job(job_id, require_user(request))
        path = job.files.get(file_id)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="ไม่พบไฟล์ผลลัพธ์")
        return FileResponse(path, filename=path.name)

    @app.get("/api/jobs/{job_id}/events")
    def job_events(job_id: str, request: Request) -> StreamingResponse:
        job = _get_job(job_id, require_user(request))

        def stream():
            last = -1
            while True:
                if job.seq != last:
                    last = job.seq
                    payload = json.dumps(job.snapshot(), ensure_ascii=False)
                    yield f"event: update\ndata: {payload}\n\n"
                    if job.status in {"done", "error"}:
                        yield f"event: {job.status}\ndata: {payload}\n\n"
                        return
                job.updated.clear()
                job.updated.wait(timeout=1.0)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()


def main() -> int:
    try:
        import uvicorn
    except ImportError:
        print("ติดตั้ง FastAPI/Uvicorn ก่อน: pip install fastapi uvicorn python-multipart", file=sys.stderr)
        return 1
    host = os.getenv("CAR_SCAN_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("CAR_SCAN_WEB_PORT", "8000"))
    graceful_shutdown = max(1, int(os.getenv("CAR_SCAN_WEB_GRACEFUL_SHUTDOWN_SECONDS", "3")))
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        timeout_graceful_shutdown=graceful_shutdown,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
