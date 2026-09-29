"""Central GPU AI worker that scans many live CCTV cameras together."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import cv2

from .config import Settings, parse_local_camera, local_camera_url, redact_stream_url, rtsp_url_candidates, save_extra_camera_host
from .compute import normalise_compute_mode, resolve_compute, save_compute_mode
from .database import DatabaseRepository, LiveScanPersister
from .realtime import publish_worker_event
from .service import LiveScanSession, ScanService

LOGGER = logging.getLogger(__name__)
INFER_MAX_DIMENSION = 1920
PREVIEW_FPS = 12.0
PREVIEW_MAX_DIMENSION = 640
PREVIEW_QUALITY = 62
FIRST_FRAME_READ_ATTEMPTS = 4
FIRST_FRAME_READ_DELAY_SECONDS = 0.20
STOP_JOIN_TIMEOUT_SECONDS = 0.05
SHUTDOWN_JOIN_TIMEOUT_SECONDS = 1.5
# Capture submits at most this often; the hub already keeps only the latest
# pending frame per lane, so extra decodes only make the live view hitch.
IDLE_SUBMIT_GAP = 0.28
HOT_SUBMIT_GAP = 0.12
# A lane stays "hot" this long after it last saw an unconfirmed plate. Hot
# lanes submit frames more often and are inferred first; idle lanes keep a
# slower heartbeat so a passing car gets several samples instead of one or two.
HOT_LANE_SECONDS = 4.0
IDLE_LANE_INTERVAL = 0.25
_LIVE_PLATE_FILE_KEYS = ("crop_image", "full_vehicle_image", "ocr_ready_image")


def same_live_registration(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Compare plate text without relying on a detector/track id."""

    def compact(value: Any) -> str:
        return "".join(str(value or "").strip().lower().split()).replace("-", "").replace("_", "").replace(".", "")

    first_key = (compact(first.get("plate_prefix")), compact(first.get("plate_number")))
    second_key = (compact(second.get("plate_prefix")), compact(second.get("plate_number")))
    return bool(first_key[0] or first_key[1]) and first_key == second_key


def live_plate_label(record: dict[str, Any]) -> str:
    country = str(record.get("country") or "")
    flag = "ไทย" if country == "thai" else "ลาว" if country == "lao" else ""
    text = f"{record.get('plate_prefix') or ''} {record.get('plate_number') or ''}".strip()
    if flag and text:
        return f"{flag} {text}"
    return text or flag


def live_public_plate(
    record: dict[str, Any],
    *,
    host: str,
    label: str,
) -> dict[str, Any]:
    """Same crop/OCR payload the video job panel expects, plus CCTV identity."""

    ocr = record.get("ocr") if isinstance(record.get("ocr"), dict) else {}
    live_id = int(record.get("id") or 0)
    try:
        image_revision = int(
            record.get("image_revision")
            or record.get("camera_confirmed_frame")
            or record.get("video_confirmed_frame")
            or 0
        )
    except (TypeError, ValueError):
        image_revision = 0
    captured_at = record.get("captured_at")
    try:
        captured_at = float(captured_at)
    except (TypeError, ValueError):
        captured_at = 0.0
    return {
        "id": live_id,
        "camera_host": host,
        "camera_label": label,
        "country": record.get("country"),
        "province": record.get("province"),
        "plate_prefix": record.get("plate_prefix"),
        "plate_number": record.get("plate_number"),
        "vehicle_type": record.get("vehicle_type"),
        "detection_confidence": float(record.get("detection_confidence") or 0.0),
        "recognition_confidence": float(record.get("recognition_confidence") or ocr.get("confidence") or 0.0),
        "overall_confidence": float(record.get("overall_confidence") or 0.0),
        "temporal_average_quality": float(record.get("temporal_average_quality") or 0.0),
        "crop_visual_quality": float(record.get("crop_visual_quality") or 0.0),
        "best_selection_score": float(record.get("best_selection_score") or 0.0),
        "camera_first_frame": record.get("camera_first_frame", record.get("video_first_frame")),
        "camera_confirmed_frame": record.get("camera_confirmed_frame", record.get("video_confirmed_frame")),
        "camera_last_frame": record.get("camera_last_frame", record.get("video_last_frame")),
        "camera_occurrences": int(record.get("camera_occurrences", record.get("video_occurrences", 0)) or 0),
        "camera_track_observations": int(record.get("camera_track_observations", record.get("video_track_observations", 0)) or 0),
        "camera_vote_score": float(record.get("camera_vote_score", record.get("video_vote_score", 0.0)) or 0.0),
        "validation": record.get("validation") if isinstance(record.get("validation"), dict) else {},
        "confirmed": bool(record.get("confirmed")),
        "captured_at": captured_at,
        "ocr": {
            "text": ocr.get("text") or "",
            "raw_text": ocr.get("raw_text") or ocr.get("text") or "",
            "method": ocr.get("method") or "",
            "confidence": float(ocr.get("confidence") or 0.0),
            "province": ocr.get("province") or record.get("province") or "",
        },
        "crop_url": f"/api/worker/cameras/{host}/plates/{live_id}/crop?v={image_revision}",
        "full_vehicle_url": f"/api/worker/cameras/{host}/plates/{live_id}/vehicle?v={image_revision}",
        "crop_image": str(record.get("crop_image") or ""),
        "full_vehicle_image": str(record.get("full_vehicle_image") or ""),
        "ocr_ready_image": str(record.get("ocr_ready_image") or ""),
    }


def client_live_plate(plate: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plate.items() if key not in _LIVE_PLATE_FILE_KEYS}


def detect_gpu_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return "CPU"


def _shrink(frame: Any, max_dimension: int = INFER_MAX_DIMENSION) -> Any:
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        return frame
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_dimension:
        return frame
    scale = max_dimension / float(longest)
    return cv2.resize(
        frame,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _encode_jpeg(frame: Any, max_dimension: int = 640, quality: int = 70) -> bytes | None:
    image = _shrink(frame, max_dimension)
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return bytes(buffer)


def preview_budget(camera_count: int, settings: Settings | None = None) -> tuple[float, int, int]:
    """Return preview FPS, max edge, and JPEG quality for the live wall.

    Encoding a 20 fps 640px JPEG on every capture thread is what makes
    cameras 05–08 hitch once several IP streams are open together.
    """

    count = max(1, int(camera_count or 1))
    fps = float(getattr(settings, "preview_fps", PREVIEW_FPS) or PREVIEW_FPS)
    dim = int(getattr(settings, "preview_max_dimension", PREVIEW_MAX_DIMENSION) or PREVIEW_MAX_DIMENSION)
    if count >= 7:
        return max(6.0, min(fps, 8.0)), min(dim, 480), 55
    if count >= 5:
        return max(7.0, min(fps, 10.0)), min(dim, 520), 58
    if count >= 3:
        return max(8.0, min(fps, 12.0)), min(dim, 640), 60
    return min(max(fps, 8.0), 15.0), min(dim, 720), int(PREVIEW_QUALITY)


def _set_ffmpeg_rtsp_options(transport: str) -> None:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;{transport}|fflags;nobuffer|flags;low_delay|"
        "max_delay;250000|stimeout;2500000"
    )


def _configure_capture(capture: Any) -> None:
    try:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass


def _open_stream(url: str) -> Any:
    capture = cv2.VideoCapture()
    open_timeout = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
    read_timeout = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
    if open_timeout is not None:
        capture.set(open_timeout, 2500)
    if read_timeout is not None:
        capture.set(read_timeout, 2500)
    backend = cv2.CAP_FFMPEG if hasattr(cv2, "CAP_FFMPEG") else cv2.CAP_ANY
    if not capture.open(url, backend):
        capture.release()
        return None
    if not capture.isOpened():
        capture.release()
        return None
    _configure_capture(capture)
    return capture


def _read_first_frame(capture: Any) -> Any:
    """Give RTSP decoders a few reads to receive SPS/PPS and the first frame."""

    for attempt in range(FIRST_FRAME_READ_ATTEMPTS):
        try:
            ok, frame = capture.read()
        except (MemoryError, SystemError):
            ok, frame = False, None
        if ok and frame is not None:
            return frame
        if attempt + 1 < FIRST_FRAME_READ_ATTEMPTS:
            time.sleep(FIRST_FRAME_READ_DELAY_SECONDS)
    return None


def _open_capture(url: str) -> tuple[Any, str, Any]:
    local = parse_local_camera(url)
    if local is not None:
        index = local[1]
        backend = cv2.CAP_DSHOW if os.name == "nt" and hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        capture = cv2.VideoCapture(index, backend)
        if not capture.isOpened() and backend != cv2.CAP_ANY:
            capture.release()
            capture = cv2.VideoCapture(index)
        if capture.isOpened():
            _configure_capture(capture)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            frame = _read_first_frame(capture)
            if frame is not None:
                return capture, url, frame
            capture.release()
            return cv2.VideoCapture(), url, None
        return capture, url, None

    for candidate in rtsp_url_candidates(url):
        _set_ffmpeg_rtsp_options("tcp")
        capture = _open_stream(candidate)
        if capture is None:
            continue
        frame = _read_first_frame(capture)
        if frame is not None:
            _configure_capture(capture)
            LOGGER.info("opened camera stream %s over tcp", redact_stream_url(candidate))
            return capture, candidate, frame
        capture.release()
    _set_ffmpeg_rtsp_options("udp")
    capture = _open_stream(url)
    if capture is not None:
        frame = _read_first_frame(capture)
        if frame is not None:
            _configure_capture(capture)
            LOGGER.info("opened camera stream %s over udp", redact_stream_url(url))
            return capture, url, frame
        capture.release()
    return cv2.VideoCapture(), url, None


def _open_video_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[Any, Path] | tuple[None, Path]:
    rate = max(8.0, min(float(fps or 15.0), 25.0))
    for suffix, code in ((".mp4", "mp4v"), (".avi", "MJPG"), (".avi", "XVID")):
        candidate = path.with_suffix(suffix)
        writer = cv2.VideoWriter(str(candidate), cv2.VideoWriter_fourcc(*code), rate, size)
        if writer.isOpened():
            return writer, candidate
        writer.release()
    return None, path


def camera_label(index: int, host: str) -> str:
    if parse_local_camera(host):
        return "กล้องคอม"
    return f"Camera {index:02d}"


@dataclass
class CameraLane:
    host: str
    url: str
    index: int
    settings: Settings
    hub: "GpuHub"
    operator_id: int | None = None
    operator_name: str = ""
    operator_username: str = ""
    stop: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    ready: threading.Event = field(default_factory=threading.Event)
    jpeg: bytes | None = None
    frame: Any = None
    seq: int = 0
    error: str = ""
    opened: bool = False
    starting: bool = False
    stopping: bool = False
    scanning: bool = True
    recording: bool = False
    record_path: Path | None = None
    writer: Any = None
    plates: list[dict[str, Any]] = field(default_factory=list)
    last_plate: str = ""
    frame_index: int = 0
    sampled_frames: int = 0
    capture_thread: threading.Thread | None = None
    capture: Any = None
    roi: dict[str, Any] | None = None
    last_detect_count: int = 0
    fps: float = 0.0
    # Monotonic deadline while this lane is tracking a plate that has not
    # been confirmed yet; the hub and the capture loop sample it more often.
    hot_until: float = 0.0
    last_infer_at: float = 0.0
    session_finished: bool = False

    @property
    def hot(self) -> bool:
        return time.monotonic() < self.hot_until

    def _lane_count(self) -> int:
        try:
            return max(1, len(self.hub.lanes))
        except Exception:
            return 1

    def __post_init__(self) -> None:
        self.session = LiveScanSession(
            last_rejected_evidence_frame=-self.settings.rejected_evidence_frame_gap,
        )
        self.session_stem = f"camera_{self.host}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.persister = LiveScanPersister(
            database_url=self.settings.database_url,
            output_dir=self.settings.output_dir,
            media_type="camera",
            source=redact_stream_url(self.url),
            operator_id=self.operator_id,
            operator_name=self.operator_name,
            operator_username=self.operator_username,
        )

    @property
    def label(self) -> str:
        return camera_label(self.index, self.host)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            last_plate = self.last_plate
            if not last_plate:
                cached = self.session.cached_plates
                if cached:
                    plate = cached[-1]
                    last_plate = live_plate_label(plate)
            return {
                "id": f"cam-{self.index:02d}",
                "host": self.host,
                "label": self.label,
                "live": self.opened,
                "starting": self.starting,
                "stopping": self.stopping,
                "scanning": self.scanning and self.opened,
                "recording": self.recording,
                "seq": self.seq,
                "error": self.error,
                "frame_index": self.frame_index,
                "plate_count": len(self.plates),
                "last_plate": last_plate,
                "sampled_frames": self.session.sampled_frames,
                "last_detect_count": self.session.last_detect_count,
                "plates": [client_live_plate(plate) for plate in self.plates],
                "has_frame": self.jpeg is not None,
                "roi": dict(self.roi) if self.roi else None,
            }

    def start(self) -> None:
        with self.lock:
            self.starting = True
            self.error = ""
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"cctv-{self.host}",
        )
        self.capture_thread.start()

    def _finish_session(self) -> None:
        with self.lock:
            if self.session_finished:
                return
            self.session_finished = True
        try:
            self.persister.finish({"media_type": "camera", "plates": self.plates, "plate_count": len(self.plates)})
        except Exception:
            LOGGER.exception("finish live persist for %s", self.host)

    def stop_lane(self, *, timeout: float | None = None) -> None:
        self.stop.set()
        thread = self.capture_thread
        writer = None
        capture = None
        with self.lock:
            writer = self.writer
            self.writer = None
            capture = self.capture
            self.recording = False
            self.opened = False
            self.starting = False
            self.stopping = True
        if writer is not None:
            writer.release()
        # Release the decoder as well as signalling the loop.  RTSP reads can
        # otherwise remain blocked long enough for a quick off/on action to
        # create a second connection for the same camera.
        if capture is not None:
            try:
                capture.release()
            except Exception:
                LOGGER.debug("release capture while stopping %s", self.host, exc_info=True)
        # Return quickly. A blocking RTSP read or decoder shutdown must not
        # hold the HTTP request; the capture thread cleans up in finally.
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=STOP_JOIN_TIMEOUT_SECONDS if timeout is None else max(0.0, timeout))
        if thread is None or not thread.is_alive():
            with self.lock:
                self.stopping = False
            self._finish_session()

    def start_recording(self) -> Path:
        with self.lock:
            if not self.opened:
                raise RuntimeError("เปิดกล้องก่อนแล้วค่อยอัด")
            if self.recording:
                raise RuntimeError("กำลังอัดคลิปกล้องนี้อยู่แล้ว")
            stamp = time.strftime("%Y%m%d_%H%M%S")
            folder = self.settings.output_dir / "web_uploads" / "live"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{self.host}_{stamp}.mp4"
            self.record_path = path
            self.recording = True
            return path

    def stop_recording(self) -> Path:
        with self.lock:
            path = self.record_path
            writer = self.writer
            self.writer = None
            self.recording = False
            self.record_path = None
        if writer is not None:
            writer.release()
            time.sleep(0.15)
        if path is None or not path.is_file() or path.stat().st_size < 32:
            raise RuntimeError("ยังไม่มีคลิปที่อัดได้")
        return path

    def plate_file(self, plate_id: int, kind: str) -> str:
        with self.lock:
            plate = next(
                (item for item in self.plates if int(item.get("id") or 0) == int(plate_id)),
                None,
            )
            if plate is None:
                return ""
            if kind == "vehicle":
                return str(plate.get("full_vehicle_image") or "")
            return str(plate.get("crop_image") or plate.get("ocr_ready_image") or "")

    def publish_records(self, records: list[dict[str, Any]]) -> None:
        service = self.hub.service
        for record in records:
            record["camera_host"] = self.host
            record["camera_label"] = self.label
            label = live_plate_label(record)
            if not service._has_complete_registration(record):
                continue
            if not record.get("captured_at"):
                record["captured_at"] = time.time()
            public = live_public_plate(record, host=self.host, label=self.label)
            accepted = False
            with self.lock:
                existing = next(
                    (
                        item
                        for item in self.plates
                        if same_live_registration(item, public)
                    ),
                    None,
                )
                if existing is None:
                    existing = next(
                        (
                            item
                            for item in self.plates
                            if int(item.get("id") or 0) == int(public.get("id") or 0)
                        ),
                        None,
                    )
                if existing is not None and not service._should_replace_live_record(existing, public):
                    # Keep the strongest reading for this ROI track. A noisy
                    # frame must not overwrite a clearer plate already shown.
                    continue
                if existing is None:
                    self.plates.append(public)
                else:
                    captured = existing.get("captured_at")
                    existing.update(public)
                    if captured:
                        existing["captured_at"] = captured
                if label:
                    self.last_plate = label
                accepted = True
            if not accepted:
                continue
            saved = None
            try:
                saved = self.persister.save_plate(record)
            except Exception:
                LOGGER.exception("persist plate from %s", self.host)
            if saved is not None:
                LOGGER.info("saved plate %s on %s", label, self.host)
            self.hub.notify_update("plate detected")

    def refresh_published_record(self, record: dict[str, Any]) -> None:
        """Replace an on-screen plate after background OCR finishes."""

        if not self.hub.service._has_complete_registration(record):
            return
        public = live_public_plate(record, host=self.host, label=self.label)
        label = live_plate_label(record)
        with self.lock:
            existing = next(
                (
                    item
                    for item in self.plates
                    if same_live_registration(item, public)
                ),
                None,
            )
            if existing is None:
                existing = next(
                    (
                        item
                        for item in self.plates
                        if int(item.get("id") or 0) == int(public.get("id") or 0)
                    ),
                    None,
                )
            if existing is None:
                self.plates.append(public)
            elif self.hub.service._should_replace_live_record(existing, public):
                captured = existing.get("captured_at")
                existing.update(public)
                if captured:
                    existing["captured_at"] = captured
            else:
                return
            if label:
                self.last_plate = label
        self.hub.service._refresh_plate_manifest(record)
        self.hub.notify_update("plate refined")
        try:
            self.persister.save_plate(record)
        except Exception:
            LOGGER.exception("persist refined OCR for %s", self.host)

    def snapshot_jpeg(self) -> bytes | None:
        with self.lock:
            frame = self.frame
            fallback = self.jpeg
        return _encode_jpeg(frame, max_dimension=1600, quality=86) or fallback

    def _capture_loop(self) -> None:
        try:
            capture, opened_url, first_frame = _open_capture(self.url)
        except Exception as error:
            with self.lock:
                self.starting = False
                self.stopping = False
                self.error = f"เปิดกล้อง {self.host} ไม่ได้: {error}"
            self.ready.set()
            self._finish_session()
            self.hub.notify_update("camera state changed")
            return
        with self.lock:
            self.capture = capture
        if self.stop.is_set():
            capture.release()
            with self.lock:
                self.capture = None
                self.starting = False
                self.stopping = False
            self.ready.set()
            self._finish_session()
            self.hub.notify_update("camera state changed")
            return
        if opened_url and opened_url != self.url:
            self.url = opened_url
            try:
                if self.settings.database_url:
                    repository = DatabaseRepository(self.settings.database_url)
                    repository.ensure_camera_storage()
                    local = parse_local_camera(opened_url)
                    repository.upsert_camera(
                        self.host,
                        opened_url,
                        kind="local" if local else "ip",
                        device_index=local[1] if local else None,
                    )
                else:
                    save_extra_camera_host(self.settings.output_dir, self.host, opened_url)
            except Exception:
                LOGGER.exception("persist working RTSP for %s", self.host)
        if not capture.isOpened() or first_frame is None:
            with self.lock:
                self.error = (
                    "ไม่สามารถเปิดกล้องคอมพิวเตอร์ได้ กรุณาอนุญาตกล้องของเครื่องนี้"
                    if parse_local_camera(self.url or self.host)
                    else (
                        f"เปิดกล้อง {self.host} ไม่ได้ — ตรวจ IP, user/password "
                        "และ path RTSP ของยี่ห้อกล้องนั้น"
                    )
                )
                self.starting = False
                self.stopping = False
            self.ready.set()
            capture.release()
            with self.lock:
                self.capture = None
            self._finish_session()
            self.hub.notify_update("camera state changed")
            return
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 15.0
        if fps > 120.0:
            fps = 25.0
        self.fps = fps
        last_preview = 0.0
        last_keep = 0.0
        last_submit = 0.0
        pending = first_frame
        try:
            while not self.stop.is_set():
                now = time.monotonic()
                preview_fps, preview_dim, quality = preview_budget(self._lane_count(), self.settings)
                preview_gap = 1.0 / max(4.0, preview_fps)
                with self.lock:
                    recording = self.recording
                need_preview = now - last_preview >= preview_gap or not self.ready.is_set()
                need_keep = now - last_keep >= 0.45
                infer_due = self.scanning and (
                    now - last_submit >= (HOT_SUBMIT_GAP if self.hot else IDLE_SUBMIT_GAP)
                )
                if pending is not None:
                    frame = pending
                    pending = None
                elif need_preview or need_keep or recording or infer_due:
                    try:
                        ok, frame = capture.read()
                    except (MemoryError, SystemError) as error:
                        # FFmpeg/OpenCV may surface an allocation failure as a
                        # SystemError (with the original MemoryError attached).
                        # Release the decoder before retrying; keeping it alive
                        # retains its 1080p buffers and makes recovery unlikely.
                        LOGGER.error("capture memory error for %s: %s", self.host, error)
                        capture.release()
                        with self.lock:
                            self.error = "หน่วยความจำไม่พอสำหรับภาพกล้อง; กำลังเปิด stream ใหม่"
                        if self.stop.wait(0.5):
                            break
                        capture, opened_url, replacement = _open_capture(self.url)
                        with self.lock:
                            self.capture = capture
                        if not capture.isOpened() or replacement is None:
                            with self.lock:
                                self.error = "หน่วยความจำไม่พอหรือเปิด stream กล้องใหม่ไม่ได้"
                            break
                        pending = replacement
                        continue
                    if not ok or frame is None:
                        time.sleep(0.02)
                        continue
                else:
                    # Drain the RTSP buffer without JPEG-encoding every frame.
                    if not capture.grab():
                        time.sleep(0.02)
                    continue
                height, width = frame.shape[:2]
                preview = None
                keep_frame = False
                if need_preview:
                    preview = _encode_jpeg(frame, max_dimension=preview_dim, quality=quality)
                    last_preview = now
                if need_keep:
                    keep_frame = True
                    last_keep = now
                writer = None
                with self.lock:
                    self.frame_index += 1
                    index = self.frame_index
                    if keep_frame:
                        # The full-resolution frame is not needed by the live
                        # API (the JPEG preview is the normal response). Keep a
                        # bounded copy so every 1080p camera does not retain a
                        # second ~6 MiB array indefinitely.
                        self.frame = _shrink(
                            frame,
                            max_dimension=max(640, int(self.settings.preview_max_dimension)),
                        ).copy()
                    if preview:
                        self.jpeg = preview
                        self.seq += 1
                    self.opened = True
                    self.starting = False
                    self.error = ""
                    if self.recording:
                        if self.writer is None and self.record_path is not None:
                            opened_writer, path = _open_video_writer(self.record_path, fps, (width, height))
                            self.writer = opened_writer
                            self.record_path = path
                            if opened_writer is None:
                                self.recording = False
                                self.error = "เปิดไฟล์อัดวิดีโอไม่ได้"
                        writer = self.writer
                if writer is not None:
                    writer.write(frame)
                if not self.ready.is_set() and self.jpeg:
                    self.ready.set()
                if infer_due:
                    infer_frame = _shrink(
                        frame,
                        max(640, int(self.settings.camera_infer_max_dimension)),
                    )
                    self.hub.submit(self.host, infer_frame, index, lane=self)
                    last_submit = now
        finally:
            capture.release()
            leftover = None
            with self.lock:
                leftover = self.writer
                self.writer = None
                self.capture = None
                self.opened = False
                self.starting = False
                self.stopping = False
            if leftover is not None:
                leftover.release()
            self._finish_session()
            self.hub.notify_update("camera state changed")
            self.ready.set()


class GpuHub:
    """One GPU-owned inference loop shared by every camera lane."""

    def __init__(
        self,
        settings: Settings,
        busy: Callable[[], bool] | None = None,
        on_update: Callable[[str], None] | None = None,
    ) -> None:
        self.settings = settings
        self.service = ScanService(settings)
        self.busy = busy or (lambda: False)
        self.on_update = on_update
        self.device = "CPU"
        self.ready = False
        self.loading = False
        self.error = ""
        self._load_lock = threading.Lock()
        self.infer_lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.pending: dict[str, tuple[Any, int, Any]] = {}
        self.lanes: dict[str, CameraLane] = {}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="gpu-hub")
        self.ocr_queue: queue.Queue[tuple[CameraLane, dict[str, Any], Any] | None] = queue.Queue()
        self.ocr_thread = threading.Thread(target=self._ocr_loop, daemon=True, name="gpu-ocr")
        self.plan = resolve_compute(settings.compute_mode)
        self.cpu_service: ScanService | None = None
        self.cpu_lock = threading.Lock()
        self.cpu_thread = threading.Thread(target=self._cpu_loop, daemon=True, name="cpu-hub")

    def notify_update(self, reason: str) -> None:
        callback = self.on_update
        if callback is None:
            return
        try:
            callback(reason)
        except Exception:
            LOGGER.exception("worker update notification failed")

    def start(self) -> None:
        self.stop.clear()
        if not self.thread.is_alive():
            self.thread = threading.Thread(target=self._loop, daemon=True, name="gpu-hub")
            self.thread.start()
        if not self.ocr_thread.is_alive():
            self.ocr_thread = threading.Thread(target=self._ocr_loop, daemon=True, name="gpu-ocr")
            self.ocr_thread.start()
        if not self.cpu_thread.is_alive():
            self.cpu_thread = threading.Thread(target=self._cpu_loop, daemon=True, name="cpu-hub")
            self.cpu_thread.start()

    def shutdown(self, timeout: float = SHUTDOWN_JOIN_TIMEOUT_SECONDS) -> None:
        """Stop inference threads before the process closes the Prisma engine."""

        self.stop.set()
        with self.pending_lock:
            self.pending.clear()
        deadline = time.monotonic() + max(0.0, timeout)
        for thread in (self.thread, self.ocr_thread, self.cpu_thread):
            if thread is threading.current_thread() or not thread.is_alive():
                continue
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)

    def ensure_loaded(self) -> None:
        with self._load_lock:
            if self.ready:
                return
            self.loading = True
            try:
                self.service.settings = self.settings
                self.plan = resolve_compute(self.settings.compute_mode)
                self.service.load_runtime()
                self.device = self.plan.gpu_name or detect_gpu_device()
                self.ready = True
                self.error = ""
            except Exception as error:
                self.error = str(error)
                raise
            finally:
                self.loading = False

    def reload_compute(self) -> None:
        """Drop GPU/CPU models so the next load pins the selected devices."""

        with self.infer_lock:
            with self.cpu_lock:
                self.ready = False
                self.loading = True
                self.plan = resolve_compute(self.settings.compute_mode)
                self.device = self.plan.gpu_name or "CPU"
                self.service.settings = self.settings
                self.service.unload_runtime()
                if self.cpu_service is not None:
                    self.cpu_service.unload_runtime()
                    self.cpu_service = None
                self.loading = False

    def ensure_cpu_loaded(self) -> None:
        if not self._using_hybrid():
            return
        with self._load_lock:
            if self.cpu_service is not None and self.cpu_service._scanner is not None:
                return
            from dataclasses import replace

            self.cpu_service = ScanService(replace(self.settings, compute_mode="cpu"))
            self.cpu_service.load_runtime()

    def _using_hybrid(self) -> bool:
        plan = getattr(self, "plan", None)
        if plan is None:
            plan = resolve_compute(self.settings.compute_mode)
            self.plan = plan
        return bool(plan.hybrid_cpu_yolo)

    def submit(self, host: str, frame: Any, frame_index: int, *, lane: Any = None) -> None:
        with self.pending_lock:
            current_lane = self.lanes.get(host)
            if current_lane is None or (lane is not None and current_lane is not lane):
                return
            lane = current_lane if lane is None else lane
            # ``capture.read()`` returns a new ndarray for the next frame, so
            # copying here only doubles the 1080p allocation. The pending map
            # owns the reference until the inference loop consumes/replaces it.
            # Keep the lane identity with the frame. A late frame from an old
            # decoder must not be processed by a newly restarted lane.
            self.pending[host] = (frame, frame_index, lane)

    def materialise(self, candidate: dict[str, Any], record_id: int, stem: str, archive_camera_id: str) -> dict[str, Any] | None:
        write_image = self.service._write_image
        scanner = self.service._scanner
        if write_image is None or scanner is None:
            return None
        with self.infer_lock:
            return self.service._materialise_candidate(
                candidate,
                record_id,
                stem,
                write_image,
                scanner,
                run_full_ocr=True,
                archive_camera_id=archive_camera_id,
            )

    def status(self) -> dict[str, Any]:
        plan = getattr(self, "plan", None) or resolve_compute(self.settings.compute_mode)
        payload = {
            "gpu": self.device or plan.gpu_name or "CPU",
            "ready": self.ready,
            "loading": self.loading,
            "error": self.error,
            "paused": bool(self.busy()),
            "hub_alive": self.thread.is_alive(),
            "active_cameras": sum(1 for lane in self.lanes.values() if lane.opened or lane.starting),
            "ocr": self.service.ocr_engine_status(),
        }
        payload.update(plan.as_dict())
        payload["gpu"] = plan.gpu_name or self.device or "CPU"
        payload["cpu_hub"] = bool(self.cpu_service is not None and self.cpu_service._scanner is not None)
        return payload

    def _next_batch(self, role: str = "primary") -> list[tuple[str, tuple[Any, int, Any]]]:
        """Pick pending frames, hot lanes first, idle lanes throttled.

        The hub is the throughput limit (about one frame per second per
        camera on CPU). Spending that budget on the lane that is currently
        reading a plate yields several samples per passing vehicle, which is
        what temporal voting needs; idle lanes keep a slower heartbeat.
        Hybrid mode lets the GPU keep hot plates while a CPU copy scans idle
        cameras in parallel.
        """

        now = time.monotonic()
        hybrid = role in ("gpu", "cpu") or self._using_hybrid()
        with self.pending_lock:
            hot: list[tuple[str, tuple[Any, int, Any]]] = []
            idle: list[tuple[str, tuple[Any, int, Any]]] = []
            for host, item in list(self.pending.items()):
                lane = self.lanes.get(host)
                submitted_lane = item[2] if len(item) > 2 else None
                if lane is None or (submitted_lane is not None and lane is not submitted_lane):
                    self.pending.pop(host, None)
                    continue
                lane_stop = getattr(lane, "stop", None)
                if lane_stop is not None and lane_stop.is_set():
                    self.pending.pop(host, None)
                    continue
                if lane is not None and lane.hot:
                    hot.append((host, item))
                else:
                    idle.append((host, item))
            if not hybrid or role in ("primary", ""):
                if not hot:
                    batch = idle
                    self.pending.clear()
                    return batch
                batch = list(hot)
                for host, item in idle:
                    lane = self.lanes.get(host)
                    if lane is None or now - lane.last_infer_at >= IDLE_LANE_INTERVAL:
                        batch.append((host, item))
                for host, _ in batch:
                    self.pending.pop(host, None)
                return batch
            if role == "cpu":
                batch = []
                for host, item in idle:
                    lane = self.lanes.get(host)
                    if lane is None or now - lane.last_infer_at >= IDLE_LANE_INTERVAL:
                        batch.append((host, item))
                for host, _ in batch:
                    self.pending.pop(host, None)
                return batch
            if hot:
                batch = list(hot)
            else:
                batch = []
                for host, item in idle:
                    lane = self.lanes.get(host)
                    if lane is None or now - lane.last_infer_at >= IDLE_LANE_INTERVAL:
                        batch.append((host, item))
                        break
            for host, _ in batch:
                self.pending.pop(host, None)
            return batch

    @staticmethod
    def _tracking_unconfirmed(lane: CameraLane, frame_index: int) -> bool:
        """Return whether the lane just saw a plate whose text is not confirmed yet."""

        session = lane.session
        if session.last_detect_count <= 0:
            return False
        for track in session.tracks:
            if int(track.get("last_frame", -1)) != int(frame_index):
                continue
            if int(track.get("track_id", 0)) not in session.confirmed_tracks:
                return True
        return False

    def _infer_batch(
        self,
        batch: list[tuple[str, tuple[Any, int, Any]]],
        service: ScanService,
        lock: threading.Lock,
        label: str,
    ) -> None:
        for host, item in batch:
            frame, frame_index = item[:2]
            submitted_lane = item[2] if len(item) > 2 else None
            lane = self.lanes.get(host)
            if (
                lane is None
                or (submitted_lane is not None and lane is not submitted_lane)
                or lane.stop.is_set()
                or service._scanner is None
            ):
                continue
            try:
                with lock:
                    service.process_live_frame(
                        lane.session,
                        service._scanner,
                        frame,
                        frame_index,
                        write_image=service._write_image,
                        preprocess_plate_crop=service._preprocess_plate_crop,
                        stride=max(1, int(self.settings.camera_frame_stride)),
                        min_confirmations=self.settings.camera_min_confirmations,
                        output_stem=lane.session_stem,
                        archive_camera_id=str(lane.index),
                        media="camera",
                        source_fps=float(lane.fps or 15.0),
                        persist_archive=True,
                        scan_roi=lane.roi,
                        apply_roi=True,
                        run_full_ocr=self.settings.full_accuracy_mode,
                    )
                    published = list(lane.session.last_published)
                    ocr_jobs = list(lane.session.pending_ocr)
                    lane.session.pending_ocr = []
                lane.sampled_frames = lane.session.sampled_frames
                lane.last_detect_count = lane.session.last_detect_count
                lane.last_infer_at = time.monotonic()
                if self._tracking_unconfirmed(lane, frame_index):
                    lane.hot_until = lane.last_infer_at + HOT_LANE_SECONDS
                if published:
                    lane.publish_records(published)
                for record, crop in ocr_jobs:
                    self.ocr_queue.put((lane, record, crop))
            except Exception:
                LOGGER.exception("%s infer failed for %s", label, host)
                with lane.lock:
                    lane.error = "วิเคราะห์เฟรมไม่สำเร็จ"

    def _loop(self) -> None:
        while not self.stop.is_set():
            try:
                if self.busy():
                    time.sleep(0.05)
                    continue
                role = "gpu" if self._using_hybrid() else "primary"
                batch = self._next_batch(role)
                if not batch:
                    time.sleep(0.01)
                    continue
                try:
                    self.ensure_loaded()
                except Exception:
                    time.sleep(0.5)
                    continue
                self._infer_batch(batch, self.service, self.infer_lock, "GPU")
            except Exception:
                LOGGER.exception("GPU hub loop error")
                time.sleep(0.2)

    def _cpu_loop(self) -> None:
        while not self.stop.is_set():
            try:
                if self.busy() or not self._using_hybrid():
                    time.sleep(0.05)
                    continue
                batch = self._next_batch("cpu")
                if not batch:
                    time.sleep(0.02)
                    continue
                try:
                    self.ensure_cpu_loaded()
                except Exception:
                    LOGGER.exception("CPU hub failed to load models")
                    time.sleep(1.0)
                    continue
                if self.cpu_service is None:
                    time.sleep(0.2)
                    continue
                self._infer_batch(batch, self.cpu_service, self.cpu_lock, "CPU")
            except Exception:
                LOGGER.exception("CPU hub loop error")
                time.sleep(0.2)

    def _ocr_loop(self) -> None:
        while not self.stop.is_set():
            try:
                job = self.ocr_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                continue
            lane, record, crop = job
            try:
                updated = self.service.refine_deferred_ocr(record, crop)
                if updated is not None:
                    lane.refresh_published_record(updated)
            except Exception:
                LOGGER.exception("deferred OCR failed for %s", lane.host)
            finally:
                self.ocr_queue.task_done()


class WorkerPool:
    """AI worker: many camera lanes, one GPU hub."""

    def __init__(self, settings: Settings, busy: Callable[[], bool] | None = None) -> None:
        self.settings = settings
        self.hub = GpuHub(settings, busy=busy, on_update=self._publish_update)
        self.guard = threading.Lock()
        self.default_roi: dict[str, Any] | None = None
        self.roi_by_host: dict[str, dict[str, Any] | None] = {}
        self._roi_repository: DatabaseRepository | None = None
        self._roi_database_url = ""
        self._roi_lock = threading.Lock()
        self._load_persisted_rois()

    def _load_persisted_rois(self) -> None:
        """Load per-camera ROI once so a worker restart keeps the scan boxes."""

        database_url = str(self.settings.database_url or "").strip()
        if not database_url or database_url == self._roi_database_url:
            return
        try:
            repository = DatabaseRepository(database_url)
            # Keep the existing database and add only the camera ROI field.
            repository.ensure_camera_storage()
            saved = repository.list_camera_rois()
            with self.guard:
                self.roi_by_host.update(saved)
            self._roi_repository = repository
            self._roi_database_url = database_url
            LOGGER.info("Loaded saved ROI for %d camera(s)", len(saved))
        except Exception:
            # A database outage must not prevent the CCTV worker from starting;
            # the current session can still use an ROI supplied by the browser.
            LOGGER.exception("Unable to load saved camera ROI")

    def _persist_camera_roi(self, host: str, roi: dict[str, Any] | None, url: str = "") -> bool:
        database_url = str(self.settings.database_url or "").strip()
        if not database_url:
            return False
        try:
            with self._roi_lock:
                if self._roi_repository is None or self._roi_database_url != database_url:
                    self._roi_repository = DatabaseRepository(database_url)
                    self._roi_repository.ensure_camera_storage()
                    self._roi_database_url = database_url
                self._roi_repository.set_camera_roi(host, roi, stream_url=url)
            return True
        except Exception:
            LOGGER.exception("Unable to persist ROI for camera %s", host)
            return False

    def _publish_update(self, reason: str) -> None:
        publish_worker_event(self.snapshot(), message=reason)

    def persist_rois(self) -> None:
        """Flush the in-memory ROI map before an orderly server shutdown."""

        with self.guard:
            lanes = {host: lane.url for host, lane in self.hub.lanes.items()}
            saved = list(self.roi_by_host.items())
        for host, roi in saved:
            self._persist_camera_roi(host, roi, lanes.get(host, ""))

    def configure(self, settings: Settings) -> None:
        self.settings = settings
        self.hub.settings = settings
        self.hub.service.settings = settings
        for lane in self.hub.lanes.values():
            lane.settings = settings
        self._load_persisted_rois()

    def set_compute(self, mode: str) -> dict[str, Any]:
        from dataclasses import replace

        mode = normalise_compute_mode(mode)
        os.environ["CAR_SCAN_COMPUTE"] = mode
        save_compute_mode(self.settings.output_dir, mode)
        self.configure(replace(self.settings, compute_mode=mode))
        was_ready = bool(self.hub.ready)
        self.hub.reload_compute()
        if was_ready:
            try:
                self.hub.ensure_loaded()
            except Exception:
                LOGGER.exception("reload compute models for %s", mode)
        self._publish_update("compute mode changed")
        return self.snapshot()

    def start_camera(
        self,
        url: str,
        host: str,
        *,
        operator_id: int | None = None,
        operator_name: str = "",
        operator_username: str = "",
        scan: bool = True,
        index: int | None = None,
        roi: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        local = parse_local_camera(url) or parse_local_camera(host)
        if local is not None:
            host = local[0]
            url = local_camera_url(local[1])
        else:
            host = (urlsplit(url).hostname or host).strip()
        previous_roi = self.roi_by_host.get(host)
        chosen_roi = roi if roi is not None else previous_roi or self.default_roi
        if chosen_roi is not None:
            self.roi_by_host[host] = chosen_roi
            if roi is not None and chosen_roi != previous_roi:
                self._persist_camera_roi(host, chosen_roi, url)
        with self.guard:
            existing = self.hub.lanes.get(host)
            if existing and (existing.opened or existing.starting):
                existing.scanning = scan
                if chosen_roi is not None:
                    existing.roi = chosen_roi
                return existing.snapshot()
            if existing:
                existing.stop_lane(timeout=0.0)
                self.hub.lanes.pop(host, None)
            lane_index = int(index or 0) or max((item.index for item in self.hub.lanes.values()), default=0) + 1
            lane = CameraLane(
                host=host,
                url=url,
                index=lane_index,
                settings=self.settings,
                hub=self.hub,
                operator_id=operator_id,
                operator_name=operator_name,
                operator_username=operator_username,
            )
            lane.scanning = scan
            lane.roi = chosen_roi
            self.hub.lanes[host] = lane
            self.hub.start()
        try:
            lane.start()
        except Exception:
            with self.guard:
                self.hub.lanes.pop(host, None)
            lane.stop_lane()
            raise
        self._publish_update("camera starting")
        return lane.snapshot()

    def start_all(
        self,
        cameras: list[tuple[str, str]],
        *,
        operator_id: int | None = None,
        operator_name: str = "",
        operator_username: str = "",
        scan: bool = True,
        roi: dict[str, Any] | None = None,
        rois: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        errors: list[dict[str, str]] = []
        if roi is not None:
            self.default_roi = roi

        def boot(index: int, url: str, host: str) -> None:
            try:
                self.start_camera(
                    url,
                    host,
                    operator_id=operator_id,
                    operator_name=operator_name,
                    operator_username=operator_username,
                    scan=scan,
                    index=index,
                    roi=(rois or {}).get(host) or self.roi_by_host.get(host) or roi,
                )
            except Exception as error:
                errors.append({"host": host, "error": str(error)})

        threads = [
            threading.Thread(target=boot, args=(index, url, host), daemon=True, name=f"boot-{host}")
            for index, (url, host) in enumerate(cameras, start=1)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)
        snapshot = self.snapshot()
        snapshot["errors"] = errors
        self._publish_update("cameras started")
        return snapshot

    def stop_camera(self, host: str) -> dict[str, Any]:
        with self.guard:
            lane = self.hub.lanes.pop(host, None)
        if lane is None:
            self._publish_update("camera stopped")
            return {"host": host, "live": False}
        lane.stop_lane(timeout=0.0)
        self._publish_update("camera stopped")
        return {"host": host, "live": False}

    def stop_all(self, *, wait: bool = False) -> None:
        with self.guard:
            lanes = list(self.hub.lanes.values())
            self.hub.lanes.clear()
        for lane in lanes:
            # Signal every decoder first so a slow RTSP read cannot delay the
            # next camera from receiving its stop request.
            lane.stop_lane(timeout=0.0)
        if wait:
            deadline = time.monotonic() + SHUTDOWN_JOIN_TIMEOUT_SECONDS
            for lane in lanes:
                thread = lane.capture_thread
                if thread is None or not thread.is_alive() or thread is threading.current_thread():
                    continue
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
            self.hub.shutdown()
        self._publish_update("cameras stopped")

    def get(self, host: str) -> CameraLane | None:
        return self.hub.lanes.get(host)

    def plate_image(self, host: str, plate_id: int, kind: str) -> str:
        lane = self.get(host)
        if lane is None:
            return ""
        return lane.plate_file(plate_id, kind)

    def set_roi(self, host: str, roi: dict[str, Any] | None) -> dict[str, Any]:
        if host:
            self.roi_by_host[host] = roi
            lane = self.get(host)
            stream_url = lane.url if lane is not None else ""
            persisted = self._persist_camera_roi(host, roi, stream_url)
            if lane is None:
                return {"host": host, "live": False, "roi": roi, "roi_persisted": persisted}
            lane.roi = roi
            self._publish_update("camera ROI changed")
            snapshot = lane.snapshot()
            snapshot["roi_persisted"] = persisted
            return snapshot
        self.default_roi = roi
        self._publish_update("camera ROI changed")
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        cameras = [lane.snapshot() for lane in self.hub.lanes.values()]
        cameras.sort(key=lambda item: item.get("label") or "")
        plates: list[dict[str, Any]] = []
        for camera in cameras:
            plates.extend(camera.get("plates") or [])
        plates.sort(
            key=lambda item: (
                float(item.get("captured_at") or 0.0),
                str(item.get("camera_host") or ""),
                int(item.get("id") or 0),
            )
        )
        return {
            "worker": "ai-worker",
            "gpu": self.hub.status(),
            "cameras": cameras,
            "plates": plates,
        }
