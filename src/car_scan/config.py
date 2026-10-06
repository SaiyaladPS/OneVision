"""Application configuration loaded from environment variables.

Secrets are intentionally not stored in source code.  For local development
use a .env file loaded by the shell or set CAR_SCAN_DATABASE_URL directly.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit


def redact_stream_url(url: str) -> str:
    """Drop userinfo from an RTSP/HTTP camera URL before logs or storage."""

    text = str(url or "").strip()
    if not text:
        return ""
    parsed = urlsplit(text)
    if not (parsed.username or parsed.password):
        return text
    host = parsed.hostname or ""
    netloc = f"{host}:{parsed.port}" if parsed.port else host
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


_LOCAL_CAMERA_ALIASES = {
    "local",
    "webcam",
    "pc",
    "this-pc",
    "thispc",
    "กล้องคอม",
    "กล้องคอมพิวเตอร์",
}


def is_camera_stream_url(url: str) -> bool:
    parsed = urlsplit(str(url or "").strip())
    return parsed.scheme.lower() in {"rtsp", "rtsps", "http", "https"} and bool(parsed.hostname)


def local_camera_host(index: int = 0) -> str:
    device = max(0, int(index))
    return "local" if device == 0 else f"local-{device}"


def local_camera_url(index: int = 0) -> str:
    return f"local://{max(0, int(index))}"


def parse_local_camera(value: str) -> tuple[str, int] | None:
    """Return (lane host, device index) for this PC's webcam, if the value names one."""

    text = str(value or "").strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in _LOCAL_CAMERA_ALIASES:
        return local_camera_host(0), 0
    parsed = urlsplit(lowered)
    if parsed.scheme in {"local", "webcam"}:
        index_text = str(parsed.hostname or parsed.path or "0").strip("/")
        try:
            index = int(index_text or 0)
        except ValueError:
            index = 0
        return local_camera_host(index), index
    match = re.fullmatch(r"(?:local|webcam)[-:](\d+)", lowered)
    if match:
        index = int(match.group(1))
        return local_camera_host(index), index
    return None


def is_camera_host(value: str) -> bool:
    text = str(value or "").strip()
    if not text or "://" in text or "/" in text or "@" in text or " " in text:
        return False
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", text):
        return True
    return bool(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?", text))


def parse_camera_urls(*chunks: str) -> tuple[str, ...]:
    found: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        normalized = str(chunk or "").replace(";", ",").replace("\n", ",")
        for part in normalized.split(","):
            url = part.strip()
            if not is_camera_stream_url(url):
                continue
            host = (urlsplit(url).hostname or "").lower()
            if not host or host in seen:
                continue
            seen.add(host)
            found.append(url)
    return tuple(found)


DEFAULT_RTSP_PORT = 554
DEFAULT_RTSP_PATH = "/Streaming/Channels/101"


def rewrite_camera_host(template: str, host: str) -> str:
    parsed = urlsplit(template)
    userinfo = ""
    if parsed.username:
        userinfo = quote(parsed.username, safe="")
        if parsed.password is not None:
            userinfo += ":" + quote(parsed.password, safe="")
        userinfo += "@"
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{userinfo}{host}{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def replace_camera_userinfo(url: str, username: str, password: str) -> str:
    """Set RTSP/HTTP userinfo without changing host, port, or path."""

    parsed = urlsplit(url)
    user = str(username or "")
    secret = str(password or "")
    if not user and secret == "":
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        return urlunsplit((parsed.scheme, f"{host}{port}", parsed.path, parsed.query, parsed.fragment))
    userinfo = quote(user, safe="")
    userinfo += ":" + quote(secret, safe="")
    userinfo += "@"
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    return urlunsplit((parsed.scheme, f"{userinfo}{host}{port}", parsed.path, parsed.query, parsed.fragment))


def default_rtsp_url(host: str) -> str:
    return f"rtsp://{host}:{DEFAULT_RTSP_PORT}{DEFAULT_RTSP_PATH}"


COMMON_RTSP_PATHS = (
    "/Streaming/Channels/101",
    "/Streaming/Channels/102",
    "/Streaming/Channels/1",
    "/h264/ch1/main/av_stream",
    "/h264/ch1/sub/av_stream",
    "/cam/realmonitor?channel=1&subtype=0",
    "/cam/realmonitor?channel=1&subtype=1",
    "/h264Preview_01_main",
    "/media/video1",
    "/live",
    "/live/ch00_0",
    "/stream1",
    "/onvif1",
    "/profile1/media.smp",
    "/",
)


def apply_rtsp_path(url: str, path: str) -> str:
    """Replace the RTSP path (and query) while keeping host and credentials."""

    parsed = urlsplit(url)
    text = str(path or "").strip()
    if not text:
        return url
    if is_camera_stream_url(text):
        return text
    if text.startswith("?"):
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", text[1:], parsed.fragment))
    if not text.startswith("/"):
        text = "/" + text
    path_part, query = (text.split("?", 1) + [""])[:2]
    return urlunsplit((parsed.scheme, parsed.netloc, path_part, query, parsed.fragment))


def rtsp_url_candidates(url: str) -> list[str]:
    """Return the given RTSP URL first, then common vendor paths for that camera."""

    text = str(url or "").strip()
    if not text:
        return []
    parsed = urlsplit(text)
    if parsed.scheme.lower() not in {"rtsp", "rtsps"}:
        return [text]
    found: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        if candidate and candidate not in seen:
            seen.add(candidate)
            found.append(candidate)

    add(text)
    original_path = parsed.path or "/"
    if parsed.query:
        original_path = f"{original_path}?{parsed.query}"
    for path in (original_path, *COMMON_RTSP_PATHS):
        add(apply_rtsp_path(text, path))
    user = parsed.username or ""
    secret = parsed.password or ""
    if user or secret:
        encoded = (
            f"/user={quote(user, safe='')}&password={quote(secret, safe='')}"
            "&channel=1&stream=0.sdp"
        )
        add(apply_rtsp_path(replace_camera_userinfo(text, "", ""), encoded))
    return found


def compose_camera_url(
    value: str,
    *,
    username: str = "",
    password: str = "",
    template: str = "",
    path: str = "",
) -> str:
    """Build a camera URL from IP/RTSP plus optional per-camera credentials."""

    text = str(value or "").strip()
    user = str(username or "").strip()
    secret = str(password or "")
    stream_path = str(path or "").strip()
    local = parse_local_camera(text)
    if local is not None:
        return local_camera_url(local[1])
    if is_camera_stream_url(text):
        url = replace_camera_userinfo(text, user, secret) if user or secret else text
    elif is_camera_host(text):
        if is_camera_stream_url(template):
            url = rewrite_camera_host(template, text)
        else:
            url = default_rtsp_url(text)
        if user or secret:
            url = replace_camera_userinfo(url, user, secret)
    else:
        return ""
    if stream_path:
        return apply_rtsp_path(url, stream_path)
    return url


def resolve_camera_url(value: str, configured: tuple[str, ...] | list[str] = ()) -> str:
    text = str(value or "").strip()
    cameras = tuple(configured or ())
    local = parse_local_camera(text)
    if local is not None:
        return local_camera_url(local[1])
    if is_camera_stream_url(text):
        return text
    if is_camera_host(text):
        wanted = text.lower()
        for url in cameras:
            if (urlsplit(url).hostname or "").lower() == wanted:
                return url
        if cameras:
            return rewrite_camera_host(cameras[0], text)
        return ""
    if not text and cameras:
        return cameras[0]
    return ""


def extra_camera_store(output_dir: Path) -> Path:
    return Path(output_dir) / "ip_cameras.json"


def _camera_entry_host(value: str, url: str = "") -> str:
    local = parse_local_camera(url) or parse_local_camera(value)
    if local is not None:
        return local[0]
    if is_camera_stream_url(url):
        return urlsplit(url).hostname or str(value or "").strip()
    if is_camera_stream_url(value):
        return urlsplit(value).hostname or ""
    text = str(value or "").strip()
    return text if is_camera_host(text) else ""


def load_extra_cameras(output_dir: Path) -> list[dict[str, str]]:
    path = extra_camera_store(output_dir)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    items: list[Any] = []
    if isinstance(payload, dict):
        cameras = payload.get("cameras")
        if isinstance(cameras, list):
            items.extend(cameras)
        hosts = payload.get("hosts")
        if isinstance(hosts, list):
            items.extend(hosts)
    elif isinstance(payload, list):
        items.extend(payload)
    found: list[dict[str, str]] = []
    by_host: dict[str, int] = {}
    for item in items:
        host = ""
        url = ""
        if isinstance(item, dict):
            url = str(item.get("url") or "").strip()
            host = str(item.get("host") or "").strip()
        else:
            host = str(item or "").strip()
        host = _camera_entry_host(host, url)
        if not host:
            continue
        key = host.lower()
        entry = {"host": host, "url": url, "label": str(item.get("label") or "").strip() if isinstance(item, dict) else ""}
        index = by_host.get(key)
        if index is None:
            by_host[key] = len(found)
            found.append(entry)
            continue
        if url and not found[index].get("url"):
            found[index]["url"] = url
        if entry["label"]:
            found[index]["label"] = entry["label"]
    return found


def load_extra_camera_hosts(output_dir: Path) -> list[str]:
    return [item["host"] for item in load_extra_cameras(output_dir)]


def _write_extra_cameras(output_dir: Path, cameras: list[dict[str, str]]) -> None:
    path = extra_camera_store(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "hosts": [item["host"] for item in cameras],
        "cameras": [
            {
                "host": item["host"],
                **({"url": item["url"]} if item.get("url") else {}),
                **({"label": item["label"]} if item.get("label") else {}),
            }
            for item in cameras
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_extra_camera_host(
    output_dir: Path,
    host: str,
    url: str = "",
    label: str | None = None,
) -> list[str]:
    cameras = load_extra_cameras(output_dir)
    wanted = str(host or "").strip()
    stored = str(url or "").strip()
    key = wanted.lower()
    updated = False
    for item in cameras:
        if item["host"].lower() != key:
            continue
        if stored:
            item["url"] = stored
        if label is not None:
            item["label"] = str(label).strip()
        updated = True
        break
    if not updated:
        cameras.append({"host": wanted, "url": stored, "label": str(label or "").strip()})
    _write_extra_cameras(output_dir, cameras)
    return [item["host"] for item in cameras]


def remove_extra_camera_host(output_dir: Path, host: str) -> list[str]:
    wanted = str(host or "").strip().lower()
    cameras = [item for item in load_extra_cameras(output_dir) if item["host"].lower() != wanted]
    _write_extra_cameras(output_dir, cameras)
    return [item["host"] for item in cameras]


def listed_camera_urls(settings: "Settings") -> tuple[str, ...]:
    urls = list(settings.ip_camera_urls)
    seen = {(urlsplit(url).hostname or "").lower() for url in urls}
    for extra in load_extra_cameras(settings.output_dir):
        host = extra["host"]
        stored = extra.get("url") or ""
        if host.lower() in seen:
            if is_camera_stream_url(stored):
                for index, current in enumerate(urls):
                    if (urlsplit(current).hostname or "").lower() == host.lower():
                        urls[index] = stored
                        break
            continue
        local = parse_local_camera(stored) or parse_local_camera(host)
        if local is not None:
            urls.append(local_camera_url(local[1]))
            seen.add(local[0].lower())
            continue
        resolved = stored if is_camera_stream_url(stored) else resolve_camera_url(host, settings.ip_camera_urls)
        if not is_camera_stream_url(resolved):
            continue
        urls.append(resolved)
        seen.add(host.lower())
    return tuple(urls)


def public_cameras(
    urls: tuple[str, ...] | list[str],
    builtin_hosts: set[str] | None = None,
) -> list[dict[str, str | bool | int]]:
    builtin = {item.lower() for item in (builtin_hosts or set())}
    cameras: list[dict[str, str | bool | int]] = []
    for url in urls:
        local = parse_local_camera(url)
        if local is not None:
            host, device_index = local
            index = len(cameras) + 1
            cameras.append(
                {
                    "host": host,
                    "label": f"Camera {index:02d}",
                    "stream_url": redact_stream_url(url),
                    "index": index,
                    "builtin": False,
                    "kind": "local",
                    "device_index": device_index,
                }
            )
            continue
        host = urlsplit(url).hostname or ""
        if host:
            index = len(cameras) + 1
            cameras.append(
                {
                    "host": host,
                    "label": f"Camera {index:02d}",
                    "stream_url": redact_stream_url(url),
                    "index": index,
                    "builtin": host.lower() in builtin,
                    "kind": "ip",
                    "has_auth": bool(urlsplit(url).username),
                }
            )
    return cameras


def latest_ocr_weights(root: Path, model: str, legacy_name: str) -> Path | None:
    """Pick the newest trained PaddleOCR checkpoint for a language model.

    Isolated GUI/CLI runs live in ``runs/ocr/<model>/<timestamp>/`` while older
    trainings used ``runs/ocr/<legacy_name>/best_accuracy``.
    """

    ocr_root = Path(root) / "runs" / "ocr"
    candidates: list[Path] = []
    legacy = ocr_root / legacy_name / "best_accuracy.pdparams"
    if legacy.is_file():
        candidates.append(legacy)
    isolated = ocr_root / model
    if isolated.is_dir():
        candidates.extend(isolated.glob("*/best_accuracy.pdparams"))
    if not candidates:
        return None
    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    return newest.with_suffix("")


def resource_root() -> Path:
    """Return the directory containing bundled models/resources."""

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    # config.py lives in <root>/src/car_scan during development.
    return Path(__file__).resolve().parents[2]


def _path(value: str | None, default: Path) -> Path:
    return Path(value).expanduser() if value else default


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE settings without requiring python-dotenv."""

    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the GUI and scan service."""

    root: Path
    output_dir: Path
    database_url: str
    # Identifier included in archived image names. Camera scans override
    # this with the actual selected camera index.
    archive_camera_id: str = "0"
    detector_confidence: float = 0.35
    vehicle_type_confidence: float = 0.25
    plate_type_confidence: float = 0.25
    character_confidence: float = 0.20
    padding: float = 0.03
    imgsz: int = 640
    video_frame_stride: int = 1
    video_target_scans_per_second: float = 2.0
    video_min_confirmations: int = 3
    camera_frame_stride: int = 1
    # Two observations reduce false positives while still allowing plates
    # from fast-moving vehicles to be confirmed before they leave the frame.
    camera_min_confirmations: int = 2
    camera_infer_max_dimension: int = 1920
    # A normal lane is sampled often enough to spot an approaching vehicle.
    # Once a plate is near/in the ROI it temporarily receives more frequent
    # inference so a fast-moving vehicle has several usable frames.
    camera_idle_submit_interval: float = 0.08
    camera_hot_submit_interval: float = 0.04
    temporal_min_quality: float = 0.58
    target_fps: float = 24.0
    # Preview delivery is deliberately independent from capture/inference.
    # Sending full-resolution frames at camera speed can starve the Qt event
    # loop even when recognition itself runs in a worker thread.
    preview_fps: float = 10.0
    preview_max_dimension: int = 960
    # Keep uncertain examples useful for training without repeatedly encoding
    # full-resolution images while a video/camera stream is running.
    max_rejected_evidence_per_run: int = 20
    rejected_evidence_frame_gap: int = 24
    # Vehicle type changes much more slowly than plate text. Reusing it for a
    # few sampled frames removes one full-frame YOLO pass from most scans.
    stream_vehicle_refresh_scans: int = 3
    # Sample video/camera frames by default; enable full accuracy explicitly
    # when every frame must be analysed.
    full_accuracy_mode: bool = False
    pipeline_mode: str = "auto"
    pipeline_config_override: Path | None = None
    plate_type_model_override: Path | None = None
    debug_enabled: bool = False
    scan_roi_override: dict[str, Any] | None = None
    ip_camera_urls: tuple[str, ...] = ()
    # auto = CUDA YOLO when present, OCR on CPU; gpu/cpu force one device;
    # hybrid = GPU YOLO for live plates plus a CPU YOLO copy for idle cameras.
    compute_mode: str = "auto"

    @classmethod
    def from_env(cls) -> "Settings":
        _load_env_file(Path.cwd() / ".env")
        _load_env_file(Path(__file__).resolve().parents[2] / ".env")
        root = _path(os.getenv("CAR_SCAN_ROOT"), resource_root()).resolve()
        # Keep raw scan evidence separate from generated training exports.
        # The resulting layout is ``scan/data/YYYYMMDD/...``.
        output = _path(os.getenv("CAR_SCAN_OUTPUT_DIR"), root / "scan" / "data").resolve()
        # Docker Compose and the deployment environment commonly expose
        # DATABASE_URL. Keep the application-specific name as the preferred
        # override, but accept the standard name for direct Windows runs too.
        database_url = (
            os.getenv("CAR_SCAN_DATABASE_URL", "").strip()
            or os.getenv("DATABASE_URL", "").strip()
        )
        from .compute import load_saved_compute_mode, normalise_compute_mode

        compute_mode = normalise_compute_mode(
            os.getenv("CAR_SCAN_COMPUTE") or load_saved_compute_mode(output) or "auto"
        )
        return cls(
            root=root,
            output_dir=output,
            database_url=database_url,
            archive_camera_id=os.getenv("CAR_SCAN_ARCHIVE_CAMERA_ID", "0").strip() or "0",
            detector_confidence=float(os.getenv("CAR_SCAN_DETECTOR_CONFIDENCE", "0.35")),
            vehicle_type_confidence=float(os.getenv("CAR_SCAN_VEHICLE_TYPE_CONFIDENCE", "0.25")),
            plate_type_confidence=float(os.getenv("CAR_SCAN_PLATE_TYPE_CONFIDENCE", "0.25")),
            character_confidence=float(os.getenv("CAR_SCAN_CHARACTER_CONFIDENCE", "0.20")),
            padding=float(os.getenv("CAR_SCAN_PADDING", "0.03")),
            imgsz=int(os.getenv("CAR_SCAN_IMGSZ", "640")),
            video_frame_stride=max(1, int(os.getenv("CAR_SCAN_VIDEO_FRAME_STRIDE", "1"))),
            video_target_scans_per_second=max(
                0.5, float(os.getenv("CAR_SCAN_VIDEO_TARGET_SCANS_PER_SECOND", "2.0"))
            ),
            video_min_confirmations=max(
                1, int(os.getenv("CAR_SCAN_VIDEO_MIN_CONFIRMATIONS", "3"))
            ),
            camera_frame_stride=max(1, int(os.getenv("CAR_SCAN_CAMERA_FRAME_STRIDE", "1"))),
            camera_min_confirmations=max(
                1, int(os.getenv("CAR_SCAN_CAMERA_MIN_CONFIRMATIONS", "2"))
            ),
            camera_infer_max_dimension=max(
                640, int(os.getenv("CAR_SCAN_CAMERA_INFER_MAX_DIMENSION", "1920"))
            ),
            camera_idle_submit_interval=max(
                0.04, float(os.getenv("CAR_SCAN_CAMERA_IDLE_SUBMIT_INTERVAL", "0.08"))
            ),
            camera_hot_submit_interval=max(
                0.03, float(os.getenv("CAR_SCAN_CAMERA_HOT_SUBMIT_INTERVAL", "0.04"))
            ),
            temporal_min_quality=max(
                0.0, float(os.getenv("CAR_SCAN_TEMPORAL_MIN_QUALITY", "0.58"))
            ),
            target_fps=max(1.0, float(os.getenv("CAR_SCAN_TARGET_FPS", "24.0"))),
            preview_fps=max(1.0, float(os.getenv("CAR_SCAN_PREVIEW_FPS", "10.0"))),
            preview_max_dimension=max(
                160, int(os.getenv("CAR_SCAN_PREVIEW_MAX_DIMENSION", "960"))
            ),
            max_rejected_evidence_per_run=max(
                0, int(os.getenv("CAR_SCAN_MAX_REJECTED_EVIDENCE_PER_RUN", "20"))
            ),
            rejected_evidence_frame_gap=max(
                1, int(os.getenv("CAR_SCAN_REJECTED_EVIDENCE_FRAME_GAP", "24"))
            ),
            stream_vehicle_refresh_scans=max(
                1, int(os.getenv("CAR_SCAN_STREAM_VEHICLE_REFRESH_SCANS", "3"))
            ),
            full_accuracy_mode=os.getenv("CAR_SCAN_FULL_ACCURACY_MODE", "0").strip().lower()
            in ("1", "true", "yes", "on"),
            pipeline_mode=os.getenv("CAR_SCAN_PIPELINE_MODE", "auto").strip().lower(),
            pipeline_config_override=_path(
                os.getenv("CAR_SCAN_PIPELINE_CONFIG"), root / "pipeline_config.yaml"
            ).resolve(),
            plate_type_model_override=(
                Path(os.environ["CAR_SCAN_PLATE_TYPE_MODEL"]).expanduser().resolve()
                if os.getenv("CAR_SCAN_PLATE_TYPE_MODEL")
                else None
            ),
            debug_enabled=os.getenv("CAR_SCAN_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on"),
            ip_camera_urls=parse_camera_urls(
                os.getenv("CAR_SCAN_IP_CAMERA_URL", ""),
                os.getenv("CAR_SCAN_IP_CAMERA_URLS", ""),
            ),
            compute_mode=compute_mode,
        )

    @property
    def ip_camera_url(self) -> str:
        return self.ip_camera_urls[0] if self.ip_camera_urls else ""

    @property
    def detector_model(self) -> Path:
        return self.root / "model" / "detect_license" / "weights" / "best.pt"

    @property
    def vehicle_type_model(self) -> Path:
        return self.root / "model" / "type_car_license" / "iruvd_run1" / "weights" / "best.pt"

    @property
    def plate_type_model(self) -> Path | None:
        return self.plate_type_model_override

    @property
    def pipeline_config(self) -> Path:
        return self.pipeline_config_override or (self.root / "pipeline_config.yaml")

    @property
    def debug_dir(self) -> Path:
        return self.output_dir / "debug"

    @property
    def thai_model(self) -> Path:
        return self.root / "model" / "thai_license_plate" / "weights" / "best.pt"

    @property
    def lao_model(self) -> Path:
        return self.root / "model" / "lao_license_plate" / "weights" / "best.pt"

    @property
    def ocr_source(self) -> Path:
        return self.root / "model"

    @property
    def ocr_config(self) -> Path:
        return self.thai_model

    @property
    def ocr_weights(self) -> Path:
        return self.thai_model

    @property
    def ocr_lao_config(self) -> Path:
        return self.lao_model

    def _latest_ocr_weights(self, model: str, legacy_name: str) -> Path | None:
        return latest_ocr_weights(self.root, model, legacy_name)

    @property
    def ocr_lao_weights(self) -> Path | None:
        return self.lao_model

    @property
    def ocr_thai_config(self) -> Path:
        return self.thai_model

    @property
    def ocr_thai_weights(self) -> Path | None:
        return self.thai_model

    @property
    def torch_ocr_model(self) -> Path:
        return self.root / "runs" / "ocr" / "plate_crnn.pt"
