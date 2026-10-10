"""Background sync for scan artifacts stored by the OneVision report server."""

from __future__ import annotations

import logging
import http.client
import json
import mimetypes
import os
import queue
import re
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

LOGGER = logging.getLogger(__name__)
_UPLOADS: queue.Queue[tuple[Path, list[Path]]] = queue.Queue(maxsize=256)
_PLATE_IMAGE_UPLOADS: queue.Queue[tuple[str, str, bytes, str]] = queue.Queue(maxsize=512)
_WORKER_STARTED = False
_WORKER_LOCK = threading.Lock()
_PLATE_IMAGE_WORKER_STARTED = False
_INDEX_LOCK = threading.Lock()
_PROTECTED_REPORT_IMAGES: set[str] = set()
_PROTECTED_REPORT_IMAGES_LOCK = threading.Lock()
_FILE_ID_PATTERN = re.compile(r"/api/files/([0-9a-f-]{36})/download(?:$|[?#])", re.I)


def _config() -> tuple[str, str, str]:
    endpoint = os.getenv("STORAGE_SERVICE_BASE_URL", "http://10.0.200.205:8080").strip().rstrip("/")
    if Path("/.dockerenv").exists():
        parsed = urlsplit(endpoint)
        if parsed.hostname in {"localhost", "127.0.0.1"}:
            host = "host.docker.internal"
            netloc = f"{host}:{parsed.port}" if parsed.port else host
            endpoint = parsed._replace(netloc=netloc).geturl().rstrip("/")
    project = os.getenv("STORAGE_SERVICE_PROJECT_CODE", "onevision").strip()
    token = os.getenv("STORAGE_SERVICE_API_TOKEN", "").strip()
    return endpoint, project, token


def _report_data_endpoint() -> str:
    return os.getenv("ONEVISION_REPORT_DATA_URL", "http://10.0.200.205:3008/api/dataset/storage").strip().rstrip("/")


def _report_image_endpoint() -> tuple[str, str]:
    endpoint = os.getenv(
        "ONEVISION_REPORT_IMAGE_URL",
        "http://10.0.200.205:3008/api/plates/image",
    ).strip().rstrip("/")
    token = os.getenv("CAR_SCAN_TRAIN_API_TOKEN", "").strip()
    return endpoint, token


def protect_report_images(*paths: str | Path | None) -> None:
    """Prevent the storage worker removing files before the plate-ID upload is queued."""

    _, token = _report_image_endpoint()
    if not token:
        return
    with _PROTECTED_REPORT_IMAGES_LOCK:
        for raw_path in paths:
            if raw_path:
                _PROTECTED_REPORT_IMAGES.add(str(Path(raw_path).resolve()))


def _release_report_images(output_dir: Path, *paths: str | Path | None) -> None:
    resolved_paths = []
    with _PROTECTED_REPORT_IMAGES_LOCK:
        for raw_path in paths:
            if not raw_path:
                continue
            resolved = str(Path(raw_path).resolve())
            _PROTECTED_REPORT_IMAGES.discard(resolved)
            resolved_paths.append(Path(resolved))
    for path in resolved_paths:
        try:
            relative = path.relative_to(output_dir.resolve()).as_posix()
        except (OSError, ValueError):
            continue
        if central_storage_file_id(relative, output_dir):
            path.unlink(missing_ok=True)


def _upload_plate_image(endpoint: str, token: str, plate_id: str, variant: str, content: bytes, mime: str) -> None:
    request = Request(
        f"{endpoint}?{urlencode({'id': plate_id, 'variant': variant})}",
        data=content,
        headers={"Authorization": f"Bearer {token}", "Content-Type": mime},
        method="POST",
    )
    with urlopen(request, timeout=20) as response:
        body = response.read(16 * 1024)
        if not 200 <= response.status < 300:
            raise OSError(f"Report image API returned HTTP {response.status}")
        try:
            result = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OSError("Report image API did not return a JSON upload confirmation") from error
        if not isinstance(result, dict) or result.get("success") is not True:
            raise OSError("Report image API did not confirm that the image was stored")


def _run_plate_image_worker() -> None:
    while True:
        endpoint, token, plate_id, variant, content, mime = _PLATE_IMAGE_UPLOADS.get()
        try:
            for attempt in range(5):
                try:
                    _upload_plate_image(endpoint, token, plate_id, variant, content, mime)
                    LOGGER.info("Uploaded scanned %s image for plate id %s to OneVision Report", variant, plate_id)
                    break
                except (HTTPError, URLError, OSError, TimeoutError, ValueError) as error:
                    if attempt == 4:
                        LOGGER.warning(
                            "Could not upload scanned %s image for plate id %s to OneVision Report after retries: %s",
                            variant,
                            plate_id,
                            error,
                        )
                    else:
                        time.sleep(min(2 ** attempt, 16))
        finally:
            _PLATE_IMAGE_UPLOADS.task_done()


def enqueue_plate_images(
    plate_id: int | str | None,
    *,
    output_dir: Path | None = None,
    **images: str | Path | None,
) -> None:
    """Queue scanned images for asynchronous storage on the Report server."""

    if plate_id is None:
        return
    endpoint, token = _report_image_endpoint()
    if not token:
        return
    global _PLATE_IMAGE_WORKER_STARTED
    with _WORKER_LOCK:
        if not _PLATE_IMAGE_WORKER_STARTED:
            threading.Thread(target=_run_plate_image_worker, name="report-plate-image-sync", daemon=True).start()
            _PLATE_IMAGE_WORKER_STARTED = True

    variants = {"full": images.get("full"), "crop": images.get("crop"), "ocr": images.get("ocr")}
    queued_paths: list[Path] = []
    for variant, raw_path in variants.items():
        if not raw_path:
            continue
        path = _resolve_local_image_path(raw_path, output_dir)
        if path is None:
            LOGGER.warning("Could not find local scanned %s image: %s", variant, raw_path)
            continue
        try:
            content = path.read_bytes()
        except OSError as error:
            LOGGER.warning("Could not read scanned %s image for Report upload: %s", variant, error)
            continue
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        try:
            _PLATE_IMAGE_UPLOADS.put_nowait((endpoint, token, str(plate_id), variant, content, mime))
            queued_paths.append(path)
        except queue.Full:
            LOGGER.error("Report plate image upload queue is full; image %s was not queued", path.name)
    _release_report_images(output_dir or Path(os.getenv("CAR_SCAN_OUTPUT_DIR", "scan/data")), *queued_paths)


def _resolve_local_image_path(raw_path: str | Path, output_dir: Path | None = None) -> Path | None:
    """Resolve an archived image path from either an absolute or relative value."""

    text = str(raw_path or "").strip()
    if not text or text.lower().startswith(("http://", "https://", "storage://")):
        return None
    path = Path(text)
    candidates = [path] if path.is_absolute() else []
    root = Path(output_dir or os.getenv("CAR_SCAN_OUTPUT_DIR", "scan/data"))
    if not path.is_absolute():
        candidates.extend((root / path, Path.cwd() / path))
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def central_storage_file_id(raw: str | None, output_dir: Path | None = None) -> str | None:
    """Resolve a central file id from an API URL, storage URI, or local path index."""

    if not raw:
        return None
    value = str(raw).strip()
    match = _FILE_ID_PATTERN.search(value)
    if match:
        return match.group(1)
    if value.lower().startswith("storage://"):
        candidate = value[10:].split("/", 1)[0]
        return candidate if _FILE_ID_PATTERN.fullmatch(f"/api/files/{candidate}/download") else None
    if output_dir is None:
        return None
    try:
        relative = Path(value).resolve().relative_to(Path(output_dir).resolve()).as_posix()
    except (OSError, ValueError):
        relative = value.replace("\\", "/").lstrip("/")
    try:
        index = json.loads((Path(output_dir) / ".storage_index.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    file_id = index.get(relative) if isinstance(index, dict) else None
    return str(file_id) if isinstance(file_id, str) and _FILE_ID_PATTERN.fullmatch(f"/api/files/{file_id}/download") else None


def _record_file_id(root: Path, relative: str, file_id: str) -> None:
    index_path = root / ".storage_index.json"
    with _INDEX_LOCK:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            index = {}
        if not isinstance(index, dict):
            index = {}
        index[relative] = file_id
        temp_path = index_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        temp_path.replace(index_path)


def _indexed_file_id(root: Path, relative: str) -> str | None:
    try:
        index = json.loads((root / ".storage_index.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    file_id = index.get(relative) if isinstance(index, dict) else None
    return str(file_id) if isinstance(file_id, str) and _FILE_ID_PATTERN.fullmatch(
        f"/api/files/{file_id}/download"
    ) else None


def _sync_plate_database_reference(root: Path, relative: str, file_id: str) -> bool:
    """Make the shared PostgreSQL row point at the durable Storage Service object."""

    filename = Path(relative).name.lower()
    if not any(marker in filename for marker in ("-plate_crops.", "-full_vehicle.", "-ocr_ready.")) and not re.search(
        r"_plate_\d+\.", filename
    ):
        return True
    database_url = (os.getenv("CAR_SCAN_DATABASE_URL") or os.getenv("DATABASE_URL") or "").strip()
    if not database_url:
        LOGGER.error("Cannot publish stored plate image %s: shared PostgreSQL is not configured", relative)
        return False
    try:
        from .database import DatabaseRepository

        repository = DatabaseRepository(database_url)
        # The camera persistence and artifact queue are asynchronous. Retry briefly
        # so the upload can finish even if the row is committed just after the file.
        for attempt in range(6):
            updated = repository.replace_plate_image_reference(relative, f"storage://{file_id}")
            if updated > 0:
                return True
            if attempt < 5:
                time.sleep(0.25 * (2 ** attempt))
        LOGGER.error("Stored plate image %s has no matching shared database row; keeping the local copy", relative)
        return False
    except Exception:
        LOGGER.exception("Could not link stored plate image %s to the shared database", relative)
        return False


def _replace_json_paths(root: Path, file_ids: dict[str, str], manifests: list[Path]) -> None:
    """Keep the report's lightweight JSON index, but point media to Storage IDs."""

    try:
        saved_ids = json.loads((root / ".storage_index.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        saved_ids = {}
    if isinstance(saved_ids, dict):
        file_ids = {**saved_ids, **file_ids}
    if not file_ids or not manifests:
        return
    for manifest in manifests:
        try:
            relative_manifest = manifest.resolve().relative_to(root.resolve()).as_posix().split("/")
        except (OSError, ValueError):
            continue
        if len(relative_manifest) != 3 or len(relative_manifest[0]) != 8 or relative_manifest[1] != "json":
            continue
        if not manifest.is_file():
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        changed = False

        def visit(value):
            nonlocal changed
            if isinstance(value, dict):
                for key, child in value.items():
                    if isinstance(child, str) and key in {
                        "full_vehicle_image", "crop_image", "ocr_ready_image", "ocr_ready_archive_image",
                        "annotated_image", "output_video",
                    }:
                        try:
                            rel = Path(child).resolve().relative_to(root.resolve()).as_posix()
                        except (OSError, ValueError):
                            rel = child.replace("\\", "/").lstrip("/")
                        file_id = file_ids.get(rel)
                        if file_id:
                            value[key] = f"/api/files/{file_id}/download"
                            changed = True
                    else:
                        visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(payload)
        if changed:
            temporary = manifest.with_suffix(".json.tmp")
            try:
                temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                temporary.replace(manifest)
            except OSError:
                LOGGER.exception("Could not update newly generated storage references in %s", manifest)


def _sync_report_manifest(endpoint: str, token: str, root: Path, manifest: Path) -> bool:
    """Send small plate metadata to Report so its Dataset index can discover remote images."""

    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    media_keys = {"full_vehicle_image", "crop_image", "ocr_ready_image", "ocr_ready_archive_image"}
    media_values: list[str] = []

    def collect(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in media_keys and isinstance(child, str):
                    media_values.append(child)
                else:
                    collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(payload)
    if not media_values or any(not central_storage_file_id(value, root) for value in media_values):
        return False
    try:
        relative = manifest.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return False
    boundary = f"----OneVision{uuid4().hex}"
    body = b"".join(
        (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"path_0\"\r\n\r\n{relative}\r\n".encode(),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file_0\"; filename=\"{manifest.name}\"\r\nContent-Type: application/json\r\n\r\n".encode(),
            manifest.read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        )
    )
    request = Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=15) as response:
            return 200 <= response.status < 300
    except (HTTPError, URLError, OSError, TimeoutError) as error:
        LOGGER.warning("Could not publish plate metadata to OneVision Report: %s", error)
        return False


def _relative_files(root: Path, paths: list[Path]) -> list[tuple[Path, str]]:
    accepted = []
    for path in paths:
        try:
            resolved = path.resolve()
            rel = resolved.relative_to(root.resolve()).as_posix()
        except (OSError, ValueError):
            continue
        pieces = rel.split("/")
        legacy_layout = (
            len(pieces) == 3
            and re.fullmatch(r"\d{8}", pieces[0]) is not None
            and pieces[1] in {"thai", "laos", "json", "log", "video"}
        )
        camera_archive_layout = (
            len(pieces) == 6
            and bool(re.fullmatch(r"[A-Za-z0-9_-]+", pieces[0]))
            and re.fullmatch(r"\d{4}", pieces[1]) is not None
            and re.fullmatch(r"\d{2}", pieces[2]) is not None
            and re.fullmatch(r"\d{2}", pieces[3]) is not None
            and pieces[4] in {"thai", "laos", "vietnam", "unknown"}
        )
        if not (legacy_layout or camera_archive_layout):
            continue
        if resolved.is_file():
            accepted.append((resolved, rel))
    return accepted


def _upload_one(endpoint: str, project: str, token: str, path: Path, relative: str) -> str:
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("STORAGE_SERVICE_BASE_URL must be an http(s) URL")
    boundary = f"----OneVision{uuid4().hex}"
    safe_name = path.name.replace('"', "")
    encoded_name = quote(safe_name.encode("utf-8"), safe="")
    folder = "/".join(relative.split("/")[:-1])
    fields = b"".join(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        for name, value in (("project", project), ("folder", folder))
    )
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="upload{path.suffix.lower()}"; filename*=UTF-8\'\'{encoded_name}\r\n'
        f"Content-Type: {mimetypes.guess_type(safe_name)[0] or 'application/octet-stream'}\r\n\r\n"
    ).encode()
    suffix = f"\r\n--{boundary}--\r\n".encode()
    content_length = len(fields) + len(prefix) + path.stat().st_size + len(suffix)
    connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(parsed.hostname, parsed.port, timeout=60)
    route = parsed.path or "/"
    if parsed.query:
        route += f"?{parsed.query}"
    try:
        route = route.rstrip("/") + "/api/files"
        connection.putrequest("POST", route)
        connection.putheader("Authorization", f"Bearer {token}")
        connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
        connection.putheader("Content-Length", str(content_length))
        connection.endheaders()
        connection.send(fields)
        connection.send(prefix)
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                connection.send(chunk)
        connection.send(suffix)
        response = connection.getresponse()
        response_body = response.read(64 * 1024)
        if response.status < 200 or response.status >= 300:
            detail = response_body.decode("utf-8", errors="replace").strip()
            if token:
                detail = detail.replace(token, "[redacted]")
            detail = " ".join(detail.split())[:1200]
            reason = f": {detail}" if detail else ""
            raise OSError(f"storage server returned HTTP {response.status}{reason}")
        try:
            envelope = json.loads(response_body.decode("utf-8"))
            file_id = str(envelope.get("data", {}).get("id") or "")
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            envelope = {}
            file_id = ""
        if not envelope.get("success") or not _FILE_ID_PATTERN.fullmatch(f"/api/files/{file_id}/download"):
            raise OSError("storage server returned an invalid upload response")
        return file_id
    finally:
        connection.close()


def _upload_batch(root: Path, paths: list[Path]) -> bool:
    endpoint, project, token = _config()
    if not token:
        LOGGER.warning("Storage Service sync is disabled: STORAGE_SERVICE_API_TOKEN is not configured")
        return False
    pending = _relative_files(root, paths)
    if not pending:
        return True
    all_uploaded = True
    uploaded_ids: dict[str, str] = {}
    images = [item for item in pending if item[0].suffix.lower() in {".webp", ".jpg", ".jpeg", ".png"}]
    metadata = [item for item in pending if item not in images]
    for path, relative in images:
        try:
            file_id = _indexed_file_id(root, relative)
            if not file_id:
                file_id = _upload_one(endpoint, project, token, path, relative)
                _record_file_id(root, relative, file_id)
            uploaded_ids[relative] = file_id
            if _sync_plate_database_reference(root, relative, file_id):
                path.unlink(missing_ok=True)
            else:
                all_uploaded = False
        except (HTTPError, URLError, OSError, TimeoutError, ValueError) as error:
            LOGGER.warning("Could not sync scan artifact %s to Report: %s", relative, error)
            all_uploaded = False
    current_manifests = [path for path, relative in pending if relative.split("/")[1:2] == ["json"] and path.suffix.lower() == ".json"]
    _replace_json_paths(root, uploaded_ids, current_manifests)
    for path, relative in metadata:
        try:
            _upload_one(endpoint, project, token, path, relative)
            if path.suffix.lower() != ".json":
                path.unlink(missing_ok=True)
            if relative.split("/")[1:2] == ["json"] and path.name.endswith("_plate.json"):
                if not _sync_report_manifest(_report_data_endpoint(), token, root, path):
                    all_uploaded = False
        except (HTTPError, URLError, OSError, TimeoutError, ValueError) as error:
            LOGGER.warning("Could not sync scan artifact %s to Storage Service: %s", relative, error)
            all_uploaded = False
    return all_uploaded


def _run_upload_worker() -> None:
    while True:
        root, paths = _UPLOADS.get()
        try:
            for attempt in range(5):
                if _upload_batch(root, paths):
                    break
                time.sleep(min(2 ** attempt, 30))
            else:
                LOGGER.error("Giving up syncing scan files after retries; local copies were retained")
        except Exception:
            LOGGER.exception("Unexpected error while syncing scan files; local copies were retained")
        finally:
            _UPLOADS.task_done()


def enqueue_artifacts(output_dir: Path, *paths: str | Path | None) -> None:
    """Upload generated data in the background; never block the camera loop."""

    files = [Path(path) for path in paths if path]
    if not files:
        return
    endpoint, _, token = _config()
    if not endpoint or not token:
        return
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if not _WORKER_STARTED:
            threading.Thread(target=_run_upload_worker, name="report-storage-sync", daemon=True).start()
            _WORKER_STARTED = True
    try:
        _UPLOADS.put_nowait((Path(output_dir), files))
    except queue.Full:
        LOGGER.error("Report storage upload queue is full; local copies were retained")


def fetch_remote_image(file_id: str | None) -> tuple[bytes, str] | None:
    """Fetch a remote evidence image for a Car Scan image response."""

    if not file_id:
        return None
    if not _FILE_ID_PATTERN.fullmatch(f"/api/files/{file_id}/download"):
        return None
    endpoint, _, token = _config()
    if not token:
        return None
    url = f"{endpoint}/api/files/{file_id}/download"
    request = Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urlopen(request, timeout=10) as response:
            content_type = response.headers.get("Content-Type", "image/webp").split(";", 1)[0]
            if not content_type.startswith("image/"):
                return None
            return response.read(), content_type
    except (HTTPError, URLError, OSError, TimeoutError):
        return None
