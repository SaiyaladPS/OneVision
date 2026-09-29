from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.car_scan.web import _parse_roi, create_app, reset_runtime_state


class WebAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self._old_env = {
            key: os.environ.get(key)
            for key in (
                "CAR_SCAN_AUTH_DB",
                "CAR_SCAN_AUTH_SECRET",
                "CAR_SCAN_ADMIN_USERNAME",
                "CAR_SCAN_ADMIN_PASSWORD",
                "CAR_SCAN_DATABASE_URL",
                "CAR_SCAN_IP_CAMERA_URL",
                "CAR_SCAN_IP_CAMERA_URLS",
                "CAR_SCAN_OUTPUT_DIR",
            )
        }
        os.environ["CAR_SCAN_AUTH_DB"] = str(Path(self._tmp.name) / "auth.sqlite")
        os.environ["CAR_SCAN_AUTH_SECRET"] = "test-secret"
        os.environ["CAR_SCAN_ADMIN_USERNAME"] = "admin"
        os.environ["CAR_SCAN_ADMIN_PASSWORD"] = "secret1"
        os.environ["CAR_SCAN_DATABASE_URL"] = ""
        os.environ["CAR_SCAN_IP_CAMERA_URL"] = ""
        os.environ["CAR_SCAN_IP_CAMERA_URLS"] = ""
        os.environ["CAR_SCAN_OUTPUT_DIR"] = str(Path(self._tmp.name) / "out")
        self.client = TestClient(create_app())

    def tearDown(self) -> None:
        reset_runtime_state()
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._tmp.cleanup()

    def _login(self, username: str = "admin", password: str = "secret1") -> None:
        response = self.client.post("/api/auth/login", json={"username": username, "password": password})
        self.assertEqual(response.status_code, 200, response.text)

    def test_new_login_revokes_the_previous_session(self) -> None:
        self._login()
        other = TestClient(create_app())
        response = other.post("/api/auth/login", json={"username": "admin", "password": "secret1"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)
        self.assertEqual(other.get("/api/auth/me").status_code, 200)

    def test_roi_update_is_broadcast_over_websocket(self) -> None:
        self._login()
        created = self.client.post(
            "/api/users",
            json={
                "username": "roioperator",
                "display_name": "ROI Operator",
                "role": "operator",
                "password": "secret1",
            },
        )
        self.assertEqual(created.status_code, 200, created.text)
        other = TestClient(create_app())
        login = other.post("/api/auth/login", json={"username": "roioperator", "password": "secret1"})
        self.assertEqual(login.status_code, 200, login.text)
        with self.client.websocket_connect("/ws") as sender, other.websocket_connect("/ws") as receiver:
            sender.receive_json()
            receiver.receive_json()
            update = self.client.post(
                "/api/worker/roi",
                json={
                    "host": "192.168.100.50",
                    "roi": {"enabled": True, "shape": "rectangle", "x": 0.2, "y": 0.3, "width": 0.4, "height": 0.5},
                },
            )
            self.assertEqual(update.status_code, 200, update.text)
            payload = receiver.receive_json()
        self.assertEqual(payload["type"], "ROI_UPDATED")
        self.assertEqual(payload["data"]["host"], "192.168.100.50")
        self.assertEqual(payload["data"]["roi"]["x"], 0.2)

    def test_home_page_redirects_when_anonymous(self) -> None:
        response = self.client.get("/", follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["location"], "/login")

    def test_login_page(self) -> None:
        response = self.client.get("/login")
        self.assertEqual(response.status_code, 200)
        self.assertIn("เข้าสู่ระบบ", response.text)
        self.assertIn("cdn.tailwindcss.com", response.text)
        self.assertIn("tailwind-theme.js", response.text)

    def test_home_page_after_login(self) -> None:
        self._login()
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Car Scan", response.text)
        self.assertIn("cdn.tailwindcss.com", response.text)
        self.assertIn("tailwind-theme.js", response.text)
        self.assertIn("historyBtn", response.text)
        self.assertIn("reportBtn", response.text)
        self.assertIn("reportFrom", response.text)
        self.assertIn("historyQuery", response.text)
        self.assertIn("passwordBtn", response.text)
        self.assertIn("usersBtn", response.text)
        self.assertIn("usersHint", response.text)
        self.assertIn("userCancelEdit", response.text)
        self.assertIn("ipCameraUrl", response.text)
        self.assertIn("ipCameraUser", response.text)
        self.assertIn("ipCameraPassword", response.text)
        self.assertIn("ipCameraPath", response.text)
        self.assertIn("ipCameraSelect", response.text)
        self.assertIn("ipCameraAddBtn", response.text)
        self.assertIn("ipCameraRemoveBtn", response.text)
        self.assertIn("localCameraBtn", response.text)
        self.assertIn("cameraLiveStatus", response.text)
        self.assertIn('id="camStart"', response.text)
        self.assertIn("cameraGrid", response.text)
        self.assertIn("gpuChip", response.text)
        self.assertIn("computeMode", response.text)
        self.assertIn("roiPanel", response.text)
        self.assertIn("app.js?v=ops41", response.text)
        self.assertIn("app.css?v=ops24", response.text)
        self.assertIn("OneVision", response.text)
        self.assertIn("Noto+Sans+Thai", response.text)
        self.assertIn("camerasBtn", response.text)
        self.assertIn("camerasPanel", response.text)
        self.assertIn("camerasClose", response.text)
        self.assertIn("camerasToolbarBtn", response.text)
        self.assertIn("cameraWallPager", response.text)
        self.assertIn("cameraWallMeta", response.text)
        script = self.client.get("/static/app.js")
        self.assertEqual(script.status_code, 200)
        self.assertIn("bindVisibleCameraStreams", script.text)
        self.assertIn("is-offpage", script.text)
        self.assertIn("compute_switched", script.text)
        self.assertIn("cropMeta", response.text)
        self.assertIn("tableMeta", response.text)
        self.assertIn('id="crops"', response.text)
        self.assertIn("result-table-panel", response.text)
        self.assertIn("crop-panel", response.text)

    def test_scan_log_requires_login(self) -> None:
        response = self.client.get("/api/scans")
        self.assertEqual(response.status_code, 401)

    def test_reports_require_login(self) -> None:
        response = self.client.get("/api/reports")
        self.assertEqual(response.status_code, 401)
        csv_response = self.client.get("/api/reports.csv")
        self.assertEqual(csv_response.status_code, 401)

    def test_reports_empty_without_database(self) -> None:
        self._login()
        response = self.client.get("/api/reports?from=2026-09-08&to=2026-09-14")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["from"], "2026-09-08")
        self.assertEqual(payload["to"], "2026-09-14")
        self.assertEqual(payload["vehicle_count"], 0)
        self.assertEqual(payload["hours"][0]["hour"], 0)
        self.assertEqual(len(payload["hours"]), 24)
        self.assertEqual(payload["database"], "missing")
        self.assertNotIn("_rows", payload)

    def test_reports_reject_inverted_range(self) -> None:
        self._login()
        response = self.client.get("/api/reports?from=2026-09-14&to=2026-09-01")
        self.assertEqual(response.status_code, 400)

    def test_reports_csv_without_database(self) -> None:
        self._login()
        response = self.client.get("/api/reports.csv?from=2026-09-08&to=2026-09-14")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("text/csv", response.headers.get("content-type", ""))
        self.assertTrue(response.content.startswith(b"\xef\xbb\xbf"))
        self.assertIn(b"plate_number", response.content)

    def test_scan_log_empty_without_database(self) -> None:
        self._login()
        response = self.client.get("/api/scans?day=2026-09-14")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["date"], "2026-09-14")
        self.assertEqual(payload["groups"], [])
        self.assertEqual(payload["database"], "missing")

    def test_scan_log_accepts_unassigned_operator_filter(self) -> None:
        self._login()
        response = self.client.get("/api/scans?day=2026-09-14&operator_id=unassigned")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["groups"], [])

    def test_scan_log_rejects_bad_operator(self) -> None:
        self._login()
        response = self.client.get("/api/scans?operator_id=not-a-user")
        self.assertEqual(response.status_code, 400)

    def test_scan_search_empty_without_database(self) -> None:
        self._login()
        response = self.client.get("/api/scans?q=1356")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["query"], "1356")
        self.assertEqual(payload["groups"], [])
        self.assertIsNone(payload["date"])

    def test_scan_detail_requires_login(self) -> None:
        response = self.client.get("/api/scans/1")
        self.assertEqual(response.status_code, 401)

    def test_scan_image_requires_login(self) -> None:
        response = self.client.get("/api/scans/1/image")
        self.assertEqual(response.status_code, 401)

    def test_password_change_requires_current_password(self) -> None:
        self._login()
        wrong = self.client.post(
            "/api/auth/password",
            json={"current_password": "nope-nope", "new_password": "secret2"},
        )
        self.assertEqual(wrong.status_code, 400)
        changed = self.client.post(
            "/api/auth/password",
            json={"current_password": "secret1", "new_password": "secret2"},
        )
        self.assertEqual(changed.status_code, 200, changed.text)
        self.client.post("/api/auth/logout")
        self._login("admin", "secret2")

    def test_health(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["app"], "car-scan-web")
        self.assertIn("max_upload", payload)
        self.assertIn("ip_camera", payload)
        self.assertIn("ip_cameras", payload)
        self.assertEqual(payload["ip_cameras"], [])
        self.assertEqual(payload.get("worker"), "ai-worker")
        self.assertIn("gpu", payload)

    def test_redacts_camera_credentials_from_url(self) -> None:
        from src.car_scan.config import is_camera_stream_url, redact_stream_url, resolve_camera_url

        self.assertEqual(
            redact_stream_url("rtsp://admin:secret@192.168.100.50:554/Streaming/Channels/101"),
            "rtsp://192.168.100.50:554/Streaming/Channels/101",
        )
        self.assertTrue(is_camera_stream_url("rtsp://192.168.100.50:554/Streaming/Channels/101"))
        self.assertFalse(is_camera_stream_url("file:///etc/passwd"))
        cloned = resolve_camera_url(
            "192.168.100.191",
            ("rtsp://admin:secret@192.168.100.50:554/Streaming/Channels/101",),
        )
        self.assertEqual(cloned, "rtsp://admin:secret@192.168.100.191:554/Streaming/Channels/101")
        self.assertEqual(
            redact_stream_url(cloned),
            "rtsp://192.168.100.191:554/Streaming/Channels/101",
        )

    def test_camera_job_requires_url(self) -> None:
        self._login()
        response = self.client.post("/api/jobs", data={"media_type": "camera"})
        self.assertEqual(response.status_code, 400)

    def test_add_camera_from_host(self) -> None:
        os.environ["CAR_SCAN_IP_CAMERA_URL"] = "rtsp://admin:secret@192.168.100.50:554/Streaming/Channels/101"
        self._login()
        missing = self.client.post("/api/cameras", json={"host": ""})
        self.assertEqual(missing.status_code, 400)
        response = self.client.post("/api/cameras", json={"host": "192.168.100.191"})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["host"], "192.168.100.191")
        hosts = {item["host"] for item in payload["cameras"]}
        self.assertEqual(hosts, {"192.168.100.50", "192.168.100.191"})
        health = self.client.get("/api/health").json()
        self.assertEqual({item["host"] for item in health["ip_cameras"]}, hosts)
        builtin = {item["host"]: item["builtin"] for item in payload["cameras"]}
        self.assertTrue(builtin["192.168.100.50"])
        self.assertFalse(builtin["192.168.100.191"])
        blocked = self.client.delete("/api/cameras", params={"host": "192.168.100.50"})
        self.assertEqual(blocked.status_code, 400)
        removed = self.client.delete("/api/cameras", params={"host": "192.168.100.191"})
        self.assertEqual(removed.status_code, 200, removed.text)
        self.assertEqual({item["host"] for item in removed.json()["cameras"]}, {"192.168.100.50"})

    def test_add_camera_with_own_username_password(self) -> None:
        from src.car_scan.config import Settings, compose_camera_url, listed_camera_urls, redact_stream_url

        os.environ["CAR_SCAN_IP_CAMERA_URL"] = "rtsp://admin:secret@192.168.100.50:554/Streaming/Channels/101"
        self.assertEqual(
            compose_camera_url(
                "10.0.75.24",
                username="admin",
                password="VLPcam123",
                template="rtsp://admin:secret@192.168.100.50:554/Streaming/Channels/101",
            ),
            "rtsp://admin:VLPcam123@10.0.75.24:554/Streaming/Channels/101",
        )
        self._login()
        response = self.client.post(
            "/api/cameras",
            json={"host": "10.0.75.24", "username": "admin", "password": "VLPcam123"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["host"], "10.0.75.24")
        dumped = json.dumps(payload)
        self.assertNotIn("VLPcam123", dumped)
        self.assertNotIn("secret", dumped)
        added = next(item for item in payload["cameras"] if item["host"] == "10.0.75.24")
        self.assertTrue(added["has_auth"])
        self.assertFalse(added["builtin"])
        settings = Settings.from_env()
        urls = listed_camera_urls(settings)
        own = next(url for url in urls if "10.0.75.24" in url)
        self.assertEqual(own, "rtsp://admin:VLPcam123@10.0.75.24:554/Streaming/Channels/101")
        self.assertEqual(redact_stream_url(own), "rtsp://10.0.75.24:554/Streaming/Channels/101")
        first = next(url for url in urls if "192.168.100.50" in url)
        self.assertIn("secret", first)
        self.assertNotIn("VLPcam123", first)

    def test_rtsp_candidates_try_other_vendor_paths(self) -> None:
        from src.car_scan.config import apply_rtsp_path, compose_camera_url, rtsp_url_candidates

        hik = "rtsp://admin:VLPcam123@10.0.75.24:554/Streaming/Channels/101"
        candidates = rtsp_url_candidates(hik)
        self.assertEqual(candidates[0], hik)
        self.assertIn("rtsp://admin:VLPcam123@10.0.75.24:554/cam/realmonitor?channel=1&subtype=0", candidates)
        self.assertIn("rtsp://admin:VLPcam123@10.0.75.24:554/h264/ch1/main/av_stream", candidates)
        self.assertEqual(
            compose_camera_url(
                "10.0.75.24",
                username="admin",
                password="VLPcam123",
                path="/cam/realmonitor?channel=1&subtype=0",
            ),
            "rtsp://admin:VLPcam123@10.0.75.24:554/cam/realmonitor?channel=1&subtype=0",
        )
        self.assertEqual(
            apply_rtsp_path(hik, "/Streaming/Channels/102"),
            "rtsp://admin:VLPcam123@10.0.75.24:554/Streaming/Channels/102",
        )

    def test_add_local_computer_camera(self) -> None:
        from src.car_scan.config import (
            Settings,
            listed_camera_urls,
            parse_local_camera,
            public_cameras,
            resolve_camera_url,
        )

        self.assertEqual(parse_local_camera("local"), ("local", 0))
        self.assertEqual(parse_local_camera("webcam:1"), ("local-1", 1))
        self.assertEqual(resolve_camera_url("local"), "local://0")
        self._login()
        response = self.client.post("/api/cameras", json={"host": "local"})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["host"], "local")
        local = next(item for item in payload["cameras"] if item["host"] == "local")
        self.assertEqual(local["kind"], "local")
        self.assertEqual(local["device_index"], 0)
        self.assertFalse(local["builtin"])
        settings = Settings.from_env()
        self.assertIn("local://0", listed_camera_urls(settings))
        cameras = public_cameras(listed_camera_urls(settings))
        self.assertTrue(any(item.get("kind") == "local" for item in cameras))
        removed = self.client.delete("/api/cameras", params={"host": "local"})
        self.assertEqual(removed.status_code, 200, removed.text)
        self.assertFalse(any(item["host"] == "local" for item in removed.json()["cameras"]))

    def test_live_camera_requires_login_and_url(self) -> None:
        anonymous = self.client.post("/api/cameras/live", json={"host": "192.168.100.50"})
        self.assertEqual(anonymous.status_code, 401)
        self._login()
        missing = self.client.post("/api/cameras/live", json={"host": "192.168.100.50"})
        self.assertEqual(missing.status_code, 400)
        frame = self.client.get("/api/cameras/live.jpg")
        self.assertEqual(frame.status_code, 404)
        motion = self.client.get("/api/cameras/live.mjpeg")
        self.assertEqual(motion.status_code, 404)

    def test_worker_status_requires_login(self) -> None:
        response = self.client.get("/api/worker")
        self.assertEqual(response.status_code, 401)
        self._login()
        payload = self.client.get("/api/worker").json()
        self.assertEqual(payload["worker"], "ai-worker")
        self.assertEqual(payload["cameras"], [])
        self.assertIn("gpu", payload)
        empty = self.client.post("/api/worker/start-all")
        self.assertEqual(empty.status_code, 400)

    def test_worker_roi_stays_per_host(self) -> None:
        from src.car_scan.config import Settings
        from src.car_scan.worker import WorkerPool

        pool = WorkerPool(
            Settings(
                root=Path.cwd(),
                output_dir=Path(self._tmp.name),
                database_url="",
            )
        )
        pool.set_roi("192.168.100.50", {"enabled": True, "x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4})
        pool.set_roi("192.168.100.191", {"enabled": True, "x": 0.5, "y": 0.5, "width": 0.4, "height": 0.4})
        pool.set_roi("", {"enabled": True, "x": 0, "y": 0, "width": 1, "height": 1})
        self.assertEqual(pool.roi_by_host["192.168.100.50"]["x"], 0.1)
        self.assertEqual(pool.roi_by_host["192.168.100.191"]["x"], 0.5)
        self.assertEqual(pool.default_roi["width"], 1)

    def test_gpu_hub_prioritises_lane_tracking_an_unconfirmed_plate(self) -> None:
        import time
        from types import SimpleNamespace

        from src.car_scan.config import Settings
        from src.car_scan.worker import IDLE_LANE_INTERVAL, GpuHub

        hub = GpuHub(
            Settings(
                root=Path.cwd(),
                output_dir=Path(self._tmp.name),
                database_url="",
            )
        )
        now = time.monotonic()
        hot = SimpleNamespace(hot=True, last_infer_at=now)
        idle_recent = SimpleNamespace(hot=False, last_infer_at=now)
        idle_stale = SimpleNamespace(hot=False, last_infer_at=now - IDLE_LANE_INTERVAL - 1.0)
        hub.lanes = {"hot": hot, "recent": idle_recent, "stale": idle_stale}
        for host in ("recent", "stale", "hot"):
            hub.submit(host, None, 1)

        batch = [host for host, _ in hub._next_batch()]

        # The lane with an unconfirmed plate goes first, a stale idle lane
        # keeps its heartbeat, and the recently inferred idle lane waits.
        self.assertEqual(batch[0], "hot")
        self.assertIn("stale", batch)
        self.assertNotIn("recent", batch)
        self.assertEqual(set(hub.pending), {"recent"})

        # Without any hot lane every pending frame is drained as before.
        hot.hot = False
        hub.submit("hot", None, 2)
        self.assertEqual(sorted(host for host, _ in hub._next_batch()), ["hot", "recent"])
        self.assertEqual(hub.pending, {})

    def test_hybrid_hub_keeps_hot_lanes_on_gpu_and_idle_on_cpu(self) -> None:
        import time
        from types import SimpleNamespace

        from src.car_scan.config import Settings
        from src.car_scan.worker import IDLE_LANE_INTERVAL, GpuHub

        hub = GpuHub(
            Settings(
                root=Path.cwd(),
                output_dir=Path(self._tmp.name),
                database_url="",
                compute_mode="hybrid",
            )
        )
        hub.plan = SimpleNamespace(hybrid_cpu_yolo=True)
        now = time.monotonic()
        hub.lanes = {
            "hot": SimpleNamespace(hot=True, last_infer_at=now),
            "idle": SimpleNamespace(hot=False, last_infer_at=now - IDLE_LANE_INTERVAL - 1.0),
        }
        hub.submit("hot", None, 1)
        hub.submit("idle", None, 2)
        self.assertEqual([host for host, _ in hub._next_batch("gpu")], ["hot"])
        self.assertEqual([host for host, _ in hub._next_batch("cpu")], ["idle"])
        self.assertEqual(hub.pending, {})

    def test_preview_budget_slows_down_when_many_cameras_are_open(self) -> None:
        from src.car_scan.config import Settings
        from src.car_scan.worker import preview_budget

        settings = Settings(
            root=Path.cwd(),
            output_dir=Path(self._tmp.name),
            database_url="",
            preview_fps=10.0,
            preview_max_dimension=960,
        )
        few_fps, few_dim, _ = preview_budget(1, settings)
        wall_fps, wall_dim, _ = preview_budget(8, settings)
        self.assertGreaterEqual(few_fps, wall_fps)
        self.assertLessEqual(wall_fps, 8.0)
        self.assertLessEqual(wall_dim, 480)
        self.assertGreater(few_dim, wall_dim)

    def test_worker_compute_requires_login_and_can_switch_to_cpu(self) -> None:
        anonymous = self.client.post("/api/worker/compute", json={"mode": "cpu"})
        self.assertEqual(anonymous.status_code, 401)
        self._login()
        response = self.client.post("/api/worker/compute", json={"mode": "cpu"})
        self.assertEqual(response.status_code, 200, response.text)
        gpu = response.json()["gpu"]
        self.assertEqual(gpu["requested"], "cpu")
        self.assertEqual(gpu["mode"], "cpu")
        self.assertEqual(gpu["yolo_device"], "cpu")
        self.assertEqual(gpu["ocr_device"], "cpu")

    def test_worker_roi_requires_login(self) -> None:
        anonymous = self.client.post("/api/worker/roi", json={"host": "", "roi": {"enabled": True, "x": 0.2, "y": 0.2, "width": 0.5, "height": 0.5}})
        self.assertEqual(anonymous.status_code, 401)
        self._login()
        response = self.client.post(
            "/api/worker/roi",
            json={"host": "", "roi": {"enabled": True, "shape": "rectangle", "x": 0.2, "y": 0.2, "width": 0.5, "height": 0.5}},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["cameras"], [])

    def test_live_cctv_plate_payload_includes_crop_and_ocr(self) -> None:
        from src.car_scan.worker import client_live_plate, live_public_plate

        payload = live_public_plate(
            {
                "id": 3,
                "country": "thai",
                "province": "นนทบุรี",
                "plate_prefix": "68",
                "plate_number": "3197",
                "vehicle_type": "car",
                "recognition_confidence": 0.91,
                "captured_at": 1710000000.5,
                "ocr": {"text": "68-3197", "method": "structured-yolo", "confidence": 0.88},
                "crop_image": str(Path(self._tmp.name) / "crop.jpg"),
                "full_vehicle_image": str(Path(self._tmp.name) / "car.jpg"),
            },
            host="192.168.100.50",
            label="Camera 01",
        )
        self.assertEqual(payload["ocr"]["text"], "68-3197")
        self.assertEqual(payload["ocr"]["method"], "structured-yolo")
        self.assertEqual(payload["province"], "นนทบุรี")
        self.assertEqual(payload["captured_at"], 1710000000.5)
        self.assertEqual(payload["crop_url"], "/api/worker/cameras/192.168.100.50/plates/3/crop?v=0")
        self.assertEqual(payload["full_vehicle_url"], "/api/worker/cameras/192.168.100.50/plates/3/vehicle?v=0")
        client = client_live_plate(payload)
        self.assertNotIn("crop_image", client)
        self.assertEqual(client["crop_url"], payload["crop_url"])

    def test_worker_plate_crop_requires_login(self) -> None:
        anonymous = self.client.get("/api/worker/cameras/192.168.100.50/plates/1/crop")
        self.assertEqual(anonymous.status_code, 401)
        self._login()
        missing = self.client.get("/api/worker/cameras/192.168.100.50/plates/1/crop")
        self.assertEqual(missing.status_code, 404)

    def test_camera_job_requires_login(self) -> None:
        response = self.client.post("/api/jobs", data={"media_type": "camera"})
        self.assertEqual(response.status_code, 401)

    def test_parse_roi_defaults_to_enabled(self) -> None:
        roi = _parse_roi('{"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}')
        self.assertIsNotNone(roi)
        self.assertTrue(roi["enabled"])
        self.assertEqual(roi["x"], 0.1)
        self.assertEqual(roi["width"], 0.3)

    def test_parse_roi_honours_explicit_disable(self) -> None:
        roi = _parse_roi('{"enabled": false, "x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}')
        self.assertIsNotNone(roi)
        self.assertFalse(roi["enabled"])

    def test_create_job_requires_login(self) -> None:
        response = self.client.post("/api/jobs", data={"media_type": "image"})
        self.assertEqual(response.status_code, 401)

    def test_create_job_accepts_uploaded_image(self) -> None:
        self._login()
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        response = self.client.post(
            "/api/jobs",
            files={"file": ("plate.png", png, "image/png")},
            data={
                "media_type": "image",
                "roi": '{"enabled": false, "x": 0, "y": 0, "width": 1, "height": 1}',
            },
        )
        payload = response.json()
        self.assertNotEqual(payload.get("detail"), "ต้องแนบไฟล์ในฟิลด์ file")
        self.assertEqual(response.status_code, 202)
        self.assertIn("id", payload)

    def test_create_job_rejects_missing_file(self) -> None:
        self._login()
        response = self.client.post("/api/jobs", data={"media_type": "image"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "ต้องแนบไฟล์ในฟิลด์ file")

    def test_safe_scan_image_resolves_relative_paths(self) -> None:
        from src.car_scan.config import Settings
        from src.car_scan.web import _safe_scan_image

        output = Path(self._tmp.name) / "scan-data"
        image = output / "20260914" / "lao" / "car.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"jpeg")
        previous = os.environ.get("CAR_SCAN_OUTPUT_DIR")
        os.environ["CAR_SCAN_OUTPUT_DIR"] = str(output)
        try:
            settings = Settings.from_env()
            self.assertEqual(_safe_scan_image(settings, "20260914/lao/car.jpg"), image.resolve())
            self.assertEqual(_safe_scan_image(settings, str(image)), image.resolve())
        finally:
            if previous is None:
                os.environ.pop("CAR_SCAN_OUTPUT_DIR", None)
            else:
                os.environ["CAR_SCAN_OUTPUT_DIR"] = previous

    def test_viewer_cannot_scan_or_manage_users(self) -> None:
        self._login()
        created = self.client.post(
            "/api/users",
            json={
                "username": "gateview",
                "display_name": "ผู้ดูผล",
                "role": "viewer",
                "password": "secret1",
            },
        )
        self.assertEqual(created.status_code, 200, created.text)
        self.client.post("/api/auth/logout")
        self._login("gateview", "secret1")
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        scan = self.client.post(
            "/api/jobs",
            files={"file": ("plate.png", png, "image/png")},
            data={"media_type": "image"},
        )
        self.assertEqual(scan.status_code, 403)
        users = self.client.get("/api/users")
        self.assertEqual(users.status_code, 403)

    def test_admin_can_create_update_and_delete_user(self) -> None:
        self._login()
        created = self.client.post(
            "/api/users",
            json={
                "username": "gateop",
                "display_name": "พนักงานด่าน",
                "role": "operator",
                "password": "secret1",
            },
        )
        self.assertEqual(created.status_code, 200, created.text)
        user_id = created.json()["id"]
        updated = self.client.patch(
            f"/api/users/{user_id}",
            json={"display_name": "พนักงานเวรเช้า", "role": "viewer"},
        )
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertEqual(updated.json()["display_name"], "พนักงานเวรเช้า")
        self.assertEqual(updated.json()["role"], "viewer")
        listed = self.client.get("/api/users")
        self.assertEqual(listed.status_code, 200, listed.text)
        names = [item["username"] for item in listed.json()["users"]]
        self.assertIn("gateop", names)
        deleted = self.client.delete(f"/api/users/{user_id}")
        self.assertEqual(deleted.status_code, 200, deleted.text)
        self_id = self.client.get("/api/auth/me").json()["id"]
        blocked = self.client.delete(f"/api/users/{self_id}")
        self.assertEqual(blocked.status_code, 400)


if __name__ == "__main__":
    unittest.main()
