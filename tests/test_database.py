from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from src.car_scan.database import (
    compact_plate_record,
    flatten_scans_to_passages,
    build_scan_report,
    group_scans_by_operator,
    jsonable_value,
    LiveScanPersister,
    live_plate_keys,
    merge_history_operators,
    plate_text_key,
    plate_query_terms,
    resolve_stored_path,
    scan_create_data,
    serialize_scan,
    stored_output_path,
)


class FakeArray:
    def __init__(self, values, shape=None):
        self._values = values
        if shape is not None:
            self.shape = shape
        elif hasattr(values, "__len__") and not isinstance(values, (str, bytes)):
            self.shape = (len(values),)
        else:
            self.shape = ()
        self.dtype = "float64"
        self.ndim = len(self.shape)
        self.size = 1
        for dim in self.shape:
            self.size *= dim

    def item(self):
        if self.size != 1:
            raise ValueError("can only convert an array of size 1")
        return self._values

    def tolist(self):
        return self._values


class ScanCreateDataTests(unittest.TestCase):
    def test_maps_scan_and_nested_plates(self) -> None:
        result = {
            "input": "scan/data/car.jpg",
            "annotated_image": "scan/data/car_annotated.jpg",
            "media_type": "image",
            "plate_count": 1,
            "plates": [
                {
                    "id": 3,
                    "country": "thai",
                    "province": "BKK",
                    "province_code": "10",
                    "plate_prefix": "กข",
                    "plate_prefix_code": "กข",
                    "plate_number": "1234",
                    "vehicle_type": "car",
                    "detection_confidence": 0.91,
                    "recognition_confidence": 0.88,
                    "full_vehicle_image": "scan/data/car-full_vehicle.jpg",
                    "crop_image": "scan/data/car-plate_crops.jpg",
                    "ocr": {"text": "กข 1234", "confidence": 0.88},
                }
            ],
        }
        data = scan_create_data(result)
        self.assertEqual(data["sourcePath"], "scan/data/car.jpg")
        self.assertEqual(data["annotatedImage"], "scan/data/car_annotated.jpg")
        self.assertEqual(data["mediaType"], "image")
        self.assertEqual(data["plateCount"], 1)
        self.assertEqual(data["rawResult"]["plates"][0]["plate_number"], "1234")
        plates = data["plates"]["create"]
        self.assertEqual(len(plates), 1)
        self.assertEqual(plates[0]["plateIndex"], 3)
        self.assertEqual(plates[0]["country"], "thai")
        self.assertEqual(plates[0]["platePrefix"], "กข")
        self.assertEqual(plates[0]["provinceCode"], "10")
        self.assertEqual(plates[0]["ocrText"], "กข 1234")
        self.assertEqual(plates[0]["ocrConfidence"], 0.88)
        self.assertEqual(plates[0]["vehicleType"], "car")
        self.assertEqual(plates[0]["vehicleImage"], "scan/data/car-full_vehicle.jpg")
        self.assertEqual(plates[0]["cropImage"], "scan/data/car-plate_crops.jpg")
        self.assertEqual(plates[0]["rawPlate"]["plate_number"], "1234")
        json.dumps(data, ensure_ascii=False)

    def test_wraps_json_columns_for_prisma(self) -> None:
        from prisma import Json

        from src.car_scan.database import wrap_prisma_json_fields

        data = wrap_prisma_json_fields(
            scan_create_data(
                {
                    "input": "scan/data/car.jpg",
                    "media_type": "image",
                    "plates": [
                        {
                            "id": 1,
                            "country": "thai",
                            "plate_number": "1234",
                            "ocr": {"text": "กข 1234", "confidence": 0.88},
                        }
                    ],
                }
            )
        )
        self.assertIsInstance(data["rawResult"], Json)
        self.assertEqual(data["rawResult"].data["plates"][0]["plate_number"], "1234")
        self.assertIsInstance(data["plates"]["create"][0]["rawPlate"], Json)
        self.assertEqual(data["plates"]["create"][0]["rawPlate"].data["plate_number"], "1234")

    def test_live_plate_keys_keep_stable_identity(self) -> None:
        plate = {
            "live_result_key": "video:1",
            "country": "lao",
            "plate_prefix": "ບຄ",
            "plate_number": "1356",
            "full_vehicle_image": "20260914/laos/a-full_vehicle.jpg",
        }
        self.assertEqual(live_plate_keys(plate), ["live:video:1", "reg:lao|ບຄ|1356"])
        compact = compact_plate_record(plate)
        self.assertEqual(compact["live_result_key"], "video:1")
        self.assertEqual(scan_create_data({"plates": [plate]})["plates"]["create"][0]["rawPlate"]["live_result_key"], "video:1")

    def test_live_plate_text_key_survives_country_flip_and_formatting(self) -> None:
        first = {"country": "thai", "plate_prefix": " กข ", "plate_number": "12-34"}
        second = {"country": "lao", "plate_prefix": "กข", "plate_number": "1234"}

        self.assertEqual(plate_text_key(first), "reg-text:กข|1234")
        self.assertEqual(plate_text_key(first), plate_text_key(second))

    def test_live_persister_updates_same_text_when_track_and_country_change(self) -> None:
        class FakeRepository:
            def __init__(self) -> None:
                self.calls = []

            def save_scan(self, _result, output_dir=None) -> int:
                return 41

            def upsert_live_plate(self, scan_id, plate, output_dir=None, plate_pk=None) -> int:
                self.calls.append((scan_id, plate_pk, plate))
                return int(plate_pk or 99)

        repository = FakeRepository()
        persister = LiveScanPersister(
            database_url="postgresql://test",
            output_dir=None,
            media_type="camera",
            source="camera",
        )
        persister._repository = lambda: repository  # type: ignore[method-assign]
        first = {"live_result_key": "camera:1", "country": "thai", "plate_prefix": "กข", "plate_number": "1234"}
        second = {"live_result_key": "camera:2", "country": "lao", "plate_prefix": " กข ", "plate_number": "12-34"}

        self.assertEqual(persister.save_plate(first), 99)
        self.assertEqual(persister.save_plate(second), 99)
        self.assertEqual([call[1] for call in repository.calls], [None, 99])
        self.assertEqual(persister.saved_count, 1)

    def test_omits_nested_create_when_no_plates(self) -> None:
        data = scan_create_data({"input": "empty.jpg", "plates": []})
        self.assertEqual(data["sourcePath"], "empty.jpg")
        self.assertEqual(data["plateCount"], 0)
        self.assertNotIn("plates", data)

    def test_truncates_country_to_varchar_limit(self) -> None:
        data = scan_create_data(
            {
                "input": "x.jpg",
                "plates": [{"id": 1, "country": "unknown_province_name"}],
            }
        )
        self.assertEqual(data["plates"]["create"][0]["country"], "unknown_province")

    def test_maps_operator_on_duty(self) -> None:
        data = scan_create_data(
            {
                "input": "gate.jpg",
                "operator_id": 7,
                "operator_name": "สมชาย",
                "operator_username": "somchai",
            }
        )
        self.assertEqual(data["operatorId"], 7)
        self.assertEqual(data["operatorName"], "สมชาย")
        self.assertEqual(data["operatorUsername"], "somchai")

    def test_strips_numpy_like_values_from_prisma_payload(self) -> None:
        result = {
            "input": "car.jpg",
            "plates": [
                {
                    "id": 1,
                    "country": "lao",
                    "plate_number": "1356",
                    "box": FakeArray([10, 20, 30, 40], shape=(4,)),
                    "best_crop": FakeArray([[1, 2], [3, 4]], shape=(2, 2)),
                    "ocr": {"text": "ບຄ 1356", "confidence": FakeArray(0.91, shape=())},
                }
            ],
        }
        data = scan_create_data(result)
        plate = data["plates"]["create"][0]
        self.assertEqual(plate["rawPlate"]["box"], [10, 20, 30, 40])
        self.assertNotIn("best_crop", plate["rawPlate"])
        self.assertEqual(plate["ocrConfidence"], 0.91)
        json.dumps(data, ensure_ascii=False)

    def test_stores_image_paths_relative_to_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            vehicle = output / "20260914" / "lao" / "cam-001-full_vehicle.jpg"
            crop = output / "20260914" / "lao" / "cam-001-plate_crops.jpg"
            annotated = output / "20260914" / "log" / "input_annotated.jpg"
            source = output / "web_uploads" / "abc" / "input.png"
            vehicle.parent.mkdir(parents=True)
            annotated.parent.mkdir(parents=True)
            source.parent.mkdir(parents=True)
            for path in (vehicle, crop, annotated, source):
                path.write_bytes(b"x")
            data = scan_create_data(
                {
                    "input": str(source),
                    "annotated_image": str(annotated),
                    "media_type": "image",
                    "plates": [
                        {
                            "id": 1,
                            "country": "lao",
                            "full_vehicle_image": str(vehicle),
                            "crop_image": str(crop),
                        }
                    ],
                },
                output_dir=output,
            )
            self.assertEqual(data["sourcePath"], "web_uploads/abc/input.png")
            self.assertEqual(data["annotatedImage"], "20260914/log/input_annotated.jpg")
            plate = data["plates"]["create"][0]
            self.assertEqual(plate["vehicleImage"], "20260914/lao/cam-001-full_vehicle.jpg")
            self.assertEqual(plate["cropImage"], "20260914/lao/cam-001-plate_crops.jpg")
            self.assertNotIn(":", plate["vehicleImage"])
            self.assertFalse(Path(plate["vehicleImage"]).is_absolute())

    def test_resolves_relative_stored_paths(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            image = output / "20260914" / "lao" / "plate.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"x")
            resolved = resolve_stored_path("20260914/lao/plate.jpg", output)
            self.assertEqual(resolved, image.resolve())
            self.assertEqual(stored_output_path(image, output), "20260914/lao/plate.jpg")

    def test_jsonable_drops_image_arrays(self) -> None:
        payload = jsonable_value(
            {
                "crop_image": "a.jpg",
                "best_crop": FakeArray([[1, 2], [3, 4]], shape=(2, 2)),
                "box": FakeArray([1, 2, 3, 4], shape=(4,)),
            }
        )
        self.assertEqual(payload["crop_image"], "a.jpg")
        self.assertEqual(payload["box"], [1, 2, 3, 4])
        self.assertNotIn("best_crop", payload)

    def test_serialize_scan_prefers_image_columns(self) -> None:
        run = SimpleNamespace(
            id=9,
            scannedAt=datetime(2026, 9, 14, 8, 30, tzinfo=timezone.utc),
            plateCount=1,
            sourcePath="web_uploads/abc/input.png",
            annotatedImage="20260914/log/input_annotated.jpg",
            outputVideo=None,
            mediaType="image",
            operatorId=1,
            operatorName="ผู้ดูแล",
            operatorUsername="admin",
            rawResult={},
            plates=[
                SimpleNamespace(
                    id=21,
                    country="lao",
                    province="ນະຄອນຫຼວງ",
                    provinceCode="01",
                    platePrefix="ບຄ",
                    platePrefixCode="ບຄ",
                    plateNumber="1356",
                    ocrText="ບຄ 1356",
                    ocrConfidence=0.9,
                    detectionConfidence=0.8,
                    recognitionConfidence=0.9,
                    overallConfidence=0.88,
                    confidenceLevel="HIGH",
                    vehicleType="car",
                    vehicleTypeConfidence=0.7,
                    plateType="car",
                    vehicleImage="20260914/lao/full.jpg",
                    cropImage="20260914/lao/crop.jpg",
                    ocrReadyImage="20260914/lao/ocr.jpg",
                    rawPlate={"full_vehicle_image": "D:/old/absolute.jpg"},
                )
            ],
        )
        row = serialize_scan(run)
        self.assertEqual(row["source_name"], "input.png")
        self.assertEqual(row["media_type"], "image")
        self.assertEqual(row["plates"][0]["full_vehicle_image"], "20260914/lao/full.jpg")
        self.assertEqual(row["plates"][0]["vehicle_type"], "car")
        self.assertEqual(row["plates"][0]["province_code"], "01")

    def test_compact_plate_keeps_relative_paths(self) -> None:
        plate = compact_plate_record(
            {
                "plate_number": "1234",
                "best_crop": FakeArray([1], shape=(2, 2)),
                "full_vehicle_image": r"D:\project\car-scan\scan\data\20260914\lao\a-full_vehicle.jpg",
            }
        )
        self.assertEqual(plate["plate_number"], "1234")
        self.assertNotIn("best_crop", plate)

    def test_omits_nested_create_when_no_plates(self) -> None:
        data = scan_create_data({"input": "empty.jpg", "plates": []})
        self.assertEqual(data["sourcePath"], "empty.jpg")
        self.assertEqual(data["plateCount"], 0)
        self.assertNotIn("plates", data)

    def test_truncates_country_to_varchar_limit(self) -> None:
        data = scan_create_data(
            {
                "input": "x.jpg",
                "plates": [{"id": 1, "country": "unknown_province_name"}],
            }
        )
        self.assertEqual(data["plates"]["create"][0]["country"], "unknown_province")

    def test_maps_operator_on_duty(self) -> None:
        data = scan_create_data(
            {
                "input": "gate.jpg",
                "operator_id": 7,
                "operator_name": "สมชาย",
                "operator_username": "somchai",
            }
        )
        self.assertEqual(data["operatorId"], 7)
        self.assertEqual(data["operatorName"], "สมชาย")
        self.assertEqual(data["operatorUsername"], "somchai")

    def test_groups_scans_by_operator(self) -> None:
        groups = group_scans_by_operator(
            [
                {"operator_id": 1, "operator_username": "admin", "operator_name": "ผู้ดูแล", "plate_count": 2},
                {"operator_id": 1, "operator_username": "admin", "operator_name": "ผู้ดูแล", "plate_count": 1},
                {"operator_id": None, "operator_username": "", "operator_name": "", "plate_count": 3},
            ]
        )
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["scan_count"], 2)
        self.assertEqual(groups[0]["plate_count"], 3)
        self.assertEqual(groups[1]["operator_id"], None)
        self.assertEqual(groups[1]["scan_count"], 1)

    def test_merges_roster_and_unassigned_operators(self) -> None:
        merged = merge_history_operators(
            [
                {"id": None, "username": "", "display_name": ""},
                {"id": 2, "username": "somchai", "display_name": "สมชาย"},
            ],
            [
                {"id": 1, "username": "admin", "display_name": "ผู้ดูแล", "active": True},
                {"id": 9, "username": "old", "display_name": "เลิกใช้", "active": False},
            ],
        )
        self.assertEqual([item["username"] for item in merged], ["somchai", "admin", ""])
        self.assertIsNone(merged[-1]["id"])

    def test_plate_query_keeps_script_and_digits(self) -> None:
        self.assertEqual(plate_query_terms("  ບຄ-1356  "), ["ບຄ-1356", "ບຄ1356", "1356"])
        self.assertEqual(plate_query_terms(""), [])

    def test_flattens_one_row_per_vehicle(self) -> None:
        passages = flatten_scans_to_passages(
            [
                {
                    "id": 10,
                    "operator_id": 1,
                    "operator_username": "admin",
                    "operator_name": "ผู้ดูแล",
                    "scanned_time": "10:32:44",
                    "plates": [
                        {"id": 1, "plate_prefix": "ບຄ", "plate_number": "1356", "province_code": "01", "vehicle_url": "/v1"},
                        {"id": 2, "plate_prefix": "ບບ", "plate_number": "9955", "vehicle_url": "/v2"},
                    ],
                }
            ]
        )
        self.assertEqual(len(passages), 2)
        self.assertEqual(passages[0]["plate_number"], "1356")
        self.assertEqual(passages[1]["plate_number"], "9955")
        self.assertEqual(passages[0]["vehicle_url"], "/v1")
        self.assertEqual(passages[0]["province_code"], "01")

    def test_flatten_search_keeps_only_matching_plate(self) -> None:
        passages = flatten_scans_to_passages(
            [
                {
                    "id": 10,
                    "plates": [
                        {"id": 1, "plate_prefix": "ບຄ", "plate_number": "1356"},
                        {"id": 2, "plate_prefix": "ບບ", "plate_number": "9955"},
                    ],
                }
            ],
            query="1356",
        )
        self.assertEqual(len(passages), 1)
        self.assertEqual(passages[0]["plate_number"], "1356")

    def test_builds_scan_report_buckets(self) -> None:
        report = build_scan_report(
            [
                {
                    "id": 10,
                    "date": "2026-09-14",
                    "scanned_time": "08:15:00",
                    "operator_id": 1,
                    "operator_username": "admin",
                    "operator_name": "ผู้ดูแล",
                    "media_type": "image",
                    "plates": [
                        {
                            "id": 1,
                            "country": "lao",
                            "province": "ນະຄອນຫຼວງວຽງຈັນ",
                            "vehicle_type": "car",
                            "plate_prefix": "ບຄ",
                            "plate_number": "1356",
                            "confidence": 0.91,
                            "confidence_level": "HIGH",
                        },
                        {
                            "id": 2,
                            "country": "thai",
                            "province": "กรุงเทพมหานคร",
                            "vehicle_type": "truck",
                            "plate_prefix": "กข",
                            "plate_number": "1234",
                            "confidence": 0.4,
                            "confidence_level": "LOW",
                        },
                    ],
                }
            ]
        )
        self.assertEqual(report["vehicle_count"], 2)
        self.assertEqual(report["scan_count"], 1)
        self.assertEqual(report["operator_count"], 1)
        self.assertEqual(report["days"][0]["count"], 2)
        self.assertEqual(report["hours"][8]["count"], 2)
        self.assertEqual(report["countries"][0]["count"], 1)
        self.assertEqual({row["label"] for row in report["confidence"]}, {"high", "low"})
        self.assertFalse(report["truncated"])
        self.assertEqual(len(report["passages"]), 2)


if __name__ == "__main__":
    unittest.main()
