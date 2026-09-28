from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from auto.auto import Detection, Verification, consensus, image_paths, intersection_over_union, label_status, source_status, yolo_label


class AutoLabelingTests(unittest.TestCase):
    def test_yolo_label_uses_normalised_xywh(self) -> None:
        detection = Detection(2, "license_plate", 0.9, (10, 20, 50, 40))

        label = yolo_label(detection, 100, 80)

        self.assertEqual(label, "2 0.300000 0.375000 0.400000 0.250000")

    def test_pass_requires_an_independent_positive_verifier(self) -> None:
        score, positive = consensus(0.99, [])
        status, reason = label_status(score, positive, 0.95, 0.80)

        self.assertEqual((score, status, reason), (0.99, "REVIEW", "needs_independent_verification"))

        score, positive = consensus(
            0.99,
            [Verification("gemini", 0.97, True, {"is_license_plate": True})],
        )
        status, reason = label_status(score, positive, 0.95, 0.80)

        self.assertEqual((score, status, reason), (0.9857, "PASS", "high_confidence_with_independent_verification"))

    def test_confidence_ranges_match_the_review_policy(self) -> None:
        self.assertEqual(label_status(0.95, 1, 0.95, 0.80)[0], "PASS")
        self.assertEqual(label_status(0.80, 1, 0.95, 0.80)[0], "REVIEW")
        self.assertEqual(label_status(0.7999, 1, 0.95, 0.80)[0], "REJECT")

    def test_iou_is_used_to_verify_the_same_plate(self) -> None:
        self.assertAlmostEqual(intersection_over_union((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)
        self.assertEqual(intersection_over_union((0, 0, 2, 2), (3, 3, 4, 4)), 0.0)

    def test_source_with_any_uncertain_box_never_enters_pass(self) -> None:
        self.assertEqual(source_status(["PASS", "PASS"]), "PASS")
        self.assertEqual(source_status(["PASS", "REVIEW"]), "REVIEW")
        self.assertEqual(source_status(["PASS", "REJECT"]), "REJECT")

    def test_default_dataset_scan_excludes_prior_generated_folders(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "source.jpg"
            raw.write_bytes(b"image")
            generated = root / "PASS" / "images"
            generated.mkdir(parents=True)
            (generated / "old.jpg").write_bytes(b"image")

            paths = image_paths(root, excluded_roots=[root / "PASS"])

        self.assertEqual(paths, [raw])
