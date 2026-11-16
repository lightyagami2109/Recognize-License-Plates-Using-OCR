"""Metric checks with hand-computable boxes, duplicate detections and misses."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluate import evaluate, iou


class EvaluationTests(unittest.TestCase):
    def write_table(self, path, rows, predictions=False):
        fields = ["frame_number", "plate_number", "x1", "y1", "x2", "y2"]
        if predictions:
            fields.append("detection_confidence")
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)

    def test_matching_duplicates_misses_and_ocr_denominators(self):
        with TemporaryDirectory() as folder:
            truth, predicted = Path(folder) / "truth.csv", Path(folder) / "predicted.csv"
            self.write_table(truth, [
                [0, "00123", 0, 0, 10, 10], [0, "MH12", 20, 0, 30, 10],
                [1, "", "", "", "", ""],
            ])
            self.write_table(predicted, [
                [0, "WRONG", 0, 0, 10, 10, 0.6],
                [0, "00123", 0, 0, 10, 10, 0.9],
                [1, "FALSE", 0, 0, 10, 10, 0.8],
                [2, "UNLABELLED", 0, 0, 10, 10, 0.9],
            ], predictions=True)
            result = evaluate(truth, predicted)
            self.assertEqual((result["true_positives"], result["false_positives"], result["false_negatives"]), (1, 2, 1))
            self.assertAlmostEqual(result["detection_precision"], 1 / 3)
            self.assertEqual(result["detection_recall"], 0.5)
            self.assertEqual(result["ocr_exact_match_accuracy"], 1.0)
            self.assertEqual(result["end_to_end_exact_match_accuracy"], 0.5)
            self.assertEqual(result["ignored_prediction_rows"], 1)

    def test_no_detections_and_unreadable_ground_truth(self):
        with TemporaryDirectory() as folder:
            truth, predicted = Path(folder) / "truth.csv", Path(folder) / "predicted.csv"
            self.write_table(truth, [[0, "ABC", 0, 0, 10, 10]])
            self.write_table(predicted, [], predictions=True)
            result = evaluate(truth, predicted)
            self.assertIsNone(result["detection_precision"])
            self.assertEqual(result["detection_recall"], 0)
            self.assertIsNone(result["ocr_exact_match_accuracy"])
            self.assertEqual(result["end_to_end_exact_match_accuracy"], 0)
            self.write_table(truth, [[0, "", 0, 0, 10, 10]])
            self.write_table(predicted, [[0, "ABC", 0, 0, 10, 10, 0.9]], predictions=True)
            result = evaluate(truth, predicted)
            self.assertEqual(result["true_positives"], 1)
            self.assertIsNone(result["ocr_exact_match_accuracy"])
            self.assertIsNone(result["end_to_end_exact_match_accuracy"])

    def test_invalid_and_contradictory_labels(self):
        with TemporaryDirectory() as folder:
            truth, predicted = Path(folder) / "truth.csv", Path(folder) / "predicted.csv"
            self.write_table(predicted, [], predictions=True)
            for box in [(0, 0, 0, 10), (0, 0, float("nan"), 10), (-1, 0, 10, 10)]:
                self.write_table(truth, [[0, "ABC", *box]])
                with self.assertRaisesRegex(ValueError, "row 2"):
                    evaluate(truth, predicted)
            self.write_table(truth, [[0, "ABC", 0, 0, 10, 10], [0, "", "", "", "", ""]])
            with self.assertRaisesRegex(ValueError, "contradictory"):
                evaluate(truth, predicted)
            predicted.write_text("frame_number,plate_number\n0,ABC\n")
            with self.assertRaisesRegex(ValueError, "required columns"):
                evaluate(truth, predicted)

    def test_iou(self):
        self.assertEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1)
        self.assertEqual(iou((0, 0, 10, 10), (20, 0, 30, 10)), 0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)


if __name__ == "__main__":
    unittest.main()
