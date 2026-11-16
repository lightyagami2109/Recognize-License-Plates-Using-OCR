"""Evaluate plate boxes and exact OCR text on manually labelled video frames."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path


BOX_FIELDS = ("x1", "y1", "x2", "y2")


def load_rows(path, predictions=False):
    rows = []
    required = {"frame_number", "plate_number", *BOX_FIELDS}
    if predictions:
        required.add("detection_confidence")
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        table = csv.DictReader(stream)
        if not required.issubset(table.fieldnames or []):
            raise ValueError(f"{path}: required columns are {', '.join(sorted(required))}")
        for line, row in enumerate(table, 2):
            try:
                if None in row or any(row.get(key) is None for key in required):
                    raise ValueError("wrong number of CSV fields")
                frame = int(row["frame_number"])
                if frame < 0:
                    raise ValueError("frame_number must be non-negative")
                text = "".join(row["plate_number"].upper().split()).replace("-", "")
                if any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" for character in text):
                    raise ValueError("plate text must use Latin letters, digits, spaces or hyphens")
                box = None
                if predictions or any(row[key].strip() for key in BOX_FIELDS):
                    box = tuple(float(row[key]) for key in BOX_FIELDS)
                    if (not all(math.isfinite(value) and value >= 0 for value in box)
                            or box[2] <= box[0] or box[3] <= box[1]):
                        raise ValueError("box must be finite, non-negative and have positive area")
                elif text:
                    raise ValueError("a labelled plate needs a bounding box")
                confidence = float(row["detection_confidence"]) if predictions else 1.0
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError("detection_confidence must be between 0 and 1")
                rows.append({"frame": frame, "text": text, "box": box, "confidence": confidence})
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}, row {line}: {error}") from error
    return rows


def iou(first, second):
    intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1]),
    )
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def evaluate(ground_truth, predictions):
    truth = load_rows(ground_truth)
    predicted = load_rows(predictions, predictions=True)
    if not truth:
        raise ValueError("Ground truth must explicitly label at least one frame")
    frames = defaultdict(list)
    for row in truth:
        frames[row["frame"]].append(row)
    for frame, rows in frames.items():
        boxes = [row["box"] for row in rows]
        if (None in boxes and len(rows) != 1) or len(set(boxes)) != len(boxes):
            raise ValueError(f"Ground truth frame {frame}: duplicate or contradictory labels")
    by_frame = defaultdict(list)
    ignored = 0
    for row in predicted:
        if row["frame"] in frames:
            by_frame[row["frame"]].append(row)
        else:
            ignored += 1
    tp = fp = fn = correct = matched_readable = readable = 0
    for frame, labels in frames.items():
        labels = [label for label in labels if label["box"] is not None]
        readable += sum(bool(label["text"]) for label in labels)
        remaining = set(range(len(labels)))
        for prediction in sorted(by_frame[frame], key=lambda row: row["confidence"], reverse=True):
            best = max(remaining, key=lambda index: iou(prediction["box"], labels[index]["box"]), default=None)
            if best is not None and iou(prediction["box"], labels[best]["box"]) >= 0.5:
                remaining.remove(best)
                tp += 1
                if labels[best]["text"]:
                    matched_readable += 1
                    correct += prediction["text"] == labels[best]["text"]
            else:
                fp += 1
        fn += len(remaining)
    return {
        "evaluated_frames": len(frames), "ignored_prediction_rows": ignored,
        "ground_truth_plates": tp + fn, "readable_ground_truth_plates": readable,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "matched_readable_plates": matched_readable, "correct_plate_reads": correct,
        "iou_threshold": 0.5,
        "detection_precision": tp / (tp + fp) if tp + fp else None,
        "detection_recall": tp / (tp + fn) if tp + fn else None,
        "ocr_exact_match_accuracy": correct / matched_readable if matched_readable else None,
        "end_to_end_exact_match_accuracy": correct / readable if readable else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    try:
        metrics = evaluate(args.ground_truth, args.predictions)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
