"""Detect plates in a full image and save boxes and crops (optionally with OCR)."""

import argparse
import hashlib
import math
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

import cv2

from read_plate import create_reader, read_plate_image

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "models" / "ultralytics"))
from ultralytics import YOLO

MODEL_PATH = ROOT / "models" / "license-plate-finetune-v1n.pt"
MODEL_URL = (
    "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/"
    "251a30d7daedca065f56e04b0af04052c907c68f/license-plate-finetune-v1n.pt"
)
MODEL_SHA256 = "0aec75976c56eb6f26dfb274c430620ec65137915ff1ae47c3a48c7af8afb7b2"


def ensure_model():
    if MODEL_PATH.is_file():
        data = MODEL_PATH.read_bytes()
    else:
        print("Downloading the plate detector (first run only)...")
        with urlopen(MODEL_URL, timeout=60) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError(f"Detector checksum failed. Remove {MODEL_PATH} and retry.")
    if not MODEL_PATH.is_file():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.write_bytes(data)
    return MODEL_PATH


def annotate_plates(image, boxes, reader=None):
    """Share plate crops, OCR, and labels between image and video processing."""
    annotated = image.copy()
    height, width = image.shape[:2]
    plates = []
    for left, top, right, bottom, confidence, _ in boxes:
        x1, y1 = max(0, math.floor(left)), max(0, math.floor(top))
        x2, y2 = min(width, math.ceil(right)), min(height, math.ceil(bottom))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2]
        label = f"Plate {confidence:.2f}"
        text, score = "", None
        if reader is not None:
            matches = read_plate_image(crop, reader=reader)
            label = "Unreadable"
            if matches:
                text, score = max(matches, key=lambda match: match[1])
                label = f"{text} | OCR {score:.2f}"
        plates.append({
            "crop": crop, "label": label, "detection_confidence": confidence,
            "plate_text": text, "ocr_confidence": score, "box": (x1, y1, x2, y2),
        })
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        label_x = min(x1, max(0, width - label_width - 2))
        label_y = max(20, y1 - 8)
        cv2.rectangle(annotated, (label_x, label_y - 16),
                      (label_x + label_width + 2, label_y + 4), (0, 0, 0), -1)
        cv2.putText(annotated, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return annotated, plates


def detect_plates(image_path, output_root=Path("outputs"), ocr=False):
    image_path = Path(image_path)
    if not image_path.is_file():
        raise ValueError(f"Image file not found: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot decode image: {image_path}")

    result = YOLO(str(ensure_model())).predict(
        image, device="cpu", conf=0.25, verbose=False,
    )[0]
    boxes = result.boxes.data.tolist()
    reader = create_reader() if ocr and boxes else None
    annotated, plates = annotate_plates(image, boxes, reader=reader)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix=f"{image_path.stem}_", dir=output_root))
    crops, readings = [], []
    for plate in plates:
        crop_path = output_dir / f"plate_{len(crops) + 1}.png"
        if not cv2.imwrite(str(crop_path), plate["crop"]):
            raise OSError(f"Could not save crop: {crop_path}")
        crops.append(crop_path)
        if ocr:
            reading = (f"{crop_path.name}: {plate['label']} | "
                       f"Detection {plate['detection_confidence']:.2f}")
            readings.append(reading)
            print(reading)
    if not cv2.imwrite(str(output_dir / "detected.png"), annotated):
        raise OSError(f"Could not save annotated image in {output_dir}")
    if ocr:
        (output_dir / "results.txt").write_text(
            "\n".join(readings or ["No plates detected."]) + "\n", encoding="utf-8",
        )
    return output_dir, crops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="A full vehicle/junction JPG or PNG")
    parser.add_argument("--ocr", action="store_true", help="Read and label each detected plate")
    args = parser.parse_args()
    try:
        output_dir, crops = detect_plates(args.image, ocr=args.ocr)
    except (ValueError, OSError, RuntimeError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Detected {len(crops)} plate(s). Results: {output_dir.resolve()}")
    if not crops:
        print("No plates found above confidence 0.25. Try a clearer or closer image.")
    for crop in crops:
        print(f"Crop: {crop}")


if __name__ == "__main__":
    main()
