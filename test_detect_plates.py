"""Offline checks for crop boundaries, output files, and model integrity."""

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import detect_plates as detector


class DetectPlateTests(unittest.TestCase):
    def test_crops_and_empty_detection(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "car.png"
            self.assertTrue(cv2.imwrite(str(image), np.full((100, 200, 3), 255, np.uint8)))
            with patch.object(detector, "ensure_model", return_value="model.pt"), \
                    patch.object(detector, "YOLO") as yolo:
                boxes = yolo.return_value.predict.return_value[0].boxes.data
                boxes.tolist.return_value = [
                    [-5, 70.2, 220, 110, 0.9, 0], [50, 50, 40, 40, 0.8, 0],
                ]
                output, crops = detector.detect_plates(image, root / "output")
                self.assertEqual(len(crops), 1)
                self.assertEqual(cv2.imread(str(crops[0])).shape, (30, 200, 3))
                self.assertTrue((output / "detected.png").is_file())
                boxes.tolist.return_value = []
                second_output, crops = detector.detect_plates(image, root / "output")
                self.assertEqual(crops, [])
                self.assertNotEqual(output, second_output)
                self.assertTrue((second_output / "detected.png").is_file())

    def test_ocr_reuses_reader_and_records_unreadable_plates(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "car.png"
            self.assertTrue(cv2.imwrite(str(image), np.full((100, 200, 3), 255, np.uint8)))
            with patch.object(detector, "ensure_model", return_value="model.pt"), \
                    patch.object(detector, "YOLO") as yolo, \
                    patch.object(detector, "create_reader") as create_reader:
                boxes = yolo.return_value.predict.return_value[0].boxes.data
                boxes.tolist.return_value = [
                    [5, 20, 95, 50, 0.9, 0], [105, 20, 195, 50, 0.8, 0],
                ]
                create_reader.return_value.recognize.side_effect = [
                    [([], "MH 12 AB 1234", 0.92)], [],
                ]
                output, crops = detector.detect_plates(image, root / "output", ocr=True)
                create_reader.assert_called_once()
                self.assertEqual(len(crops), 2)
                report = (output / "results.txt").read_text()
                self.assertIn("plate_1.png: MH12AB1234 | OCR 0.92 | Detection 0.90", report)
                self.assertIn("plate_2.png: Unreadable | Detection 0.80", report)
                self.assertTrue((output / "detected.png").is_file())
                create_reader.reset_mock()
                boxes.tolist.return_value = []
                output, crops = detector.detect_plates(image, root / "output", ocr=True)
                self.assertEqual(crops, [])
                create_reader.assert_not_called()
                self.assertEqual((output / "results.txt").read_text(), "No plates detected.\n")

    def test_invalid_input_does_not_load_model(self):
        with TemporaryDirectory() as folder, patch.object(detector, "ensure_model") as model:
            image = Path(folder) / "car.png"
            with self.assertRaisesRegex(ValueError, "not found"):
                detector.detect_plates(image)
            image.write_text("not an image")
            with self.assertRaisesRegex(ValueError, "Cannot decode"):
                detector.detect_plates(image)
            model.assert_not_called()

    def test_download_checksum_and_cache(self):
        with TemporaryDirectory() as folder:
            model = Path(folder) / "model.pt"
            data = b"test model"
            response = MagicMock()
            response.__enter__.return_value.read.return_value = data
            with patch.object(detector, "MODEL_PATH", model), \
                    patch.object(detector, "MODEL_SHA256", hashlib.sha256(data).hexdigest()), \
                    patch.object(detector, "urlopen", return_value=response) as download:
                self.assertEqual(detector.ensure_model(), model)
                self.assertEqual(detector.ensure_model(), model)
                download.assert_called_once()
                model.write_bytes(b"corrupted")
                with self.assertRaisesRegex(ValueError, "checksum failed"):
                    detector.ensure_model()


if __name__ == "__main__":
    unittest.main()
