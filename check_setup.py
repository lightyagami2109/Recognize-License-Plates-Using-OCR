"""Check the Step 1 environment without downloading model weights."""

import sys

import cv2
import easyocr
import numpy as np
import torch


def main():
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    assert gray.shape == (32, 64), "OpenCV image processing failed"
    assert callable(easyocr.Reader), "EasyOCR Reader is unavailable"
    assert torch.ones(2).sum().item() == 2, "PyTorch computation failed"

    print(f"Python: {sys.version.split()[0]}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"EasyOCR: {easyocr.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print("Setup passed. Ready for Step 2: read a cropped plate image.")


if __name__ == "__main__":
    main()
