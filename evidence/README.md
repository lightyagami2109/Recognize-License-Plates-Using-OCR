# Local execution evidence

These files document the preliminary development checks in `../REPORT.md`.
They do not constitute independent traffic evaluation or proof of reliable OCR.

| File | Origin |
| --- | --- |
| `input_frame_0.png` | First decoded frame of the repeated-photo input clip |
| `annotated_frame_0.png` | First decoded frame of the actual annotated MP4 |
| `plate_crop_0.png` | Input frame cropped to the exported box `(307,188,388,211)` |
| `annotated_demo.mp4` | Actual six-frame, 3 FPS video output; no audio |
| `video_verification.json` | Decoded frame count, FPS, code revision and frame-0 result |
| `tests.txt` | Fresh combined stdout/stderr from `python -m unittest -v` |
| `synthetic_ocr.txt` | Fresh output from OCR on `samples/synthetic_plate.png` |
| `evaluation.txt` | Fresh output from the committed preliminary CSV evaluation |
| `FASTALPR_LICENSE.txt` | Upstream MIT license for the sample source repository |

Log headers record the command, source-code revision, and exit status. Test logs
include mock-generated plate readings/download notices; these are test fixtures,
not recognition results on real traffic. The synthetic OCR transcript is a
separate real-model invocation. The video corresponds to
`../evaluation/preliminary_predictions.csv`, including its incorrect OCR results.
The annotation rounds the OCR score 0.1064 to 0.11 for display.

The photo-derived inputs and outputs originate from
[ankandrew/fast-alpr, assets/test_image.png](https://github.com/ankandrew/fast-alpr/blob/master/assets/test_image.png).
The upstream project is Copyright (c) 2024 ankandrew, published under the included
MIT license. Transformations: decoded/re-encoded into a repeated-photo clip,
plate-box crop, and model-generated annotation overlays. This is a development
sample, not moving traffic; overlap with model training data is unknown.
