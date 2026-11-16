# Vehicle Number Plate Recognition System

Computer Vision Project — 5th Semester

Evaluation date: 14 November 2026

Student name / roll number / faculty: complete before submission

## Abstract

This Python prototype detects vehicle number plates, reads single-line plate
text with OCR, and produces annotated images, annotated videos, and timestamped
CSV files. It uses pretrained YOLO11n and EasyOCR models on CPU. It does not
train a model from scratch. The file-processing pipeline works, but the current
OCR fails on the evaluated real vehicle sample. The results below are a
preliminary diagnostic, not a representative accuracy benchmark.

## Objective and scope

Implement the plate detection and OCR portion of Assignment 2 using OpenCV,
a plate-trained object detector, and an OCR library. The prototype supports
cropped images, complete vehicle images, and recorded video.

Traffic violation detection is not implemented. Reading a plate does not establish
a violation. If the faculty requires this part, a specific rule such as crossing
a stop line during a red interval must be implemented and evaluated separately.

## Tools and architecture

| Component | Implementation |
| --- | --- |
| Runtime used for local verification | Python 3.11.17 |
| Image/video processing | opencv-python-headless 5.0.0.93 |
| Array processing | NumPy 2.4.6 |
| Detector inference | ultralytics-opencv-headless 8.4.210 |
| Plate detector | morsetechlab YOLO11n plate-finetuned weights |
| OCR | EasyOCR 1.7.2, English model, CPU |
| Result storage | Python standard-library CSV and JSON |

Processing flow:

`Input image/video → YOLO plate boxes → clipped plate crops → grayscale OCR → text and confidence → annotated output + CSV`

Detection uses a minimum score of 0.25. Bounding boxes are clipped to the image.
OCR considers each crop to be one text line and restricts output to uppercase
Latin letters and digits. It removes whitespace but does not correct ambiguous
characters or verify regional plate formats. No perspective correction,
tracking, frame voting, or automatic confidence rejection is implemented.

For video, the detector and OCR reader load once. Every decoded frame is
processed; timestamps are calculated from the zero-based frame index divided
by nominal FPS. Output is an MP4 without audio. A run writes partial files until
video verification succeeds. Input files are not overwritten.

## Test material and ground truth

The initial preliminary evaluation used the following development material:

1. A generated text image reading `MH12AB1234`, used for an OCR setup check.
2. One public FastALPR vehicle photo, reused during development.
3. A six-frame, 3 FPS video made by repeating that same photo, used to check
   video encoding, frame retention, and CSV output.

Only frame 0 of the repeated-photo clip is evaluated for recognition. Counting
all six identical-source frames as independent observations would exaggerate
the evidence. Five prediction rows are therefore excluded by the evaluator.

The visible target plate is manually transcribed as `5AU5341`. Its manually
estimated axis-aligned box is `(306, 188, 390, 212)` in pixels. This preliminary
annotation covers the fully visible target plate, not a comprehensive annotation
of small or partly occluded plates elsewhere in the scene. Source-image overlap
with the pretrained model's training set is unknown.

The detector's model card separately reports train/test contamination in its
upstream dataset. Its published accuracy figures are not used in this report.
The annotated demo, its first-frame input/output images, and the plate crop
are included under `evidence/`, with upstream attribution and license.

## Added real traffic footage collection

Ten real moving-traffic clips have now been added under `samples/footage/`.
Each is five seconds at 15 FPS: **10 clips, 50 seconds, 750 decoded frames**.
These clips are separate from the synthetic image and repeated-photo diagnostic.
They come from ten source videos on Wikimedia Commons, with per-file author
credits, source links, license links, and transformation details.

| Clip | Scene |
| --- | --- |
| [traffic_01.mp4](samples/footage/traffic_01.mp4) | Night highway traffic |
| [traffic_02.mp4](samples/footage/traffic_02.mp4) | Signal-controlled urban junction |
| [traffic_03.mp4](samples/footage/traffic_03.mp4) | Approaching car on a rural road |
| [traffic_04.mp4](samples/footage/traffic_04.mp4) | Airport rental-car exit and crossing |
| [traffic_05.mp4](samples/footage/traffic_05.mp4) | Hanoi intersection traffic |
| [traffic_06.mp4](samples/footage/traffic_06.mp4) | Street-level approaching cars |
| [traffic_07.mp4](samples/footage/traffic_07.mp4) | Wide street intersection |
| [traffic_08.mp4](samples/footage/traffic_08.mp4) | Munich street junction |
| [traffic_09.mp4](samples/footage/traffic_09.mp4) | Approaching ambulance and cars |
| [traffic_10.mp4](samples/footage/traffic_10.mp4) | Snow-covered urban street |

[View the ten-clip preview gallery and credits](samples/footage/README.md).
[File provenance and hashes](samples/footage/manifest.json) and
[full decoding checks](samples/footage/validation.json) are included.

All ten videos decode successfully and show changing frames; three positions
per clip were visually inspected. The collection includes junctions, approaching
vehicles, night scenes, and winter conditions. Some available renditions are
low resolution and plate text may be unreadable. These are footage samples,
not ten proven successful recognition examples or ten labelled violations.

Clips 06 and 07 are related street-intersection recordings and share one source
group. They must stay together when splitting development/test data. The clips
have not been manually labelled or evaluated; overlap with model training data
is unknown. The preliminary metrics below remain based on the original single
labelled development target, and are not recomputed or extrapolated to these
new clips. A held-out accuracy evaluation is still pending.

## Evaluation method

`evaluate.py` compares labelled boxes and predicted boxes within the same frame.
It considers only frames explicitly present in the ground-truth CSV. Predictions
are sorted by detector confidence; each is matched to the unmatched ground-truth
box with greatest intersection over union (IoU). A match requires IoU ≥ 0.5.
Each box can be matched once; duplicate predictions become false positives.

- True positive (TP): a matched plate detection.
- False positive (FP): an unmatched prediction in an evaluated frame.
- False negative (FN): an unmatched ground-truth plate.
- Detection precision: `TP / (TP + FP)`.
- Detection recall: `TP / (TP + FN)`.
- OCR exact-match accuracy: correct complete strings divided by matched,
  manually readable ground-truth plates.
- End-to-end exact-match accuracy: correct matched strings divided by all
  readable ground-truth plates, including those missed by the detector.

Text comparison ignores case, whitespace, and hyphen separators; it never
substitutes characters such as O/0. Unreadable ground-truth plates contribute
to detection counts but not OCR denominators. Undefined rates are JSON `null`.
Confidence scores are model outputs, not measured accuracy percentages.

## Measured preliminary results

**Sample size: one labelled target in one development frame. These numbers
must not be presented as overall project accuracy.**

| Measure | Observed result |
| --- | --- |
| Evaluated frames / labelled targets | 1 / 1 |
| TP / FP / FN | 1 / 0 / 0 |
| Box IoU for the target | approximately 0.924 |
| Detection precision / recall on this annotation | 1/1 = 100% / 1/1 = 100% |
| Correct complete plate reads | 0/1 |
| OCR exact-match accuracy on this annotation | 0% |
| End-to-end exact-match accuracy on this annotation | 0% |
| Ground-truth plate | `5AU5341` |
| Predicted plate in frame 0 | `LEV31` |
| Detection / OCR confidence in frame 0 | 0.7305 / 0.1064 |

The plate is localised successfully, but the text is wrong. Subsequent repeated
frames contain other low-confidence readings such as `4EV531`; they are not
independent test cases. Earlier direct-image OCR returned `4E05`, also wrong.
Resizing and contrast experiments did not reliably correct the error, and no
sample-specific replacement was added.

Separately, the synthetic cropped-text check correctly reads `MH12AB1234` at
approximately 0.99 OCR confidence. That verifies a simple input path, not
performance on natural plates. The six-frame smoke video is readable at 3 FPS
with all six frames retained and six timestamped CSV rows.

All 14 automated tests passed locally. They check image validation, box clipping,
model checksum/cache handling, OCR reuse, empty/unreadable results, CSV values,
frame timestamps, video round-trip/cleanup, IoU, duplicate detections, missed
plates, metric denominators, and invalid labels. Mock-based tests establish code
behaviour; they do not establish model accuracy.

## Execution evidence

The images and files below come from local execution of the project code, not
illustrative mockups. They support the limited checks described above; the
real-photo OCR failure is intentionally retained.

### 1. Synthetic cropped-image OCR check

![Generated single-line plate used for the OCR check](samples/synthetic_plate.png)

Actual captured result:

```text
Plate text: MH12AB1234 | OCR confidence: 0.99
```

[Full OCR execution log](evidence/synthetic_ocr.txt). This is generated text,
so it proves only the simple cropped-image path works.

### 2. Real-photo detection and OCR failure

Input frame 0 from the repeated-photo clip:

![Unannotated input frame 0](evidence/input_frame_0.png)

Actual first frame decoded from the annotated output video:

![Actual annotated output showing the wrong OCR reading LEV31](evidence/annotated_frame_0.png)

The exported plate crop, enlarged for viewing:

<img src="evidence/plate_crop_0.png" width="324" alt="Actual detector crop: visible plate 5AU5341">

The visible plate is `5AU5341`; OCR predicted `LEV31` with score 0.1064
(displayed as 0.11). The box is a successful target detection; the text is a
failed read. This is not evidence of successful real-plate recognition.

### 3. Video and timestamped export

[Download the actual annotated demo MP4](evidence/annotated_demo.mp4).
It contains six repeated-photo frames at 3 FPS, not moving traffic.
[Saved video verification](evidence/video_verification.json) records the decoded
frame count and FPS. The annotated frame above is extracted from that MP4.

The first three actual prediction rows are:

```csv
frame_number,video_time_seconds,plate_number,detection_confidence,ocr_confidence,x1,y1,x2,y2
0,0.000,LEV31,0.7305,0.1064,307,188,388,211
1,0.333,4EV531,0.7316,0.0397,307,188,388,211
2,0.667,4EV531,0.7318,0.0434,307,188,388,211
```

[Complete prediction CSV](evaluation/preliminary_predictions.csv),
[manual ground truth](evaluation/preliminary_ground_truth.csv), and
[calculated metrics](evaluation/preliminary_metrics.json) are available for
inspection. [The evaluator execution log](evidence/evaluation.txt) shows the
reproducible result: one matched detection and zero correct reads in the one
labelled frame; the other five repeated-frame rows are excluded.

### 4. Automated test evidence

Fresh test run summary:

```text
Ran 14 tests in 0.141s

OK
```

[Full test transcript](evidence/tests.txt), including the command, source revision,
and exit status. Some test cases mock detector/OCR responses; their printed
sample readings verify code behaviour, not real-world recognition accuracy.

[Evidence origins and upstream license](evidence/README.md) identify the photo
source, transformations, and the origin of each artifact.

## Limitations and improvements

The immediate priority is improving real-plate OCR using a separate development
set: better crop quality, perspective handling, text-line segmentation, or a
plate-specific OCR model should be compared before choosing a change. Two-line
plates and non-Latin scripts are unsupported by the present recognition path.
Blur, small plates, unusual fonts, and low light can reduce recognition quality.

Repeated detections remain separate CSV rows; they do not count unique vehicles.
There is no tracker or temporal voting. CPU processing may be slower than the
source FPS. Fixed-rate footage is preferred because timestamps use nominal FPS.
No traffic-rule decision is made.

A final benchmark should use independent scenes with exhaustively labelled
visible plates and explicit negative frames. Keep clips from the same vehicle
or scene together when separating development and test data. Include different
lighting, distances, angles, and plate formats. Lock settings before running the
held-out test, and report sample counts, detection precision/recall, exact OCR
accuracy, end-to-end accuracy, failures, and runtime. That benchmark remains
pending; it cannot be inferred from this one-photo diagnostic.

## How to reproduce

From the activated project environment:

```sh
python -m unittest -v
python read_plate.py samples/synthetic_plate.png
python process_video.py "path/to/traffic_video.mp4"
python evaluate.py evaluation/preliminary_ground_truth.csv evaluation/preliminary_predictions.csv
```

The preliminary CSV files and JSON metrics are committed under `evaluation/`.
They let the metric calculation be reproduced without downloading model weights.
New video runs append box-coordinate columns needed for evaluation; older CSV
files without those columns must be regenerated. See `evaluation/README.md`
for the label format and preparation procedure.

## Sources

- [OpenCV project](https://opencv.org/)
- [Ultralytics prediction documentation](https://docs.ultralytics.com/modes/predict/)
- [Detector model card and licensing](https://huggingface.co/morsetechlab/yolov11-license-plate-detection)
- [EasyOCR API](https://www.jaided.ai/easyocr/documentation/)
- [FastALPR source photo](https://github.com/ankandrew/fast-alpr/blob/master/assets/test_image.png)

The detector and Ultralytics are published under AGPL-3.0. Original dependency
and model licenses apply. No pretrained weights are committed to this project.
