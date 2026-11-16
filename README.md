# Vehicle Number Plate Recognition System

A computer vision semester project to detect vehicle number plates and read
their text using Python, OpenCV, and EasyOCR.

## Sample footage

[Ten real traffic MP4 clips with previews and source credits](samples/footage/README.md)
are included under `samples/footage/`. Each lasts five seconds: 50 seconds and
750 fully decoded frames in total. They come from ten source videos; the two
Nikon street clips are related and must remain in one development/test group.

```sh
python process_video.py samples/footage/traffic_06.mp4
```

These clips have not been manually labelled or benchmarked. Some plates are
small, blurred, or dark. The original repeated-photo smoke clip is separate
and is not counted as real traffic footage. License details and clip hashes are
in the footage folder.

## Step 1: Environment setup

Use Python 3.11 or 3.12. Python 3.11 is used for local verification.
No Google Colab or GPU is required for this step.

### macOS / Linux

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_setup.py
```

If Python 3.12 is installed instead, replace `python3.11` with `python3.12`.

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python check_setup.py
```

The check imports the libraries, processes a small image with OpenCV, and runs
a PyTorch calculation. It should end with:

```text
Setup passed. Ready for Step 2: read a cropped plate image.
```

EasyOCR installs PyTorch as a dependency. Installation needs an internet
connection and may take a few minutes. This step does not download OCR weights
or run recognition; EasyOCR downloads its weights when a Reader is first created
in Step 2.

OpenCV's headless package supports image processing and saved video output,
but does not open desktop windows with `cv2.imshow`. Keep just this OpenCV
package installed to avoid conflicting `cv2` installations.

## Step 2: Read a cropped plate image

Activate the environment from Step 1, then run:

```sh
python read_plate.py samples/synthetic_plate.png
python read_plate.py "path/to/cropped_plate.jpg"
```

The included sample is generated text, not a real vehicle photograph.

Use a clear JPG or PNG containing only one horizontal line of plate characters.
Crop away the vehicle and any small country labels. Full vehicle photographs
and two-line plates need detection/segmentation in later steps.

The first run downloads English OCR recognition weights into `models/easyocr/`.
Internet access is required for that first run; later runs reuse the local model.
Processing runs on CPU. Model files are ignored by Git.

Verified output for the included synthetic sample on the local CPU:

```text
Plate text: MH12AB1234 | OCR confidence: 0.99
```

Confidence is the model's score, not measured accuracy or proof of a valid plate.
This step restricts output to uppercase Latin letters and digits. It does not
correct ambiguous characters such as O/0 or validate regional plate formats.

Run the offline checks with:

```sh
python -m unittest -v
```

These tests mock OCR to check image validation and result handling. Evaluate
recognition separately using real labelled plate images.

## Step 3: Detect plates in a full image

Update the environment, then provide a vehicle or junction image:

```sh
python -m pip install -r requirements.txt
python detect_plates.py "path/to/vehicle.jpg"
```

The first run downloads the 5.47 MB YOLO11n plate detector into `models/`.
The source revision and SHA-256 checksum are pinned in `detect_plates.py`;
subsequent runs check and reuse the local file. The script runs on CPU.

Each run creates a new folder under `outputs/` containing:

- `detected.png`: the full image with green plate boxes and detection scores.
- `plate_1.png`, `plate_2.png`, etc.: unannotated crops for the OCR step.

Runs use separate output folders to avoid overwriting earlier results. A run
with no detections still saves the full image and prints a message. The fixed
minimum detection confidence is 0.25; this is not measured accuracy.

Without `--ocr`, this command performs detection only. For a single-line detected plate:

```sh
python read_plate.py "outputs/<run-folder>/plate_1.png"
```

The OCR sample from Step 2 is just text, not a full vehicle scene; use an actual
vehicle image for detection. Two-line crops still need the later OCR improvements.
The offline tests cover crop boundaries, empty detections, invalid images,
non-overwriting outputs, and model download/cache checksum handling.

### Detector source and limitations

The detector is the nano model published by
[morsetechlab](https://huggingface.co/morsetechlab/yolov11-license-plate-detection),
revision `251a30d7daedca065f56e04b0af04052c907c68f`. The weights and Ultralytics
are published under AGPL-3.0. Their original licenses apply; model weights are
not committed to this repository.

The model author reports train/test contamination in the upstream dataset.
Do not use the model card's accuracy numbers in this project's report. Evaluate
on independent labelled images/video, especially for Indian plates, motorcycles,
small plates, and low light. This step is a prototype, not an accuracy evaluation.

`ultralytics-opencv-headless` supplies the same `ultralytics` Python API while
using the existing headless OpenCV package. Do not install the regular
`ultralytics` package alongside it. NumPy and OpenCV were updated together to
satisfy the YOLO package's macOS compatibility requirements.

Local smoke test: the detector found one plate (score 0.73) in the public
[FastALPR vehicle sample](https://github.com/ankandrew/fast-alpr/blob/master/assets/test_image.png).
The image was used locally and is not included in this repository. This is a
functionality check, not an independent accuracy benchmark.

## Step 4: Detect and read plates together

```sh
python detect_plates.py "path/to/vehicle.jpg" --ocr
```

One command now detects plates, reads each crop, and writes the recognised text
above its box in `detected.png`. It also saves each crop and a `results.txt` file
mapping crop filenames to recognised text, OCR confidence, and detection
confidence. Each run uses a fresh output folder under `outputs/`.

One OCR reader is shared across all plates in the image. It loads only when at
least one plate is detected. First use may download the OCR model as described
in Step 2. No additional dependencies are needed for this step.

If OCR returns no text, the crop is retained and labelled `Unreadable`. Low
confidence readings are displayed with their score; they are not automatically
accepted as correct. The existing single-line, Latin letters/digits restriction
still applies. Country labels, two-line plates, blur, and O/0 or B/8 ambiguity
can produce incorrect text. Check the crop against the displayed result.

The combined flow was run on the FastALPR vehicle sample: detection succeeded
(score 0.73), but OCR returned `4E05` at score 0.10, which does not match the
visible plate. Resizing and contrast checks did not give a reliable correction,
so no sample-specific correction was added. This documents a recognition failure,
not a successful accuracy result; use clearer images and evaluate independently.
The synthetic cropped sample still reads `MH12AB1234` at score 0.99.

Run all offline checks with `python -m unittest -v`. The integration test covers
multiple plates, one shared reader, an unreadable crop, saved text results, and
an image with no detections.

## Step 5: Process a video

```sh
python process_video.py "path/to/traffic_video.mp4"
```

This runs the same detector and OCR on every decoded frame, with both models
loaded once for the video. It saves `outputs/<new-run-folder>/annotated.mp4`
with plate boxes, recognised text, and confidence scores. Frames without plates
are also retained. No additional dependencies are required.

The output uses the input's nominal FPS; processing itself can be much slower
than real time on CPU. Progress prints every 30 frames. Use a short clip first.
The MP4 contains video only: audio is not copied. Odd frame dimensions are
padded by one black pixel for the MP4 encoder. Prefer fixed-frame-rate footage.

The script checks the input, FPS, and encoder, and detects early decoder failure
when the source reports a frame count. It decodes the complete saved output to
verify the frame count before marking it finished. Each run uses a separate
folder. On failure or Ctrl+C, resources are released and any unfinished video
keeps an `.partial.mp4` name; the source video is never overwritten.

The same OCR limitations from Step 4 apply. There is no vehicle tracking or
frame-to-frame voting yet, so text may flicker or be wrong. Per-frame crops are
processed in memory rather than saved to disk. CSV export is described in Step 6.

Run `python -m unittest -v` for the offline checks, including an encoded video
round-trip, model reuse, invalid input/FPS, early decoding failure, and encoder
cleanup. Local real-model smoke test: all six frames of a repeated-photo clip were
processed, saved, and decoded at 3 FPS. This verifies the file pipeline, not
recognition performance on moving traffic; low-confidence OCR errors remain.

## Step 6: Export video readings to CSV

The existing command now saves both `annotated.mp4` and `plates.csv` in the
same output folder:

```sh
python process_video.py "path/to/traffic_video.mp4"
```

CSV columns:

| Column | Meaning |
| --- | --- |
| `frame_number` | Zero-based decoded frame index; the first frame is 0 |
| `video_time_seconds` | Frame index / source FPS, rounded to milliseconds |
| `plate_number` | Raw normalised OCR text; blank if OCR returned no text |
| `detection_confidence` | Plate detector score, rounded to four decimals |
| `ocr_confidence` | OCR score, rounded to four decimals; blank if unreadable |
| `x1`, `y1`, `x2`, `y2` | Clipped plate box in original-frame pixels (added for evaluation) |

Timestamps are relative to the start of the video, not clock time, and use the
same nominal FPS as the saved MP4. Use fixed-frame-rate footage for accurate
timing. Each detected plate gets one row; frames with no detections add no rows.
A video with no plates produces a CSV containing only the header.

Repeated sightings are retained as separate rows. There is no vehicle tracking,
deduplication, or voting; row counts are not counts of unique vehicles. Low
confidence and incorrect OCR readings are retained for review, not certified as
valid plates. The CSV is written incrementally with Python's standard library,
so no new dependencies are needed.

Each run has a fresh folder. An interrupted/failed run keeps `plates.partial.csv`
for inspection; it is renamed to `plates.csv` after the output video passes
verification. CSV streams and video handles are closed on failure. Offline tests
cover multiple plates in a frame, empty frames, timestamps, leading zeros,
unreadable crops, header-only exports, and unfinished-run naming. The local
six-frame real-model smoke run produced six CSV rows, with timestamps from
0.000 to 1.667 seconds at 3 FPS; its low-confidence OCR readings remain incorrect.

## Step 7: Evaluate and read the report

```sh
python evaluate.py evaluation/preliminary_ground_truth.csv evaluation/preliminary_predictions.csv
```

The evaluator reports detection precision/recall, OCR exact-match accuracy,
end-to-end exact-match accuracy, and raw counts using one-to-one box matching
at IoU 0.5. It evaluates only explicitly labelled frames. See
[evaluation instructions](evaluation/README.md) for manual labels and new footage.

[The project report](REPORT.md) documents the architecture, tests, actual results,
limitations, and reproduction commands. The committed preliminary evaluation
contains **one manually labelled target in one development frame**: detection
matched it, but OCR read it incorrectly. This is not a representative accuracy
benchmark. The repeated-photo clip is not six independent examples.

All 14 automated tests pass. Ten unlabelled real-footage clips are now included;
a final accuracy report still requires independent, manually labelled footage. The report is provided as Markdown; complete the
student details before submission.

Image/video detection, OCR, CSV export, an evaluator, and a preliminary report
are implemented. Independent benchmarking and traffic violation detection remain;
violations need a defined rule.

## References

- [EasyOCR installation and usage](https://github.com/JaidedAI/EasyOCR)
- [OpenCV Python packages](https://pypi.org/project/opencv-python-headless/)
- [EasyOCR API documentation](https://www.jaided.ai/easyocr/documentation/)
- [Ultralytics prediction API](https://docs.ultralytics.com/modes/predict/)
