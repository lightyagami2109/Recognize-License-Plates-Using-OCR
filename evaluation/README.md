# Accuracy evaluation

The committed `preliminary_*` files are a one-frame development diagnostic,
not a held-out benchmark. `preliminary_predictions.csv` records the actual
six-frame repeated-photo video output. Only frame 0 is labelled; the remaining
five rows are deliberately ignored. See `../REPORT.md` for limitations and results.

Recalculate the saved metrics without model downloads:

```sh
python evaluate.py evaluation/preliminary_ground_truth.csv evaluation/preliminary_predictions.csv
```

## Evaluate new footage

Ten real-footage clips are now included under `samples/footage/`; see the
[gallery and source credits](../samples/footage/README.md). They are unlabelled.
Keep related clips 06/07 together in development/test splits; this acquisition
does not establish that the clips are independent of model training data.

1. Obtain separate vehicle/junction footage and run `process_video.py`.
2. Choose frames before reviewing predictions. For a final test, keep scenes
   independent of development footage and label every visible target plate.
3. Manually record each plate's text and bounding box from the original frame,
   independently of the predicted boxes. Frame numbers start at 0, coordinates
   are pixels in the original image, and x2/y2 are the exclusive right/bottom edge.
4. Save a CSV with this header:

   ```csv
   frame_number,plate_number,x1,y1,x2,y2
   ```

   Add one row per plate. For an explicitly negative frame, give the frame number
   and leave the other five fields blank. For a visible plate whose text cannot
   be transcribed, give its coordinates but leave `plate_number` blank. Do not
   combine a negative-frame row with plate rows for the same frame.
5. Run:

   ```sh
   python evaluate.py "path/to/ground_truth.csv" "outputs/<run-folder>/plates.csv"
   ```

The JSON output includes counts and rates between 0 and 1. A rate with no valid
denominator is `null`, not an invented zero. IoU matching uses a fixed threshold
of 0.5 and detector-confidence ordering. Duplicate detections count as false
positives. OCR accuracy uses matched, readable ground-truth plates; end-to-end
accuracy additionally penalises missed readable plates. Only explicitly labelled
frames count. Unlabelled frames are not silently treated as negatives.

Ground truth and predictions must refer to the same source video. CSVs do not
contain a video identifier, so check this pairing yourself. The evaluator uses
all exported predictions; no extra confidence cutoff is applied. New prediction
CSVs include x1/y1/x2/y2; regenerate exports created before this addition.

The source photo is linked in the report. To repeat the pipeline smoke test,
download it separately, pad its odd height to an even value, and encode six
copies at 3 FPS. This checks I/O only. Do not substitute this clip for independent
traffic footage in an accuracy claim.
