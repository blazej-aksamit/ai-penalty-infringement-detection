# Evaluation Scripts

This folder contains small utilities for reporting thesis metrics in a way that is easy to defend.

## Recommended Scripts

- `binary_classifier_report.py`
  Use this for binary decision tasks where you already have ground truth and predictions in a CSV.
  It writes:
  - confusion matrix
  - accuracy / balanced accuracy
  - precision / recall / F1 for the chosen positive class
  - no-skill baselines

- `apply_uncertain_policy.py`
  Use this to convert borderline line decisions into an explicit `uncertain` label.
  It writes an augmented CSV with policy decisions, reasons, and flags.

- `abstaining_classifier_report.py`
  Use this when predictions can be `uncertain`.
  It reports:
  - coverage
  - selective accuracy
  - lower-bound accuracy over all samples
  - confusion counts plus abstentions by truth class

- `audit_yolo_dataset.py`
  Use this to audit the `data/yolo_gk_ball` dataset.
  It reports:
  - image and label counts per split
  - missing / extra / empty label files
  - annotation counts per class
  - metadata counts from the selected metadata CSV, preferably `frames_metadata_canonical.csv`
  - a short summary of the chosen YOLO training run

- `build_yolo_canonical_index.py`
  Use this to export a disk-based image index for `data/yolo_gk_ball`.
  It does not modify the dataset.
  It writes one row per image with:
  - disk split
  - label presence and box counts
  - joined metadata fields when available
  - metadata status (`match`, `split_mismatch`, `missing_metadata`)

- `build_canonical_yolo_metadata.py`
  Use this after the disk audit to produce a safer metadata layer for YOLO work.
  It writes:
  - canonical metadata with split taken from the actual disk layout
  - a manifest of images that still do not have YOLO labels
  It does not overwrite the original `frames_metadata.csv`.

- `prepare_yolo_eval_subset.py`
  Use this when you want a self-contained YOLO evaluation subset.
  It copies matched image-label pairs into a clean YOLO subset so you can run evaluation safely or share a frozen eval copy.

- `run_yolo_detection_eval.py`
  Use this inside a YOLO-capable environment to run quantitative detection evaluation and save a small JSON summary.

- `evaluate_kick_detection.py`
  Use this inside a YOLO-capable environment to evaluate automatic kick-frame detection against `data/meta/kick_times.csv`.
  It writes:
  - per-clip prediction results
  - frame-error metrics such as exact and within `+/-N`
  - a Markdown report and JSON summary

- `batch_run_final_pipeline.py`
  Use this inside a YOLO-capable environment to run the full final pipeline over a labeled split of kick-window clips.
  It writes:
  - one aggregated CSV over the split
  - a JSON summary with decision counts and exact-match rate

- `summarize_label_balance.py`
  Use this to report class priors for the labeled dataset and for each split.
  It also reports the majority-class baseline accuracy.

## Current Thesis Use

- line-logic pilot comparison:
  `runs/final_hybrid_eval/comparison_after_cleanup.csv`
- final violation labels:
  `data/meta/keeper_violation_labels_final.csv`
- split file:
  `data/meta/splits_violation.csv`
