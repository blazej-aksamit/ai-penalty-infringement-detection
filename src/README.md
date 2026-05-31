# Scripts Overview

This directory contains both active thesis scripts and historical experiments.

## Active Script Groups

### `pipeline/`

Use these first for dataset preparation and the current prototype flow.

Recommended files:
- `extract_penalties.py`
- `download_original_halves_for_penalties.py`
- `cut_clips.py`
- `pick_kick_times.py`
- `make_kick_windows_720p.py`
- `review_kick_windows_720p.py`
- `label_violation.py`
- `run_full_penalty_pipeline.py`

Current note:
- `run_full_penalty_pipeline.py` now supports both manual `--frame-idx` and automatic `--auto-kick`

### `kick_detection/`

Kick-moment detection experiments.

Main file:
- `ball_motion_detector.py`

### `evaluation/`

Reporting helpers for thesis metrics.

Recommended files:
- `binary_classifier_report.py`
- `summarize_label_balance.py`
- `apply_uncertain_policy.py`
- `abstaining_classifier_report.py`
- `evaluate_kick_detection.py`
- `batch_run_final_pipeline.py`

### `line_logic/`

Goal-line reference and goalkeeper decision logic.

Recommended files:
- `prototype_line_decision.py`
- `hybrid_line_decision.py`
- `goalframe_homography_probe.py`
- `compare_with_friend.py`

### `ml/`

Clip-level classification experiments.

Files:
- `train_r2plus1d.py`
- `eval_r2plus1d.py`

### `pose/`

Pose-related helper scripts and exploratory utilities.

Recommended files:
- `prepare_pose_pilot.py`
- `build_pose_pilot_review_sheet.py`
- `run_yolo_pose_inference.py`

### `yolo/`

Dataset preparation helpers for YOLO.

## Legacy Areas

### `archive/`

Old utility scripts and abandoned experiments.
Keep for reference only.

### `pipeline/pipeline/`

Historical nested copy of pipeline work.
It has been moved to:
- `archive/pipeline_nested_legacy/`

### `archive/line_logic_nested_legacy/`

Contains the duplicate prototype file that used to live under `line_logic/line_logic/`.
Prefer the top-level `scripts/line_logic/prototype_line_decision.py`.

## Editing Rule

When updating logic:

1. edit the top-level script in the relevant area
2. do not start from nested duplicate copies
3. treat `archive/` as read-only history unless absolutely needed
