# Data Overview

This directory contains raw footage references, extracted clips, annotations, and metadata.

## Main Subdirectories

- `raw/`
  Raw match sources and downloaded material.
- `clips/`
  Extracted penalty clips and kick-centered windows.
- `annotations/`
  Frame-level annotations used during detection work.
- `yolo/`
  Older YOLO dataset artifacts.
- `yolo_gk_ball/`
  Main goalkeeper + ball detection dataset.
- `meta/`
  CSV metadata used to connect all stages of the pipeline.
- `line_logic_blazej_evaluation/`
  Goal-line decision evaluation subset.
- `line_logic_friend_eval/`
  Manual comparison subset for line-logic evaluation.
- `pose_dev/`
  Temporary pose experiment assets.

## Canonical Metadata Files

Use these first:

- `meta/kick_times.csv`
- `meta/kick_windows_720p.csv`
- `meta/keeper_violation_labels_final.csv`
- `meta/splits_violation.csv`
- `meta/to_review_uncertain.csv`

For YOLO-specific work inside `yolo_gk_ball/meta/`:

- `yolo_gk_ball/meta/frames_metadata_canonical.csv`
  Preferred repaired metadata layer based on the actual disk split layout and current label presence.
- `yolo_gk_ball/meta/frames_metadata.csv`
  Older metadata snapshot kept for traceability.

## Metadata Roles

- `kick_times.csv`
  Manual kick frame selections for full penalty clips.
- `kick_windows_720p.csv`
  Kick-centered short clip windows derived from kick timing.
- `keeper_violation_labels_final.csv`
  Final keeper violation labels for windows.
- `splits_violation.csv`
  Train / val / test split assignment.
- `to_review_uncertain.csv`
  Cases that still need manual review or uncertainty handling.

## Legacy or Partial Metadata

- `keeper_violation_labels.csv`
  Earlier labeling pass.
- `gk_offline_labels.csv`
  Small partial helper file.
- `gk_line_labels.csv`
  Placeholder / not yet populated.
- `penalties_all.csv`
  Broader raw penalty listing.
- `penalties.csv`
  Legacy file with unresolved merge markers.

## Important Practical Rule

If two CSVs seem to describe the same stage, prefer the file whose name contains:

- `final`
- `720p`
- `splits`

before using older helper CSVs.

## Practical Default Choices

If you need:

- source matches:
  use `raw/SoccerNet/`
- full penalty clips:
  use `clips/penalties_720p/`
- final short windows:
  use `clips/kick_windows_720p_v2/`
- goalkeeper + ball training data:
  use `yolo_gk_ball/`
- YOLO metadata:
  use `yolo_gk_ball/meta/frames_metadata_canonical.csv`
- final thesis labels:
  use `meta/keeper_violation_labels_final.csv`
- final thesis splits:
  use `meta/splits_violation.csv`
