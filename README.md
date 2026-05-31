# AI-Based Decision Support for Goalkeeper Goal-Line Infringements in Football Penalty Kicks

Bachelor thesis project — a computer-vision pipeline that automatically determines whether a goalkeeper left the goal line before a penalty kick was taken (a foul under FIFA Laws of the Game).

---

## Overview

During a penalty kick the goalkeeper must remain on the goal line until the ball is kicked. Detecting violations manually is difficult even for match officials and VAR operators. This project builds a single-camera, fully-automated decision-support system that:

1. Detects the kick moment using ball-motion analysis
2. Localises the goal-line in the frame via homography
3. Detects the goalkeeper with a fine-tuned YOLOv8 model
4. Measures the goalkeeper's distance from the goal line in real-world units
5. Outputs a three-way verdict: **Legal / Infringement / Uncertain**

A secondary module detects **encroachment** (outfield players entering the penalty area before the kick).

---

## Results

Evaluated on 49 penalty clips (22 from SoccerNet + 27 original Video Project clips):

| Policy | Coverage | Selective Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Relaxed | 0.980 (48/49) | 0.938 | 0.875 | 0.778 | 0.824 |
| Conservative | 0.918 (45/49) | 0.933 | 0.833 | 0.714 | 0.769 |

The **conservative policy** adds an uncertainty buffer around the ±2 px near-boundary zone, trading a small coverage drop for improved precision.

---

## Repository Structure

```
src/                   Main pipeline source code
  pipeline/            End-to-end pipeline orchestration
  line_logic/          Goal-line localisation and distance measurement
  kick_detection/      Ball-motion kick detector
  yolo/                YOLOv8 training and inference wrappers
  pose/                Pose estimation (experimental)
  ml/                  Video-classifier experiments (R2+1D)
  evaluation/          Batch runners and metric/report scripts
  tools/               Utilities (video tools, batch runners)

data/
  annotations/         Manual bounding-box annotations (YOLO .txt labels), organised in batches
  meta/                Ground-truth violation/encroachment labels and canonical metadata CSVs
  yolo/                YOLO dataset config (dataset.yaml; source frames excluded)
  yolo_gk_ball/        Goalkeeper/ball labels + dataset config (source frames excluded)

models/                Fine-tuned detector weights (train4_best.pt, via Git LFS)
requirements.txt       Python dependencies
```

> **Note on data & licensing.** Only our own annotations (YOLO `.txt` labels, CSV
> metadata, dataset configs) are published here. The underlying penalty frames and
> clips are **not** redistributed: SoccerNet footage is governed by its Non-Commercial
> Research License, and the additional clips are YouTube-sourced. The extraction
> pipeline is included so the datasets can be regenerated from the original sources.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline on a single clip
python src/pipeline/run_full_penalty_pipeline.py \
    --clip path/to/penalty_clip.mp4 \
    --auto-kick
```

---

## Tech Stack

- **Detection**: YOLOv8 (fine-tuned on custom goalkeeper/ball dataset, ~400 annotated frames)
- **Geometry**: Automated Hough-transform goal-line detection with geometric plausibility filtering
- **Kick detection**: Ball-motion (velocity-onset) detector with a -1 frame correction
- **Uncertainty**: Selective-classification policy (relaxed vs conservative)
- **Pose**: MediaPipe / YOLOv8-pose (investigated but rejected, not in final pipeline)
- **Dataset**: SoccerNet penalty clips + additional YouTube-sourced clips

---

## Key Files

| File | Description |
|---|---|
| `src/pipeline/run_full_penalty_pipeline.py` | End-to-end pipeline entry point |
| `src/line_logic/hybrid_line_decision.py` | Goal-line decision logic |
| `src/line_logic/uncertainty_policy.py` | Relaxed/conservative uncertainty policy |
| `src/kick_detection/ball_motion_detector.py` | Automatic kick-moment detection |
| `src/evaluation/batch_run_final_pipeline.py` | Batch evaluation + metrics runner |
| `data/meta/keeper_violation_labels_final.csv` | Ground-truth violation labels |
