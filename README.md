# AI Penalty Infringement Detection

AI-based decision-support prototype for detecting goalkeeper goal-line infringements during football penalty kicks from single-camera broadcast footage.

Developed as a Bachelor's Thesis at the University of Southern Denmark.

## Overview

This project investigates whether goalkeeper goal-line compliance during football penalties can be assessed automatically using standard broadcast video.

The system combines:

- automatic kick-moment estimation through ball-motion analysis
- YOLO-based goalkeeper and ball detection
- Hough-transform-based goal-line localisation
- geometric goalkeeper-line reasoning
- explicit uncertainty-aware decision policies

The system is designed as a decision-support tool, not a fully automated referee replacement.

It produces one of three outputs:

- Legal
- Potential infringement
- Uncertain

---

## Final Adopted Pipeline

1. Input penalty clip
2. Automatic kick detection
3. -1 frame temporal correction
4. YOLO goalkeeper-ball detection
5. Goal-line localisation using geometric filtering
6. Goalkeeper-line classification
7. Uncertainty-aware final decision

---

## Main Results

### 22-clip benchmark

**Relaxed policy**
- Coverage: 0.955
- Selective accuracy: 0.952
- Infringement recall: 1.000

**Conservative policy**
- Coverage: 0.864
- Selective accuracy: 0.947
- Infringement recall: 1.000

### Expanded 49-clip robustness evaluation

**Relaxed policy**
- Selective accuracy: 0.938
- Coverage: 0.980

**Conservative policy**
- Selective accuracy: 0.933
- Coverage: 0.918

This demonstrates the practical trade-off between coverage and abstention under ambiguous geometric conditions.

---

## Installation

```bash
git clone https://github.com/blazej-aksamit/ai-penalty-infringement-detection.git
cd ai-penalty-infringement-detection
pip install -r requirements.txt
```

---

## Quick Start

```bash
python src/pipeline/run_full_penalty_pipeline.py \
    --video-path path/to/penalty_clip.mp4 \
    --auto-kick \
    --kick-frame-adjust -1
```

---

## Technical Details

### Object Detection

YOLOv8n trained for:

- goalkeeper
- ball

Final adopted detector: `runs/detect/train4/weights/best.pt`

### Goal-Line Localisation

The final adopted method uses:

- Hough line transform
- geometric plausibility filtering
- local line candidate scoring

No homography calibration is used.

### Decision Logic

The system measures goalkeeper position relative to the detected goal line using pixel-space geometric reasoning.

Two uncertainty policies are available:

**Relaxed policy** — prioritises coverage.

**Conservative policy** — uses abstention on geometrically ambiguous near-boundary cases.

---

## Limitations

Performance depends strongly on:

- kick-frame accuracy
- goalkeeper lower-body visibility
- goal-line visibility
- broadcast perspective quality

The system is optimised for professional broadcast footage and does not generalise reliably to amateur smartphone recordings without retraining.

---

## Thesis Information

Bachelor's Thesis  
University of Southern Denmark  
Faculty of Engineering

**Title:** AI-Based Decision Support for Goalkeeper Goal-Line Infringements in Football Penalty Kicks

**Authors:**
- Błażej Aksamit
- Maciej Gawłowski

**Supervisor:**
- Rodrigo Furlan de Assis

**Submission:** June 2026
