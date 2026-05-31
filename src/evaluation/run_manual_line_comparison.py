"""
RQ2 comparison: automatic goal-line detection vs. manual reference line.

For each clip we already have:
  - Auto-pipeline result (final_result.json)  -- decision using Hough-detected line
  - Manual annotation (manual_goal_lines.json) -- human-drawn goal line

We re-run the classification logic using the manual line (keeping everything else
identical: same goalkeeper bbox, same foot proxies, same threshold, same policy).

Outputs
-------
runs/rq2_manual_comparison/
    comparison_results.csv      -- per-clip comparison
    comparison_summary.json     -- aggregate metrics
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.line_logic.uncertainty_policy import apply_uncertainty_policy

LINE_DIST_THRESH_PX = 10.0
UNCERTAINTY_MARGIN_PX = 2.0
LOCAL_Y_ERR_THRESH_PX = 8.0
BBOX_PROXY_SPREAD_THRESH_PX = 17.0


# ── helpers ────────────────────────────────────────────────────────────────────

def load_yolo_boxes(label_path: Path, img_w: int, img_h: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        xc = float(parts[1]) * img_w
        yc = float(parts[2]) * img_h
        w  = float(parts[3]) * img_w
        h  = float(parts[4]) * img_h
        conf = float(parts[5]) if len(parts) >= 6 else 1.0
        boxes.append({
            "cls": cls_id, "conf": conf,
            "x1": int(round(xc - w / 2)), "y1": int(round(yc - h / 2)),
            "x2": int(round(xc + w / 2)), "y2": int(round(yc + h / 2)),
        })
    return boxes


def pick_goalkeeper(boxes, cls=0, conf_min=0.25):
    gk = [b for b in boxes if b["cls"] == cls and b["conf"] >= conf_min]
    return sorted(gk, key=lambda b: b["conf"], reverse=True)[0] if gk else None


def point_to_line_dist(px, py, line):
    x1, y1, x2, y2 = line
    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    den = math.hypot(y2 - y1, x2 - x1)
    return num / den if den > 0 else 1e9


def line_y_at_x(line, x):
    x1, y1, x2, y2 = line
    if abs(x2 - x1) < 1e-6:
        return (y1 + y2) / 2.0
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def classify_with_line(gk_box, manual_line):
    """Return decision dict using the given manual line directly (no Hough)."""
    if gk_box is None:
        return {"decision": "uncertain", "reason": "no_goalkeeper",
                "min_dist": None, "point_name": None, "all_dists": {}, "local_y_err": None}
    if manual_line is None:
        return {"decision": "uncertain", "reason": "no_manual_line",
                "min_dist": None, "point_name": None, "all_dists": {}, "local_y_err": None}

    x1b, y1b, x2b, y2b = gk_box["x1"], gk_box["y1"], gk_box["x2"], gk_box["y2"]
    pts = {
        "left_bottom":   (float(x1b), float(y2b)),
        "center_bottom": ((x1b + x2b) / 2.0, float(y2b)),
        "right_bottom":  (float(x2b), float(y2b)),
    }
    gk_ybot = float(y2b)

    all_dists = {}
    local_y_errs = {}
    for name, (px, py) in pts.items():
        all_dists[name] = point_to_line_dist(px, py, manual_line)
        local_y_errs[name] = abs(line_y_at_x(manual_line, px) - gk_ybot)

    best_name = min(pts.keys(), key=lambda n: (local_y_errs[n], all_dists[n]))
    min_dist   = all_dists[best_name]
    local_y_err = local_y_errs[best_name]

    decision = "on_line" if min_dist <= LINE_DIST_THRESH_PX else "off_line"
    return {
        "decision": decision,
        "reason": "manual_line",
        "min_dist": min_dist,
        "point_name": best_name,
        "point_source": "bbox",
        "all_dists": all_dists,
        "local_y_err": local_y_err,
    }


def auto_line_y_at_gk(auto_result: dict, gk_box) -> float | None:
    """Reconstruct the auto-detected line Y position at the goalkeeper's X from stored distances."""
    # We can't recover the exact line from the stored JSON without re-running Hough.
    # Instead we return None so the line-offset metric uses NaN for that clip.
    # (The overlay image is available but parsing it would be fragile.)
    return None


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dirs", nargs="+",
                    default=[
                        "runs/evaluation/batch_final_orig/test",
                        "runs/evaluation/batch_final_ext/test_ext",
                    ])
    ap.add_argument("--annotations", default="data/meta/manual_goal_lines.json")
    ap.add_argument("--labels-csv-orig",
                    default="data/meta/keeper_violation_labels.csv")
    ap.add_argument("--labels-csv-ext",
                    default="data/meta/keeper_violation_labels_ext.csv")
    ap.add_argument("--out-dir", default="runs/rq2_manual_comparison")
    ap.add_argument("--draw-overlays", action="store_true",
                    help="Save side-by-side overlay images for inspection")
    args = ap.parse_args()

    ann_path = REPO_ROOT / args.annotations
    if not ann_path.exists():
        print(f"ERROR: annotations file not found: {ann_path}")
        print("Run scripts/tools/annotate_goal_lines.py first.")
        sys.exit(1)

    with open(ann_path, encoding="utf-8") as f:
        annotations = json.load(f)

    # Load ground-truth labels
    gt_map = {}
    for csv_path in [args.labels_csv_orig, args.labels_csv_ext]:
        p = REPO_ROOT / csv_path
        if p.exists():
            df = pd.read_csv(p)
            for _, row in df.iterrows():
                wf = Path(str(row["window_file"])).name
                stem = Path(wf).stem
                if pd.notna(row["violation"]):
                    gt_map[stem] = int(row["violation"])

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = out_dir / "overlays"
    if args.draw_overlays:
        overlays_dir.mkdir(exist_ok=True)

    rows = []

    for bd_rel in args.batch_dirs:
        bd = REPO_ROOT / bd_rel
        if not bd.exists():
            print(f"  [WARN] batch dir not found: {bd}")
            continue

        for clip_dir in sorted(bd.iterdir()):
            if not clip_dir.is_dir():
                continue
            clip_name = clip_dir.name

            # Load auto result
            auto_json = clip_dir / "final_result.json"
            if not auto_json.exists():
                print(f"  [SKIP - no final_result.json] {clip_name}")
                continue
            with open(auto_json, encoding="utf-8") as f:
                auto = json.load(f)

            auto_decision = auto.get("decision", "uncertain")

            # Load frame
            frame_path = Path(auto.get("frame_path", ""))
            if not frame_path.exists():
                # try relative
                frames_dir = clip_dir / "frames"
                imgs = sorted(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
                frame_path = imgs[0] if imgs else None

            img = cv2.imread(str(frame_path)) if frame_path and frame_path.exists() else None
            if img is None:
                print(f"  [SKIP - no frame image] {clip_name}")
                continue
            h, w = img.shape[:2]

            # Load YOLO detections
            label_path = Path(auto.get("label_path", ""))
            if not label_path.exists():
                label_path = clip_dir / "detect" / "labels" / (frame_path.stem + ".txt")
            boxes = load_yolo_boxes(label_path, w, h)
            gk_box = pick_goalkeeper(boxes)

            # Get manual annotation
            manual_coords = annotations.get(clip_name)
            manual_line = tuple(manual_coords) if manual_coords else None  # (x1,y1,x2,y2)

            # Re-classify using manual line
            manual_result = classify_with_line(gk_box, manual_line)
            manual_result = apply_uncertainty_policy(
                manual_result,
                line_dist_thresh_px=LINE_DIST_THRESH_PX,
                uncertainty_margin_px=UNCERTAINTY_MARGIN_PX,
                local_y_err_thresh_px=LOCAL_Y_ERR_THRESH_PX,
                bbox_proxy_spread_thresh_px=BBOX_PROXY_SPREAD_THRESH_PX,
            )
            manual_decision = manual_result["decision"]

            # Line offset at goalkeeper x (approximate using distances stored in auto result)
            # We use auto min_dist as proxy for line offset
            auto_min_dist   = auto.get("min_dist_px")
            manual_min_dist = manual_result.get("min_dist")

            # GT label
            gt_violation = gt_map.get(clip_name.replace("_KICK", ""))
            if gt_violation is None:
                gt_violation = gt_map.get(clip_name)

            # pred label conversion
            def decision_to_pred(d):
                if d == "off_line": return "violation"
                if d == "on_line":  return "valid"
                return "uncertain"

            gt_label    = "violation" if gt_violation == 1 else ("valid" if gt_violation == 0 else None)
            auto_pred   = decision_to_pred(auto_decision)
            manual_pred = decision_to_pred(manual_decision)

            decision_changed = (auto_decision != manual_decision)

            row = {
                "clip": clip_name,
                "gt_violation": gt_violation,
                "gt_label": gt_label,
                "auto_decision": auto_decision,
                "auto_pred": auto_pred,
                "auto_min_dist_px": auto_min_dist,
                "manual_annotation": "yes" if manual_coords else "skipped",
                "manual_decision": manual_decision,
                "manual_pred": manual_pred,
                "manual_min_dist_px": manual_min_dist,
                "decision_changed": decision_changed,
                "auto_correct": (auto_pred == gt_label) if gt_label else None,
                "manual_correct": (manual_pred == gt_label) if gt_label else None,
            }
            rows.append(row)

            changed_str = " <-- CHANGED" if decision_changed else ""
            print(f"  {clip_name[:50]:<50}  auto={auto_decision:<10}  manual={manual_decision:<10}{changed_str}")

            # Draw overlay if requested
            if args.draw_overlays and img is not None:
                vis = img.copy()
                # Draw manual line
                if manual_line:
                    x1m, y1m, x2m, y2m = manual_line
                    cv2.line(vis, (x1m, y1m), (x2m, y2m), (0, 255, 0), 2)
                    # extend
                    if abs(x2m - x1m) > 1:
                        mm = (y2m - y1m) / (x2m - x1m)
                        bm = y1m - mm * x1m
                        cv2.line(vis, (0, int(mm * 0 + bm)), (w-1, int(mm*(w-1)+bm)), (0, 200, 0), 1)
                # Draw GK box
                if gk_box:
                    cv2.rectangle(vis, (gk_box["x1"], gk_box["y1"]),
                                  (gk_box["x2"], gk_box["y2"]), (0, 165, 255), 2)
                label = f"auto={auto_decision} | manual={manual_decision}"
                if decision_changed:
                    label += " [CHANGED]"
                cv2.putText(vis, label, (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imwrite(str(overlays_dir / f"{clip_name}.jpg"), vis)

    # ── save results ───────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_out = out_dir / "comparison_results.csv"
    df.to_csv(csv_out, index=False)

    annotated = df[df["manual_annotation"] == "yes"]
    n_total = len(df)
    n_ann   = len(annotated)
    n_changed = int(annotated["decision_changed"].sum())
    agreement = 1 - n_changed / n_ann if n_ann > 0 else 0.0

    # accuracy breakdown
    auto_correct   = annotated["auto_correct"].sum() if "auto_correct" in annotated else None
    manual_correct = annotated["manual_correct"].sum() if "manual_correct" in annotated else None
    n_gt = annotated["gt_label"].notna().sum()

    # exact match rates
    auto_acc   = float(auto_correct   / n_gt) if n_gt > 0 and auto_correct   is not None else None
    manual_acc = float(manual_correct / n_gt) if n_gt > 0 and manual_correct is not None else None

    # per-decision change breakdown
    changes = annotated[annotated["decision_changed"] == True][
        ["clip", "auto_decision", "manual_decision", "gt_label", "auto_correct", "manual_correct"]
    ].to_dict("records")

    summary = {
        "clips_total": n_total,
        "clips_annotated": n_ann,
        "clips_skipped": n_total - n_ann,
        "decision_agreement_rate": round(agreement, 4),
        "decisions_changed": n_changed,
        "auto_exact_match_rate": round(auto_acc, 4) if auto_acc is not None else None,
        "manual_exact_match_rate": round(manual_acc, 4) if manual_acc is not None else None,
        "changed_clips": changes,
    }

    json_out = out_dir / "comparison_summary.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Clips total          : {n_total}")
    print(f"Clips annotated      : {n_ann}")
    print(f"Decision agreement   : {agreement:.1%}  ({n_ann - n_changed}/{n_ann})")
    print(f"Decisions changed    : {n_changed}")
    if auto_acc is not None:
        print(f"Auto exact-match     : {auto_acc:.1%}  ({int(auto_correct)}/{n_gt})")
    if manual_acc is not None:
        print(f"Manual exact-match   : {manual_acc:.1%}  ({int(manual_correct)}/{n_gt})")
    print(f"\nCSV  -> {csv_out}")
    print(f"JSON -> {json_out}")


if __name__ == "__main__":
    main()
