import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.run_full_penalty_pipeline import auto_detect_kick_frame


def _stack_overlays(goalkeeper_overlay: Path, encroachment_overlay: Path, out_path: Path) -> str | None:
    img1 = cv2.imread(str(goalkeeper_overlay)) if goalkeeper_overlay.exists() else None
    img2 = cv2.imread(str(encroachment_overlay)) if encroachment_overlay.exists() else None
    if img1 is None and img2 is None:
        return None
    if img1 is None:
        combined = img2
    elif img2 is None:
        combined = img1
    else:
        target_w = max(img1.shape[1], img2.shape[1])
        if img1.shape[1] != target_w:
            scale = target_w / img1.shape[1]
            img1 = cv2.resize(img1, (target_w, int(round(img1.shape[0] * scale))))
        if img2.shape[1] != target_w:
            scale = target_w / img2.shape[1]
            img2 = cv2.resize(img2, (target_w, int(round(img2.shape[0] * scale))))
        separator = 255 * (cv2.imread(str(goalkeeper_overlay))[:8, :target_w] * 0 + 1) if False else None
        combined = cv2.vconcat([img1, img2])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined)
    return str(out_path).replace("\\", "/")


def _run_subprocess(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_kick_frame(video_path: Path, args, out_dir: Path) -> tuple[int, str, dict | None, str | None]:
    if args.frame_idx is not None and not args.auto_kick:
        return args.frame_idx, "manual", None, None

    if not args.auto_kick:
        raise RuntimeError("Provide --frame-idx or enable --auto-kick.")

    kick_args = SimpleNamespace(
        kick_model_path=args.kick_model_path,
        model_path=args.model_path,
        kick_window_start_s=args.kick_window_start_s,
        kick_window_end_s=args.kick_window_end_s,
        kick_min_confidence=args.kick_min_confidence,
        kick_onset_factor=args.kick_onset_factor,
        kick_max_tracking_jump_px=args.kick_max_tracking_jump_px,
        kick_min_sustained_velocity=args.kick_min_sustained_velocity,
        kick_disable_peak_fallback=args.kick_disable_peak_fallback,
        kick_frame_adjust=args.kick_frame_adjust,
    )
    kick_details, kick_json_path = auto_detect_kick_frame(video_path, kick_args, out_dir)
    if kick_details.get("kick_frame") is None:
        raise RuntimeError(
            "Automatic kick detection failed and no manual --frame-idx was provided. "
            f"Reason: {kick_details.get('reason')}"
        )
    return int(kick_details["kick_frame"]), "auto_ball_motion", kick_details, str(kick_json_path).replace("\\", "/")


def main():
    parser = argparse.ArgumentParser(description="Run goalkeeper line and player encroachment checks on the same kick frame.")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--frame-idx", type=int, default=None)
    parser.add_argument("--auto-kick", action="store_true")
    parser.add_argument("--kick-model-path", default=None)
    parser.add_argument("--kick-window-start-s", type=float, default=4.0)
    parser.add_argument("--kick-window-end-s", type=float, default=12.0)
    parser.add_argument("--kick-min-confidence", type=float, default=0.25)
    parser.add_argument("--kick-onset-factor", type=float, default=2.5)
    parser.add_argument("--kick-min-sustained-velocity", type=float, default=2.0)
    parser.add_argument("--kick-max-tracking-jump-px", type=float, default=180.0)
    parser.add_argument("--kick-disable-peak-fallback", action="store_true")
    parser.add_argument("--kick-frame-adjust", type=int, default=-1)
    parser.add_argument("--model-path", default="runs/detect/train4/weights/best.pt")
    parser.add_argument("--pose-model-path", default=None)
    parser.add_argument("--player-model-path", default="yolo26n.pt")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--line-dist-thresh", type=float, default=10.0)
    parser.add_argument("--apply-uncertain-policy", action="store_true")
    parser.add_argument("--uncertainty-margin-px", type=float, default=2.0)
    parser.add_argument("--uncertainty-local-y-err-px", type=float, default=8.0)
    parser.add_argument("--uncertainty-bbox-proxy-spread-px", type=float, default=17.0)
    parser.add_argument("--out-root", default="runs/combined_officiating")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    out_dir = Path(args.out_root) / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx, kick_source, kick_details, kick_json_path = _resolve_kick_frame(video_path, args, out_dir)

    goalkeeper_out_root = out_dir / "goalkeeper_line"
    encroachment_out_root = out_dir / "encroachment"

    gk_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "pipeline" / "run_full_penalty_pipeline.py"),
        "--video-path",
        str(video_path),
        "--frame-idx",
        str(frame_idx),
        "--model-path",
        str(args.model_path),
        "--conf",
        str(args.conf),
        "--line-dist-thresh",
        str(args.line_dist_thresh),
        "--out-root",
        str(goalkeeper_out_root),
    ]
    if args.pose_model_path:
        gk_cmd.extend(["--pose-model-path", str(args.pose_model_path)])
    if args.apply_uncertain_policy:
        gk_cmd.extend(
            [
                "--apply-uncertain-policy",
                "--uncertainty-margin-px",
                str(args.uncertainty_margin_px),
                "--uncertainty-local-y-err-px",
                str(args.uncertainty_local_y_err_px),
                "--uncertainty-bbox-proxy-spread-px",
                str(args.uncertainty_bbox_proxy_spread_px),
            ]
        )

    enc_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "pipeline" / "run_player_encroachment_probe.py"),
        "--video-path",
        str(video_path),
        "--frame-idx",
        str(frame_idx),
        "--kick-model-path",
        str(args.model_path),
        "--player-model-path",
        str(args.player_model_path),
        "--out-root",
        str(encroachment_out_root),
    ]

    gk_stdout = _run_subprocess(gk_cmd)
    enc_stdout = _run_subprocess(enc_cmd)

    gk_json = goalkeeper_out_root / video_path.stem / "final_result.json"
    enc_json = encroachment_out_root / video_path.stem / "encroachment_result.json"
    gk_result = _load_json(gk_json)
    enc_result = _load_json(enc_json)

    combined_overlay_path = out_dir / "combined_overlay.jpg"
    combined_overlay = _stack_overlays(
        goalkeeper_out_root / video_path.stem / "hybrid" / "final_overlay.jpg",
        encroachment_out_root / video_path.stem / "encroachment_overlay.jpg",
        combined_overlay_path,
    )

    combined = {
        "video_path": str(video_path).replace("\\", "/"),
        "frame_idx": frame_idx,
        "kick_source": kick_source,
        "kick_detection_json": kick_json_path,
        "kick_details": kick_details,
        "goalkeeper_line": gk_result,
        "encroachment": enc_result,
        "combined_overlay_path": combined_overlay,
        "goalkeeper_stdout": gk_stdout,
        "encroachment_stdout": enc_stdout,
    }

    json_path = out_dir / "combined_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    flat = {
        "video_path": combined["video_path"],
        "frame_idx": frame_idx,
        "kick_source": kick_source,
        "goalkeeper_decision": gk_result.get("decision"),
        "goalkeeper_reason": gk_result.get("reason"),
        "encroachment_decision": enc_result.get("decision"),
        "encroachment_reason": enc_result.get("decision_reason"),
        "combined_overlay_path": combined_overlay,
        "goalkeeper_result_json": str(gk_json).replace("\\", "/"),
        "encroachment_result_json": str(enc_json).replace("\\", "/"),
    }
    csv_path = out_dir / "combined_result.csv"
    pd.DataFrame([flat]).to_csv(csv_path, index=False)

    print("\nCombined pipeline finished.")
    print(f"Frame: {frame_idx}")
    print(f"Kick source: {kick_source}")
    print(f"Goalkeeper decision: {gk_result.get('decision')} ({gk_result.get('reason')})")
    print(f"Encroachment decision: {enc_result.get('decision')} ({enc_result.get('decision_reason')})")
    print(f"Saved combined JSON: {json_path}")
    print(f"Saved combined CSV:  {csv_path}")
    if combined_overlay:
        print(f"Saved combined overlay: {combined_overlay_path}")


if __name__ == "__main__":
    main()
