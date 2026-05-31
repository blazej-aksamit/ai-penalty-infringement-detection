from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import cv2
import pandas as pd


RESULTS_CSV = Path("runs/evaluation/encroachment_gt_test/test_encroachment_gt_results.csv")
OUT_CSV = Path("data/meta/encroachment_labels.csv")


def load_done(out_csv: Path) -> Dict[str, Dict[str, str]]:
    done: Dict[str, Dict[str, str]] = {}
    if out_csv.exists() and out_csv.stat().st_size > 0:
        with out_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row.get('clip_name','')}::{row.get('frame_idx_gt','')}"
                done[key] = row
    return done


def append_row(row: Dict[str, str], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not out_csv.exists()) or out_csv.stat().st_size == 0
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        fieldnames = [
            "clip_name",
            "frame_idx_gt",
            "encroachment",
            "uncertain",
            "notes",
            "overlay_path",
            "result_json",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Manually label encroachment on GT kick-frame overlays.")
    parser.add_argument("--results-csv", default=str(RESULTS_CSV))
    parser.add_argument("--out-csv", default=str(OUT_CSV))
    args = parser.parse_args()

    results_csv = Path(args.results_csv)
    out_csv = Path(args.out_csv)

    if not results_csv.exists():
        raise SystemExit(f"Missing results CSV: {results_csv}")

    df = pd.read_csv(results_csv)
    df = df.loc[df["pipeline_ok"] == True].copy()
    done = load_done(out_csv)

    print(f"Rows available: {len(df)} | Already labeled: {len(done)}")
    print("Controls:")
    print("  0 = no encroachment")
    print("  1 = encroachment")
    print("  u = uncertain")
    print("  n = skip")
    print("  q = quit")

    cv2.namedWindow("Encroachment label", cv2.WINDOW_NORMAL)

    for row in df.itertuples(index=False):
        clip_name = str(row.clip_name)
        frame_idx = int(row.frame_idx_gt)
        key = f"{clip_name}::{frame_idx}"
        if key in done:
            continue

        overlay_path = Path(str(row.overlay_path))
        if not overlay_path.is_absolute():
            overlay_path = Path.cwd() / overlay_path
        image = cv2.imread(str(overlay_path))
        if image is None:
            print(f"[SKIP] missing overlay: {overlay_path}")
            continue

        while True:
            show = image.copy()
            lines = [
                clip_name,
                f"GT frame={frame_idx} | module={row.decision} | reason={row.decision_reason}",
                "0=no encroachment  1=encroachment  u=uncertain  n=skip  q=quit",
            ]
            y = 28
            for text in lines:
                cv2.putText(show, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(show, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                y += 28

            cv2.imshow("Encroachment label", show)
            key_code = cv2.waitKey(0) & 0xFF

            if key_code == ord("q"):
                cv2.destroyAllWindows()
                print("Stopped.")
                return
            if key_code == ord("n"):
                print(f"[SKIP] {clip_name}")
                break
            if key_code in {ord("0"), ord("1"), ord("u")}:
                encroachment = ""
                uncertain = "0"
                if key_code == ord("0"):
                    encroachment = "0"
                elif key_code == ord("1"):
                    encroachment = "1"
                else:
                    uncertain = "1"
                append_row(
                    {
                        "clip_name": clip_name,
                        "frame_idx_gt": str(frame_idx),
                        "encroachment": encroachment,
                        "uncertain": uncertain,
                        "notes": "",
                        "overlay_path": str(overlay_path).replace("\\", "/"),
                        "result_json": str(row.result_json).replace("\\", "/"),
                    },
                    out_csv,
                )
                print(f"[OK] {clip_name} -> {'UNCERTAIN' if uncertain == '1' else ('ENCROACHMENT' if encroachment == '1' else 'NO ENCROACHMENT')}")
                break

    cv2.destroyAllWindows()
    print(f"Done. Saved: {out_csv}")


if __name__ == "__main__":
    main()
