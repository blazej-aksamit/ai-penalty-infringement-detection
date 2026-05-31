import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "SoccerNet"
OUT_CSV = ROOT / "data" / "meta" / "penalties.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

LABEL_FILENAME = "Labels-v2.json"


def parse_game_time(game_time: str):
    """
    Expected common format: '1 - 12:34'
    Returns: (half, t_seconds) or None
    """
    m = re.match(r"^\s*([12])\s*-\s*(\d{1,2}):(\d{2})\s*$", str(game_time).strip())
    if not m:
        return None
    half = int(m.group(1))
    t_seconds = int(m.group(2)) * 60 + int(m.group(3))
    return half, float(t_seconds)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def main():
    print("RAW_DIR:", RAW_DIR)
    label_files = sorted(RAW_DIR.rglob(LABEL_FILENAME))
    print(f"Found {len(label_files)} label files named {LABEL_FILENAME}")

    rows = []

    for lf in label_files:
        try:
            data = json.loads(lf.read_text(encoding="utf-8"))
        except Exception as e:
            print("Skip unreadable:", lf, e)
            continue

        annotations = data.get("annotations")
        if not isinstance(annotations, list):
            continue

        game_id = lf.parent.name

        for ev in annotations:
            if not isinstance(ev, dict):
                continue

            label = ev.get("label", "")
            if not isinstance(label, str):
                continue

            if "penalty" not in label.lower():
                continue

            half = None
            t_seconds = None

            game_time = ev.get("gameTime")
            if isinstance(game_time, str):
                parsed = parse_game_time(game_time)
                if parsed:
                    half, t_seconds = parsed

            if t_seconds is None:
                pos = ev.get("position")
                posf = safe_float(pos)
                if posf is not None:
                    t_seconds = posf / 1000.0 if posf > 10000 else posf

                h = ev.get("half") or ev.get("period")
                try:
                    half = int(h) if h is not None else half
                except Exception:
                    pass

            if t_seconds is None or half not in (1, 2):
                continue

            rows.append(
                {
                    "game_id": game_id,
                    "half": int(half),
                    "t_seconds": round(float(t_seconds), 3),
                    "label": label,
                    "gameTime": ev.get("gameTime", ""),
                    "labels_file": str(lf),
                }
            )

    rows.sort(key=lambda r: (r["game_id"], r["half"], r["t_seconds"], r["label"]))

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["game_id", "half", "t_seconds", "label", "gameTime", "labels_file"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Extracted {len(rows)} penalty-like events.")
    print("Saved:", OUT_CSV)
    if rows:
        print("Example:", rows[0])


if __name__ == "__main__":
    main()
