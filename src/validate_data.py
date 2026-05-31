import csv
from pathlib import Path

CSV = Path("data/meta/kick_windows_720p.csv")
CLIPS_DIR = Path("data/clips/kick_windows_720p_v2")

print("=== VALIDATING kick_windows_720p.csv ===\n")

if not CSV.exists():
    print(f"[FAIL] {CSV} nie istnieje")
    exit(1)

rows = []
errors = []

with CSV.open(newline="", encoding="utf-8") as f:
    for i, row in enumerate(csv.DictReader(f), start=2):
        rows.append(row)
        
        if not row.get("kick_in_window_s", "").strip():
            errors.append(f"Row {i}: kick_in_window_s is empty")
        
        window_file = row.get("window_file", "").strip()
        if not window_file:
            errors.append(f"Row {i}: window_file is empty")
        else:
            filepath = Path(window_file)
            if not filepath.exists():
                errors.append(f"Row {i}: {filepath} does not exist")

actual_clips = list(CLIPS_DIR.glob("*.mp4"))

print(f"CSV rows: {len(rows)}")
print(f"Actual clips in {CLIPS_DIR}: {len(actual_clips)}")
print(f"Errors found: {len(errors)}\n")

if errors:
    print("ERRORS:")
    for e in errors[:10]:
        print(f"  - {e}")
    if len(errors) > 10:
        print(f"  ... and {len(errors)-10} more")
    exit(1)
else:
    print("[PASS] All checks passed!")
