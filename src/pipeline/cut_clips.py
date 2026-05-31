import csv
import re
import subprocess
from collections import defaultdict
from pathlib import Path

# ---------- settings ----------
CSV_PATH = Path("data/meta/penalties.csv")
OUT_DIR = Path("data/clips/penalties_720p")
INDEX_PATH = OUT_DIR / "clips_index.csv"

PRE_SECONDS = 5
POST_SECONDS = 5

REENCODE_TO_MP4 = True
DELETE_SOURCE_AFTER_SUCCESS = True

# preferred source file names
VIDEO_SUFFIXES = ["_720p.mkv", "_224p.mkv", ".mkv"]
# -----------------------------


def slug(s, max_len=120):
    s = str(s).strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    return s[:max_len] if len(s) > max_len else s


def safe_float(value):
    try:
        return round(float(value), 3)
    except Exception:
        return None


def normalize_key(game_id, half, t_seconds):
    return (str(game_id).strip(), int(half), round(float(t_seconds), 3))


def make_clip_name(game_id, half, t_seconds):
    safe_game = slug(game_id)
    ext = ".mp4" if REENCODE_TO_MP4 else ".mkv"
    return f"{safe_game}_H{int(half)}_{int(round(float(t_seconds))):06d}s{ext}"


def resolve_video_path(game_dir, half):
    candidates = [game_dir / f"{half}{suffix}" for suffix in VIDEO_SUFFIXES]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # best guess even if not downloaded yet


def run(cmd):
    subprocess.run(cmd, check=True)


def load_existing_index():
    done_keys = set()

    if not INDEX_PATH.exists() or INDEX_PATH.stat().st_size == 0:
        return done_keys

    with INDEX_PATH.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            game_id = str(row.get("game_id", "")).strip()
            half = str(row.get("half", "")).strip()
            t = safe_float(row.get("t_seconds", ""))
            if game_id and half in {"1", "2"} and t is not None:
                done_keys.add((game_id, int(half), t))

    return done_keys


def open_index_writer():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = (not INDEX_PATH.exists()) or INDEX_PATH.stat().st_size == 0

    f = INDEX_PATH.open("a", newline="", encoding="utf-8")
    fieldnames = [
        "clip_path",
        "game_id",
        "half",
        "t_seconds",
        "label",
        "gameTime",
        "labels_file",
        "video_path",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    if write_header:
        writer.writeheader()

    return f, writer


def load_jobs():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing {CSV_PATH}.")
    if CSV_PATH.stat().st_size == 0:
        raise RuntimeError(f"{CSV_PATH} is empty.")

    jobs_by_video = defaultdict(list)
    skipped_bad_row = 0

    with CSV_PATH.open("r", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)

        for row in reader:
            try:
                game_id = str(row["game_id"]).strip()
                half = int(row["half"])
                t = round(float(row["t_seconds"]), 3)
                label = row.get("label", "")
                game_time = row.get("gameTime", "")
                labels_file = str(row["labels_file"]).strip()
            except Exception:
                skipped_bad_row += 1
                continue

            if half not in (1, 2) or not labels_file:
                skipped_bad_row += 1
                continue

            game_dir = Path(labels_file).parent
            video_path = resolve_video_path(game_dir, half)
            clip_path = OUT_DIR / make_clip_name(game_id, half, t)

            jobs_by_video[str(video_path)].append(
                {
                    "key": normalize_key(game_id, half, t),
                    "game_id": game_id,
                    "half": half,
                    "t_seconds": t,
                    "label": label,
                    "gameTime": game_time,
                    "labels_file": labels_file,
                    "video_path": video_path,
                    "clip_path": clip_path,
                }
            )

    for _, items in jobs_by_video.items():
        items.sort(key=lambda x: x["t_seconds"])

    return jobs_by_video, skipped_bad_row


def write_index_row(writer, item):
    writer.writerow(
        {
            "clip_path": str(item["clip_path"]).replace("\\", "/"),
            "game_id": item["game_id"],
            "half": item["half"],
            "t_seconds": f'{item["t_seconds"]:.3f}',
            "label": item["label"],
            "gameTime": item["gameTime"],
            "labels_file": item["labels_file"],
            "video_path": str(item["video_path"]).replace("\\", "/"),
        }
    )


def item_is_done(item, done_keys):
    return item["key"] in done_keys and item["clip_path"].exists()


def build_ffmpeg_cmd(video_path, out_path, start, duration):
    if REENCODE_TO_MP4:
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-crf",
            "28",
            "-preset",
            "veryfast",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            str(out_path),
        ]

    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-c",
        "copy",
        str(out_path),
    ]


def main():
    jobs_by_video, skipped_bad_row = load_jobs()
    done_keys = load_existing_index()
    index_file, writer = open_index_writer()

    created = 0
    indexed_existing = 0
    skipped_missing_video = 0
    deleted_sources = 0

    try:
        for video_key in sorted(jobs_by_video.keys()):
            items = jobs_by_video[video_key]
            video_path = items[0]["video_path"]

            # if a clip already exists in penalties_720p but isn't indexed yet, register it
            for item in items:
                if item["clip_path"].exists() and item["key"] not in done_keys:
                    write_index_row(writer, item)
                    done_keys.add(item["key"])
                    indexed_existing += 1

            index_file.flush()

            pending_items = [item for item in items if not item_is_done(item, done_keys)]

            # nothing left for this source video
            if not pending_items:
                if DELETE_SOURCE_AFTER_SUCCESS and video_path.exists():
                    try:
                        video_path.unlink()
                        deleted_sources += 1
                        print(f"[DEL] {video_path.name}")
                    except Exception as e:
                        print(f"[WARN] Could not delete {video_path}: {e}")
                continue

            # source video not downloaded yet
            if not video_path.exists():
                skipped_missing_video += len(pending_items)
                continue

            batch_ok = True

            for item in pending_items:
                start = max(item["t_seconds"] - PRE_SECONDS, 0.0)
                duration = PRE_SECONDS + POST_SECONDS

                cmd = build_ffmpeg_cmd(video_path, item["clip_path"], start, duration)

                try:
                    run(cmd)
                    created += 1
                    print(f"[OK] {item['clip_path'].name}")

                    if item["key"] not in done_keys:
                        write_index_row(writer, item)
                        done_keys.add(item["key"])
                        index_file.flush()

                except subprocess.CalledProcessError:
                    batch_ok = False
                    print(
                        f"[FAIL] ffmpeg failed for "
                        f"{item['game_id']} H{item['half']} t={item['t_seconds']}"
                    )

            # delete source half only when all clips for that half are done
            if DELETE_SOURCE_AFTER_SUCCESS and batch_ok and video_path.exists():
                all_done_now = all(item_is_done(item, done_keys) for item in items)
                if all_done_now:
                    try:
                        video_path.unlink()
                        deleted_sources += 1
                        print(f"[DEL] {video_path.name}")
                    except Exception as e:
                        print(f"[WARN] Could not delete {video_path}: {e}")

    finally:
        index_file.close()

    print("\nDone.")
    print(f"Created clips: {created}")
    print(f"Indexed already-existing clips: {indexed_existing}")
    print(f"Skipped (missing source video right now): {skipped_missing_video}")
    print(f"Skipped (bad csv rows): {skipped_bad_row}")
    print(f"Deleted source half-videos: {deleted_sources}")
    print(f"Index saved to: {INDEX_PATH}")


if __name__ == "__main__":
    main()