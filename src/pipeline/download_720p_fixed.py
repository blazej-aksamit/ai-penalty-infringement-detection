import csv
import os
from collections import OrderedDict, defaultdict
from pathlib import Path

from SoccerNet.Downloader import SoccerNetDownloader

RAW = Path("data/raw/SoccerNet")
INP = Path("data/meta/penalties.csv")
CLIPS_INDEX = Path("data/clips/penalties_720p/clips_index.csv")

BATCH_GAMES = 20
SPLITS = ["train", "valid", "test", "challenge"]
PW = os.getenv("SOCCERNET_PW", "")


def safe_float(value):
    try:
        return round(float(value), 3)
    except Exception:
        return None


def load_done_clip_keys():
    done = set()

    if not CLIPS_INDEX.exists() or CLIPS_INDEX.stat().st_size == 0:
        return done

    with CLIPS_INDEX.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            game_id = str(row.get("game_id", "")).strip()
            half = str(row.get("half", "")).strip()
            t = safe_float(row.get("t_seconds", ""))
            if game_id and half in {"1", "2"} and t is not None:
                done.add((game_id, half, t))

    return done


def build_game_targets():
    if not INP.exists():
        raise SystemExit(f"Missing {INP}")

    done_keys = load_done_clip_keys()

    games = OrderedDict()
    expected_by_game = defaultdict(set)

    with INP.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            game_id = str(row.get("game_id", "")).strip()
            half = str(row.get("half", "")).strip()
            t = safe_float(row.get("t_seconds", ""))
            labels_file = Path(str(row.get("labels_file", "")).strip())

            if not game_id or half not in {"1", "2"} or t is None:
                continue

            try:
                parts = labels_file.parts
                sn_idx = parts.index("SoccerNet")
                league = parts[sn_idx + 1]
                season = parts[sn_idx + 2]
            except (ValueError, IndexError):
                print(f"[WARN] Cannot parse league/season from: {labels_file}")
                continue

            game_key = (league, season, game_id)

            if game_key not in games:
                games[game_key] = {
                    "league": league,
                    "season": season,
                    "game_id": game_id,
                    "half_files": set(),
                }

            games[game_key]["half_files"].add(f"{half}_720p.mkv")
            expected_by_game[game_key].add((game_id, half, t))

    selected = []

    for game_key, info in games.items():
        expected = expected_by_game[game_key]
        done_count = sum(1 for key in expected if key in done_keys)

        if expected and done_count >= len(expected):
            continue

        info["pending_penalties"] = len(expected) - done_count
        selected.append(info)

        if len(selected) >= BATCH_GAMES:
            break

    return selected


def invoke_download(dl, game_rel, filename, split, league):
    attempts = [
        lambda: dl.downloadGame(game=game_rel, files=[filename], split=split, task=league),
        lambda: dl.downloadGame(game=game_rel, files=[filename], split=split),
        lambda: dl.downloadGame(game_rel, files=[filename], split=split, task=league),
        lambda: dl.downloadGame(game_rel, files=[filename], split=split),
        lambda: dl.downloadGame(game_rel, files=[filename], spl=split, verbose=True),
    ]

    for fn in attempts:
        try:
            fn()
            return
        except Exception:
            continue


def try_download(dl, info, half_file):
    expected = RAW / info["league"] / info["season"] / info["game_id"] / half_file

    if expected.exists() and expected.stat().st_size > 0:
        print(f"[SKIP] already exists: {expected}")
        return True

    expected.parent.mkdir(parents=True, exist_ok=True)
    game_rel = f'{info["league"]}/{info["season"]}/{info["game_id"]}'

    print(f'[DL] {game_rel} | {half_file}')

    for split in SPLITS:
        invoke_download(dl, game_rel, half_file, split, info["league"])

        if expected.exists() and expected.stat().st_size > 0:
            print(f"[OK] {expected}")
            return True

    print(f"[FAIL] missing after all splits: {expected}")
    return False


def main():
    batch = build_game_targets()

    if not batch:
        print("No unfinished games left. Nothing to download.")
        return

    print(f"Next batch: {len(batch)} match(es) (limit={BATCH_GAMES})")
    for i, info in enumerate(batch, start=1):
        halves = ", ".join(sorted(info["half_files"]))
        print(
            f'  {i:02d}. {info["league"]}/{info["season"]}/{info["game_id"]} '
            f'| halves: {halves} | pending penalties: {info["pending_penalties"]}'
        )

    dl = SoccerNetDownloader(LocalDirectory=str(RAW))
    if PW and hasattr(dl, "password"):
        dl.password = PW
    elif not PW:
        print("WARNING: SOCCERNET_PW is not set.")

    ok = 0
    fail = 0

    for info in batch:
        for half_file in sorted(info["half_files"]):
            if try_download(dl, info, half_file):
                ok += 1
            else:
                fail += 1

    print(f"\nDone. Downloaded/available files OK: {ok} | FAIL: {fail}")


if __name__ == "__main__":
    main()