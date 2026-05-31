import os, csv
from pathlib import Path
from SoccerNet.Downloader import SoccerNetDownloader

ROOT = Path("data/raw/SoccerNet")
META = Path("data/meta/penalties.csv")  # your current filtered penalties.csv
PW = os.environ.get("SOCCERNET_PW", "")

if not PW:
    raise SystemExit("SOCCERNET_PW is not set in your environment.")

if not META.exists():
    raise SystemExit(f"Missing: {META}")

dl = SoccerNetDownloader(LocalDirectory=str(ROOT))
if hasattr(dl, "password"):
    dl.password = PW

splits = ["train", "valid", "test", "challenge"]

def game_rel_from_labels(labels_file: str) -> Path:
    p = Path(labels_file)
    game_dir = p.parent

    # Try: absolute -> relative to ROOT absolute
    try:
        return game_dir.resolve().relative_to(ROOT.resolve())
    except Exception:
        pass

    # Fallback: find ".../SoccerNet/<task>/<season>/<game>/Labels..."
    parts = list(game_dir.parts)
    if "SoccerNet" in parts:
        i = parts.index("SoccerNet")
        return Path(*parts[i+1:])

    # Last fallback: assume labels_file already relative
    return game_dir

def try_download(game_rel: Path, filename: str) -> bool:
    # SoccerNet API usually wants forward slashes
    game_key = game_rel.as_posix()
    target = ROOT / game_rel / filename

    # If already downloaded, skip
    if target.exists() and target.stat().st_size > 0:
        print(f"[SKIP] already have {target}")
        return True

    for spl in splits:
        try:
            dl.downloadGame(game_key, files=[filename], spl=spl, verbose=True)
        except Exception as e:
            # some versions raise, some only print errors
            print(f"   (exception on split={spl}): {type(e).__name__}: {e}")

        if target.exists() and target.stat().st_size > 0:
            mb = target.stat().st_size / (1024*1024)
            print(f"[OK] {game_key} | {filename} | split={spl} | {mb:.1f} MB")
            return True

    print(f"[FAIL] {game_key} | {filename} (not found in any split)")
    return False

# Collect unique (game_rel, half)
targets = {}
with META.open(newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        labels_file = row.get("labels_file", "")
        half = str(row.get("half", "")).strip()
        if not labels_file or half not in ("1","2"):
            continue
        game_rel = game_rel_from_labels(labels_file)
        targets[(game_rel.as_posix(), half)] = game_rel

print(f"Targets (unique game+half): {len(targets)}")

ok = fail = 0
for (game_key, half), game_rel in targets.items():
    # Try both naming conventions (some servers/packages use one or the other)
    candidates = [f"{half}_720p.mkv", f"{half}.mkv"]
    got = False
    for fn in candidates:
        if try_download(game_rel, fn):
            got = True
            break
    if got: ok += 1
    else: fail += 1

print(f"\nDone. OK: {ok}  FAIL: {fail}")
