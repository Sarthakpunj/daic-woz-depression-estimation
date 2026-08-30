"""
download_baseline.py — download only the participants in the official
train/dev splits, keep only the small files needed (baseline + transcript),
and clean up as it goes so local disk never fills.

Workflow per participant:
  1. download {id}_P.zip from the USC index with curl
  2. unzip into {id}_P/
  3. trim to the KEEP_SUFFIXES files (via trim_participants logic)
  4. delete the zip
Safe to stop and re-run: participants already trimmed are skipped.

Run from inside your pipeline folder, with the split CSVs reachable:
    python3 download_baseline.py

Edit the two paths below if needed.
"""

import subprocess
import zipfile
from pathlib import Path
import pandas as pd

import config
from trim_participants import keep_file, human

BASE_URL = "https://dcapswoz.ict.usc.edu/wwwdaicwoz"

# Where to put the trimmed dataset. Defaults to config.DATA_ROOT so the rest
# of your pipeline (build_dataset.py) finds it without changes.
DEST = config.DATA_ROOT

# The split files live in DEST (you already downloaded them there).
SPLIT_FILES = [config.TRAIN_SPLIT, config.DEV_SPLIT]


def needed_ids():
    ids = []
    for fn in SPLIT_FILES:
        path = DEST / fn
        if not path.exists():
            raise FileNotFoundError(
                f"Split file not found: {path}. "
                f"Make sure the train/dev split CSVs are in {DEST}.")
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        ids += [int(x) for x in df[config.ID_COL].tolist()]
    return sorted(set(ids))


def already_done(pid):
    """Re-download pass for transcripts: 'done' means the transcript exists."""
    folder = DEST / f"{pid}_P"
    return folder.exists() and (folder / f"{pid}_TRANSCRIPT.csv").exists()


def trim_folder(folder: Path, pid: int):
    freed = 0
    for f in folder.iterdir():
        if f.is_file() and not keep_file(f.name):
            freed += f.stat().st_size
            f.unlink()
    return freed


def download_one(pid: int):
    zip_path = DEST / f"{pid}_P.zip"
    url = f"{BASE_URL}/{pid}_P.zip"
    print(f"[{pid}] downloading...", flush=True)
    # -f fail on error, -L follow redirects, -o output, --retry for robustness
    result = subprocess.run(
        ["curl", "-fL", "--retry", "3", "-o", str(zip_path), url],
        capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[{pid}] download FAILED (skipping): {result.stderr.strip()[:200]}")
        if zip_path.exists():
            zip_path.unlink()
        return False

    print(f"[{pid}] unzipping...", flush=True)
    folder = DEST / f"{pid}_P"
    folder.mkdir(exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            # Some zips contain a top-level "{pid}_P/" folder; others extract
            # files loose. Detect which, and normalise so files always end up
            # directly inside our dedicated {pid}_P/ folder.
            prefix = f"{pid}_P/"
            has_wrapper = any(n.startswith(prefix) for n in names)
            if has_wrapper:
                z.extractall(DEST)
            else:
                z.extractall(folder)
    except zipfile.BadZipFile:
        print(f"[{pid}] bad zip (skipping)")
        zip_path.unlink()
        return False

    if not folder.exists() or not any(folder.iterdir()):
        print(f"[{pid}] WARNING: no files found after unzip")
        zip_path.unlink()
        return False

    freed = trim_folder(folder, pid)
    zip_path.unlink()
    print(f"[{pid}] done (trimmed {human(freed)}, zip removed)")
    return True


def main():
    ids = needed_ids()
    print(f"{len(ids)} participants needed (train+dev).")
    done = [p for p in ids if already_done(p)]
    todo = [p for p in ids if not already_done(p)]
    print(f"  already done: {len(done)}")
    print(f"  to download : {len(todo)}\n")

    ok, fail = 0, []
    for pid in todo:
        if download_one(pid):
            ok += 1
        else:
            fail.append(pid)

    print(f"\nFinished. Downloaded+trimmed {ok}. Failed: {len(fail)}.")
    if fail:
        print(f"  failed ids: {fail}")
        print("  re-run the script to retry just those.")


if __name__ == "__main__":
    main()