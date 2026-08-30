"""
trim_participants.py — shrink DAIC-WOZ folders to only the files this
audio+visual baseline actually uses, to save disk space.

For each xxx_P/ folder under DATA_ROOT it KEEPS:
    xxx_COVAREP.csv      (audio)
    xxx_FORMANT.csv      (audio)
    xxx_CLNF_gaze.txt    (visual)
    xxx_CLNF_pose.txt    (visual)
    xxx_CLNF_AUs.txt     (visual)
and DELETES everything else in the folder (AUDIO.wav, CLNF_features.txt,
CLNF_features3D.txt, CLNF_hog.txt, TRANSCRIPT.csv, etc.), which is where
almost all the size lives.

SAFE BY DEFAULT: runs in dry-run mode and only PRINTS what it would delete.
To actually delete, run with --apply.

Usage:
    python3 trim_participants.py            # dry run: shows what would be deleted + space saved
    python3 trim_participants.py --apply    # actually delete

You can run this repeatedly as you download more participants in batches.
"""

import sys
from pathlib import Path
import daic_woz_pipeline.src.config as config

KEEP_SUFFIXES = {
    "_COVAREP.csv",
    "_FORMANT.csv",
    "_CLNF_gaze.txt",
    "_CLNF_pose.txt",
    "_CLNF_AUs.txt",
    "_TRANSCRIPT.csv",   # kept: needed for the text modality and interviewer-
                          # segment removal (see IMPROVEMENT_ROADMAP.md). Tiny.
}


def keep_file(name: str) -> bool:
    return any(name.endswith(suf) for suf in KEEP_SUFFIXES)


def human(nbytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if nbytes < 1024:
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}TB"


def main():
    apply = "--apply" in sys.argv
    root = config.DATA_ROOT
    if not root.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {root}")

    total_freed = 0
    folders = sorted(root.glob("*_P"))
    if not folders:
        print(f"No *_P folders found under {root}")
        return

    for folder in folders:
        if not folder.is_dir():
            continue
        for f in folder.iterdir():
            if f.is_file() and not keep_file(f.name):
                size = f.stat().st_size
                total_freed += size
                if apply:
                    f.unlink()
                    print(f"  deleted {f.relative_to(root)} ({human(size)})")
                else:
                    print(f"  would delete {f.relative_to(root)} ({human(size)})")

    mode = "Freed" if apply else "Would free"
    print(f"\n{mode} {human(total_freed)} across {len(folders)} participant folders.")
    if not apply:
        print("Dry run only. Re-run with --apply to actually delete.")


if __name__ == "__main__":
    main()