"""
inspect_raw.py — deep look at ONE participant's raw DAIC-WOZ data.

This is a "data understanding" tool: it shows what the raw per-frame files
actually contain and how your preprocessing decisions affect them, BEFORE
the functionals collapse everything into one vector. It needs only a single
participant, so it works fine under a storage crunch.

It reports / plots, for one participant:
  - COVAREP: total frames, voiced vs unvoiced split (the VUV flag), sampling
  - FORMANT: frame count
  - CLNF gaze/pose/AUs: total frames, how many survive the confidence/success
    filter, i.e. how much low-quality tracking you discard
  - A feature-over-time plot (e.g. F0 across the interview) to show the raw
    signal your functionals summarise

Usage:
    python3 inspect_raw.py            # uses the first participant found
    python3 inspect_raw.py 303        # specific participant ID

Figures saved to outputs/figures/.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import daic_woz_pipeline.src.config as config

FIG_DIR = config.OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
BLUE = "#4C72B0"
RED = "#C44E52"


def find_participant():
    if len(sys.argv) > 1:
        return int(sys.argv[1])
    folders = sorted(config.DATA_ROOT.glob("*_P"))
    if not folders:
        sys.exit(f"No *_P folders under {config.DATA_ROOT}")
    return int(folders[0].name.split("_")[0])


def read_clnf(folder, pid, suffix):
    path = folder / f"{pid}_CLNF_{suffix}.txt"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def main():
    pid = find_participant()
    folder = config.DATA_ROOT / f"{pid}_P"
    if not folder.exists():
        sys.exit(f"Participant folder not found: {folder}")

    print(f"=== Raw data inspection: participant {pid} ===\n")

    # ---- COVAREP (audio) ----
    cov_path = folder / f"{pid}_COVAREP.csv"
    if cov_path.exists():
        cov = pd.read_csv(cov_path, header=None).values.astype(float)
        n_frames = cov.shape[0]
        vuv = cov[:, config.COVAREP_VUV_COL_INDEX]
        n_voiced = int((vuv == 1).sum())
        n_unvoiced = n_frames - n_voiced
        # COVAREP is sampled at 100 Hz (10 ms hop) in DAIC-WOZ
        dur_sec = n_frames / 100.0
        print(f"COVAREP (audio):")
        print(f"  total frames        : {n_frames:,}")
        print(f"  ~duration           : {dur_sec/60:.1f} min (at 100 Hz)")
        print(f"  voiced frames       : {n_voiced:,} ({100*n_voiced/n_frames:.1f}%)")
        print(f"  unvoiced frames     : {n_unvoiced:,} ({100*n_unvoiced/n_frames:.1f}%)")
        print(f"  -> we keep voiced only (DROP_UNVOICED={config.DROP_UNVOICED})\n")

        # Feature-over-time plot: F0 (column 0) across the interview
        f0 = cov[:, 0]
        fig, ax = plt.subplots(figsize=(8, 3))
        t = np.arange(n_frames) / 100.0 / 60.0  # minutes
        ax.plot(t, f0, color=BLUE, lw=0.4)
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("F0 (COVAREP col 0)")
        ax.set_title(f"Participant {pid}: raw F0 over the interview")
        fig.tight_layout()
        out = FIG_DIR / f"raw_f0_timeseries_{pid}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}\n")

        # Voiced/unvoiced bar
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.bar(["Voiced", "Unvoiced"], [n_voiced, n_unvoiced], color=[BLUE, RED])
        ax.set_ylabel("Frames")
        ax.set_title(f"Participant {pid}: voiced vs unvoiced (COVAREP)")
        for i, v in enumerate([n_voiced, n_unvoiced]):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom")
        fig.tight_layout()
        out = FIG_DIR / f"voiced_unvoiced_{pid}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}\n")

    # ---- FORMANT (audio) ----
    form_path = folder / f"{pid}_FORMANT.csv"
    if form_path.exists():
        form = pd.read_csv(form_path, header=None).values
        print(f"FORMANT (audio):")
        print(f"  total frames        : {form.shape[0]:,}")
        print(f"  formants per frame  : {form.shape[1]}\n")

    # ---- CLNF visual: confidence filtering effect ----
    print("CLNF visual (frames kept after confidence/success filter):")
    filt_rows = []
    for suffix in ["gaze", "pose", "AUs"]:
        df = read_clnf(folder, pid, suffix)
        if df is None:
            continue
        total = len(df)
        kept = df.copy()
        if "success" in kept.columns:
            kept = kept[kept["success"] == 1]
        if "confidence" in kept.columns:
            kept = kept[kept["confidence"] >= 0.5]
        n_kept = len(kept)
        pct = 100 * n_kept / total if total else 0
        print(f"  {suffix:5s}: {total:,} total -> {n_kept:,} kept ({pct:.1f}%)")
        filt_rows.append((suffix, total, n_kept))
    print()

    if filt_rows:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        x = np.arange(len(filt_rows))
        totals = [r[1] for r in filt_rows]
        kepts = [r[2] for r in filt_rows]
        ax.bar(x - 0.2, totals, 0.4, label="Total frames", color="#cccccc")
        ax.bar(x + 0.2, kepts, 0.4, label="Kept (high-confidence)", color=BLUE)
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in filt_rows])
        ax.set_ylabel("Frames")
        ax.set_title(f"Participant {pid}: CLNF confidence filtering")
        ax.legend()
        fig.tight_layout()
        out = FIG_DIR / f"clnf_filtering_{pid}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}\n")

    print("Done. These figures document your preprocessing on real data and")
    print("are valid evidence for Deliverable 1 even with one participant.")


if __name__ == "__main__":
    main()