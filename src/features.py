"""
Feature extraction for original DAIC-WOZ (CLNF files).

The corpus gives ONE PHQ-8 label per participant but MANY frames of
per-frame features. We collapse each participant's multivariate time
series into a single fixed-length vector by computing statistical
"functionals" (mean, std, percentiles, skew, kurtosis, ...) over every
feature column. This is the standard, well-understood AVEC-style baseline
approach and keeps the model small and interpretable.

Audio  = COVAREP (74 features/frame) + FORMANT (5 formants/frame)
Visual = CLNF gaze + pose + action units (AUs)

We deliberately skip the raw 2D/3D landmark files for the baseline: gaze,
pose and AUs are lower-dimensional, more interpretable, and known to carry
most of the depression-relevant signal. You can add landmarks later.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis

import config


def _functionals(arr: np.ndarray, prefix: str) -> dict:
    """Compute functional statistics over a 2D array (frames x features)."""
    if arr.size == 0 or arr.shape[0] == 0:
        # Participant had no usable frames; functionals are undefined.
        # Return NaNs so the caller can decide how to impute.
        n_feat = arr.shape[1] if arr.ndim == 2 else 0
        feats = {}
        for j in range(n_feat):
            for fn in config.FUNCTIONALS:
                feats[f"{prefix}_f{j}_{fn}"] = np.nan
        return feats

    feats = {}
    for j in range(arr.shape[1]):
        col = arr[:, j]
        col = col[~np.isnan(col)]
        if col.size == 0:
            vals = {fn: np.nan for fn in config.FUNCTIONALS}
        else:
            vals = {
                "mean": np.mean(col),
                "std": np.std(col),
                "min": np.min(col),
                "max": np.max(col),
                "median": np.median(col),
                "q25": np.percentile(col, 25),
                "q75": np.percentile(col, 75),
                "skew": skew(col) if col.size > 2 else 0.0,
                "kurtosis": kurtosis(col) if col.size > 2 else 0.0,
            }
        for fn in config.FUNCTIONALS:
            feats[f"{prefix}_f{j}_{fn}"] = vals[fn]
    return feats


def _read_covarep(folder: Path, pid: int) -> np.ndarray:
    """COVAREP: comma-separated, no header. Optionally drop unvoiced frames."""
    path = folder / f"{pid}_COVAREP.csv"
    if not path.exists():
        return np.empty((0, 0))
    arr = pd.read_csv(path, header=None).values.astype(float)
    if config.DROP_UNVOICED and arr.shape[1] > config.COVAREP_VUV_COL_INDEX:
        voiced = arr[:, config.COVAREP_VUV_COL_INDEX] == 1
        if voiced.sum() > 0:
            arr = arr[voiced]
    return arr


def _read_formant(folder: Path, pid: int) -> np.ndarray:
    path = folder / f"{pid}_FORMANT.csv"
    if not path.exists():
        return np.empty((0, 0))
    return pd.read_csv(path, header=None).values.astype(float)


def _read_clnf(folder: Path, pid: int, suffix: str) -> np.ndarray:
    """
    CLNF files are comma-separated WITH a header row. The first few columns
    are frame, timestamp, confidence, success. We drop those metadata columns
    and keep only the actual feature columns, and we keep only rows the
    tracker marked successful (success == 1) with confidence above a floor.
    """
    path = folder / f"{pid}_CLNF_{suffix}.txt"
    if not path.exists():
        return np.empty((0, 0))
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Filter to confidently tracked frames when those columns exist.
    if "success" in df.columns:
        df = df[df["success"] == 1]
    if "confidence" in df.columns:
        df = df[df["confidence"] >= 0.5]

    drop_cols = [c for c in ["frame", "timestamp", "confidence", "success"]
                 if c in df.columns]
    df = df.drop(columns=drop_cols)
    if df.shape[0] == 0:
        return np.empty((0, 0))
    return df.values.astype(float)


def extract_audio(folder: Path, pid: int) -> dict:
    cov = _read_covarep(folder, pid)
    form = _read_formant(folder, pid)
    feats = {}
    feats.update(_functionals(cov, "covarep"))
    feats.update(_functionals(form, "formant"))
    return feats


def extract_visual(folder: Path, pid: int) -> dict:
    gaze = _read_clnf(folder, pid, "gaze")
    pose = _read_clnf(folder, pid, "pose")
    aus = _read_clnf(folder, pid, "AUs")
    feats = {}
    feats.update(_functionals(gaze, "gaze"))
    feats.update(_functionals(pose, "pose"))
    feats.update(_functionals(aus, "au"))
    return feats


def extract_participant(data_root: Path, pid: int) -> dict:
    """Return a flat dict of all audio + visual functionals for one participant."""
    folder = data_root / f"{pid}_P"
    row = {config.ID_COL: pid}
    if not folder.exists():
        print(f"  [warn] folder missing for participant {pid}: {folder}")
        return row
    row.update(extract_audio(folder, pid))
    row.update(extract_visual(folder, pid))
    return row
