"""
explore.py — generate visualizations from the extracted DAIC-WOZ features.

Works on whatever you've built so far:
  - With your small test batch (a few participants): shows that extraction
    produced real, populated features (proof the pipeline works).
  - With the full dataset: shows the class imbalance, feature distributions,
    and an audio-vs-visual overview — the figures that belong in deliverable 1.

Run AFTER build_dataset.py (it reads outputs/train_features.parquet).

    python3 explore.py

All figures are saved into outputs/figures/.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

FIG_DIR = config.OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#4C72B0"
RED = "#C44E52"


def load():
    tr = config.OUT_DIR / "train_features.parquet"
    dv = config.OUT_DIR / "dev_features.parquet"
    if not tr.exists():
        sys.exit("No train_features.parquet found. Run build_dataset.py first.")
    train = pd.read_parquet(tr)
    dev = pd.read_parquet(dv) if dv.exists() else pd.DataFrame()
    return train, dev


def feature_columns(df):
    audio = [c for c in df.columns if c.startswith(("covarep_", "formant_"))]
    visual = [c for c in df.columns if c.startswith(("gaze_", "pose_", "au_"))]
    return audio, visual


def fig_class_balance(train, dev):
    """The key figure for deliverable 1: shows the imbalance SMOTE addresses."""
    panels = [("Train", train)]
    if not dev.empty:
        panels.append(("Dev", dev))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 3.6))
    if len(panels) == 1:
        axes = [axes]
    for ax, (name, df) in zip(axes, panels):
        counts = df[config.LABEL_COL].value_counts().reindex([0, 1]).fillna(0)
        ax.bar(["Not depressed", "Depressed"], counts.values, color=[BLUE, RED])
        ax.set_title(f"{name}  (n={len(df)})")
        ax.set_ylabel("Participants")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 0.3, str(int(v)), ha="center")
    fig.suptitle("Class balance (before SMOTE)")
    fig.tight_layout()
    out = FIG_DIR / "class_balance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_feature_coverage(train, audio, visual):
    """Proof-of-extraction figure: how many features are populated (not NaN)
    per participant. Useful even with a tiny batch to show extraction worked."""
    cols = audio + visual
    coverage = train[cols].notna().mean(axis=1) * 100
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ids = train[config.ID_COL].astype(str) if config.ID_COL in train else train.index.astype(str)
    ax.bar(ids, coverage.values, color=BLUE)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% features populated")
    ax.set_xlabel("Participant")
    ax.set_title(f"Feature extraction coverage  ({len(cols)} features each)")
    ax.axhline(100, color="gray", ls="--", lw=0.8)
    fig.tight_layout()
    out = FIG_DIR / "feature_coverage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_modality_breakdown(audio, visual):
    """Simple bar of how many features come from each modality / source."""
    groups = {
        "COVAREP (audio)": sum(c.startswith("covarep_") for c in audio),
        "FORMANT (audio)": sum(c.startswith("formant_") for c in audio),
        "Gaze (visual)": sum(c.startswith("gaze_") for c in visual),
        "Pose (visual)": sum(c.startswith("pose_") for c in visual),
        "AUs (visual)": sum(c.startswith("au_") for c in visual),
    }
    fig, ax = plt.subplots(figsize=(6, 3.6))
    colors = [BLUE, BLUE, RED, RED, RED]
    ax.bar(list(groups.keys()), list(groups.values()), color=colors)
    ax.set_ylabel("Number of features")
    ax.set_title("Feature count by source")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    out = FIG_DIR / "modality_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_feature_distributions(train, audio, visual):
    """Distributions of a few example features. With many participants this
    shows real spread; with few it just confirms values are sensible."""
    examples = []
    for prefix in ["covarep_f0_mean", "formant_f0_mean", "gaze_f0_mean",
                   "pose_f0_mean", "au_f0_mean"]:
        match = [c for c in (audio + visual) if c == prefix]
        if not match:
            match = [c for c in (audio + visual) if c.startswith(prefix.split("_f0")[0])]
        if match:
            examples.append(match[0])
    examples = examples[:4]
    if not examples:
        return None
    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, examples):
        vals = train[col].dropna().values
        ax.hist(vals, bins=min(10, max(3, len(vals))), color=BLUE, edgecolor="white")
        ax.set_title(col, fontsize=8)
        ax.set_ylabel("count")
    fig.suptitle("Example feature distributions (train)")
    fig.tight_layout()
    out = FIG_DIR / "feature_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    train, dev = load()
    audio, visual = feature_columns(train)
    n = len(train)
    print(f"Loaded {n} train participants, {len(dev)} dev.")
    print(f"Features: {len(audio)} audio + {len(visual)} visual = {len(audio)+len(visual)}")

    saved = []
    saved.append(fig_class_balance(train, dev))
    saved.append(fig_feature_coverage(train, audio, visual))
    saved.append(fig_modality_breakdown(audio, visual))
    d = fig_feature_distributions(train, audio, visual)
    if d:
        saved.append(d)

    print("\nSaved figures:")
    for s in saved:
        print(f"  {s}")

    if n < 20 or (not dev.empty and (dev[config.LABEL_COL].nunique() < 2)):
        print("\nNOTE: This looks like a small/single-class test batch.")
        print("The 'feature coverage' and 'feature count by source' figures are")
        print("meaningful now (they prove extraction works). The class-balance and")
        print("distribution figures only become informative on the full dataset.")


if __name__ == "__main__":
    main()