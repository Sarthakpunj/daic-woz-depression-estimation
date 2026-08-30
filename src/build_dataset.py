"""
Build participant-level feature tables for the official train and dev splits,
then cache them to disk so feature extraction (the slow part) runs only once.

Run:  python build_dataset.py
"""

import pandas as pd
from pathlib import Path

import config
from features import extract_participant


def load_split(data_root: Path, filename: str) -> pd.DataFrame:
    df = pd.read_csv(data_root / filename)
    df.columns = [c.strip() for c in df.columns]
    # The official files use these exact headers; assert so failures are loud.
    for col in (config.ID_COL, config.LABEL_COL):
        if col not in df.columns:
            raise KeyError(
                f"Expected column '{col}' in {filename}. "
                f"Found columns: {list(df.columns)}"
            )
    return df


def build_split(data_root: Path, filename: str, tag: str) -> pd.DataFrame:
    split = load_split(data_root, filename)
    rows = []
    skipped = 0
    print(f"[{tag}] extracting features for {len(split)} participants...")
    for _, r in split.iterrows():
        pid = int(r[config.ID_COL])
        folder = data_root / f"{pid}_P"
        if not folder.exists():
            skipped += 1
            continue
        feats = extract_participant(data_root, pid)
        feats[config.LABEL_COL] = int(r[config.LABEL_COL])
        if config.SCORE_COL in split.columns:
            feats[config.SCORE_COL] = r[config.SCORE_COL]
        if getattr(config, "GENDER_COL", None) and config.GENDER_COL in split.columns:
            feats[config.GENDER_COL] = int(r[config.GENDER_COL])
        rows.append(feats)
        print(f"  done {pid}")
    print(f"[{tag}] extracted {len(rows)}, skipped {skipped} missing.")
    return pd.DataFrame(rows)


def main():
    data_root = config.DATA_ROOT
    if not data_root.exists():
        raise FileNotFoundError(
            f"DATA_ROOT does not exist: {data_root}\n"
            f"Edit config.py and set DATA_ROOT to your DAIC-WOZ folder."
        )

    train = build_split(data_root, config.TRAIN_SPLIT, "train")
    dev = build_split(data_root, config.DEV_SPLIT, "dev")

    train_path = config.OUT_DIR / "train_features.parquet"
    dev_path = config.OUT_DIR / "dev_features.parquet"
    train.to_parquet(train_path)
    dev.to_parquet(dev_path)

    print(f"\nSaved:\n  {train_path}  shape={train.shape}"
          f"\n  {dev_path}  shape={dev.shape}")
    print("\nTrain class balance:")
    print(train[config.LABEL_COL].value_counts())
    print("\nDev class balance:")
    print(dev[config.LABEL_COL].value_counts())


if __name__ == "__main__":
    main()