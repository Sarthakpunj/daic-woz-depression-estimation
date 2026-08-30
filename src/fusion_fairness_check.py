"""
fusion_fairness_check.py — does adding the group-even (but null) audio modality
change the text model's gender disparity?

Text alone showed a held-out MAE gap of 2.16 between gender groups; audio was even
across groups. This asks whether stacking text with the group-even audio dilutes
text's disparity at all — a genuinely novel observation if it does, and an honest
"no" if it doesn't.

Uses the out-of-fold predictions saved by run_fusion.py (fusion_oof_predictions.npy),
so no retraining. Columns are: text, audio, av, stack, avg, early (in that order).

Run:
    python3 fusion_fairness_check.py
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import daic_woz_pipeline.src.config as config
from sklearn.metrics import mean_absolute_error, f1_score

PHQ8_CUTOFF = 10
COLS = ["text", "audio", "av", "stack", "avg", "early"]


def load_labels_and_gender():
    rows = []
    for fn in [config.TRAIN_SPLIT, config.DEV_SPLIT]:
        d = pd.read_csv(config.DATA_ROOT / fn); d.columns = [c.strip() for c in d.columns]
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    gcol = "Gender" if "Gender" in df.columns else getattr(config, "GENDER_COL", "Gender")
    return df[[config.ID_COL, config.SCORE_COL, gcol]], gcol


def rebuild_id_order():
    """Reproduce the exact participant order used by run_fusion.py (intersection,
    sorted by ID) so predictions line up with labels/gender."""
    import pandas as pd
    def ids_of(fn):
        return set(pd.read_parquet(config.OUT_DIR / fn)[config.ID_COL])
    text_ids = ids_of("text_features_mpnet.parquet")
    audio_ids = ids_of("audio_w2v2_participant_layerLAST.parquet")
    frames = [pd.read_parquet(config.OUT_DIR / f)
              for f in ["train_features.parquet", "dev_features.parquet"]
              if (config.OUT_DIR / f).exists()]
    av_ids = set(pd.concat(frames, ignore_index=True)[config.ID_COL])
    lab = []
    for fn in [config.TRAIN_SPLIT, config.DEV_SPLIT]:
        d = pd.read_csv(config.DATA_ROOT / fn); d.columns = [c.strip() for c in d.columns]
        lab.append(d)
    lab_ids = set(pd.concat(lab, ignore_index=True)[config.ID_COL])
    ids = sorted(text_ids & audio_ids & av_ids & lab_ids)
    return ids


def main():
    preds = np.load(config.OUT_DIR / "fusion_oof_predictions.npy")
    ids = rebuild_id_order()
    if len(ids) != preds.shape[0]:
        print(f"WARNING: {len(ids)} ids vs {preds.shape[0]} predictions; order may differ.")

    labgen, gcol = load_labels_and_gender()
    labgen = labgen.set_index(config.ID_COL)
    y = np.array([labgen.loc[i, config.SCORE_COL] for i in ids], dtype=float)
    g = np.array([labgen.loc[i, gcol] for i in ids])

    print(f"n={len(ids)}  (cross-validation OOF predictions; gender groups as coded 0/1)\n")
    print(f"{'config':28s} {'grp0 MAE':>9s} {'grp1 MAE':>9s} {'gap':>6s} "
          f"{'grp0 F1':>8s} {'grp1 F1':>8s}")

    groups = sorted(np.unique(g[~pd.isna(g)]))
    for ci, name in enumerate(COLS):
        p = preds[:, ci]
        row = {}
        for grp in groups:
            m = (g == grp)
            row[grp] = (mean_absolute_error(y[m], p[m]),
                        f1_score(y[m] >= PHQ8_CUTOFF, p[m] >= PHQ8_CUTOFF, zero_division=0))
        if len(groups) == 2:
            a, b = groups
            gap = abs(row[a][0] - row[b][0])
            print(f"{name:28s} {row[a][0]:9.3f} {row[b][0]:9.3f} {gap:6.3f} "
                  f"{row[a][1]:8.3f} {row[b][1]:8.3f}")

    print("\nInterpretation:")
    print("  Compare the 'gap' for TEXT alone versus STACKING/average/early fusion.")
    print("  - If the gap SHRINKS when audio is added, that is a novel (if modest)")
    print("    observation: a group-even null modality partially offsets text's bias.")
    print("  - If the gap is unchanged, adding audio does not help fairness either —")
    print("    report honestly. (These are CV OOF estimates; subgroup n is small,")
    print("    so read as indicative, consistent with the held-out fairness caveat.)")


if __name__ == "__main__":
    main()