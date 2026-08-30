"""
fairness_confound_check.py — distinguishes model bias from label-distribution
effects in the held-out gender gap.

The Section 7.7 gap (group 0 MAE 3.37 vs group 1 MAE 5.53) could mean either:
  (a) genuine model bias — the model is worse for group 1, OR
  (b) group 1 simply has harder scores (higher mean / wider spread), so ANY
      model — including the mean predictor — does worse on it.

Diagnostic: compute, per group, the PHQ-8 mean and spread, the model's MAE, AND
the mean-predictor's MAE. If the mean predictor shows a similar gap, the
disparity is largely label-distribution-driven. If the mean predictor is even
across groups but the model is not, that is stronger evidence of model bias.

Run:
    python3 fairness_confound_check.py
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import config
from sklearn.metrics import mean_absolute_error, f1_score

PHQ8_CUTOFF = 10


def main():
    pred = pd.read_csv(config.OUT_DIR / "heldout_test_predictions.csv")
    pred.columns = [c.strip() for c in pred.columns]

    cands = list(config.DATA_ROOT.glob("*test*split*.csv"))
    ts = pd.read_csv(cands[0]); ts.columns = [c.strip() for c in ts.columns]
    gcol = "Gender" if "Gender" in ts.columns else getattr(config, "GENDER_COL", "Gender")
    idcol = "Participant_ID" if "Participant_ID" in ts.columns else config.ID_COL

    df = pred.merge(ts[[idcol, gcol]].rename(columns={idcol: pred.columns[0], gcol: "gender"}),
                    on=pred.columns[0], how="left")

    # the mean predictor the model was compared against uses the TRAINING mean.
    # Load train+dev to get the true training mean (~6.7).
    rows = []
    for fn in [config.TRAIN_SPLIT, config.DEV_SPLIT]:
        d = pd.read_csv(config.DATA_ROOT / fn); d.columns = [c.strip() for c in d.columns]
        rows.append(d)
    train_mean = pd.concat(rows, ignore_index=True)[config.SCORE_COL].astype(float).mean()
    print(f"Training-set mean PHQ-8 (mean predictor value): {train_mean:.2f}\n")

    print(f"{'group':6s} {'n':>3s} {'PHQ mean':>9s} {'PHQ std':>8s} "
          f"{'model MAE':>10s} {'meanpred MAE':>13s} {'model F1':>9s}")
    stats = {}
    for g, sub in df.groupby("gender"):
        yt = sub["true"].values; yp = sub["pred"].values
        model_mae = mean_absolute_error(yt, yp)
        mp_mae = mean_absolute_error(yt, np.full(len(yt), train_mean))
        f1 = f1_score(yt >= PHQ8_CUTOFF, yp >= PHQ8_CUTOFF, zero_division=0)
        stats[int(g)] = (model_mae, mp_mae)
        print(f"{int(g):<6d} {len(sub):>3d} {yt.mean():>9.2f} {yt.std():>8.2f} "
              f"{model_mae:>10.3f} {mp_mae:>13.3f} {f1:>9.3f}")

    if len(stats) == 2:
        g0, g1 = sorted(stats)
        model_gap = abs(stats[g0][0] - stats[g1][0])
        mp_gap = abs(stats[g0][1] - stats[g1][1])
        print(f"\n  model MAE gap        = {model_gap:.3f}")
        print(f"  mean-predictor gap   = {mp_gap:.3f}")
        print("\nInterpretation:")
        if mp_gap >= 0.6 * model_gap:
            print("  The mean predictor shows a comparable gap -> the disparity is")
            print("  LARGELY LABEL-DISTRIBUTION-DRIVEN (group 1 has harder scores),")
            print("  not primarily model bias. Report this nuance.")
        elif mp_gap <= 0.3 * model_gap:
            print("  The mean predictor is roughly even but the MODEL is not ->")
            print("  stronger evidence of GENUINE MODEL BIAS. Flag clearly.")
        else:
            print("  Mixed: part label-distribution, part possible model effect.")
            print("  Report both contributions honestly.")


if __name__ == "__main__":
    main()