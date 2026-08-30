"""
severe_weighting_text.py — Item 4: score-aware sample weighting on the TEXT
features, to test whether the severe-case weakness (severe-band MAE ~7-10) can
be reduced now that the features actually carry signal.

Honest framing: this failed at baseline because there was no signal to reweight.
Text has signal, so the logic differs. BUT weighting typically TRADES OFF overall
accuracy for severe-case accuracy, so BOTH are reported: overall MAE/R2 AND
severe-band MAE, plain vs weighted. No cherry-picking.

Weighting scheme: each training sample is weighted by (1 + score/scale), so
higher-PHQ-8 participants count more. Weights computed inside each training fold
only (no leakage). Ridge and SVR accept sample_weight in .fit().

Run:
    python3 severe_weighting_text.py text_features_mpnet.parquet
"""

import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd

import daic_woz_pipeline.src.config as config
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

SEVERE_LO = 15   # severe band = PHQ-8 15-24
WEIGHT_SCALE = 10.0


def load(path):
    p = path if "/" in path else str(config.OUT_DIR / path)
    feats = pd.read_parquet(p)
    if config.SCORE_COL in feats.columns:
        df = feats
    else:
        rows = []
        for fn in [config.TRAIN_SPLIT, config.DEV_SPLIT]:
            d = pd.read_csv(config.DATA_ROOT / fn); d.columns=[c.strip() for c in d.columns]
            rows.append(d)
        lab = pd.concat(rows, ignore_index=True)[[config.ID_COL, config.SCORE_COL]]
        df = feats.merge(lab, on=config.ID_COL, how="inner")
    cols = [c for c in feats.columns if c.startswith("emb_") or c.startswith("ling_")]
    return df[cols].values.astype(float), df[config.SCORE_COL].values.astype(float)


def severe_mae(y_true, y_pred):
    mask = y_true >= SEVERE_LO
    if mask.sum() == 0:
        return float("nan")
    return mean_absolute_error(y_true[mask], y_pred[mask])


def run(X, y, model_name, weighted):
    outer = KFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    at, ap = [], []
    for tr, te in outer.split(X):
        imp = SimpleImputer(strategy="median").fit(X[tr])
        sc = StandardScaler().fit(imp.transform(X[tr]))
        Xtr = sc.transform(imp.transform(X[tr]))
        Xte = sc.transform(imp.transform(X[te]))
        ytr = y[tr]
        # weights computed inside the training fold only
        w = (1.0 + ytr / WEIGHT_SCALE) if weighted else None
        if model_name == "ridge":
            best, best_mae = None, 1e9
            for alpha in [1.0, 10.0, 100.0]:
                m = Ridge(alpha=alpha, random_state=config.RANDOM_STATE)
                m.fit(Xtr, ytr, sample_weight=w)
                # quick inner check on train fold (kept simple)
                mae = mean_absolute_error(ytr, m.predict(Xtr))
                if mae < best_mae:
                    best, best_mae = m, mae
            pred = best.predict(Xte)
        else:  # svr
            best, best_mae = None, 1e9
            for Cc in [1.0, 10.0]:
                m = SVR(C=Cc)
                m.fit(Xtr, ytr, sample_weight=w)
                mae = mean_absolute_error(ytr, m.predict(Xtr))
                if mae < best_mae:
                    best, best_mae = m, mae
            pred = best.predict(Xte)
        at.extend(y[te]); ap.extend(pred)
    at, ap = np.array(at), np.array(ap)
    return mean_absolute_error(at, ap), r2_score(at, ap), severe_mae(at, ap)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "text_features_mpnet.parquet"
    X, y = load(path)
    n_severe = int((y >= SEVERE_LO).sum())
    print(f"Feature file: {path}   n={len(y)}, severe (>= {SEVERE_LO}): {n_severe}")
    print(f"(Severe band is small, so severe MAE is noisy — read as indicative.)\n")

    print(f"{'model':12s} {'weight':8s} {'MAE':>7s} {'R2':>7s} {'MAE_severe':>11s}")
    for model_name in ["ridge", "svr"]:
        for weighted in [False, True]:
            mae, r2, smae = run(X, y, model_name, weighted)
            tag = "weighted" if weighted else "plain"
            print(f"{model_name:12s} {tag:8s} {mae:7.3f} {r2:+7.3f} {smae:11.3f}")
    print("\nInterpretation: compare plain vs weighted for each model.")
    print("  - If severe MAE drops while overall MAE stays similar -> weighting helps.")
    print("  - If severe MAE drops but overall MAE rises -> a trade-off (report both).")
    print("  - If neither moves much -> weighting adds little even with real signal.")
    print("Report BOTH the severe-band change AND the overall-MAE change honestly.")


if __name__ == "__main__":
    main()