"""
repeated_cv_significance.py — Item 2: proper significance via repeated CV +
the Nadeau-Bengio (2003) corrected resampled t-test.

Why: a single 5-fold Wilcoxon has a floor of p=0.0625 (it cannot reach p<0.05
with 5 folds), which is the weakest point in the text validation. Repeated CV
(e.g. 5 repeats x 5 folds = 25 paired differences) plus the variance-corrected
t-test is the standard, correct tool for comparing two models across CV folds.
It accounts for the dependence between overlapping training sets, which an
ordinary t-test ignores (and which would otherwise overstate significance).

This turns a defensive footnote ("p=0.0625 is just the floor") into a proper,
reportable significance result.

Run:
    python3 repeated_cv_significance.py text_features_mpnet.parquet
"""

import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd
from scipy import stats

import daic_woz_pipeline.src.config as config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error

N_REPEATS = 5
N_FOLDS = 5


def corrected_ttest(diffs, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test.
       diffs: per-fold (mean_predictor_MAE - model_MAE). Positive => model better.
       Correction inflates the variance by (1/k + n_test/n_train) to account for
       the overlap between training sets across folds."""
    k = len(diffs)
    mean = np.mean(diffs)
    var = np.var(diffs, ddof=1)
    if var == 0:
        return np.inf, 0.0
    t = mean / np.sqrt(var * (1.0/k + n_test/n_train))
    p = 2 * stats.t.sf(abs(t), k - 1)
    return t, p


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
    if not cols:
        gcol = getattr(config,"GENDER_COL","Gender")
        excl = {config.ID_COL,gcol,config.SCORE_COL,"PHQ8_Binary"}
        cols = [c for c in df.columns if c not in excl and pd.api.types.is_numeric_dtype(df[c])]
    return df[cols].values.astype(float), df[config.SCORE_COL].values.astype(float)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "text_features_mpnet.parquet"
    X, y = load(path)
    print(f"Feature file: {path}   n={len(y)}, features={X.shape[1]}")
    print(f"Repeated CV: {N_REPEATS} repeats x {N_FOLDS} folds "
          f"= {N_REPEATS*N_FOLDS} paired differences\n")

    diffs = []
    model_maes, mean_maes = [], []
    for rep in range(N_REPEATS):
        kf = KFold(N_FOLDS, shuffle=True, random_state=rep)
        for tr, te in kf.split(X):
            pipe = Pipeline([("imp",SimpleImputer(strategy="median")),
                             ("sc",StandardScaler()),
                             ("clf",Ridge(random_state=config.RANDOM_STATE))])
            gs = GridSearchCV(pipe, {"clf__alpha":[1.0,10.0,100.0]},
                              scoring="neg_mean_absolute_error",
                              cv=KFold(3,shuffle=True,random_state=config.RANDOM_STATE),
                              n_jobs=-1)
            gs.fit(X[tr], y[tr])
            model_mae = mean_absolute_error(y[te], gs.best_estimator_.predict(X[te]))
            mean_mae = mean_absolute_error(y[te], np.full(len(te), y[tr].mean()))
            diffs.append(mean_mae - model_mae)   # positive => model better
            model_maes.append(model_mae); mean_maes.append(mean_mae)

    diffs = np.array(diffs)
    n = len(y); n_test = n // N_FOLDS; n_train = n - n_test
    t, p = corrected_ttest(diffs, n_train, n_test)

    print(f"Model  MAE (mean over {len(diffs)} folds): {np.mean(model_maes):.3f}")
    print(f"Mean-predictor MAE:                        {np.mean(mean_maes):.3f}")
    print(f"Mean improvement (mean-pred - model):      {np.mean(diffs):+.3f}")
    print(f"Folds where model beats mean predictor:    {(diffs>0).sum()}/{len(diffs)}")
    print(f"\nNadeau-Bengio corrected resampled t-test:")
    print(f"  t = {t:.3f},  p = {p:.4f}")
    if p < 0.05:
        print(f"  -> SIGNIFICANT at alpha=0.05: the text model beats the mean")
        print(f"     predictor by a statistically significant margin.")
    else:
        print(f"  -> not significant at alpha=0.05 (report honestly).")
    print("\n(Note: the correction inflates variance to account for overlapping")
    print(" training sets across folds, so this p-value is conservative and")
    print(" more defensible than an uncorrected repeated t-test.)")


if __name__ == "__main__":
    main()