"""
fusion_significance.py — Nadeau-Bengio test of fusion variants AGAINST TEXT.
Benchmark is the text model, not the mean predictor. Nested stacking throughout.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
import daic_woz_pipeline.src.config as config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error

N_REPEATS, N_OUTER, N_INNER = 5, 5, 3
SEED = config.RANDOM_STATE
TEXT_PARQUET = "text_features_mpnet.parquet"
AUDIO_PARQUET = "audio_w2v2_participant_layerLAST.parquet"
BASELINE_PARQUETS = ["train_features.parquet", "dev_features.parquet"]


def corrected_ttest(diffs, n_train, n_test):
    k = len(diffs); mean = np.mean(diffs); var = np.var(diffs, ddof=1)
    if var == 0: return np.inf, 0.0
    t = mean / np.sqrt(var * (1.0/k + n_test/n_train))
    return t, 2 * stats.t.sf(abs(t), k - 1)


def load_all():
    rows = []
    for fn in [config.TRAIN_SPLIT, config.DEV_SPLIT]:
        d = pd.read_csv(config.DATA_ROOT / fn); d.columns=[c.strip() for c in d.columns]
        rows.append(d)
    labels = pd.concat(rows, ignore_index=True)[[config.ID_COL, config.SCORE_COL]]
    t = pd.read_parquet(config.OUT_DIR / TEXT_PARQUET)
    tc = [c for c in t.columns if c.startswith("emb_") or c.startswith("ling_")]
    text = t[[config.ID_COL] + tc]
    a = pd.read_parquet(config.OUT_DIR / AUDIO_PARQUET)
    ac = [c for c in a.columns if c.startswith("emb_")]
    audio = a[[config.ID_COL] + ac]
    frames = [pd.read_parquet(config.OUT_DIR / f) for f in BASELINE_PARQUETS
              if (config.OUT_DIR / f).exists()]
    b = pd.concat(frames, ignore_index=True)
    gcol = getattr(config, "GENDER_COL", "Gender")
    drop = {gcol, config.SCORE_COL, "PHQ8_Binary", config.ID_COL}
    bc = [c for c in b.columns if c not in drop and pd.api.types.is_numeric_dtype(b[c])]
    av = b[[config.ID_COL] + bc]
    ids = (set(labels[config.ID_COL]) & set(text[config.ID_COL])
           & set(audio[config.ID_COL]) & set(av[config.ID_COL]))
    ids = sorted(ids)
    def sel(df):
        d = df[df[config.ID_COL].isin(ids)].sort_values(config.ID_COL)
        return d.drop(columns=[config.ID_COL]).values.astype(float)
    lab = labels[labels[config.ID_COL].isin(ids)].sort_values(config.ID_COL)
    return {"text": sel(text), "audio": sel(audio), "av": sel(av)}, lab[config.SCORE_COL].values.astype(float)


def fit_predict(kind, Xtr, ytr, Xte, seed):
    if kind == "text":
        est, grid = Ridge(random_state=SEED), {"clf__alpha":[1.0,10.0,100.0]}
    else:
        est, grid = SVR(), {"clf__C":[1.0,10.0]}
    pipe = Pipeline([("imp",SimpleImputer(strategy="median")),
                     ("sc",StandardScaler()),("clf",est)])
    gs = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error",
                      cv=KFold(N_INNER, shuffle=True, random_state=seed), n_jobs=-1)
    gs.fit(Xtr, ytr)
    return gs.best_estimator_.predict(Xte)


def main():
    X, y = load_all(); n = len(y)
    print(f"n={n} participants common to all modalities")
    print(f"Repeated CV: {N_REPEATS}x{N_OUTER} folds. Benchmark: TEXT ALONE\n")
    d_stack, d_avg = [], []
    text_maes, stack_maes, avg_maes = [], [], []
    for rep in range(N_REPEATS):
        for tr, te in KFold(N_OUTER, shuffle=True, random_state=rep).split(y):
            ytr = y[tr]
            base_te = {k: fit_predict(k, X[k][tr], ytr, X[k][te], rep) for k in X}
            meta_tr = {k: np.zeros(len(tr)) for k in X}
            for itr, ite in KFold(N_INNER, shuffle=True, random_state=rep).split(tr):
                for k in X:
                    meta_tr[k][ite] = fit_predict(k, X[k][tr][itr], ytr[itr], X[k][tr][ite], rep)
            Zt = np.column_stack([meta_tr[k] for k in X])
            Ze = np.column_stack([base_te[k] for k in X])
            stack_pred = LinearRegression().fit(Zt, ytr).predict(Ze)
            avg_pred = Ze.mean(axis=1)
            t_mae = mean_absolute_error(y[te], base_te["text"])
            s_mae = mean_absolute_error(y[te], stack_pred)
            a_mae = mean_absolute_error(y[te], avg_pred)
            text_maes.append(t_mae); stack_maes.append(s_mae); avg_maes.append(a_mae)
            d_stack.append(t_mae - s_mae); d_avg.append(t_mae - a_mae)
    n_test = n // N_OUTER; n_train = n - n_test
    print(f"text alone MAE: {np.mean(text_maes):.3f}")
    print(f"stacking   MAE: {np.mean(stack_maes):.3f}")
    print(f"simple avg MAE: {np.mean(avg_maes):.3f}\n")
    for name, d in [("STACKING vs text", d_stack), ("SIMPLE AVG vs text", d_avg)]:
        d = np.array(d); t, p = corrected_ttest(d, n_train, n_test)
        print(f"{name}:")
        print(f"  mean improvement over text: {np.mean(d):+.3f} MAE")
        print(f"  folds fusion beats text: {(d>0).sum()}/{len(d)}")
        print(f"  Nadeau-Bengio: t={t:.3f}, p={p:.4f}")
        if p < 0.05 and np.mean(d) > 0:
            print("  -> Fusion SIGNIFICANTLY improves on text alone.")
        elif p < 0.05 and np.mean(d) < 0:
            print("  -> Fusion significantly WORSE than text alone.")
        else:
            print("  -> No significant difference: fusion does NOT add over text alone.")
        print()


if __name__ == "__main__":
    main()