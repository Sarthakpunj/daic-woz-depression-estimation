"""
run_fusion.py — Phase 4: does ANY modality add anything over text alone?
Late fusion / nested stacking, text as the anchor.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import daic_woz_pipeline.src.config as config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, f1_score

PHQ8_CUTOFF = 10; SEVERE_LO = 15
N_OUTER, N_INNER = 5, 3
SEED = config.RANDOM_STATE
TEXT_PARQUET = "text_features_mpnet.parquet"
AUDIO_PARQUET = "audio_w2v2_participant_layerLAST.parquet"
BASELINE_PARQUETS = ["train_features.parquet", "dev_features.parquet"]


def load_labels():
    rows = []
    for fn in [config.TRAIN_SPLIT, config.DEV_SPLIT]:
        d = pd.read_csv(config.DATA_ROOT / fn); d.columns=[c.strip() for c in d.columns]
        rows.append(d)
    return pd.concat(rows, ignore_index=True)[[config.ID_COL, config.SCORE_COL]]


def load_modalities():
    mods = {}
    t = pd.read_parquet(config.OUT_DIR / TEXT_PARQUET)
    tcols = [c for c in t.columns if c.startswith("emb_") or c.startswith("ling_")]
    mods["text"] = t[[config.ID_COL] + tcols]
    a = pd.read_parquet(config.OUT_DIR / AUDIO_PARQUET)
    acols = [c for c in a.columns if c.startswith("emb_")]
    mods["audio"] = a[[config.ID_COL] + acols].rename(columns={c: f"aud_{c}" for c in acols})
    frames = [pd.read_parquet(config.OUT_DIR / fn) for fn in BASELINE_PARQUETS
              if (config.OUT_DIR / fn).exists()]
    if frames:
        b = pd.concat(frames, ignore_index=True)
        gcol = getattr(config, "GENDER_COL", "Gender")
        drop = {gcol, config.SCORE_COL, "PHQ8_Binary"}
        bcols = [c for c in b.columns if c not in drop and c != config.ID_COL
                 and pd.api.types.is_numeric_dtype(b[c])]
        mods["av"] = b[[config.ID_COL] + bcols].rename(columns={c: f"av_{c}" for c in bcols})
    return mods


def make_model(kind):
    if kind == "text":
        return Ridge(random_state=SEED), {"clf__alpha": [1.0, 10.0, 100.0]}
    return SVR(), {"clf__C": [1.0, 10.0]}


def fit_predict(kind, Xtr, ytr, Xte):
    est, grid = make_model(kind)
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), ("clf", est)])
    gs = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error",
                      cv=KFold(N_INNER, shuffle=True, random_state=SEED), n_jobs=-1)
    gs.fit(Xtr, ytr)
    return gs.best_estimator_.predict(Xte)


def metrics(y, p, label):
    sev = y >= SEVERE_LO
    return {"config": label, "mae": round(mean_absolute_error(y, p), 3),
            "r2": round(r2_score(y, p), 3),
            "f1": round(f1_score(y >= PHQ8_CUTOFF, p >= PHQ8_CUTOFF, zero_division=0), 3),
            "mae_severe": round(mean_absolute_error(y[sev], p[sev]), 2) if sev.sum() else float("nan")}


def main():
    labels = load_labels(); mods = load_modalities()
    print("Modalities loaded:", list(mods))
    ids = set(labels[config.ID_COL])
    for df in mods.values():
        ids &= set(df[config.ID_COL])
    ids = sorted(ids)
    print(f"Participants common to all modalities: {len(ids)}\n")
    lab = labels[labels[config.ID_COL].isin(ids)].sort_values(config.ID_COL)
    y = lab[config.SCORE_COL].values.astype(float)
    X = {}
    for k, df in mods.items():
        d = df[df[config.ID_COL].isin(ids)].sort_values(config.ID_COL)
        X[k] = d.drop(columns=[config.ID_COL]).values.astype(float)
        print(f"  {k:5s}: {X[k].shape}")
    print()
    outer = KFold(N_OUTER, shuffle=True, random_state=SEED)
    oof = {k: np.zeros(len(y)) for k in ["text","audio","av","stack","avg","early"]}
    for tr, te in outer.split(y):
        ytr = y[tr]
        base_te = {}
        for k in X:
            base_te[k] = fit_predict(k, X[k][tr], ytr, X[k][te]); oof[k][te] = base_te[k]
        meta_tr = {k: np.zeros(len(tr)) for k in X}
        for itr, ite in KFold(N_INNER, shuffle=True, random_state=SEED).split(tr):
            for k in X:
                meta_tr[k][ite] = fit_predict(k, X[k][tr][itr], ytr[itr], X[k][tr][ite])
        Zt = np.column_stack([meta_tr[k] for k in X])
        Ze = np.column_stack([base_te[k] for k in X])
        oof["stack"][te] = LinearRegression().fit(Zt, ytr).predict(Ze)
        oof["avg"][te] = Ze.mean(axis=1)
        Xe_tr = np.hstack([X["text"][tr], X["audio"][tr]])
        Xe_te = np.hstack([X["text"][te], X["audio"][te]])
        oof["early"][te] = fit_predict("text", Xe_tr, ytr, Xe_te)
    lm = {"text":"TEXT ALONE (anchor)","audio":"audio alone (null)","av":"baseline A+V alone (null)",
          "stack":"STACKING (nested meta)","avg":"simple average","early":"early fusion (text+audio)"}
    results = [metrics(y, oof[k], lm[k]) for k in ["text","audio","av","stack","avg","early"]]
    df = pd.DataFrame(results)
    anchor = df[df.config == lm["text"]].iloc[0]
    df["vs_text_mae"] = (df["mae"] - anchor["mae"]).round(3)
    df["beats_text"] = np.where((df["mae"] < anchor["mae"] - 0.1) & (df["r2"] > anchor["r2"] + 0.02), "yes", "no")
    df.loc[df.config == lm["text"], ["vs_text_mae","beats_text"]] = ["-","-"]
    print("=== FUSION: does anything add over TEXT ALONE? ===")
    print(f"(n={len(y)}; benchmark is text, not the mean predictor)\n")
    print(df.to_string(index=False))
    df.to_csv(config.OUT_DIR / "fusion_results.csv", index=False)
    np.save(config.OUT_DIR / "fusion_oof_predictions.npy",
            np.column_stack([oof[k] for k in ["text","audio","av","stack","avg","early"]]))
    print(f"\nSaved fusion_results.csv and fusion_oof_predictions.npy")


if __name__ == "__main__":
    main()