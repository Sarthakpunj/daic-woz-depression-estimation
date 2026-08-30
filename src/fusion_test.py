"""
heldout_fusion.py — held-out fusion evaluation on the 47, reusing the SAME
feature-building helpers as heldout_test.py so the 47 test participants get
text + audio + baseline features built exactly the way the working held-out
scripts build them.

PRE-SPECIFIED: fusion representative = STACKING (nested meta over text+audio+av),
the primary design and strongest CV fusion strategy. Text anchor recomputed on
the same participants. Trains on the common train+dev set, evaluates once on the 47.

Run:
    python3 heldout_fusion.py
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

# reuse the exact helpers the working held-out script uses
from daic_woz_pipeline.src.build_text_features import read_participant_text, chunk_pool_embed
import daic_woz_pipeline.src.features as featmod
from daic_woz_pipeline.src.heldout_test import test_ids_scores, ensure_files, load_train_labels
try:
    from daic_woz_pipeline.src.build_text_features import MPNET
except Exception:
    MPNET = "sentence-transformers/all-mpnet-base-v2"

PHQ8_CUTOFF = 10
N_INNER = 3
SEED = config.RANDOM_STATE
N_BOOT = 10000
RNG = np.random.RandomState(SEED)

AUDIO_PARQUET = "audio_w2v2_participant_layerLAST.parquet"


def fit_predict(kind, Xtr, ytr, Xte):
    if kind == "text":
        est, grid = Ridge(random_state=SEED), {"clf__alpha": [1.0, 10.0, 100.0]}
    else:
        est, grid = SVR(), {"clf__C": [1.0, 10.0]}
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), ("clf", est)])
    gs = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error",
                      cv=KFold(N_INNER, shuffle=True, random_state=SEED), n_jobs=-1)
    gs.fit(Xtr, ytr)
    return gs.best_estimator_.predict(Xte)


def bootstrap_ci(yte, pred):
    n = len(yte); maes, r2s, f1s = [], [], []
    for _ in range(N_BOOT):
        idx = RNG.randint(0, n, n)
        yt, yp = yte[idx], pred[idx]
        maes.append(mean_absolute_error(yt, yp))
        if np.var(yt) > 0: r2s.append(r2_score(yt, yp))
        f1s.append(f1_score(yt >= PHQ8_CUTOFF, yp >= PHQ8_CUTOFF, zero_division=0))
    def ci(a): a = np.array(a); return np.percentile(a, 2.5), np.percentile(a, 97.5)
    return ci(maes), ci(r2s), ci(f1s)


def show(name, yte, pred):
    mae, r2 = mean_absolute_error(yte, pred), r2_score(yte, pred)
    f1 = f1_score(yte >= PHQ8_CUTOFF, pred >= PHQ8_CUTOFF, zero_division=0)
    (ml, mh), (rl, rh), (fl, fh) = bootstrap_ci(yte, pred)
    print(f"  {name}:")
    print(f"    MAE={mae:.3f} CI[{ml:.2f},{mh:.2f}]  R2={r2:+.3f} CI[{rl:+.2f},{rh:+.2f}]  "
          f"F1={f1:.3f} CI[{fl:.2f},{fh:.2f}]")


# baseline AV columns come from the training parquet (fixed order)
def _av_columns():
    frames = [pd.read_parquet(config.OUT_DIR / fn)
              for fn in ["train_features.parquet", "dev_features.parquet"]
              if (config.OUT_DIR / fn).exists()]
    tr = pd.concat(frames, ignore_index=True)
    gcol = getattr(config, "GENDER_COL", "Gender")
    lab = {config.ID_COL, gcol, config.SCORE_COL, "PHQ8_Binary"}
    return [c for c in tr.columns if c not in lab and pd.api.types.is_numeric_dtype(tr[c])]


AV_COLS = _av_columns()


def build_features(ids, model):
    """Build text embeddings + baseline functionals + audio for a list of ids.
    Baseline AV features are aligned to the training parquet's column order."""
    audio = pd.read_parquet(config.OUT_DIR / AUDIO_PARQUET).set_index(config.ID_COL)
    acols = [c for c in audio.columns if c.startswith("emb_")]

    text_rows, av_dicts, aud_rows, kept = [], [], [], []
    for pid in ids:
        if pid not in audio.index:
            continue
        ensure_files(pid)
        txt = read_participant_text(pid)
        if txt is None:
            continue
        try:
            avfeat = featmod.extract_participant(config.DATA_ROOT, pid)  # dict
        except Exception:
            continue
        if avfeat is None:
            continue
        temb = chunk_pool_embed(txt, model)
        text_rows.append(np.asarray(temb, dtype=float).ravel())
        av_dicts.append(avfeat)
        aud_rows.append(audio.loc[pid, acols].values.astype(float))
        kept.append(pid)

    # align AV features into the fixed training column order
    av_df = pd.DataFrame(av_dicts)
    for c in AV_COLS:
        if c not in av_df.columns:
            av_df[c] = np.nan
    Xav = av_df[AV_COLS].values.astype(float)
    return (kept, np.vstack(text_rows), np.vstack(aud_rows), Xav)


def main():
    print("Building held-out (47) features for all three modalities...")
    print("(reuses the same helpers as heldout_test.py)\n")

    te_ids_list, test_scores = test_ids_scores()  # (list, dict pid->phq)
    train_df = load_train_labels()                # DataFrame
    train_labels = dict(zip(train_df[config.ID_COL].astype(int),
                            train_df[config.SCORE_COL].astype(float)))

    te_ids_all = sorted(te_ids_list)
    tr_ids_all = sorted(train_labels)

    from sentence_transformers import SentenceTransformer
    print("loading mpnet...")
    model = SentenceTransformer(MPNET)
    kept_te, Xte_text, Xte_aud, Xte_av = build_features(te_ids_all, model)
    kept_tr, Xtr_text, Xtr_aud, Xtr_av = build_features(tr_ids_all, model)

    yte = np.array([test_scores[p] for p in kept_te], dtype=float)
    ytr = np.array([train_labels[p] for p in kept_tr], dtype=float)
    print(f"train+dev built: {len(kept_tr)}   held-out built: {len(kept_te)}\n")

    Xtr = {"text": Xtr_text, "audio": Xtr_aud, "av": Xtr_av}
    Xte = {"text": Xte_text, "audio": Xte_aud, "av": Xte_av}

    mp = np.full(len(yte), ytr.mean())
    print(f"  mean predictor: MAE={mean_absolute_error(yte,mp):.3f}  R2={r2_score(yte,mp):+.3f}\n")

    base_te = {k: fit_predict(k, Xtr[k], ytr, Xte[k]) for k in Xtr}
    show("TEXT ALONE (anchor)", yte, base_te["text"])

    meta_tr = {k: np.zeros(len(ytr)) for k in Xtr}
    for itr, ite in KFold(N_INNER, shuffle=True, random_state=SEED).split(ytr):
        for k in Xtr:
            meta_tr[k][ite] = fit_predict(k, Xtr[k][itr], ytr[itr], Xtr[k][ite])
    order = ["text", "audio", "av"]
    Zt = np.column_stack([meta_tr[k] for k in order])
    Ze = np.column_stack([base_te[k] for k in order])
    meta = LinearRegression().fit(Zt, ytr)
    stack_pred = meta.predict(Ze)
    show("STACKING (pre-specified fusion)", yte, stack_pred)

    print(f"\n  meta-model weights:")
    for k, w in zip(order, meta.coef_):
        print(f"    {k:5s}: {w:+.3f}")
    print(f"    intercept: {meta.intercept_:+.3f}")

    # ---- RECALIBRATION-ONLY CONTROL: meta sees ONLY the text prediction ----
    # If this reproduces stacking, the held-out "gain" is text recalibration,
    # demonstrated rather than argued.
    Zt_text = meta_tr["text"].reshape(-1, 1)
    Ze_text = base_te["text"].reshape(-1, 1)
    recal = LinearRegression().fit(Zt_text, ytr)
    recal_pred = recal.predict(Ze_text)
    print(f"\n  recalibration-only control (a*text + b): a={recal.coef_[0]:+.3f}, b={recal.intercept_:+.3f}")
    show("RECAL-TEXT (text prediction only, recalibrated)", yte, recal_pred)
    print(f"\n  agreement stacking vs recal-text: "
          f"mean|diff|={np.mean(np.abs(stack_pred-recal_pred)):.4f}, "
          f"corr={np.corrcoef(stack_pred, recal_pred)[0,1]:.4f}")

    # per-group fairness on held-out
    ts = pd.read_csv(list(config.DATA_ROOT.glob("*test*split*.csv"))[0])
    ts.columns = [c.strip() for c in ts.columns]
    if "Gender" in ts.columns:
        gmap = dict(zip(ts[config.ID_COL], ts["Gender"]))
        g = np.array([gmap.get(p, np.nan) for p in kept_te])
        print(f"\n  Per-group held-out (the real fairness test):")
        print(f"  {'group':6s} {'n':>3s} {'text MAE':>9s} {'stack MAE':>10s} {'text F1':>8s} {'stack F1':>9s}")
        print(f"  (also showing recal-text MAE, to test whether recalibration alone")
        print(f"   reproduces the fairness change)")
        for grp in sorted(set(g[~np.isnan(g)])):
            m = g == grp
            print(f"  {int(grp):<6d} {m.sum():>3d} "
                  f"text {mean_absolute_error(yte[m], base_te['text'][m]):.3f} | "
                  f"stack {mean_absolute_error(yte[m], stack_pred[m]):.3f} | "
                  f"recal {mean_absolute_error(yte[m], recal_pred[m]):.3f}  ||  "
                  f"F1 text {f1_score(yte[m]>=PHQ8_CUTOFF, base_te['text'][m]>=PHQ8_CUTOFF, zero_division=0):.3f} "
                  f"stack {f1_score(yte[m]>=PHQ8_CUTOFF, stack_pred[m]>=PHQ8_CUTOFF, zero_division=0):.3f}")


if __name__ == "__main__":
    main()