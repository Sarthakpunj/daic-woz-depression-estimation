"""
fairness_and_bootstrap.py — Items 3 & 4 for the text model.

(3) FAIRNESS: gender-parity check for mpnet+Ridge, mirroring the baseline
    protocol (reported as group 0 vs group 1, since the official docs do not
    define the gender coding). Now meaningful because text F1 is ~0.59 (at
    baseline F1 was ~0, so parity was uninformative).

(4) BOOTSTRAP CIs: 95% confidence intervals for the held-out MAE, R2, and F1,
    by resampling the 47 test participants with replacement. Converts the
    "n=47 is small" limitation into quantified uncertainty.

Run:
    python3 fairness_and_bootstrap.py

Requires: the held-out predictions (true, pred per test participant) and the
gender column. This script re-runs the held-out prediction itself so it is
self-contained; if you already saved heldout_test_predictions.csv it will use
that plus the test split for gender.
"""

import warnings; warnings.filterwarnings("ignore")
import os; os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import pandas as pd
import config
from sklearn.metrics import mean_absolute_error, r2_score, f1_score

PHQ8_CUTOFF = 10
GENDER_COL = getattr(config, "GENDER_COL", "Gender")
N_BOOT = 10000
RNG = np.random.RandomState(config.RANDOM_STATE)


def load_heldout_predictions():
    """Load saved held-out predictions if present; else regenerate them."""
    p = config.OUT_DIR / "heldout_test_predictions.csv"
    if p.exists():
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        return df
    print(f"{p.name} not found — regenerating held-out predictions...")
    return regenerate_predictions()


def regenerate_predictions():
    """Re-run the held-out prediction (embed 47 test, train on 142, predict)."""
    from sentence_transformers import SentenceTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, KFold

    CHUNK = 200
    DROP = ("scrubbed_entry", "<laughter>", "xxx", "<sync>", "<synch>")

    def read_txt(pid):
        f = config.DATA_ROOT / f"{pid}_P" / f"{pid}_TRANSCRIPT.csv"
        if not f.exists():
            return None
        try:
            d = pd.read_csv(f, sep="\t")
        except Exception:
            d = pd.read_csv(f)
        d.columns = [c.strip().lower() for c in d.columns]
        if "speaker" not in d.columns or "value" not in d.columns:
            return None
        d = d[d["speaker"].astype(str).str.strip().str.lower() == "participant"]
        parts = [s.strip() for s in d["value"].astype(str)
                 if s.strip() and not any(t in s.lower() for t in DROP)]
        t = " ".join(parts).strip()
        return t or None

    def embed(text, model):
        w = text.split()
        chunks = [" ".join(w[i:i+CHUNK]) for i in range(0, len(w), CHUNK)]
        return np.asarray(model.encode(chunks, show_progress_bar=False)).mean(axis=0)

    # test split (47) with scores
    cands = list(config.DATA_ROOT.glob("*test*split*.csv"))
    ts = pd.read_csv(cands[0]); ts.columns = [c.strip() for c in ts.columns]
    scol = config.SCORE_COL if config.SCORE_COL in ts.columns else next(
        c for c in ts.columns if "phq" in c.lower() and "score" in c.lower())
    idcol = config.ID_COL if config.ID_COL in ts.columns else "Participant_ID"
    test_ids = [int(x) for x in ts[idcol]]
    test_scores = dict(zip(test_ids, ts[scol].astype(float)))

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # test features
    trows, tids, tscore = [], [], []
    for pid in test_ids:
        t = read_txt(pid)
        if t is None:
            continue
        trows.append(embed(t, model)); tids.append(pid); tscore.append(test_scores[pid])
    Xte = np.vstack(trows); yte = np.array(tscore)

    # train features (reuse existing parquet)
    tp = config.OUT_DIR / "text_features_mpnet.parquet"
    tf = pd.read_parquet(tp)
    rows = []
    for fn in [config.TRAIN_SPLIT, config.DEV_SPLIT]:
        d = pd.read_csv(config.DATA_ROOT / fn); d.columns = [c.strip() for c in d.columns]
        rows.append(d)
    lab = pd.concat(rows, ignore_index=True)[[config.ID_COL, config.SCORE_COL]]
    tr = tf.merge(lab, on=config.ID_COL, how="inner")
    emb_cols = [c for c in tf.columns if c.startswith("emb_")]
    Xtr = tr[emb_cols].values.astype(float); ytr = tr[config.SCORE_COL].values.astype(float)
    # align test to emb_ cols only (drop ling_ for simplicity/consistency)
    Xte = Xte[:, :len(emb_cols)] if Xte.shape[1] >= len(emb_cols) else Xte

    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), ("clf", Ridge(random_state=config.RANDOM_STATE))])
    gs = GridSearchCV(pipe, {"clf__alpha": [1.0, 10.0, 100.0]},
                      scoring="neg_mean_absolute_error",
                      cv=KFold(5, shuffle=True, random_state=config.RANDOM_STATE), n_jobs=-1)
    gs.fit(Xtr, ytr)
    pred = gs.best_estimator_.predict(Xte)
    out = pd.DataFrame({"Participant_ID": tids, "true": yte, "pred": pred})
    out.to_csv(config.OUT_DIR / "heldout_test_predictions.csv", index=False)
    print(f"  regenerated and saved ({len(out)} predictions)")
    return out


def attach_gender(df):
    """Merge the test-split gender column onto the predictions."""
    # find the test split
    cands = list(config.DATA_ROOT.glob("*test*split*.csv"))
    if not cands:
        print("WARNING: test split not found; cannot do fairness check.")
        return df
    ts = pd.read_csv(cands[0]); ts.columns = [c.strip() for c in ts.columns]
    gcol = GENDER_COL if GENDER_COL in ts.columns else (
        "Gender" if "Gender" in ts.columns else None)
    if gcol is None:
        print("WARNING: no gender column in test split; skipping fairness.")
        return df
    idcol = config.ID_COL if config.ID_COL in ts.columns else "Participant_ID"
    return df.merge(ts[[idcol, gcol]].rename(columns={idcol: df.columns[0], gcol: "gender"}),
                    left_on=df.columns[0], right_on=df.columns[0], how="left")


def fairness(df):
    print("\n=== FAIRNESS: gender parity (held-out, mpnet+Ridge) ===")
    if "gender" not in df.columns:
        print("  gender not available; skipped."); return
    for g, sub in df.groupby("gender"):
        yt = sub["true"].values; yp = sub["pred"].values
        mae = mean_absolute_error(yt, yp)
        f1 = f1_score(yt >= PHQ8_CUTOFF, yp >= PHQ8_CUTOFF, zero_division=0)
        print(f"  group {int(g)}: n={len(sub):2d}  MAE={mae:.3f}  F1={f1:.3f}")
    # gap
    groups = sorted(df["gender"].dropna().unique())
    if len(groups) == 2:
        a = df[df.gender == groups[0]]; b = df[df.gender == groups[1]]
        f1a = f1_score(a["true"]>=PHQ8_CUTOFF, a["pred"]>=PHQ8_CUTOFF, zero_division=0)
        f1b = f1_score(b["true"]>=PHQ8_CUTOFF, b["pred"]>=PHQ8_CUTOFF, zero_division=0)
        maea = mean_absolute_error(a["true"], a["pred"])
        maeb = mean_absolute_error(b["true"], b["pred"])
        print(f"  F1 gap  = {abs(f1a-f1b):.3f}   MAE gap = {abs(maea-maeb):.3f}")
        print("  (report as group 0 vs group 1; official docs do not define coding)")


def bootstrap(df):
    print(f"\n=== BOOTSTRAP 95% CIs (held-out, {N_BOOT} resamples of n={len(df)}) ===")
    yt = df["true"].values; yp = df["pred"].values
    n = len(yt)
    maes, r2s, f1s = [], [], []
    for _ in range(N_BOOT):
        idx = RNG.randint(0, n, n)
        yt_b, yp_b = yt[idx], yp[idx]
        maes.append(mean_absolute_error(yt_b, yp_b))
        # r2 undefined if all-equal; guard
        if np.var(yt_b) > 0:
            r2s.append(r2_score(yt_b, yp_b))
        f1s.append(f1_score(yt_b >= PHQ8_CUTOFF, yp_b >= PHQ8_CUTOFF, zero_division=0))
    def ci(a):
        a = np.array(a); return np.percentile(a, 2.5), np.percentile(a, 97.5)
    mlo, mhi = ci(maes); rlo, rhi = ci(r2s); flo, fhi = ci(f1s)
    print(f"  MAE = {mean_absolute_error(yt,yp):.3f}  95% CI [{mlo:.3f}, {mhi:.3f}]")
    print(f"  R2  = {r2_score(yt,yp):+.3f}  95% CI [{rlo:+.3f}, {rhi:+.3f}]")
    print(f"  F1  = {f1_score(yt>=PHQ8_CUTOFF, yp>=PHQ8_CUTOFF, zero_division=0):.3f}"
          f"  95% CI [{flo:.3f}, {fhi:.3f}]")
    if rlo > 0:
        print("  -> R2 CI is entirely positive: the model beats the mean predictor")
        print("     on the held-out set with 95% confidence.")
    else:
        print("  -> R2 CI includes 0: held-out advantage not certain at 95% (honest).")


def main():
    df = load_heldout_predictions()
    print(f"Loaded {len(df)} held-out predictions.")
    df = attach_gender(df)
    fairness(df)
    bootstrap(df)


if __name__ == "__main__":
    main()