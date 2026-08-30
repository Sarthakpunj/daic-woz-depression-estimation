"""
audio_heldout_bootstrap.py — bootstrap 95% CIs for the audio held-out metrics.

Protocol symmetry: the text chapter attaches bootstrap intervals to its held-out
numbers; this does the same for audio. For a null result the intervals document
that even the optimistic edge of the interval is weak, which is a stronger
statement than the point estimate alone.

Uses the PRE-SPECIFIED configuration: Wav2Vec2, LAST layer, SVR.

Run:
    python3 audio_heldout_bootstrap.py audio_w2v2_participant_layerLAST.parquet
"""

import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import daic_woz_pipeline.src.config as config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, r2_score, f1_score

PHQ8_CUTOFF = 10
N_BOOT = 10000
RNG = np.random.RandomState(config.RANDOM_STATE)


def load_split(fn):
    d = pd.read_csv(config.DATA_ROOT / fn)
    d.columns = [c.strip() for c in d.columns]
    return d


def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else "audio_w2v2_participant_layerLAST.parquet"
    path = Path(fname) if "/" in fname else config.OUT_DIR / fname
    if not path.exists():
        print(f"ERROR: {path} not found."); return

    print(f"=== AUDIO HELD-OUT BOOTSTRAP: {path.name} (pre-specified: SVR) ===\n")
    audio = pd.read_parquet(path)
    emb = [c for c in audio.columns if c.startswith("emb_")]

    tr_lab = pd.concat([load_split(config.TRAIN_SPLIT),
                        load_split(config.DEV_SPLIT)], ignore_index=True)[[config.ID_COL, config.SCORE_COL]]
    tr = audio.merge(tr_lab, on=config.ID_COL, how="inner")

    ts = pd.read_csv(list(config.DATA_ROOT.glob("*test*split*.csv"))[0])
    ts.columns = [c.strip() for c in ts.columns]
    scol = config.SCORE_COL if config.SCORE_COL in ts.columns else next(
        c for c in ts.columns if "phq" in c.lower() and "score" in c.lower())
    te = audio.merge(ts[[config.ID_COL, scol]], on=config.ID_COL, how="inner")

    Xtr = tr[emb].values.astype(float); ytr = tr[config.SCORE_COL].values.astype(float)
    Xte = te[emb].values.astype(float); yte = te[scol].values.astype(float)
    print(f"  train+dev {len(ytr)}, held-out {len(yte)}\n")

    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), ("clf", SVR())])
    gs = GridSearchCV(pipe, {"clf__C": [1.0, 10.0]}, scoring="neg_mean_absolute_error",
                      cv=KFold(5, shuffle=True, random_state=config.RANDOM_STATE), n_jobs=-1)
    gs.fit(Xtr, ytr)
    pred = gs.best_estimator_.predict(Xte)

    n = len(yte)
    maes, r2s, f1s = [], [], []
    for _ in range(N_BOOT):
        idx = RNG.randint(0, n, n)
        yt, yp = yte[idx], pred[idx]
        maes.append(mean_absolute_error(yt, yp))
        if np.var(yt) > 0:
            r2s.append(r2_score(yt, yp))
        f1s.append(f1_score(yt >= PHQ8_CUTOFF, yp >= PHQ8_CUTOFF, zero_division=0))

    def ci(a):
        a = np.array(a); return np.percentile(a, 2.5), np.percentile(a, 97.5)

    mlo, mhi = ci(maes); rlo, rhi = ci(r2s); flo, fhi = ci(f1s)
    obs_mae = mean_absolute_error(yte, pred)
    obs_r2 = r2_score(yte, pred)
    obs_f1 = f1_score(yte >= PHQ8_CUTOFF, pred >= PHQ8_CUTOFF, zero_division=0)

    print(f"  MAE = {obs_mae:.3f}  95% CI [{mlo:.3f}, {mhi:.3f}]")
    print(f"  R2  = {obs_r2:+.3f}  95% CI [{rlo:+.3f}, {rhi:+.3f}]")
    print(f"  F1  = {obs_f1:.3f}  95% CI [{flo:.3f}, {fhi:.3f}]")
    print()
    if rhi < 0.05:
        print("  -> The ENTIRE 95% interval for R2 is at or below ~0: even the")
        print("     optimistic edge shows no meaningful explanatory power. The")
        print("     null is not a matter of insufficient precision.")
    elif rlo > 0:
        print("  -> R2 interval entirely positive (unexpected for a null; investigate).")
    else:
        print("  -> R2 interval spans zero; the point estimate is negative and even")
        print("     the upper bound is weak. Report honestly.")


if __name__ == "__main__":
    main()