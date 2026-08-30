import numpy as np, pandas as pd
import config
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

tr = pd.read_parquet(config.OUT_DIR / "train_features.parquet")
dv = pd.read_parquet(config.OUT_DIR / "dev_features.parquet")
df = pd.concat([tr, dv], ignore_index=True)
print("train+dev shape:", df.shape)

audio  = [c for c in df.columns if c.startswith(("covarep_", "formant_"))]
visual = [c for c in df.columns if c.startswith(("gaze_", "pose_", "au_"))]
print(f"audio feats: {len(audio)}, visual feats: {len(visual)}")

FEATS = audio
Xtr = df[FEATS].values.astype(float)
ytr = df[config.SCORE_COL].values.astype(float)

ts = pd.read_csv(list(config.DATA_ROOT.glob('*test*split*.csv'))[0]); ts.columns=[c.strip() for c in ts.columns]
idcol=config.ID_COL
gmap=dict(zip(ts[idcol], ts['Gender']))
smap=dict(zip(ts[idcol], ts['PHQ_Score'] if 'PHQ_Score' in ts.columns else ts[config.SCORE_COL]))

import glob
test_pq = glob.glob(str(config.OUT_DIR/"*test*feature*.parquet"))
if not test_pq:
    print("\n!! No test-feature parquet found. Run this and paste it:")
    print("   grep -n 'parquet|covarep|formant|test|build' heldout_test.py | head")
    raise SystemExit
te = pd.read_parquet(test_pq[0]); te.columns=[c.strip() for c in te.columns]
print("loaded test features:", test_pq[0].split('/')[-1], te.shape)

te = te[te[idcol].isin(ts[idcol])]
Xte = te[FEATS].values.astype(float)
yte = np.array([smap[p] for p in te[idcol]])
g   = np.array([gmap.get(p, np.nan) for p in te[idcol]])

col_mean = np.nanmean(Xtr, axis=0)
Xtr = np.where(np.isnan(Xtr), col_mean, Xtr)
Xte = np.where(np.isnan(Xte), col_mean, Xte)

model = Pipeline([("scale", StandardScaler()), ("svr", SVR(kernel="rbf"))])
model.fit(Xtr, ytr)
pred = model.predict(Xte)

agg = mean_absolute_error(yte, pred)
print(f"\nHeld-out aggregate MAE: {agg:.3f}   (expected ~5.344)")
if abs(agg-5.344) > 0.15:
    print("  !! aggregate does NOT match 5.344 — do NOT use these numbers; paste this output.")
print("=== BASELINE per-group held-out MAE ===")
for grp in sorted(set(g[~np.isnan(g)])):
    m=g==grp
    print(f"  group {int(grp)} (n={m.sum()}): MAE {mean_absolute_error(yte[m],pred[m]):.3f}")
gs=sorted(set(g[~np.isnan(g)]))
if len(gs)==2:
    m0,m1=g==gs[0],g==gs[1]
    print(f"  GAP: {abs(mean_absolute_error(yte[m0],pred[m0])-mean_absolute_error(yte[m1],pred[m1])):.3f}")
mp=ytr.mean()
print("=== mean-predictor per-group (context) ===")
for grp in gs:
    m=g==grp
    print(f"  group {int(grp)}: MAE {mean_absolute_error(yte[m], np.full(m.sum(),mp)):.3f}")
