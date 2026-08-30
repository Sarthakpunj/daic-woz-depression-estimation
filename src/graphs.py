import numpy as np, pandas as pd, sys
import daic_woz_pipeline.src.config as config
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import daic_woz_pipeline.src.features as featmod
from daic_woz_pipeline.src.heldout_test import test_ids_scores, ensure_files, load_train_labels

ids, scores = test_ids_scores()
print("Ensuring test files...")
for p in ids: ensure_files(p)

frames=[]
for fn in ["train_features.parquet","dev_features.parquet"]:
    p=config.OUT_DIR/fn
    if p.exists(): frames.append(pd.read_parquet(p))
tr=pd.concat(frames, ignore_index=True)
if config.SCORE_COL not in tr.columns:
    labels=load_train_labels()[[config.ID_COL, config.SCORE_COL]]
    tr=tr.merge(labels, on=config.ID_COL, how="inner")
gcol=getattr(config,"GENDER_COL","Gender")
lab={config.ID_COL, gcol, config.SCORE_COL, "PHQ8_Binary"}
cols=[c for c in tr.columns if c not in lab and pd.api.types.is_numeric_dtype(tr[c])]

ts=pd.read_csv(list(config.DATA_ROOT.glob('*test*split*.csv'))[0]); ts.columns=[c.strip() for c in ts.columns]
gmap=dict(zip(ts[config.ID_COL], ts['Gender']))
rows=[]; kept=[]
for pid in ids:
    try:
        feat=featmod.extract_participant(config.DATA_ROOT, pid)
    except Exception as e:
        print(f"  [{pid}] extract failed: {e}"); continue
    feat[config.SCORE_COL]=scores[pid]
    rows.append(feat); kept.append(pid)
te=pd.DataFrame(rows)
cols=[c for c in cols if c in te.columns]

Xtr=tr[cols].values.astype(float); ytr=tr[config.SCORE_COL].values.astype(float)
Xte=te[cols].values.astype(float); yte=te[config.SCORE_COL].values.astype(float)
g=np.array([gmap.get(p,np.nan) for p in kept])

# impute NaNs with TRAIN column means (fit on train, apply to both)
col_mean=np.nanmean(Xtr, axis=0)
col_mean=np.where(np.isnan(col_mean), 0.0, col_mean)   # cols all-NaN -> 0
Xtr=np.where(np.isnan(Xtr), col_mean, Xtr)
Xte=np.where(np.isnan(Xte), col_mean, Xte)

model=Pipeline([("scale",StandardScaler()),("svr",SVR(kernel="rbf"))])
model.fit(Xtr,ytr); pred=model.predict(Xte)

agg=mean_absolute_error(yte,pred)
print(f"\nHeld-out aggregate MAE: {agg:.3f}  (expected ~5.344)")
print("  OK matches 4.1 baseline." if abs(agg-5.344)<=0.15 else "  !! does NOT match 5.344 — paste this output.")
print("=== BASELINE per-group held-out MAE ===")
gs=sorted(set(g[~np.isnan(g)]))
for grp in gs:
    m=g==grp
    print(f"  group {int(grp)} (n={m.sum()}): MAE {mean_absolute_error(yte[m],pred[m]):.3f}")
if len(gs)==2:
    m0,m1=g==gs[0],g==gs[1]
    print(f"  GAP: {abs(mean_absolute_error(yte[m0],pred[m0])-mean_absolute_error(yte[m1],pred[m1])):.3f}")
mp=ytr.mean()
print("=== mean-predictor per-group (context) ===")
for grp in gs:
    m=g==grp
    print(f"  group {int(grp)}: MAE {mean_absolute_error(yte[m],np.full(m.sum(),mp)):.3f}")
