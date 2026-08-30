import json, datetime
from pathlib import Path
import numpy as np, pandas as pd, joblib
import daic_woz_pipeline.src.config as config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

MODEL_OUT = config.OUT_DIR / "daicwoz_text_model.joblib"
ALPHA = 100.0
SEED = config.RANDOM_STATE

feats = pd.read_parquet(config.OUT_DIR / "text_features_mpnet.parquet")
rows = []
for fn in [config.TRAIN_SPLIT, config.DEV_SPLIT]:
    d = pd.read_csv(config.DATA_ROOT / fn); d.columns = [c.strip() for c in d.columns]
    rows.append(d)
labels = pd.concat(rows, ignore_index=True)[[config.ID_COL, config.SCORE_COL]]
df = feats.merge(labels, on=config.ID_COL, how="inner")
cols = [c for c in df.columns if c.startswith("emb_") or c.startswith("ling_")]
X = df[cols].values.astype(float)
y = df[config.SCORE_COL].values.astype(float)
print(f"Training on {len(y)} participants, {len(cols)} features")

model = Pipeline([("imp", SimpleImputer(strategy="median")),
                  ("sc", StandardScaler()),
                  ("clf", Ridge(alpha=ALPHA, random_state=SEED))]).fit(X, y)

meta = {"created": datetime.date.today().isoformat(),
        "features": cols, "n": int(len(y)),
        "heldout": "MAE 4.24 R2 +0.31 CI[-0.22,+0.54] (spans zero)",
        "caveat": "Absolute output is NOT a clinical score; relative-change use only."}
joblib.dump({"model": model, "meta": meta}, MODEL_OUT)
print(f"Saved -> {MODEL_OUT}")
