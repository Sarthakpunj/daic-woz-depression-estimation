import numpy as np, pandas as pd
import daic_woz_pipeline.src.config as config
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

tr = pd.read_parquet(config.OUT_DIR / "train_features.parquet")
dv = pd.read_parquet(config.OUT_DIR / "dev_features.parquet")
df = pd.concat([tr, dv], ignore_index=True)
if config.SCORE_COL not in df.columns:
    from daic_woz_pipeline.src.heldout_test import load_train_labels
    df = df.merge(load_train_labels()[[config.ID_COL, config.SCORE_COL]], on=config.ID_COL, how="inner")

gcol = getattr(config, "GENDER_COL", "Gender")
lab = {config.ID_COL, gcol, config.SCORE_COL, "PHQ8_Binary"}
cols = [c for c in df.columns if c not in lab and pd.api.types.is_numeric_dtype(df[c])]
X = df[cols].values.astype(float)
y = df[config.SCORE_COL].values.astype(float)
cm = np.nanmean(X, axis=0); cm = np.where(np.isnan(cm), 0, cm)
X = np.where(np.isnan(X), cm, X)

mean_pred_mae = mean_absolute_error(y, np.full(len(y), y.mean()))
print(f"mean-predictor MAE (full set): {mean_pred_mae:.3f}")

rng = np.random.RandomState(config.RANDOM_STATE)
sizes = [35, 71, 106, len(X)]
rows = []
for n in sizes:
    idx = rng.choice(len(X), size=n, replace=False)
    Xs, ys = X[idx], y[idx]
    tr_maes, va_maes = [], []
    for tri, tei in KFold(5, shuffle=True, random_state=config.RANDOM_STATE).split(Xs):
        m = Pipeline([("s", StandardScaler()), ("svr", SVR(kernel="rbf"))]).fit(Xs[tri], ys[tri])
        tr_maes.append(mean_absolute_error(ys[tri], m.predict(Xs[tri])))
        va_maes.append(mean_absolute_error(ys[tei], m.predict(Xs[tei])))
    rows.append({"n": n, "train_mae": np.mean(tr_maes), "val_mae": np.mean(va_maes)})
    print(f"n={n:3d}: train {np.mean(tr_maes):.3f}  val {np.mean(va_maes):.3f}")

pd.DataFrame(rows).to_csv(config.OUT_DIR / "learning_curve.csv", index=False)
print("\nsaved learning_curve.csv")
print("-> if val_mae stays flat near the mean-predictor MAE, the figure supports the null.")
