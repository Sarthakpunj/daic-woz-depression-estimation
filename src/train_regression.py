"""
train_regression.py — proposal-compliant evaluation for PHQ-8 depression
severity estimation on DAIC-WOZ.

Implements the four design commitments from the project proposal:
  (i)   Nested participant-level cross-validation (no participant spans
        train/test; inner loop tunes, outer loop evaluates). Addresses the
        subject-level data-leakage critique (Danylenko & Unold 2026 [41]).
  (ii)  Gender-stratified reporting with a +/-5% parity check
        (Bailey & Plumbley 2021 [8]).
  (iii) Mean-predictor baseline, so models must beat trivial learning.
  (iv)  PHQ-8 used as label with limitations acknowledged (handled in writeup).

Primary task: REGRESSION on PHQ8_Score (target MAE < 4).
Also derives a binary label (PHQ8 >= 10) to report F1 for the
fusion-vs-single-modality and gender-parity criteria.

Run:  python train_regression.py
Reads outputs/train_features.parquet and dev_features.parquet, COMBINES them
(nested CV does its own splitting), and writes a results table + metrics JSON.
"""

import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score

import daic_woz_pipeline.src.config as config

PHQ8_CUTOFF = 10  # standard binary depression threshold


def load_combined():
    """Load train+dev and concatenate. Nested CV handles splitting, so we
    pool all labelled participants to maximise data on this small corpus."""
    tr = pd.read_parquet(config.OUT_DIR / "train_features.parquet")
    dv = pd.read_parquet(config.OUT_DIR / "dev_features.parquet")
    df = pd.concat([tr, dv], ignore_index=True)
    if config.SCORE_COL not in df.columns:
        raise KeyError(
            f"'{config.SCORE_COL}' not found. Nested-CV regression needs the "
            f"continuous PHQ-8 score. Re-run build_dataset.py so it is saved.")
    return df


def feature_columns(df):
    audio = [c for c in df.columns if c.startswith(("covarep_", "formant_"))]
    visual = [c for c in df.columns if c.startswith(("gaze_", "pose_", "au_"))]
    return audio, visual


def model_and_grid(name):
    """Return an (estimator, param_grid) for the inner-loop tuning."""
    if name == "rf":
        est = RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=-1)
        grid = {"clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 10, 20]}
    elif name == "ridge":
        est = Ridge(random_state=config.RANDOM_STATE)
        grid = {"clf__alpha": [1.0, 10.0, 100.0]}
    elif name == "svr":
        est = SVR(kernel="rbf")
        grid = {"clf__C": [1.0, 10.0], "clf__gamma": ["scale"]}
    else:
        raise ValueError(name)
    return est, grid


def build_pipeline(est):
    # Preprocessing fitted inside each fold (no leakage).
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", est),
    ])


def parity_gap(values_by_group):
    """Max minus min across groups (e.g. F1 across genders)."""
    vals = [v for v in values_by_group.values() if v is not None and not np.isnan(v)]
    return (max(vals) - min(vals)) if len(vals) >= 2 else float("nan")


def nested_cv_evaluate(X, y, gender, model_name, n_outer=5, n_inner=3,
                       weight_severe=False, pids=None, stratify=False):
    """Nested participant-level CV. Each row is one participant, so a standard
    KFold over rows is already participant-level (no segment leakage possible).
    Outer loop: held-out evaluation. Inner loop: hyperparameter tuning.

    weight_severe: if True, weight training samples by their PHQ-8 score so the
    model attends more to rare high-severity cases (label-skew mitigation).
    stratify: if True, stratify outer folds on the binary label so each fold
    has a similar depressed proportion (reduces fold variance).
    pids: optional participant IDs (same order as X) to record per prediction.
    Weights / tuning use TRAINING-fold data only (no leakage)."""
    if stratify:
        from sklearn.model_selection import StratifiedKFold
        outer = StratifiedKFold(n_splits=n_outer, shuffle=True,
                                random_state=config.RANDOM_STATE)
        split_iter = outer.split(X, (y >= PHQ8_CUTOFF).astype(int))
    else:
        outer = KFold(n_splits=n_outer, shuffle=True,
                      random_state=config.RANDOM_STATE)
        split_iter = outer.split(X)

    maes, rmses, f1s = [], [], []
    train_maes = []  # train-fold MAE, to expose the overfitting gap
    # accumulate out-of-fold predictions for pooled metrics + export
    all_true, all_pred, all_gender, all_pid, all_fold = [], [], [], [], []

    for fold_i, (tr_idx, te_idx) in enumerate(split_iter):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        est, grid = model_and_grid(model_name)
        pipe = build_pipeline(est)
        inner = KFold(n_splits=n_inner, shuffle=True,
                      random_state=config.RANDOM_STATE)
        search = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error",
                              cv=inner, n_jobs=-1)
        if weight_severe:
            sample_w = 1.0 + (y_tr / 6.0)
            search.fit(X_tr, y_tr, clf__sample_weight=sample_w)
        else:
            search.fit(X_tr, y_tr)
        best = search.best_estimator_

        pred = best.predict(X_te)
        train_pred = best.predict(X_tr)
        train_maes.append(mean_absolute_error(y_tr, train_pred))

        maes.append(mean_absolute_error(y_te, pred))
        rmses.append(np.sqrt(mean_squared_error(y_te, pred)))
        f1s.append(f1_score(y_te >= PHQ8_CUTOFF, pred >= PHQ8_CUTOFF,
                            zero_division=0))

        all_true.extend(y_te.tolist())
        all_pred.extend(pred.tolist())
        all_fold.extend([fold_i] * len(te_idx))
        if gender is not None:
            all_gender.extend(gender[te_idx].tolist())
        if pids is not None:
            all_pid.extend(pids[te_idx].tolist())

    at_all = np.array(all_true); ap_all = np.array(all_pred)
    result = {
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "mae_per_fold": [float(m) for m in maes],   # for paired test (#1)
        "rmse_mean": float(np.mean(rmses)),
        "f1_mean": float(np.mean(f1s)),             # fold-averaged (biased low)
        # pooled F1 over all out-of-fold predictions (the honest headline F1, #2)
        "f1_pooled": float(f1_score(at_all >= PHQ8_CUTOFF,
                                    ap_all >= PHQ8_CUTOFF, zero_division=0)),
        "train_mae_mean": float(np.mean(train_maes)),
        "overfit_gap": float(np.mean(maes) - np.mean(train_maes)),
    }

    # Per-score-band MAE (label-skew exposure), shared helper
    result["mae_by_score_band"] = _band_mae(all_true, all_pred)

    # Out-of-fold predictions for export (#4): id, true, pred, gender, fold
    result["_oof"] = {
        "true": all_true, "pred": all_pred,
        "gender": all_gender if all_gender else None,
        "pid": all_pid if all_pid else None,
        "fold": all_fold,
    }

    # Gender-stratified metrics over pooled out-of-fold predictions
    if gender is not None and len(all_gender) == len(all_true):
        at = np.array(all_true); ap = np.array(all_pred); ag = np.array(all_gender)
        by_gender_mae, by_gender_f1 = {}, {}
        for g in np.unique(ag):
            m = ag == g
            if m.sum() == 0:
                continue
            by_gender_mae[str(g)] = float(mean_absolute_error(at[m], ap[m]))
            by_gender_f1[str(g)] = float(
                f1_score(at[m] >= PHQ8_CUTOFF, ap[m] >= PHQ8_CUTOFF,
                         zero_division=0))
        result["mae_by_gender"] = by_gender_mae
        result["f1_by_gender"] = by_gender_f1
        result["f1_gender_gap"] = parity_gap(by_gender_f1)
    return result


def _band_mae(true, pred):
    """Per-score-band MAE, shared by models and the mean predictor."""
    at = np.array(true); ap = np.array(pred)
    bands = {"low_0_4": (0, 4), "mid_5_9": (5, 9),
             "high_10_14": (10, 14), "severe_15_24": (15, 24)}
    out = {}
    for label, (lo, hi) in bands.items():
        m = (at >= lo) & (at <= hi)
        if m.sum() > 0:
            out[label] = {"n": int(m.sum()),
                          "mae": float(mean_absolute_error(at[m], ap[m]))}
    return out


def mean_predictor_baseline(X, y, n_outer=5):
    """Predict the training-fold mean PHQ-8 for everyone. Models must beat this
    to demonstrate non-trivial learning (proposal commitment iii).
    Returns overall MAE, per-fold MAEs (for the paired test), and per-band MAE
    (so the severe-band model failure can be shown to equal mean prediction)."""
    outer = KFold(n_splits=n_outer, shuffle=True, random_state=config.RANDOM_STATE)
    maes, all_true, all_pred = [], [], []
    for tr_idx, te_idx in outer.split(X):
        pred = np.full(len(te_idx), y[tr_idx].mean())
        maes.append(mean_absolute_error(y[te_idx], pred))
        all_true.extend(y[te_idx].tolist())
        all_pred.extend(pred.tolist())
    return {
        "mae_mean": float(np.mean(maes)),
        "mae_per_fold": [float(m) for m in maes],
        "mae_by_score_band": _band_mae(all_true, all_pred),
    }


def main():
    df = load_combined()
    audio, visual = feature_columns(df)
    y = df[config.SCORE_COL].values.astype(float)

    # Participant IDs for per-prediction export (#4). Excluded from features
    # because feature_columns only selects prefixed columns.
    pids = df[config.ID_COL].values if config.ID_COL in df.columns else None

    # Set True to stratify outer folds on the binary label (#5). If you switch
    # this on, ALL results (models + mean predictor) are regenerated together.
    STRATIFY = False

    gender = None
    for cand in ["Gender", "gender", "PHQ8_Gender"]:
        if cand in df.columns:
            gender = df[cand].values
            print(f"Gender column found: '{cand}' -> gender audit enabled.")
            break
    if gender is None:
        print("No gender column in features; gender audit will be skipped.")

    print(f"Participants: {len(df)}  |  PHQ-8 mean={y.mean():.2f} "
          f"std={y.std():.2f}  range=[{y.min():.0f},{y.max():.0f}]  "
          f"stratified_folds={STRATIFY}")

    configs = {"audio": audio, "visual": visual, "fusion": audio + visual}
    models = ["rf", "ridge", "svr"]

    # Mean-predictor baseline (feature-independent): overall, per-fold, per-band
    mean_res = mean_predictor_baseline(df[audio + visual].values, y)
    mean_mae = mean_res["mae_mean"]
    mean_folds = mean_res["mae_per_fold"]
    mean_sev = mean_res["mae_by_score_band"].get("severe_15_24", {}).get("mae")
    print(f"\nMean-predictor baseline MAE = {mean_mae:.3f} "
          f"(severe-band MAE = {mean_sev:.2f} if applicable)  "
          f"-- models must beat this\n")

    rows = [{"weighting": "-", "model": "mean_predictor", "features": "-",
             "mae": round(mean_mae, 3), "rmse": None, "f1_pooled": None,
             "beats_mean": "-",
             "mae_severe": round(mean_sev, 2) if mean_sev else None}]
    all_results = {"mean_predictor": mean_res}

    oof_frames = []  # collect per-participant predictions for export (#4)

    for weight_severe in [False, True]:
        wtag = "weighted" if weight_severe else "plain"
        for feat_name, cols in configs.items():
            X = df[cols].values
            for m in models:
                res = nested_cv_evaluate(X, y, gender, m,
                                         weight_severe=weight_severe,
                                         pids=pids, stratify=STRATIFY)
                # Paired significance test vs mean predictor (#1). Same folds,
                # so paired Wilcoxon is valid; weak at n=5 — report with care.
                pval = None
                try:
                    from scipy.stats import wilcoxon
                    if len(res["mae_per_fold"]) == len(mean_folds):
                        _, pval = wilcoxon(res["mae_per_fold"], mean_folds)
                except Exception:
                    pval = None

                # stash export rows
                oof = res.pop("_oof")
                frame = pd.DataFrame({
                    "participant_id": oof["pid"] if oof["pid"] else
                                      [None] * len(oof["true"]),
                    "true": oof["true"], "pred": oof["pred"],
                    "gender": oof["gender"] if oof["gender"] else
                              [None] * len(oof["true"]),
                    "fold": oof["fold"],
                    "model": m, "features": feat_name, "weighting": wtag,
                })
                oof_frames.append(frame)

                all_results[f"{m}__{feat_name}__{wtag}"] = res
                row = {
                    "weighting": wtag, "model": m, "features": feat_name,
                    "mae": round(res["mae_mean"], 3),
                    "mae_std": round(res["mae_std"], 3),
                    "train_mae": round(res["train_mae_mean"], 3),
                    "overfit_gap": round(res["overfit_gap"], 3),
                    "rmse": round(res["rmse_mean"], 3),
                    "f1_pooled": round(res["f1_pooled"], 3),
                    "beats_mean": "yes" if res["mae_mean"] < mean_mae else "NO",
                    "p_vs_mean": round(pval, 3) if pval is not None else None,
                }
                sev = res.get("mae_by_score_band", {}).get("severe_15_24")
                if sev:
                    row["mae_severe"] = round(sev["mae"], 2)
                if "f1_gender_gap" in res:
                    row["f1_gender_gap"] = round(res["f1_gender_gap"], 3)
                rows.append(row)
                print(f"  [{wtag}] {m}__{feat_name}: MAE={res['mae_mean']:.3f}"
                      f"(+/-{res['mae_std']:.3f}) F1pooled={res['f1_pooled']:.3f} "
                      f"MAE_severe={row.get('mae_severe','-')} "
                      f"p={row['p_vs_mean']} "
                      f"{'BEATS' if res['mae_mean']<mean_mae else 'worse'}")

    table = pd.DataFrame(rows)
    table.to_csv(config.OUT_DIR / "regression_results.csv", index=False)
    # strip non-serialisable nothing; _oof already popped
    with open(config.OUT_DIR / "regression_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    # per-participant out-of-fold predictions (#4)
    pd.concat(oof_frames, ignore_index=True).to_csv(
        config.OUT_DIR / "oof_predictions.csv", index=False)

    print("\n=== Regression results (nested participant-level CV) ===")
    print(table.to_string(index=False))
    print(f"\nTarget: MAE < 4.0   |   Mean-predictor MAE: {mean_mae:.3f}")
    print(f"Saved results, metrics, and oof_predictions.csv to {config.OUT_DIR}")


if __name__ == "__main__":
    main()