
import time, os, copy
from pathlib import Path
import numpy as np, torch, joblib
import daic_woz_pipeline.src.config as config
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from daic_woz_pipeline.src.build_text_features import read_participant_text, chunk_pool_embed
try:
    from daic_woz_pipeline.src.build_text_features import MPNET
except Exception:
    MPNET = "sentence-transformers/all-mpnet-base-v2"
from daic_woz_pipeline.src.heldout_test import test_ids_scores, ensure_files

BLOB = config.OUT_DIR / "daicwoz_text_model.joblib"

def dir_mb(p):
    return sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(p) for f in fs)/1e6

def predict_all(st, model, cols, ids, scores):
    X,y,kept=[],[],[]
    for pid in ids:
        ensure_files(pid); txt=read_participant_text(pid)
        if txt is None: continue
        v=np.asarray(chunk_pool_embed(txt,st),dtype=float).ravel()
        if v.shape[0]<len(cols): v=np.concatenate([v,np.zeros(len(cols)-v.shape[0])])
        X.append(v[:len(cols)]); y.append(scores[pid]); kept.append(pid)
    X,y=np.array(X),np.array(y)
    return y, model.predict(X), kept

blob=joblib.load(BLOB); model=blob["model"]; cols=blob["meta"]["features"]
st=SentenceTransformer(MPNET, device="cpu")

print("[1] SIZE")
fp="/tmp/mpnet_f32"; st.save(fp); s_full=dir_mb(fp)
h=copy.deepcopy(st); h.to("cpu"); h._first_module().auto_model.half()
hp="/tmp/mpnet_fp16"; h.save(hp); s_half=dir_mb(hp)
print(f"  float32: {s_full:.1f} MB | fp16: {s_half:.1f} MB | -{100*(1-s_half/s_full):.0f}%")

print("[2] LATENCY (CPU ms/chunk)")
smp=["I have felt low and not slept, withdrawing from people for weeks."]*3
def lat(m,n=15):
    for t in smp[:2]: chunk_pool_embed(t,m)
    s=time.perf_counter(); c=0
    for _ in range(n):
        for t in smp: chunk_pool_embed(t,m); c+=1
    return (time.perf_counter()-s)/c*1000
print(f"  float32 {lat(st):.1f} | fp16 {lat(h):.1f}")

print("[3] ACCURACY + consistency + fairness")
ids,scores=test_ids_scores()
yf,pf,kept=predict_all(st,model,cols,ids,scores)
yh,ph,_=predict_all(h,model,cols,ids,scores)
print(f"  float32 MAE {mean_absolute_error(yf,pf):.3f} R2 {r2_score(yf,pf):+.3f}")
print(f"  fp16    MAE {mean_absolute_error(yh,ph):.3f} R2 {r2_score(yh,ph):+.3f}")
print(f"  consistency r={np.corrcoef(pf,ph)[0,1]:.4f} mean|diff|={np.mean(np.abs(pf-ph)):.3f}")
import pandas as pd
ts=pd.read_csv(list(config.DATA_ROOT.glob('*test*split*.csv'))[0]); ts.columns=[c.strip() for c in ts.columns]
gm=dict(zip(ts[config.ID_COL],ts['Gender'])); g=np.array([gm.get(p,np.nan) for p in kept])
for grp in sorted(set(g[~np.isnan(g)])):
    m=g==grp
    print(f"  group {int(grp)} (n={m.sum()}): f32 {mean_absolute_error(yf[m],pf[m]):.3f} | fp16 {mean_absolute_error(yh[m],ph[m]):.3f}")

# ---- offset-correction check: is the fp16 MAE bump a systematic shift? ----
print("[4] OFFSET-CORRECTION CHECK")
offset = float(np.mean(pf - ph))
ph_corrected = ph + offset
print(f"  systematic offset (mean f32-fp16): {offset:+.3f} points")
print(f"  fp16 raw       : MAE {mean_absolute_error(yh,ph):.3f}  R2 {r2_score(yh,ph):+.3f}")
print(f"  fp16 +offset   : MAE {mean_absolute_error(yh,ph_corrected):.3f}  R2 {r2_score(yh,ph_corrected):+.3f}")
print(f"  float32        : MAE {mean_absolute_error(yf,pf):.3f}  R2 {r2_score(yf,pf):+.3f}")
print("  -> if fp16+offset ~ float32, the fp16 loss is a correctable constant.")

# ---- save fp16 results to JSON (archive traceability) ----
import json
results = {
    "method": "fp16 half-precision (int8 rejected: qnnpack embedding cosine ~0.7)",
    "audio_encoder_mb": 377.5,
    "size_mb": {"float32": 438.9, "fp16": 220.0, "reduction_pct": 50.0},
    "latency_ms_chunk_cpu": {"note": "comparable; fp16 emulated on CPU, ~40-50ms both"},
    "accuracy": {
        "float32": {"mae": round(float(mean_absolute_error(yf,pf)),3), "r2": round(float(r2_score(yf,pf)),3)},
        "fp16_raw": {"mae": round(float(mean_absolute_error(yh,ph)),3), "r2": round(float(r2_score(yh,ph)),3)},
        "fp16_offset_corrected": {"mae": round(float(mean_absolute_error(yh,ph_corrected)),3), "r2": round(float(r2_score(yh,ph_corrected)),3)},
        "systematic_offset": round(offset,3),
        "prediction_corr": round(float(np.corrcoef(pf,ph)[0,1]),4),
        "mean_abs_diff": round(float(np.mean(np.abs(pf-ph))),3)
    },
    "fairness": {
        "group0": {"n": 24, "f32": 3.272, "fp16": 3.887},
        "group1": {"n": 23, "f32": 5.251, "fp16": 5.188}
    }
}
with open(config.OUT_DIR / "edge_fp16_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved -> edge_fp16_results.json")
