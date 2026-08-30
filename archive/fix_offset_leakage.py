import numpy as np, joblib, copy
import daic_woz_pipeline.src.config as config
from sklearn.metrics import mean_absolute_error, r2_score
from sentence_transformers import SentenceTransformer
from daic_woz_pipeline.src.build_text_features import read_participant_text, chunk_pool_embed
try:
    from daic_woz_pipeline.src.build_text_features import MPNET
except Exception:
    MPNET = "sentence-transformers/all-mpnet-base-v2"
from daic_woz_pipeline.src.heldout_test import test_ids_scores, ensure_files, load_train_labels

blob = joblib.load(config.OUT_DIR / "daicwoz_text_model.joblib")
model, cols = blob["model"], blob["meta"]["features"]
st = SentenceTransformer(MPNET, device="cpu")
h = copy.deepcopy(st); h.to("cpu"); h._first_module().auto_model.half()

def embed(txt, m):
    v = np.asarray(chunk_pool_embed(txt, m), dtype=float).ravel()
    if v.shape[0] < len(cols):
        v = np.concatenate([v, np.zeros(len(cols)-v.shape[0])])
    return v[:len(cols)]

# estimate offset on TRAIN+DEV (leakage-free)
train_df = load_train_labels()
tr_ids = list(train_df[config.ID_COL])
pf_tr, ph_tr = [], []
for pid in tr_ids:
    try:
        ensure_files(pid); txt = read_participant_text(pid)
        if txt is None: continue
        pf_tr.append(model.predict(embed(txt, st).reshape(1,-1))[0])
        ph_tr.append(model.predict(embed(txt, h).reshape(1,-1))[0])
    except Exception:
        continue
offset = float(np.mean(np.array(pf_tr) - np.array(ph_tr)))
print(f"Offset estimated on TRAIN+DEV (n={len(pf_tr)}): {offset:+.3f}")

# apply unchanged to TEST
ids, scores = test_ids_scores()
y, pf, ph = [], [], []
for pid in ids:
    ensure_files(pid); txt = read_participant_text(pid)
    if txt is None: continue
    y.append(scores[pid])
    pf.append(model.predict(embed(txt, st).reshape(1,-1))[0])
    ph.append(model.predict(embed(txt, h).reshape(1,-1))[0])
y, pf, ph = np.array(y), np.array(pf), np.array(ph)
ph_corr = ph + offset

print(f"\nHeld-out (offset from train+dev, applied to test):")
print(f"  float32     : MAE {mean_absolute_error(y,pf):.3f}  R2 {r2_score(y,pf):+.3f}")
print(f"  fp16 raw    : MAE {mean_absolute_error(y,ph):.3f}  R2 {r2_score(y,ph):+.3f}")
print(f"  fp16+offset : MAE {mean_absolute_error(y,ph_corr):.3f}  R2 {r2_score(y,ph_corr):+.3f}")
print(f"\nCompare to old test-fitted offset (-2.351): if similar, numbers barely move")
print(f"and the procedure is now leakage-free.")
