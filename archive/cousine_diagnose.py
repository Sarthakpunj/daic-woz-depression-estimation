"""
cosine_diagnose.py — find WHY float32 vs int8 predictions diverged (r=0.81).

Runs three checks:
  1. Embedding cosine: same text through float32 vs int8 embedder. If ~1.0, the
     embedder is fine and the bug is downstream (feature alignment). If low, int8
     quantization genuinely mangled the representation on this backend.
  2. fp16 comparison: half-precision is near-lossless; if fp16 gives r~0.99 but
     int8 gives r~0.81, the int8/qnnpack path is the culprit (use fp16 instead).
  3. Determinism: embed the SAME text twice with the SAME model — if that isn't
     ~1.0, there's nondeterminism (e.g. tokenizer/padding) confounding everything.
"""
import numpy as np, joblib, torch, copy
import daic_woz_pipeline.src.config as config
from sentence_transformers import SentenceTransformer
from daic_woz_pipeline.src.build_text_features import chunk_pool_embed
try: from daic_woz_pipeline.src.build_text_features import MPNET
except Exception: MPNET = "sentence-transformers/all-mpnet-base-v2"

torch.backends.quantized.engine = "qnnpack"
st = SentenceTransformer(MPNET, device="cpu")
texts = ["I have felt low and unmotivated, not sleeping, withdrawing from people for weeks now.",
         "Work went okay today, saw a friend, felt reasonably good.",
         "Everything is heavy and pointless and I cannot get out of bed."]

def cos(a,b):
    a,b=np.ravel(a),np.ravel(b); m=min(len(a),len(b)); a,b=a[:m],b[:m]
    return float(np.dot(a,b)/((np.linalg.norm(a)*np.linalg.norm(b)) or 1))

# 3. determinism
print("[determinism] same model, same text, twice:")
for t in texts[:1]:
    e1=chunk_pool_embed(t,st); e2=chunk_pool_embed(t,st)
    print(f"   cos = {cos(e1,e2):.5f}  (should be 1.00000)")

# 1. int8 embedder
q = copy.deepcopy(st); q.to("cpu")
for p in q.parameters(): p.data = p.data.cpu()
torch.quantization.quantize_dynamic(q._first_module().auto_model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
print("\n[int8] embedding cosine float32 vs int8:")
for t in texts:
    print(f"   cos = {cos(chunk_pool_embed(t,st), chunk_pool_embed(t,q)):.5f}")

# 2. fp16 embedder
h = copy.deepcopy(st); h.to("cpu"); h._first_module().auto_model.half()
print("\n[fp16] embedding cosine float32 vs fp16:")
for t in texts:
    try:
        print(f"   cos = {cos(chunk_pool_embed(t,st), chunk_pool_embed(t,h)):.5f}")
    except Exception as e:
        print(f"   fp16 failed on CPU ({e}) — expected; fp16 often needs GPU. Use int8 fix instead.")
        break

print("\nINTERPRETATION:")
print("  determinism <1.0  -> nondeterministic embedding; fix that first.")
print("  int8 cos ~1.0     -> embedder fine; bug is downstream feature alignment.")
print("  int8 cos low      -> int8/qnnpack mangles this model; prefer fp16 or")
print("                        report int8 honestly as lossy on this backend.")