import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import joblib, copy
import daic_woz_pipeline.src.config as config
from sklearn.metrics import mean_absolute_error, r2_score
from sentence_transformers import SentenceTransformer
from daic_woz_pipeline.src.build_text_features import read_participant_text, chunk_pool_embed
try: from daic_woz_pipeline.src.build_text_features import MPNET
except Exception: MPNET="sentence-transformers/all-mpnet-base-v2"
from daic_woz_pipeline.src.heldout_test import test_ids_scores, ensure_files, load_train_labels
import pandas as pd

os.makedirs(config.OUT_DIR / "figures", exist_ok=True)
FIG = config.OUT_DIR / "figures"
TEXTC="#2b6cb0"; NULLC="#a0aec0"; MEANC="#e53e3e"; OKC="#38a169"; PURP="#805ad5"
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.25,
                     "axes.axisbelow":True,"savefig.dpi":200,"savefig.bbox":"tight"})

blob=joblib.load(config.OUT_DIR/"daicwoz_text_model.joblib")
model,cols=blob["model"],blob["meta"]["features"]
st=SentenceTransformer(MPNET, device="cpu")

def embed(txt,m):
    v=np.asarray(chunk_pool_embed(txt,m),dtype=float).ravel()
    if v.shape[0]<len(cols): v=np.concatenate([v,np.zeros(len(cols)-v.shape[0])])
    return v[:len(cols)]

# ---- REAL held-out text predictions ----
ids,scores=test_ids_scores()
for p in ids: ensure_files(p)
ts=pd.read_csv(list(config.DATA_ROOT.glob('*test*split*.csv'))[0]); ts.columns=[c.strip() for c in ts.columns]
gmap=dict(zip(ts[config.ID_COL], ts['Gender']))
yt,yp,g=[],[],[]
for pid in ids:
    txt=read_participant_text(pid)
    if txt is None: continue
    yt.append(scores[pid]); yp.append(model.predict(embed(txt,st).reshape(1,-1))[0])
    g.append(gmap.get(pid,np.nan))
yt,yp,g=np.array(yt),np.array(yp),np.array(g)
print(f"text held-out: n={len(yt)} MAE={mean_absolute_error(yt,yp):.3f} R2={r2_score(yt,yp):+.3f}")

# ===== FIG 1: modality comparison =====
mods=["Baseline\n(A+V)","Deep Audio","Text","Fusion","Edge\n(fp16+offset)"]
mae=[5.344,5.535,4.241,4.093,4.237]; colc=[NULLC,NULLC,TEXTC,TEXTC,TEXTC]
fig,ax=plt.subplots(figsize=(8,4.6))
bars=ax.bar(mods,mae,color=colc,edgecolor="white",width=0.62)
bars[3].set_hatch("///"); bars[4].set_hatch("///")
ax.axhline(5.43,color=MEANC,ls="--",lw=1.6)
for b,v in zip(bars,mae): ax.text(b.get_x()+b.get_width()/2,v+0.06,f"{v:.2f}",ha="center",fontweight="bold")
ax.set_ylabel("Held-out MAE (PHQ-8 points)"); ax.set_ylim(0,6.2)
ax.set_title("Held-out performance by modality\n(lower = better; only text carries recoverable signal)",fontweight="bold")
ax.legend(handles=[Patch(color=TEXTC,label="Recoverable signal (text)"),
                   Patch(facecolor=TEXTC,hatch="///",label="Text-derived (fusion/edge)"),
                   Patch(color=NULLC,label="Null (\u2248 mean predictor)"),
                   plt.Line2D([0],[0],color=MEANC,ls="--",label="Mean predictor (5.43)")],
          frameon=False,fontsize=9,loc="lower right")
plt.savefig(FIG/"real_fig1_modality_comparison.png"); plt.close(); print("real_fig1 done")

# ===== FIG 2: REAL predicted vs actual =====
fig,ax=plt.subplots(figsize=(5.6,5.4))
ax.scatter(yt,yp,s=46,color=TEXTC,alpha=0.7,edgecolor="white")
ax.plot([0,24],[0,24],"k--",lw=1,alpha=.6,label="perfect prediction")
ax.axhline(10,color=MEANC,ls=":",lw=1,alpha=.6); ax.axvline(10,color=MEANC,ls=":",lw=1,alpha=.6)
ax.set_xlim(0,24); ax.set_ylim(0,24); ax.set_aspect("equal")
ax.set_xlabel("Actual PHQ-8"); ax.set_ylabel("Predicted PHQ-8")
ax.set_title(f"Text model: predicted vs actual (held-out)\nMAE {mean_absolute_error(yt,yp):.2f}, R\u00b2 {r2_score(yt,yp):+.2f}",fontweight="bold")
ax.legend(frameon=False,fontsize=9,loc="upper left")
plt.savefig(FIG/"real_fig2_pred_vs_actual.png"); plt.close(); print("real_fig2 done (REAL)")

# ===== FIG 3: REAL fairness =====
fig,ax=plt.subplots(figsize=(6.4,4.4))
gs=sorted(set(g[~np.isnan(g)])); x=np.arange(len(gs)); w=0.5
gm=[mean_absolute_error(yt[g==gr],yp[g==gr]) for gr in gs]
ax.bar(x,gm,w,color=TEXTC,edgecolor="white")
for i,v in enumerate(gm): ax.text(i,v+0.07,f"{v:.2f}",ha="center",fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([f"Group {int(gr)}\n(n={(g==gr).sum()})" for gr in gs])
ax.set_ylabel("Held-out MAE"); ax.set_ylim(0,max(gm)+1)
gap=abs(gm[0]-gm[1]) if len(gm)==2 else 0
ax.set_title(f"Fairness: per-group MAE (text model)\nGap {gap:.2f} PHQ-8 points",fontweight="bold")
plt.savefig(FIG/"real_fig3_fairness.png"); plt.close(); print("real_fig3 done (REAL)")

# ===== FIG 4: REAL float32 vs fp16, TRAIN+DEV offset (leakage-free) =====
h=copy.deepcopy(st); h.to("cpu"); h._first_module().auto_model.half()
# offset from train+dev (40-sample, seed 42) — leakage-free
train_df=load_train_labels(); tr_ids=list(train_df[config.ID_COL])
rng=np.random.RandomState(42); sample=list(rng.choice(tr_ids,size=min(40,len(tr_ids)),replace=False))
dd=[]
for pid in sample:
    try:
        ensure_files(pid); t=read_participant_text(pid)
        if t is None: continue
        dd.append(model.predict(embed(t,st).reshape(1,-1))[0]-model.predict(embed(t,h).reshape(1,-1))[0])
    except Exception: continue
offset=float(np.mean(dd)); print(f"train+dev offset: {offset:+.3f}")
# test preds
pf,ph=[],[]
for pid in ids:
    txt=read_participant_text(pid)
    if txt is None: continue
    pf.append(model.predict(embed(txt,st).reshape(1,-1))[0])
    ph.append(model.predict(embed(txt,h).reshape(1,-1))[0])
pf,ph=np.array(pf),np.array(ph)
fig,axs=plt.subplots(1,2,figsize=(10,4.4))
axs[0].scatter(pf,ph,s=42,color=PURP,alpha=.7,edgecolor="white"); axs[0].plot([0,24],[0,24],"k--",lw=1,alpha=.5)
axs[0].set_xlim(0,24);axs[0].set_ylim(0,24);axs[0].set_aspect("equal")
axs[0].set_xlabel("float32 prediction");axs[0].set_ylabel("fp16 prediction")
axs[0].set_title(f"fp16 vs float32 (raw)\nr={np.corrcoef(pf,ph)[0,1]:.4f}; fp16 sits {abs(offset):.2f} below float32",fontweight="bold")
axs[1].scatter(pf,ph+offset,s=42,color=OKC,alpha=.7,edgecolor="white"); axs[1].plot([0,24],[0,24],"k--",lw=1,alpha=.5)
axs[1].set_xlim(0,24);axs[1].set_ylim(0,24);axs[1].set_aspect("equal")
axs[1].set_xlabel("float32 prediction");axs[1].set_ylabel("fp16 + offset")
axs[1].set_title(f"After correction (+{abs(offset):.2f}, train+dev)\nmatches full precision",fontweight="bold")
plt.suptitle("Edge: fp16 preserves the signal (correctable constant shift)",fontweight="bold",y=1.02)
plt.savefig(FIG/"real_fig4_fp16_recalibration.png"); plt.close(); print(f"real_fig4 done (offset {offset:+.3f})")

# ===== FIG 5: bootstrap CIs =====
fig,ax=plt.subplots(figsize=(7.2,4.0))
data=[("Text\n(held-out)",0.31,-0.22,0.54,TEXTC),("Deep Audio\n(held-out)",-0.130,-0.359,-0.001,NULLC)]
for i,(lab,m,lo,hi,cc) in enumerate(data):
    ax.plot([lo,hi],[i,i],color=cc,lw=3); ax.plot(m,i,"o",color=cc,ms=10,zorder=3)
    ax.text(hi+0.03,i,f"[{lo:+.2f}, {hi:+.2f}]",va="center",fontsize=9)
ax.axvline(0,color=MEANC,ls="--",lw=1.4)
ax.set_yticks([0,1]); ax.set_yticklabels([d[0] for d in data]); ax.set_xlim(-0.55,0.85)
ax.set_xlabel("Held-out R\u00b2 (95% bootstrap CI)")
ax.set_title("Bootstrap CIs: text spans zero (uncertain positive)\naudio entirely below zero (confident null)",fontweight="bold")
plt.savefig(FIG/"real_fig5_bootstrap_ci.png"); plt.close(); print("real_fig5 done")

# ===== FIG 6: REAL learning curve =====
lc_path=config.OUT_DIR/"learning_curve.csv"
if lc_path.exists():
    lc=pd.read_csv(lc_path); n,train,val=lc.iloc[:,0],lc.iloc[:,1],lc.iloc[:,2]
    fig,ax=plt.subplots(figsize=(6.8,4.2))
    ax.plot(n,train,"o-",color=TEXTC,label="Training MAE")
    ax.plot(n,val,"s-",color=MEANC,label="Validation MAE")
    ax.axhline(4.777,color=NULLC,ls="--",lw=1.3,label="Mean predictor (4.78)")
    ax.set_xlabel("Training participants"); ax.set_ylabel("MAE")
    ax.set_title("Baseline learning curve (audio+visual)\nvalidation flat at mean predictor",fontweight="bold")
    ax.legend(frameon=False,fontsize=9)
    plt.savefig(FIG/"real_fig6_learning_curve.png"); plt.close(); print("real_fig6 done (REAL)")
else:
    print("real_fig6 SKIPPED: no learning_curve.csv — run make_learning_curve.py first (below)")

print("\nDONE. Real figures (real_*.png) in", FIG)
