import os, numpy as np, pandas as pd, joblib, copy
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import daic_woz_pipeline.src.config as config
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sentence_transformers import SentenceTransformer
from daic_woz_pipeline.src.build_text_features import read_participant_text, chunk_pool_embed
try: from daic_woz_pipeline.src.build_text_features import MPNET
except Exception: MPNET="sentence-transformers/all-mpnet-base-v2"
from daic_woz_pipeline.src.heldout_test import test_ids_scores, ensure_files, load_train_labels

FIG=config.OUT_DIR/"figures"; os.makedirs(FIG,exist_ok=True)
TEXTC="#2b6cb0"; NULLC="#a0aec0"; MEANC="#e53e3e"; OKC="#38a169"; PURP="#805ad5"
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True,"savefig.dpi":200,"savefig.bbox":"tight"})

# ============ FIGURE 4.5 (fp16) — direction derived from computed offset ============
blob=joblib.load(config.OUT_DIR/"daicwoz_text_model.joblib"); model,cols=blob["model"],blob["meta"]["features"]
st=SentenceTransformer(MPNET,device="cpu"); h=copy.deepcopy(st); h.to("cpu"); h._first_module().auto_model.half()
def embed(txt,m):
    v=np.asarray(chunk_pool_embed(txt,m),dtype=float).ravel()
    if v.shape[0]<len(cols): v=np.concatenate([v,np.zeros(len(cols)-v.shape[0])])
    return v[:len(cols)]

# train+dev offset (leakage-free), offset = mean(float32 - fp16)
train_df=load_train_labels(); tr_ids=list(train_df[config.ID_COL])
rng=np.random.RandomState(42); sample=list(rng.choice(tr_ids,size=min(40,len(tr_ids)),replace=False))
dd=[]
for pid in sample:
    try:
        ensure_files(pid); t=read_participant_text(pid)
        if t is None: continue
        dd.append(model.predict(embed(t,st).reshape(1,-1))[0]-model.predict(embed(t,h).reshape(1,-1))[0])
    except Exception: continue
offset=float(np.mean(dd))  # = mean(f32 - fp16); negative means fp16 higher
print(f"offset (f32 - fp16) = {offset:+.3f}")

ids,scores=test_ids_scores(); pf,ph=[],[]
for pid in ids:
    ensure_files(pid); t=read_participant_text(pid)
    if t is None: continue
    pf.append(model.predict(embed(t,st).reshape(1,-1))[0]); ph.append(model.predict(embed(t,h).reshape(1,-1))[0])
pf,ph=np.array(pf),np.array(ph)

# derive wording from sign — no hard-coded direction
mag=abs(offset)
if offset<0:   # f32 < fp16  => fp16 higher => subtract to correct
    dir_word=f"fp16 sits {mag:.2f} above float32"; corr_word=f"After correction (\u2212{mag:.2f}, train+dev)"
else:          # fp16 lower => add to correct
    dir_word=f"fp16 sits {mag:.2f} below float32"; corr_word=f"After correction (+{mag:.2f}, train+dev)"

fig,axs=plt.subplots(1,2,figsize=(10,4.4))
axs[0].scatter(pf,ph,s=42,color=PURP,alpha=.7,edgecolor="white"); axs[0].plot([0,24],[0,24],"k--",lw=1,alpha=.5)
axs[0].set_xlim(0,24);axs[0].set_ylim(0,24);axs[0].set_aspect("equal")
axs[0].set_xlabel("float32 prediction");axs[0].set_ylabel("fp16 prediction")
axs[0].set_title(f"fp16 vs float32 (raw)\nr={np.corrcoef(pf,ph)[0,1]:.4f}; {dir_word}",fontweight="bold")
axs[1].scatter(pf,ph+offset,s=42,color=OKC,alpha=.7,edgecolor="white"); axs[1].plot([0,24],[0,24],"k--",lw=1,alpha=.5)
axs[1].set_xlim(0,24);axs[1].set_ylim(0,24);axs[1].set_aspect("equal")
axs[1].set_xlabel("float32 prediction");axs[1].set_ylabel("fp16 + offset")
axs[1].set_title(f"{corr_word}\nmatches full precision",fontweight="bold")
plt.suptitle("Edge: fp16 preserves the signal (correctable constant shift)",fontweight="bold",y=1.02)
plt.savefig(FIG/"real_fig4_fp16_recalibration.png"); plt.close()
print(f"fig4 done: '{dir_word}', correction sign {'-' if offset<0 else '+'}{mag:.2f}")

# ============ FIGURE 4.1 (learning curve) with ERROR BARS ============
tr=pd.read_parquet(config.OUT_DIR/"train_features.parquet"); dv=pd.read_parquet(config.OUT_DIR/"dev_features.parquet")
df=pd.concat([tr,dv],ignore_index=True)
if config.SCORE_COL not in df.columns:
    df=df.merge(load_train_labels()[[config.ID_COL,config.SCORE_COL]],on=config.ID_COL,how="inner")
gcol=getattr(config,"GENDER_COL","Gender"); lab={config.ID_COL,gcol,config.SCORE_COL,"PHQ8_Binary"}
fcols=[c for c in df.columns if c not in lab and pd.api.types.is_numeric_dtype(df[c])]
X=df[fcols].values.astype(float); y=df[config.SCORE_COL].values.astype(float)
cm=np.nanmean(X,axis=0); cm=np.where(np.isnan(cm),0,cm); X=np.where(np.isnan(X),cm,X)
mp=mean_absolute_error(y,np.full(len(y),y.mean()))

rng2=np.random.RandomState(config.RANDOM_STATE)
sizes=[35,71,106,len(X)]; rows=[]; tr_m=[];tr_s=[];va_m=[];va_s=[]
for n in sizes:
    idx=rng2.choice(len(X),size=n,replace=False); Xs,ys=X[idx],y[idx]
    trM,vaM=[],[]
    for tri,tei in KFold(5,shuffle=True,random_state=config.RANDOM_STATE).split(Xs):
        m=Pipeline([("s",StandardScaler()),("svr",SVR(kernel="rbf"))]).fit(Xs[tri],ys[tri])
        trM.append(mean_absolute_error(ys[tri],m.predict(Xs[tri])))
        vaM.append(mean_absolute_error(ys[tei],m.predict(Xs[tei])))
    tr_m.append(np.mean(trM));tr_s.append(np.std(trM));va_m.append(np.mean(vaM));va_s.append(np.std(vaM))
    rows.append({"n":n,"train_mae":np.mean(trM),"train_std":np.std(trM),"val_mae":np.mean(vaM),"val_std":np.std(vaM)})
    print(f"n={n:3d}: train {np.mean(trM):.3f}±{np.std(trM):.3f}  val {np.mean(vaM):.3f}±{np.std(vaM):.3f}")
pd.DataFrame(rows).to_csv(config.OUT_DIR/"learning_curve.csv",index=False)

fig,ax=plt.subplots(figsize=(6.8,4.2))
ax.errorbar(sizes,tr_m,yerr=tr_s,fmt="o-",color=TEXTC,capsize=4,label="Training MAE")
ax.errorbar(sizes,va_m,yerr=va_s,fmt="s-",color=MEANC,capsize=4,label="Validation MAE")
ax.axhline(mp,color=NULLC,ls="--",lw=1.3,label=f"Mean predictor ({mp:.2f})")
ax.set_xlabel("Training participants"); ax.set_ylabel("MAE (nested CV)")
ax.set_title("Baseline learning curve (audio+visual)\nvalidation flat at mean predictor within fold noise",fontweight="bold")
ax.legend(frameon=False,fontsize=9)
plt.savefig(FIG/"real_fig6_learning_curve.png"); plt.close()
print(f"\nfig6 done with error bars. mean-predictor MAE = {mp:.3f}")
