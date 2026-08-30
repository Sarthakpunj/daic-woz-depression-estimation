"""Exercises the real features.py logic + the real SMOTE-train-only design,
using a local minimal SMOTE since imblearn isn't installed in this sandbox."""
import sys; sys.path.insert(0, ".")
from pathlib import Path
import numpy as np, pandas as pd
import daic_woz_pipeline.src.config as config
config.DATA_ROOT = Path("_selftest/fake_daic")  # redirect to fake data

from daic_woz_pipeline.src.features import extract_participant

# --- build feature tables using the REAL extractor ---
def build(split_file):
    sp = pd.read_csv(config.DATA_ROOT/split_file)
    rows=[]
    for _,r in sp.iterrows():
        pid=int(r["Participant_ID"])
        f=extract_participant(config.DATA_ROOT,pid)
        f["PHQ8_Binary"]=int(r["PHQ8_Binary"]); rows.append(f)
    return pd.DataFrame(rows)

train=build("train_split_Depression_AVEC2017.csv")
dev=build("dev_split_Depression_AVEC2017.csv")
print("train shape",train.shape,"dev shape",dev.shape)

audio=[c for c in train.columns if c.startswith(("covarep_","formant_"))]
visual=[c for c in train.columns if c.startswith(("gaze_","pose_","au_"))]
print("n audio feats",len(audio),"| n visual feats",len(visual))

# --- minimal SMOTE (train-only) to validate the methodology end-to-end ---
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

def smote(X,y,k=4,seed=42):
    rng=np.random.RandomState(seed)
    Xr,yr=list(X),list(y)
    classes,counts=np.unique(y,return_counts=True)
    maj=counts.max()
    for c in classes:
        Xc=X[y==c]; need=maj-len(Xc)
        if need<=0 or len(Xc)<2: continue
        kk=min(k,len(Xc)-1)
        from sklearn.neighbors import NearestNeighbors
        nn=NearestNeighbors(n_neighbors=kk+1).fit(Xc)
        for _ in range(need):
            i=rng.randint(len(Xc)); nbrs=nn.kneighbors([Xc[i]],return_distance=False)[0][1:]
            j=nbrs[rng.randint(len(nbrs))]; lam=rng.rand()
            Xr.append(Xc[i]+lam*(Xc[j]-Xc[i])); yr.append(c)
    return np.array(Xr),np.array(yr)

def run(cols,name):
    Xtr=train[cols].values; Xdv=dev[cols].values
    ytr=train["PHQ8_Binary"].values; ydv=dev["PHQ8_Binary"].values
    imp=SimpleImputer(strategy="median").fit(Xtr)
    Xtr=imp.transform(Xtr); Xdv=imp.transform(Xdv)
    sc=StandardScaler().fit(Xtr); Xtr=sc.transform(Xtr); Xdv=sc.transform(Xdv)
    Xb,yb=smote(Xtr,ytr)  # SMOTE on TRAIN ONLY
    for mname,clf in [("logreg",LogisticRegression(max_iter=2000)),
                      ("svm",SVC()),
                      ("mlp",MLPClassifier(hidden_layer_sizes=(32,),max_iter=500))]:
        clf.fit(Xb,yb); p=clf.predict(Xdv)
        print(f"  {mname}/{name}: balanced_train={np.bincount(yb)} "
              f"macroF1={f1_score(ydv,p,average='macro'):.3f} acc={accuracy_score(ydv,p):.3f}")

print("\nBEFORE SMOTE train balance:",np.bincount(train["PHQ8_Binary"].values))
for cols,name in [(audio,"audio"),(visual,"visual"),(audio+visual,"fusion")]:
    run(cols,name)
print("\nSELFTEST PASSED: extraction + train-only SMOTE + 3 models all ran.")
