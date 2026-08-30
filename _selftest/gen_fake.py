"""Generate synthetic DAIC-WOZ-shaped folders + split CSVs to test the pipeline."""
import numpy as np, pandas as pd, os
from pathlib import Path
np.random.seed(0)
root = Path("_selftest/fake_daic"); root.mkdir(parents=True, exist_ok=True)

def make_participant(pid, depressed):
    d = root/f"{pid}_P"; d.mkdir(exist_ok=True)
    n = np.random.randint(200, 400)
    shift = 0.8 if depressed else 0.0
    # COVAREP: 74 cols, col5 (idx4) is VUV flag 0/1
    cov = np.random.randn(n,74)+shift
    cov[:,4] = (np.random.rand(n)>0.3).astype(float)
    pd.DataFrame(cov).to_csv(d/f"{pid}_COVAREP.csv", header=False, index=False)
    # FORMANT: 5 cols
    pd.DataFrame(np.random.randn(n,5)+shift).to_csv(d/f"{pid}_FORMANT.csv", header=False, index=False)
    # CLNF files have headers + frame/timestamp/confidence/success
    def clnf(name, k):
        df = pd.DataFrame(np.random.randn(n,k)+shift, columns=[f"{name}{i}" for i in range(k)])
        df.insert(0,"frame",np.arange(n)); df.insert(1,"timestamp",np.arange(n)*0.03)
        df.insert(2,"confidence",0.9); df.insert(3,"success",1)
        df.to_csv(d/f"{pid}_CLNF_{name}.txt", index=False)
    clnf("gaze",8); clnf("pose",6); clnf("AUs",20)

# 30 train (≈30% depressed), 12 dev
train_ids = list(range(300,330)); dev_ids = list(range(330,342))
def labels(ids, frac):
    lab = (np.random.rand(len(ids))<frac).astype(int)
    return lab
tr_lab = labels(train_ids,0.3); dv_lab = labels(dev_ids,0.3)
for pid,l in zip(train_ids,tr_lab): make_participant(pid,l)
for pid,l in zip(dev_ids,dv_lab): make_participant(pid,l)

pd.DataFrame({"Participant_ID":train_ids,"PHQ8_Binary":tr_lab,
              "PHQ8_Score":tr_lab*12+np.random.randint(0,5,len(train_ids))}
            ).to_csv(root/"train_split_Depression_AVEC2017.csv", index=False)
pd.DataFrame({"Participant_ID":dev_ids,"PHQ8_Binary":dv_lab,
              "PHQ8_Score":dv_lab*12+np.random.randint(0,5,len(dev_ids))}
            ).to_csv(root/"dev_split_Depression_AVEC2017.csv", index=False)
print("fake data:", len(train_ids),"train", len(dev_ids),"dev",
      "| train depressed:", int(tr_lab.sum()))
