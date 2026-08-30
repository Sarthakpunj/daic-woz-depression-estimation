"""
Configuration for the DAIC-WOZ multimodal depression detection baseline.

Edit DATA_ROOT to point at your DAIC-WOZ download once it arrives.
The expected layout (original DAIC-WOZ, AVEC 2017) is:

DATA_ROOT/
    300_P/
        300_COVAREP.csv
        300_FORMANT.csv
        300_CLNF_features.txt      (2D facial landmarks)
        300_CLNF_features3D.txt    (3D facial landmarks)
        300_CLNF_gaze.txt
        300_CLNF_pose.txt
        300_CLNF_AUs.txt           (action units)
        300_TRANSCRIPT.csv
        ...
    301_P/
        ...
    train_split_Depression_AVEC2017.csv
    dev_split_Depression_AVEC2017.csv
    full_test_split.csv            (labels withheld in the original release)
"""

from pathlib import Path

# >>> EDIT THIS ONE LINE once your data is downloaded <<<
DATA_ROOT = Path("/Users/sarthakpunj/Desktop/daic woz")

# Output directory for features, models, figures, metrics
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Official split files (in DATA_ROOT)
TRAIN_SPLIT = "train_split_Depression_AVEC2017.csv"
DEV_SPLIT = "dev_split_Depression_AVEC2017.csv"

# Column names in the split CSVs
ID_COL = "Participant_ID"
LABEL_COL = "PHQ8_Binary"      # 0 = not depressed, 1 = depressed
SCORE_COL = "PHQ8_Score"       # raw PHQ-8 score (0-24)
GENDER_COL = "Gender"   # 0/1 in the split CSVs; used for the fairness audit

# Reproducibility
RANDOM_STATE = 42

# Functional statistics computed per feature column to collapse a
# per-frame time series into one fixed-length vector per participant.
FUNCTIONALS = ["mean", "std", "min", "max", "median",
               "q25", "q75", "skew", "kurtosis"]

# COVAREP rows where the 5th column (VUV, voiced/unvoiced flag) == 0 are
# unvoiced frames; many pipelines drop them so silence doesn't dominate.
COVAREP_VUV_COL_INDEX = 1   # VUV (voiced/unvoiced) flag, confirmed from data
DROP_UNVOICED = True
