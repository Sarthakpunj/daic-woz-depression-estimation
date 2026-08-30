"""
build_text_features.py — Phase 2 text modality for DAIC-WOZ.

Pipeline per participant:
  1. read xxx_TRANSCRIPT.csv (TAB-separated: start_time, stop_time, speaker, value)
  2. keep ONLY participant turns (drop Ellie) -- interviewer-segment removal,
     addressing the shortcut documented by Danylenko & Unold [41] / Burdisso.
  3. light clean (drop scrubbed/empty, strip bracketed non-speech markers)
  4. embed with a sentence-transformer using CHUNK-AND-POOL so long interviews
     are not truncated to 512 tokens: split into word-chunks, embed each,
     mean-pool to one fixed-length vector per participant.
  5. save one parquet per model: text_features_<tag>.parquet, columns =
     emb_0..emb_{d-1} + Participant_ID + (optional) interpretable linguistic
     features, in the same format build_dataset.py produces so the existing
     nested-CV pipeline can consume it.

Run locally (CPU is fine; frozen embeddings, no fine-tuning):
    pip install sentence-transformers --break-system-packages
    python3 build_text_features.py

Edit MODELS to compare different encoders.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

import config

# --- models to compare (tag -> HF/sentence-transformers name) ---------------
# Frozen embeddings only. all-mpnet-base-v2 is the strong general default;
# MiniLM is smaller (tests whether lower dim helps at n=142); add a clinical
# model if installed. Comment out any you don't want to run.
MODELS = {
    "mpnet":  "sentence-transformers/all-mpnet-base-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "clinical": "emilyalsentzer/Bio_ClinicalBERT",   # ungated clinical BERT
}

CHUNK_WORDS = 200          # ~ under the 512-token limit after tokenisation
ADD_LINGUISTIC = True      # add interpretable features (pronouns, length, etc.)

# first-person singular pronouns: elevated use is associated with depression
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'll", "i'd"}


def read_participant_text(pid: int) -> str | None:
    """Return the participant's concatenated speech (Ellie removed), or None."""
    folder = config.DATA_ROOT / f"{pid}_P"
    path = folder / f"{pid}_TRANSCRIPT.csv"
    if not path.exists():
        return None
    # TAB-separated despite the .csv extension
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        # some files have stray quoting; fall back to python engine
        df = pd.read_csv(path, sep="\t", engine="python", on_bad_lines="skip")
    df.columns = [c.strip().lower() for c in df.columns]
    if "speaker" not in df.columns or "value" not in df.columns:
        return None
    # keep participant turns only (interviewer-segment removal)
    part = df[df["speaker"].astype(str).str.strip().str.lower() == "participant"]
    texts = []
    for v in part["value"].astype(str):
        v = v.strip()
        if not v or v.lower() == "nan":
            continue
        if "scrubbed_entry" in v.lower():     # privacy redaction, not speech
            continue
        v = re.sub(r"<[^>]*>", " ", v)        # drop <laughter> etc.
        v = re.sub(r"\bxxx\b", " ", v)        # unrecognised-word marker
        v = re.sub(r"\s+", " ", v).strip()
        if v:
            texts.append(v)
    joined = " ".join(texts).strip()
    return joined if joined else None


def linguistic_features(text: str) -> dict:
    words = text.split()
    n = len(words)
    if n == 0:
        return {"ling_wordcount": 0, "ling_ttr": 0.0, "ling_firstperson": 0.0}
    lower = [w.lower().strip(".,!?;:") for w in words]
    ttr = len(set(lower)) / n                       # type-token ratio
    fp = sum(1 for w in lower if w in FIRST_PERSON) / n
    return {"ling_wordcount": n, "ling_ttr": round(ttr, 5),
            "ling_firstperson": round(fp, 5)}


def chunk_pool_embed(text: str, model) -> np.ndarray:
    """Split into word-chunks, embed each, mean-pool -> one vector.
    Avoids truncating a 10-min interview to the model's 512-token limit."""
    words = text.split()
    if not words:
        return None
    chunks = [" ".join(words[i:i + CHUNK_WORDS])
              for i in range(0, len(words), CHUNK_WORDS)]
    embs = model.encode(chunks, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)
    return embs.mean(axis=0)


def split_ids():
    ids = []
    for fn in [config.TRAIN_SPLIT, config.DEV_SPLIT]:
        df = pd.read_csv(config.DATA_ROOT / fn)
        df.columns = [c.strip() for c in df.columns]
        ids += [int(x) for x in df[config.ID_COL].tolist()]
    return sorted(set(ids))


def main():
    from sentence_transformers import SentenceTransformer

    ids = split_ids()
    print(f"{len(ids)} train+dev participants.")

    # read all texts once (model-independent)
    texts, kept_ids, ling_rows = {}, [], {}
    missing = []
    for pid in ids:
        t = read_participant_text(pid)
        if t is None:
            missing.append(pid)
            continue
        texts[pid] = t
        kept_ids.append(pid)
        if ADD_LINGUISTIC:
            ling_rows[pid] = linguistic_features(t)
    print(f"  transcripts read: {len(kept_ids)}  | missing: {len(missing)}")
    if missing:
        print(f"  missing ids (no/empty transcript): {missing}")

    # word-count sanity
    wc = np.array([len(texts[p].split()) for p in kept_ids])
    print(f"  participant words: min={wc.min()} median={int(np.median(wc))} "
          f"max={wc.max()}")

    for tag, name in MODELS.items():
        print(f"\n=== embedding with {tag} ({name}) ===")
        model = SentenceTransformer(name)   # downloads once, then cached
        rows = []
        for i, pid in enumerate(kept_ids, 1):
            vec = chunk_pool_embed(texts[pid], model)
            row = {f"emb_{j}": float(v) for j, v in enumerate(vec)}
            row[config.ID_COL] = pid
            if ADD_LINGUISTIC:
                row.update(ling_rows[pid])
            rows.append(row)
            if i % 25 == 0:
                print(f"    {i}/{len(kept_ids)}")
        out = pd.DataFrame(rows)
        path = config.OUT_DIR / f"text_features_{tag}.parquet"
        out.to_parquet(path, index=False)
        print(f"  saved {path}  shape={out.shape}")

    print("\nDone. Next: run these through the nested-CV pipeline (see "
          "run_text_regression.py) and compare against the mean predictor.")


if __name__ == "__main__":
    main()
