# Leakage-Free Multimodal Depression Severity Estimation (DAIC-WOZ)

Codebase for my UCL MSc dissertation, *"Automatic Estimation of Depression Severity from
Clinical Interviews."* It implements a rigorous, participant-level nested cross-validation
pipeline to estimate continuous PHQ-8 scores on the DAIC-WOZ corpus, evaluating classical
audio-visual features, deep audio representations (Wav2Vec2, WavLM), text (MPNet), and
multimodal fusion under a single identical protocol.

**Author:** Sarthak Punj (25133018) · **Supervisor:** Tigmanshu Bhatnagar · COMP0190, UCL

## Key Findings

* **Language is the only signal.** Under leakage-free evaluation, the recoverable PHQ-8
  signal is exclusively linguistic. Classical audio-visual functionals and deep audio models
  did not beat a mean predictor.
* **No multimodal gain.** Fusion decomposed entirely into a rescaling of the text prediction
  (recalibration control r = 1.0000): the audio and visual modalities contributed no
  information, not merely that the aggregate failed to improve.
* **Text carries modest, research-grade signal.** The text model alone beat a mean predictor
  significantly (cross-validation MAE 3.72, R² +0.26, p = 0.0087; held-out MAE 4.24, with a
  held-out R² interval spanning zero).
* **Edge deployment.** Because the model's predictive inputs are text alone, the deployed
  pipeline needs no audio or visual feature extraction. The sentence-embedding model
  compresses via fp16 to ~220 MB (from 439 MB) with no loss of signal, supporting private,
  on-device inference.

## Repository Structure

```
.
├── src/          # Core pipeline: dataset building, feature extraction,
│                 #   nested CV, significance tests, fairness checks, edge quantization
├── outputs/      # Generated figures, metric logs, prediction CSVs, saved models
├── archive/      # Ad-hoc exploration and plotting scripts
├── data/         # Git-ignored — DAIC-WOZ raw data (licence-restricted, not distributed)
├── requirements.txt
└── README.md
```

### Key scripts in `src/`

**Data & features**
- `build_dataset.py` — build the `.parquet` feature matrices from raw participant folders
- `build_text_features.py` — chunk-and-pool MPNet sentence embeddings (interviewer turns removed)

**Modality evaluations (one shared protocol)**
- `train_regression.py` — baseline: nested-CV over classical audio / visual / fusion functionals
- `run_text_regression.py` — text: headline MPNet + Ridge model
- `run_audio_regression.py` — deep audio: 18 configs (Wav2Vec2 / WavLM × early/mid/last)
- `run_fusion.py` — multimodal fusion (stacking / averaging / early) vs text anchor
- `edge_fp16.py` — fp16 compression for edge deployment

**Evaluation, significance & fairness**
- `heldout_test_both.py` — final held-out evaluation on the 47 test participants
- `repeated_cv_significance.py` — Nadeau–Bengio corrected significance test
- `shuffle_test.py` — label-permutation test
- `fairness_and_bootstrap.py`, `trace_fairness.py`, `baseline_fairness_v2.py` —
  per-group fairness, confound checks, bootstrap CIs
- `fix_offset_leakage.py` — leakage-free fp16 offset (estimated on train+dev)
- `make_real_figures.py` — regenerate all figures from real predictions

## Data Privacy Notice

Under the DAIC-WOZ End User License Agreement, no raw clinical data, transcripts, or
extracted feature matrices are included in this repository. The corpus is research-use only
and must not be redistributed or used to identify participants. Obtain access from USC ICT.

## Requirements and Usage

Requires **Python 3.10+**.

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Obtain DAIC-WOZ access from USC ICT and place the raw `*_P` folders and split files in
   `data/raw/`.
3. Configure paths in `src/config.py` (`DATA_ROOT`, split files, output directory).
4. Build the feature matrices:
   ```bash
   python3 src/build_dataset.py
   python3 src/build_text_features.py
   ```
5. Run modality evaluations, then the held-out test:
   ```bash
   python3 src/train_regression.py       # baseline
   python3 src/run_text_regression.py    # text
   python3 src/run_audio_regression.py   # deep audio
   python3 src/run_fusion.py             # fusion
   python3 src/edge_fp16.py              # edge
   python3 src/heldout_test_both.py      # final held-out evaluation
   ```
6. Regenerate figures:
   ```bash
   python3 src/make_real_figures.py
   ```

*On Apple Silicon, prefix runs with `export PYTORCH_ENABLE_MPS_FALLBACK=1`.*

All reported numbers trace to the `outputs/*_results.csv` files and the held-out runs;
`RANDOM_STATE` is fixed in `src/config.py` for reproducibility.

## Key Results (held-out, 47 participants)

| Modality | MAE | R² | Verdict |
|---|---|---|---|
| Baseline (A+V) | 5.34 | −0.06 | null |
| Deep audio | 5.54 | −0.13 | null (CI < 0) |
| Text | 4.24 | +0.31 | signal (CI spans 0) |
| Fusion | 4.09 | +0.39 | = text recalibration |
| Edge (fp16 + offset) | 4.24 | +0.32 | text, deployable |

## Ethics

Secondary analysis of an existing, ethically collected, licensed dataset; no new
human-subjects data collection. The model is **research-grade, not a clinical instrument**.
See the dissertation's ethics section for full discussion.

## License

Code released for academic review. DAIC-WOZ data is governed by its own licence and is not
distributed here.