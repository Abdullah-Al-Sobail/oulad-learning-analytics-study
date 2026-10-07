# Predictive Learning Analytics on OULAD — Reproducibility Package

This repository accompanies the manuscript *"Predictive Learning Analytics for
Adaptive Learning: A Leakage-Safe OULAD Study and Proposed LLM-Based Feedback
Architecture"* (submitted to the Journal of Learning Analytics).

## 1. Data

This repository does **not** include the OULAD data files themselves.
Download them from the official source:

https://analyse.kmi.open.ac.uk/open_dataset

(CC-BY 4.0 license; Kuzilek, Hlosta, & Zdrahal, 2017,
https://doi.org/10.1038/sdata.2017.171)

Place the following CSV files in a local `./oulad/` directory at the repo
root before running any script:
`studentInfo.csv`, `studentVle.csv`, `studentAssessment.csv`,
`assessments.csv`, `vle.csv`, `courses.csv`, `studentRegistration.csv`.

## 2. Environment

See `requirements.txt`. Core pipeline (pandas/numpy/scikit-learn/matplotlib)
was run under Python 3.12.3. The XGBoost results in the paper were produced
in a separate environment with XGBoost installed; run `pip show xgboost`
there and record the exact version here before archiving a release.

## 3. Scripts (run in this order)

| Script | Produces |
|---|---|
| `scripts/oulad_pipeline.py` | Shared module: feature engineering (incl. the corrected `avg_days_before_deadline`) and the CV-safe `TrainQuantileBinner` pipeline component. Imported by the stage scripts below, not run directly. |
| `scripts/appendix_final.py` | Table 1: the main 3-model (Logistic Regression / Random Forest / XGBoost) RandomizedSearchCV-tuned benchmark. Requires `xgboost`. |
| `scripts/stage1_v2.py` | Table 2: end-of-course vs. early (day<=56) Random Forest comparison. Writes `checkpoint_v2.pkl` and `validation_results_v2.json`. Run before stage2/stage3/stage4. |
| `scripts/stage2_v2.py` | Table 3 (student-grouped split, presentation holdout, majority/stratified baselines) and Table 4 (feature-group ablation). Reads `checkpoint_v2.pkl`. |
| `scripts/stage3_v2.py` | Bootstrap 95% CIs, permutation importance, and the calibration (reliability curve) figure. Reads `checkpoint_v2.pkl`. |
| `scripts/stage4_assessment_free.py` | The assessment-free day-56 experiment (behavioral-only early-warning signal), added in response to peer review. Reads `checkpoint_v2.pkl`. |
| `scripts/make_architecture_figure.py` | Generates the standalone conceptual architecture figure (Figure 1); does not depend on OULAD data. |

Each stage script after the first reads from and/or writes to
`validation_results_v2.json`, accumulating results across runs.

## 4. Correspondence between scripts and manuscript sections

- Table 1, Section 4.4-4.5: `appendix_final.py`
- Section 4.3 (leakage-safe quartile binning): `oulad_pipeline.py` (`TrainQuantileBinner`)
- Table 2, Section 4.6 (temporal/early-warning evaluation, including the
  assessment-free variant): `stage1_v2.py`, `stage4_assessment_free.py`
- Table 3, Section 4.7 (robustness checks and baselines): `stage2_v2.py`
- Table 4, Section 4.8 (feature-group ablation and permutation importance): `stage2_v2.py`, `stage3_v2.py`
- Figure 2, Section 4.9 (calibration): `stage3_v2.py`
- Figure 1, Section 3 (architecture diagram): `make_architecture_figure.py`

## 5. Known limitations of this pipeline (see manuscript Section 6 for full discussion)

- A single temporal cutoff (day 56) is evaluated, not a full multi-point curve.
- Permutation importance is reported as a tractable substitute for SHAP, which
  was not available in the environment used for the reported results.
- Robustness-check and ablation models (Tables 2-4) use a fixed Random Forest
  configuration (matching Table 1's tuned hyperparameters, but with a reduced
  tree count of 150 vs. 400 for tractability) rather than independently
  re-tuning for each variant.

## 6. Citation

If you use this code, please cite the manuscript (full citation to be added
once the DOI/venue is finalized) and the OULAD dataset (Kuzilek, Hlosta, &
Zdrahal, 2017).
