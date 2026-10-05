# A Feature-Selection-Driven Ensemble Learning Framework for Early Prediction of Chronic Kidney Disease

Final-year B.Tech CSE (AI & ML) project. It builds a CKD classification pipeline on NHANES 1999–2018 data, compares feature-selection methods and several classifiers, checks the results with repeated cross-validation, explains the model with SHAP, and serves a 10-feature XGBoost model through a small Flask app.

NHANES is a cross-sectional survey, so the models here classify whether a participant meets the CKD definition used in this project at the time of examination. They do not predict future CKD onset or progression. "Early prediction" in the title should be read as early identification from routine measurements.

## Contents

- [Workflow](#workflow)
- [Dataset](#dataset)
- [CKD target definition](#ckd-target-definition)
- [Predictors](#predictors)
- [Feature selection](#feature-selection)
- [Models](#models)
- [Results](#results)
- [Explainability](#explainability)
- [Flask application](#flask-application)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Reproducing the pipeline](#reproducing-the-pipeline)
- [Limitations](#limitations)
- [Future work](#future-work)
- [Disclaimer](#disclaimer)
- [Author](#author)

## Workflow

```
NHANES files (10 cycles, 1999–2018)
  → merge demographics, lab, body measures, blood pressure, diabetes questionnaire
  → adult cohort (age ≥ 20) and CKD target
  → ML dataset (25 predictors)
  → imputation + scaling, stratified 80/20 split
  → feature selection (5 methods, frequency across methods)
  → model training and comparison (5 / 7 / 10 / 25 features)
  → repeated stratified cross-validation
  → SHAP and XGBoost feature importance
  → Flask app (XGBoost, 10 features)
```

## Dataset

The data comes from the public CDC NHANES releases for ten two-year cycles from 1999–2000 to 2017–2018. The download scripts in `src/` fetch the following files per cycle: demographics (DEMO), standard biochemistry profile, urine albumin/creatinine, body measures (BMX), blood pressure (BPX), diabetes questionnaire (DIQ) and blood pressure questionnaire (BPQ).

Raw and processed data are **not** committed. `data/raw/` and `data/processed/` are listed in `.gitignore`, so the dataset has to be rebuilt locally with the scripts described below.

| Item | Value |
|---|---|
| Source | NHANES 1999–2018 (10 cycles) |
| Population | Adults, age ≥ 20 with enough data to define the target |
| Final ML dataset | 51,713 participants (`src/project_integrity_test.py` checks for this row count) |
| Non-CKD / CKD | 42,883 / 8,830 (CKD prevalence about 17.08%) |
| Predictors | 25 |
| Train / test split | 80 / 20, stratified, `random_state=42` |

The class counts are what the author's local run of `create_ml_dataset.py` printed. They can't be recomputed from the tracked files, since the data isn't in the repository. Rerunning the pipeline will show the numbers on your machine.

## CKD target definition

The target is built in `src/create_nhanes_cohort.py`:

- eGFR is computed from serum creatinine, age and sex using a CKD-EPI 2009-style equation, with separate coefficients for men and women and no race coefficient.
- UACR is computed as urine albumin / urine creatinine × 100. Values with non-positive albumin or creatinine are set to missing.
- `ckd_target = 1` if **eGFR < 60 or UACR ≥ 30**, otherwise `0`. Participants with neither eGFR nor UACR available are dropped.

`create_ml_dataset.py` removes creatinine, eGFR, UACR and the urine albumin/creatinine columns from the predictor table, so the variables that define the label aren't used as inputs. That reduces direct target leakage. It doesn't rule out indirect information that correlates with kidney function. BUN, for example, is a kidney-related lab value and is among the strongest predictors.

The model learns this particular lab-based definition of CKD. It is not a clinician's diagnosis.

## Predictors

The 25 predictors come from `create_ml_dataset.py` and are the same list used in `final_analysis_nhanes.py` and `shap_analysis_nhanes.py`:

| Group | Variables |
|---|---|
| Demographic | `age`, `sex` |
| Anthropometric | `bmi`, `weight`, `waist` |
| Blood pressure | `systolic_bp`, `diastolic_bp` |
| Biochemistry | `glucose`, `bun`, `total_cholesterol`, `uric_acid`, `albumin`, `total_protein`, `calcium`, `phosphorus`, `sodium`, `potassium`, `chloride`, `bilirubin`, `triglycerides`, `ast`, `alt`, `alp`, `ggt` |
| Diabetes-related | `diabetes` (questionnaire item DIQ010, encoded 1/0) |

NHANES missing-value codes (7, 9, 77, 99, 777, 999, 7777, 9999) are converted to NaN. Remaining missing values are filled with the median in a `SimpleImputer`, followed by `StandardScaler`.

`create_ml_dataset.py` also lists a `bp_told_high` variable (BPQ020). `add_bpq_to_merged.py` writes a separate `_merged_v2.csv`, but the cohort script reads the original merged file, so this variable is not part of the 25 predictors used in the reported experiments.

## Feature selection

`src/feature_selection_nhanes.py` runs five methods on the processed training split only and keeps the top 15 variables from each:

| Method | Implementation |
|---|---|
| Pearson correlation | absolute correlation with the target |
| Mutual information | `mutual_info_classif` |
| RFE | recursive elimination with logistic regression |
| LASSO | L1-penalised logistic regression, ranked by absolute coefficient |
| Random Forest importance | 300 trees, balanced class weights |

The selected lists are saved to `results/nhanes_feature_selection_results.csv`. How often each variable appears across the five lists is saved to `results/nhanes_feature_frequency.csv`.

- **Selected by all 5 methods** (the "core 7"): `age`, `bun`, `systolic_bp`, `glucose`, `diabetes`, `uric_acid`, `waist`
- **Selected by 4 of 5 methods:** `albumin`, `bmi`, `total_cholesterol`. Adding these to the core 7 gives the 10-feature set.
- `weight` was selected by only one method.

The 5-feature set (`age`, `bun`, `systolic_bp`, `glucose`, `diabetes`) is the first five variables of the core group. Both the 5- and 7-feature sets were chosen by hand from the frequency ranking, not by an automated search.

## Models

Experiments cover:

- Logistic Regression
- Linear SVM
- Random Forest
- Gradient Boosting
- XGBoost
- Soft Voting ensemble (Logistic Regression + Random Forest + XGBoost)
- Stacking ensemble (same three base learners, Logistic Regression as the meta-learner)

`src/model_training_nhanes.py` trains all seven on the core 7, extended 10 and all 25 feature sets with 5-fold stratified cross-validation plus a held-out test set. Output is in `results/nhanes_model_results.csv`.

Only **XGBoost with 10 features** is used by the Flask app. The other models are experimental comparisons.

Decision threshold is 0.5 everywhere (0 on the decision function for the SVM). No threshold tuning was done.

## Results

### Feature-set comparison (XGBoost, held-out test set)

| Feature Set | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------------|----------|-----------|--------|----------|---------|
| 5 features  | 85.84% | 68.07% | 32.11% | 43.63% | 80.03% |
| 7 features  | 85.91% | 68.82% | 31.99% | 43.68% | 80.62% |
| 10 features | 86.17% | 69.31% | 34.14% | 45.75% | 81.15% |
| 25 features | 86.32% | 69.14% | 35.90% | 47.26% | 82.45% |

Sources: `nhanes_5feature_test_results.csv`, `nhanes_7feature_test_results.csv`, `nhanes_10feature_test_results.csv`, `nhanes_final_test_results.csv`. These scripts use `n_estimators=300`, `max_depth=5`, `learning_rate=0.05`.

5-fold cross-validation on the training split, for the 5, 7 and 10-feature models:

| Feature Set | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------------|----------|-----------|--------|----------|---------|
| 5 features  | 85.96% | 69.07% | 32.21% | 43.92% | 80.53% |
| 7 features  | 86.20% | 69.81% | 33.88% | 45.60% | 81.31% |
| 10 features | 86.26% | 69.99% | 34.22% | 45.95% | 81.82% |

The final 25-feature script doesn't run a separate CV, so there is no matching row. `nhanes_model_results.csv` has a CV result for an XGBoost 25-feature model with a different configuration (`n_estimators=200`, `max_depth=4`), with ROC-AUC 82.92%.

The 10-feature model also has a test specificity of 96.89%.

### Ensembles on 25 features (held-out test set)

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| XGBoost | 0.8632 | 0.6914 | 0.3590 | 0.9670 | 0.4726 | 0.8245 |
| Voting | 0.8474 | 0.5591 | 0.5034 | 0.9183 | 0.5298 | 0.8200 |
| Stacking | 0.7916 | 0.4298 | 0.6744 | 0.8158 | 0.5250 | 0.8216 |

Source: `results/nhanes_final_test_results.csv`.

### Robustness (repeated stratified cross-validation)

`src/robustness_nhanes.py` runs 5-fold stratified CV repeated 5 times (25 folds) on the training split. A new pipeline (median imputation, scaling, model) is fitted in each fold. Values below are mean ± standard deviation over the 25 folds, in percent.

| Model | Accuracy | Precision | Recall | Specificity | F1 | AUC |
|---|---|---|---|---|---|---|
| XGBoost Core 7 | 86.22 ± 0.26 | 70.81 ± 1.82 | 32.91 ± 1.09 | 97.20 ± 0.23 | 44.92 ± 1.20 | 81.46 ± 0.67 |
| XGBoost Extended 10 | 86.32 ± 0.26 | 71.33 ± 1.84 | 33.26 ± 1.14 | 97.24 ± 0.24 | 45.35 ± 1.22 | 81.91 ± 0.71 |
| XGBoost All 25 | 86.53 ± 0.28 | 72.18 ± 2.00 | 34.39 ± 1.21 | 97.27 ± 0.27 | 46.57 ± 1.28 | 82.88 ± 0.56 |
| Voting All 25 | 85.07 ± 0.33 | 57.06 ± 1.18 | 50.99 ± 1.45 | 92.09 ± 0.39 | 53.84 ± 1.06 | 82.47 ± 0.54 |
| Stacking All 25 | 79.22 ± 0.37 | 43.18 ± 0.63 | 68.56 ± 1.32 | 81.42 ± 0.48 | 52.98 ± 0.72 | 82.68 ± 0.53 |

Source: `results/nhanes_repeated_cv_results.csv`. Note that the hyperparameters here (`n_estimators=200`, `max_depth=4`) differ from the 300/5 setting in the single-split scripts, so numbers across the two tables aren't directly comparable.

### Reading the results

- **Feature selection:** seven variables were picked by every method. Performance improves only slightly as features are added: ROC-AUC goes from 80.03% (5 features) to 82.45% (25 features) on the test set.
- **XGBoost:** it has the highest accuracy, precision and specificity, and with all 25 features the best AUC among the XGBoost configurations. Its recall is low (about 32–36%) at a 0.5 threshold, so it misses most CKD cases at that cutoff.
- **Voting and Stacking:** they trade precision and specificity for recall. Stacking has the highest recall (68.56% in repeated CV), and Voting has the highest F1 (53.84%) among the repeated-CV configurations. Their AUC is close to XGBoost's, so the ensembles aren't better overall. They give a different operating point.
- **Robustness:** standard deviations across folds are small (AUC SD below 0.8 points), so the numbers are stable across resamples of this dataset.
- **Choice of the 10-feature model:** it keeps most of the 25-feature AUC (81.15% vs 82.45% on test) while asking a user for only 10 inputs. This is a practical trade-off for the app, not a claim that it's clinically better.

Repeated cross-validation is internal validation on the same NHANES data. It isn't external or clinical validation.

> `results/robustness_results.csv` is an older output with near-perfect scores (accuracy ~99%, AUC ~0.999) and model names such as "Selected 15" that the current scripts don't produce. It is not used in this README, and the repeated-CV results above come from `nhanes_repeated_cv_results.csv`.

## Explainability

`src/shap_analysis_nhanes.py` loads the 25-feature XGBoost pipeline, applies its imputer and scaler, and runs `shap.TreeExplainer` on a random sample of 3,000 participants (`random_state=42`).

Mean absolute SHAP value, top 10 (`results/nhanes_shap_feature_importance.csv`):

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | age | 0.458 |
| 2 | bun | 0.346 |
| 3 | systolic_bp | 0.260 |
| 4 | uric_acid | 0.206 |
| 5 | diabetes | 0.175 |
| 6 | ast | 0.160 |
| 7 | sex | 0.139 |
| 8 | glucose | 0.132 |
| 9 | albumin | 0.130 |
| 10 | total_protein | 0.118 |

The XGBoost built-in importance (`results/nhanes_xgb_feature_importance.csv`) is computed separately in `final_analysis_nhanes.py`. Its top five are `diabetes` (0.172), `age` (0.159), `bun` (0.154), `systolic_bp` (0.064) and `glucose` (0.043).

The two measures answer different questions. SHAP importance is the average size of a feature's contribution to individual predictions. XGBoost's built-in importance reflects how the trees use a feature when splitting, so it can rank features differently (here, `diabetes` is first in tree importance but fifth in SHAP). Both agree that age, BUN, systolic blood pressure, diabetes and uric acid matter most. These are statements about model behaviour, not causal effects.

Plots: `results/nhanes_shap_summary.png`, `results/nhanes_shap_bar.png`, `results/nhanes_xgb_feature_importance.png`. Also in `results/`: ROC curves, precision-recall curves and confusion matrices for the three 25-feature models.

## Flask application

`app/app.py` loads `models/nhanes_final_xgboost_10_features.joblib`, which is a scikit-learn pipeline (median imputer → scaler → XGBoost). The page is titled **CKD Risk Assessment** with the subtitle *Machine Learning Based Chronic Kidney Disease Classification*.

Inputs (form field names match the model's feature names):

`age`, `bmi`, `waist`, `diabetes`, `systolic_bp`, `bun`, `glucose`, `uric_acid`, `albumin`, `total_cholesterol`

Output:

- **CKD Detected** or **No CKD Detected**, from the model's 0.5-threshold class prediction
- the model's predicted probability for the CKD class, in percent

Empty fields are sent as missing values and filled by the pipeline's imputer. The probability is the raw XGBoost output. Calibration wasn't evaluated, so it shouldn't be read as a clinical risk percentage.

The app runs locally only. There is no hosted demo.

## Project structure

```
CKD-Feature-Ensemble/
├── app/
│   ├── app.py
│   ├── static/
│   │   └── style.css
│   └── templates/
│       └── index.html
├── models/
│   ├── nhanes_final_xgboost_5_features.joblib
│   ├── nhanes_final_xgboost_7_features.joblib
│   ├── nhanes_final_xgboost_10_features.joblib      # used by the Flask app
│   └── nhanes_final_xgboost_all_25.joblib
├── notebooks/
│   └── 01_dataset_inspection.py
├── results/
│   ├── nhanes_feature_frequency.csv
│   ├── nhanes_feature_selection_results.csv
│   ├── nhanes_model_results.csv
│   ├── nhanes_5feature_cv_results.csv
│   ├── nhanes_5feature_test_results.csv
│   ├── nhanes_7feature_cv_results.csv
│   ├── nhanes_7feature_test_results.csv
│   ├── nhanes_10feature_cv_results.csv
│   ├── nhanes_10feature_test_results.csv
│   ├── nhanes_final_test_results.csv
│   ├── nhanes_repeated_cv_results.csv
│   ├── robustness_results.csv                       # older output, see note in Results
│   ├── nhanes_xgb_feature_importance.csv
│   ├── nhanes_shap_feature_importance.csv
│   ├── nhanes_roc_curves.png
│   ├── nhanes_precision_recall_curves.png
│   ├── confusion_matrix_xgboost_all_25.png
│   ├── confusion_matrix_voting_all_25.png
│   ├── confusion_matrix_stacking_all_25.png
│   ├── nhanes_xgb_feature_importance.png
│   ├── nhanes_shap_summary.png
│   └── nhanes_shap_bar.png
├── src/
│   ├── download_nhanes_demo.py
│   ├── download_nhanes_clinical.py
│   ├── download_nhanes_extra.py
│   ├── download_nhanes_bpq.py
│   ├── merge_nhanes.py
│   ├── add_bpq_to_merged.py
│   ├── create_nhanes_cohort.py
│   ├── create_ml_dataset.py
│   ├── preprocess_nhanes.py
│   ├── feature_selection_nhanes.py
│   ├── model_training_nhanes.py
│   ├── train_nhanes_5feature.py
│   ├── train_nhanes_7feature.py
│   ├── train_nhanes_10feature.py
│   ├── robustness_nhanes.py
│   ├── final_analysis_nhanes.py
│   ├── shap_analysis_nhanes.py
│   ├── project_integrity_test.py
│   └── test_nhanes_download.py
├── app_requirements.txt
├── .gitignore
└── README.md
```

Not tracked: `data/raw/`, `data/processed/`, and the Voting and Stacking model files (`nhanes_final_voting_all_25.joblib`, `nhanes_final_stacking_all_25.joblib`). All of these are regenerated by the scripts.

`notebooks/01_dataset_inspection.py` is an early exploration script for a different, smaller CKD file (`data/raw/chronic_kidney_disease.arff`). It isn't part of the NHANES pipeline.

## Installation

Requires Python 3 and Git. Commands are for Windows PowerShell.

```powershell
git clone https://github.com/harshchandel393-ai/CKD-Feature-Ensemble.git
cd CKD-Feature-Ensemble

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install -r app_requirements.txt
```

`app_requirements.txt` lists `flask`, `joblib`, `pandas`, `numpy`, `scikit-learn` and `xgboost`, which is enough to run the app. Versions are not pinned. If the saved models fail to load, a scikit-learn or XGBoost version mismatch with the one used for training is the likely cause.

The research scripts need a few more packages that are not in that file:

```powershell
pip install requests matplotlib shap
```

### Run the app

From the repository root:

```powershell
python app/app.py
```

Then open http://127.0.0.1:5000 in a browser.

## Reproducing the pipeline

All scripts use paths relative to the repository root (`data/...`, `models/...`, `results/...`), so run them from there. The order below follows the file dependencies in the code:

| Step | Script | What it does |
|---|---|---|
| 1 | `download_nhanes_demo.py` | Downloads DEMO files for the 10 cycles |
| 2 | `download_nhanes_clinical.py` | Downloads biochemistry and urine albumin/creatinine files |
| 3 | `download_nhanes_extra.py` | Downloads body measures, blood pressure, diabetes questionnaire |
| 4 | `download_nhanes_bpq.py` | Downloads the blood pressure questionnaire |
| 5 | `merge_nhanes.py` | Merges the files per cycle and stacks cycles into `nhanes_1999_2018_merged.csv` |
| 6 | `add_bpq_to_merged.py` | Optional. Adds BPQ columns and writes `..._merged_v2.csv`; not read by later steps |
| 7 | `create_nhanes_cohort.py` | Adults ≥ 20, computes eGFR and UACR, builds `ckd_target` |
| 8 | `create_ml_dataset.py` | Selects and renames predictors, cleans missing codes, drops target-defining columns |
| 9 | `preprocess_nhanes.py` | Stratified 80/20 split, median imputation, scaling |
| 10 | `feature_selection_nhanes.py` | Five feature-selection methods and frequency table |
| 11 | `model_training_nhanes.py` | Seven models on core 7 / extended 10 / all 25 |
| 12 | `train_nhanes_5feature.py`, `train_nhanes_7feature.py`, `train_nhanes_10feature.py` | Final XGBoost models for the smaller feature sets |
| 13 | `final_analysis_nhanes.py` | 25-feature XGBoost, Voting, Stacking; test metrics, curves, importance |
| 14 | `robustness_nhanes.py` | 5×5 repeated stratified CV |
| 15 | `shap_analysis_nhanes.py` | SHAP analysis of the 25-feature XGBoost model |

`project_integrity_test.py` checks that the expected data, model and result files exist and that the models load. `test_nhanes_download.py` is a download check.

## Limitations

1. **NHANES is cross-sectional.** Predictors and the CKD label are measured at the same visit. The models classify existing CKD status as defined here, and don't predict future onset, progression or kidney failure.
2. **No external validation.** All results come from internal splits of NHANES.
3. **No prospective or clinical validation.**
4. **Not a diagnostic system.** The app is a demonstration of the model.
5. **Probabilities aren't calibrated.** Calibration wasn't evaluated, so the output shouldn't be read as clinical risk.
6. **Missing values are imputed** with the training-set median, which can be a poor guess for individuals with unusual profiles.
7. **Class imbalance.** About 17% of participants are CKD. At a 0.5 threshold, XGBoost has high specificity but recall around 32–36%, and the ensembles reach higher recall by accepting more false positives.
8. **Population dependence.** NHANES is a US survey. Performance on other populations or clinical settings is unknown.
9. **The target definition shapes the model.** The label is eGFR/UACR-based, with a single-visit measurement and a 2009-style equation, so the model learns that rule and not a clinical diagnosis.
10. **Laboratory and clinical measurements are inputs.** Several predictors (BUN, uric acid, albumin) are blood tests, so the model isn't a questionnaire-only screening tool. Even with target-defining variables removed, some predictors are physiologically related to kidney function.

## Future work

Not implemented yet:

- external validation on a separate dataset
- probability calibration (e.g. reliability curves, Platt or isotonic scaling)
- decision-threshold optimisation for different recall/precision goals
- subgroup analysis by age, sex and diabetes status
- longitudinal datasets for actual progression modelling
- prospective validation
- more detailed per-prediction explanations in the app
- deployment improvements (pinned dependencies, production server)

## Disclaimer

This project is intended for research and educational purposes only. It is not a medical diagnostic system and should not be used to make clinical decisions.

## Author

**Harsh Chandel**
B.Tech CSE (AI & ML)
GitHub: [harshchandel393-ai](https://github.com/harshchandel393-ai)
