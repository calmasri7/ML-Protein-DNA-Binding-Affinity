# Machine Learning Models Directory (`ML_models`)

This directory contains scripts, input data, and outputs for training, evaluating, and optimizing machine learning (ML) models: Random Forest (RF), Linear Regression (REG), Support Vector Machine (SVM), and Neural Networks (NN).

## Directory Structure

The main directory (`ML_models`) is organized as follows:

```
ML_models/
├── ML_RF/
│   ├── Data/
│   │   ├── gbsa_[model]_scr[frac]_trn_[rep]_[fold].csv
│   │   ├── gbsa_[model]_scr[frac]_val_[rep]_[fold].csv
│   │   ├── gbsa_[model]_scr[frac]_trn_final.csv
│   │   ├── gbsa_[model]_scr[frac]_tst_final.csv
│   │   ├── gbsa_[model]_scr[frac]_tst_preprocess.csv
│   │   ├── gbsa_[model]_train_stats.csv
│   │   └── torch_prep_kfold.py
│   │
│   ├── Model/
│   │   └── rf_fold_[rep]_[fold]_[model]_scr[frac].pkl
│   │
│   ├── run_model.py
│   ├── hyperopt.py
│   ├── run_hyperopt.sh
│   ├── run_ML.sh
│   │
│   ├── Random_Forest_metrics.csv
│   ├── summary_metrics.csv
│   ├── hyperopt_results_[model].csv
│   ├── best_hyperparams_[model].json
│   │
│   ├── final_metrics_[model]_trn_scr[frac].csv
│   ├── final_metrics_[model]_tst_scr[frac].csv
│   ├── predictions_[model]_final_avg_scr[frac].csv
│   ├── predictions_[model]_scr[frac]_rep[rep]_fold[fold].csv
│   └── predictions_test_[model]_scr[frac].csv
│
├── ML_REG/
│   ├── run_model.py
│   └── hyperopt.py
│
├── ML_SVM/
│   ├── run_model.py
│   └── hyperopt.py
│
└── ML_NN/
    ├── run_model.py
    └── hyperopt.py
```

Replace placeholders as follows:

* `[model]`: `reg`, `bin`, or `mclass`
* `[frac]`: Scramble fraction (e.g., `0p00` for 0.0)
* `[rep]`: Repetition number
* `[fold]`: Fold number

*Note:* Full datasets and outputs are in `ML_RF`. `ML_REG`, `ML_SVM`, and `ML_NN` directories contain only their respective scripts (`run_model.py` and `hyperopt.py`).

## Workflow Overview

### Phase 1: Initial Data Preparation

Run initial train/test split using `Data/torch_prep_kfold.py`, executed via:

* `run_hyperopt.sh`
* `run_ML.sh`

### Phase 2: Hyperparameter Optimization

Execute Bayesian hyperparameter tuning:

```bash
sbatch run_hyperopt.sh [reg|bin|mclass]
```

This script:

* Prepares training data
* Runs hyperparameter optimization (`hyperopt.py`)
* Outputs optimal parameters (`best_hyperparams_[model].json`)
* Logs results (`hyperopt_results_[model].csv`)

### Phase 3: Model Training and Evaluation

Perform model training and evaluation:

```bash
sbatch run_ML.sh [reg|bin|mclass] "[scramble_fractions]"
```

Example:

```bash
sbatch run_ML.sh reg "0.0 0.25 1.0"
```

This script:

* Prepares datasets with optimal parameters
* Trains models (`run_model.py`) for each repeat and fold
* Evaluates models
* Aggregates predictions and metrics

## Key Scripts

* **`Data/torch_prep_kfold.py`**: Dataset preparation and splitting.
* **`hyperopt.py`**: Hyperparameter optimization.
* **`run_model.py`**: Model training and evaluation.
* **`run_hyperopt.sh`**: Submission script for hyperparameter tuning.
* **`run_ML.sh`**: Submission script for training and evaluation.

## Script-Specific Inputs and Outputs

### Hyperparameter Optimization (`run_hyperopt.sh`)

**Inputs:**

* Raw CSV data (`exp_data_all.csv`, `rawdat.csv`)

**Outputs:**

* Optimal hyperparameters (`best_hyperparams_[model].json`)
* Hyperparameter tuning results (`hyperopt_results_[model].csv`)

### Model Training and Evaluation (`run_ML.sh`)

**Inputs:**

* Optimal hyperparameters JSON (`best_hyperparams_[model].json`)
* Prepared CSV datasets (training, validation, test)

**Outputs:**

* Trained models (`Model/rf_fold_[rep]_[fold]_[model]_scr[frac].pkl`)
* Final metrics:

  * Training (`final_metrics_[model]_trn_scr[frac].csv`)
  * Test (`final_metrics_[model]_tst_scr[frac].csv`)
* Predictions:

  * Aggregated (`predictions_[model]_final_avg_scr[frac].csv`)
  * Fold-level (`predictions_[model]_scr[frac]_rep[rep]_fold[fold].csv`)
  * Final test set (`predictions_test_[model]_scr[frac].csv`)
* Summary metrics (`summary_metrics.csv`, `Random_Forest_metrics.csv`)

## Sample Outputs

Sample outputs in `ML_RF` for scramble fraction `0p00`, repetition `0`, and fold `0` include:

* CSV datasets (e.g., `Data/gbsa_[model]_scr0p00_trn_0_0.csv`)
* Metrics (e.g., `final_metrics_[model]_trn_scr0p00.csv`)
* Predictions (e.g., `predictions_[model]_scr0p00_rep0_fold0.csv`)
* Model files (`Model/rf_fold_0_0_[model]_scr0p00.pkl`)
