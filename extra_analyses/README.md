# Robustness Experiments — Label Noise & Subsampling (RF baselines)

This repo contains **two parallel pipelines** for robustness studies built on the original prep code and Random‑Forest baselines.

* **`ML_RF_noise/`** — label‑noise sweeps (`noiXX` tags)
* **`ML_RF_subsample/`** — sequence‑level subsampling sweeps (`subXX` tags)

Both keep the existing **scramble** mechanism (`scr` tag), compute **μ,σ on clean train** once, and keep the **test set clean**.

---

## Repo layout

```
ML_RF_noise/
  Data/torch_prep_kfold_noise.py   # prep with --noise_levels
  run_model_rf.py                  # expects --noise_tags (noi00, noi05, …)

ML_RF_subsample/
  Data/torch_prep_kfold.py         # prep with --subsample_fracs
  run_model_rf.py                  # expects --subsample_tags (sub100, sub75, …)
  
ML_RF_DNA/
  Data/torch_prep_kfold.py         # uses DNA shape inputs
  run_model_rf.py                  
```

---

## Key changes vs. original

* **Noise pipeline** (`ML_RF_noise`)

  * New flag: `--noise_levels 0.00 0.05 0.10 ...`

    * **reg**: add Gaussian noise to labels (σ × level)
    * **bin/mclass**: flip a `level` fraction of **sequences**
  * Filenames include `noiXX` (e.g., `noi05` = 5%).

* **Subsample pipeline** (`ML_RF_subsample`)

  * New flag: `--subsample_fracs 1.0 0.75 0.50 ...`
  * Keeps that fraction of **unique sequences** to form smaller training sets
  * Filenames include `subXX` (e.g., `sub50` = 50%).

* **DNA shape pipelines**
 * Choice of inputs `Inputs/rawdat_DNA.csv` (DNA shape only) or `Inputs/rawdat_MMGBSA_DNA.csv` (DNA shape + MMGBSA features)
 * Minor modifications to the scripts to be compatible with the input formats
---

## Workflow

### A) Noise experiments (in `ML_RF_noise/`)

1. **Initial split** (`--initial_split`) → `..._trn_final.csv`, `..._tst_preprocess.csv`
2. **Process train** (`--process train`) → standardize using clean train; build K‑folds for each `--noise_levels` (adds `noiXX`)
3. **Process test** (`--process test`) → same preprocessing; **no noise**; write `..._tst_final.csv`
4. **RF**: train/eval using `run_model_rf.py` with matching `--noise_tags`

### B) Subsample experiments (in `ML_RF_subsample/`)

1. **Initial split** (`--initial_split`) → `..._trn_final.csv`, `..._tst_preprocess.csv`
2. **Process train** (`--process train`) → standardize; for each `--subsample_fracs` build K‑folds (adds `subXX`)
3. **Process test** (`--process test`) → same preprocessing; write `..._tst_final.csv`
4. **RF**: train/eval using `run_model_rf.py` with matching `--subsample_tags`

---

## Example commands

Provided in `ML_RF_noise/run_ML.sh`, `ML_RF_subsample/run_ML.sh` and `ML_RD_DNA/run_ML.sh`

---

## Output naming

* **Common parts**: `<prefix>_<mtype>_scrXX_..._{trn|val}_{rep}_{fold}.csv`  and  `..._tst_final.csv`
* **Noise**: `..._noi05_...`  (levels ×100)
* **Subsample**: `..._sub50_...`  (percent kept)
* RF checkpoints: `Model/rf_fold_{rep}_{fold}_{mtype}_scrXX_{noi|sub}.pkl`
* Metrics: `final_metrics_{mtype}_{trn|tst}_scrXX_{noi|sub}.csv`
* Predictions: `predictions[_test]_{mtype}_scrXX_{noi|sub}.csv`

---

## Notes

* **μ,σ** are computed **once on clean averaged train** and reused across noise/subsample conditions.
* **Test set is never noised** and is shared across conditions of the same `scr` tag.
* For classification, all operations (scramble, noise flips, subsampling) are **sequence‑level** to keep labels consistent per sequence.
* Ensure `--noise_tags` or `--subsample_tags` match the filenames generated in each pipeline.
